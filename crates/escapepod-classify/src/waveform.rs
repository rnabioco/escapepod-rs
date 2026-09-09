// SPDX-License-Identifier: MIT

//! The windowed variant: BAM scan, chunk assembly, and per-read scoring.
//!
//! A `waveform_model` bundle reads a *signal window* rather than a column
//! vector, and almost nothing above the POD5 index is shared with the
//! column-scoring path:
//!
//! | | column variants (`gbm`, `feature_model`) | windowed variant |
//! |---|---|---|
//! | map | move table, in raw signal coordinates | move table walked through the CIGAR, in the aligned reference's coordinates |
//! | anchor | junction query base, counted along the query | a reference base, `motif_start + motif_offset` |
//! | spans | straight from the move table | banded-DP refined against expected k-mer levels |
//! | frame | voted per run from the data | declared by the bundle (`reverse_signal`) |
//! | input | one flat `Vec<f64>` of selected columns | three tensors, assembled by [`escapepod_signal::chunk`] |
//!
//! So this is a second pipeline over the same two files, not a second scorer
//! on the first one. What it does *not* do is own any of the assembly: every
//! rule about what the model sees — the window, the channels, the refinement,
//! the k-mer context — lives in `escapepod-signal`, and everything here only
//! marshals a BAM record into the shape that crate takes.
//!
//! # Why the frame is declared rather than voted
//!
//! The column path votes on whether the move table indexes time-ordered or
//! reversed signal, because nothing in its bundle says. This one's bundle
//! does: `reverse_signal` is copied from the corpus's `prepare_config.json`,
//! so it is the frame the model was *trained* in rather than the frame this
//! run happens to be in. Those are the same thing whenever the data is the
//! data the model is for, and when they differ the vote would be the wrong
//! answer confidently — the window would be mirrored to match the run instead
//! of matching the model.

use std::collections::HashMap;

use anyhow::{Context, Result};
use rayon::prelude::*;
use uuid::Uuid;

use escapepod_signal::chunk::{
    Anchor, Chunk, LevelModel, ProcessConfig, ReadInputs, cut_chunk, process_read, read_rows,
};
use escapepod_signal::mapping::{CigarKind, CigarOp};

use noodles_bam as bam;
use noodles_bgzf as bgzf;
use noodles_sam::alignment::RecordBuf;
use noodles_sam::alignment::record::cigar::op::Kind;
use noodles_sam::alignment::record::data::field::Tag;

use crate::anchor::SkipReason;
use crate::bundle::WaveformSpec;
use crate::geometry::RefGeometry;
use crate::recipe::KmerLevels;

// Scoring needs the graph; the scan and the assembly above do not, and are
// what a corpus builder or a parity test links.
#[cfg(feature = "waveform-onnx")]
use crate::bundle::{AbstainRule, ChargingBundle};
#[cfg(feature = "waveform-onnx")]
use crate::pipeline::{ClassifyStats, NoCall, NoCallReason, Pod5Index, ReadCall, signal_adc};

/// One read anchored to a reference base, with everything the assembly needs.
///
/// Deliberately keeps the *move table* rather than the map it expands to: the
/// map is one `i64` per base and the moves are one byte per stride, so on a
/// 5-stride basecall the compact form is eight times smaller — and this is
/// held for every read in the BAM at once.
#[derive(Debug, Clone)]
pub struct WaveformRead {
    pub read_id: Uuid,
    pub reference: String,
    pub mapq: u8,
    /// `ns`: samples the basecaller saw.
    pub ns: i64,
    /// `ts`: samples trimmed from the front.
    pub ts: i64,
    pub stride: u32,
    pub moves: Vec<u8>,
    pub cigar: Vec<CigarOp>,
    /// 0-based reference start of the alignment.
    pub ref_start: usize,
    /// 0-based, exclusive reference end of the alignment.
    pub ref_end: usize,
    /// The anchor base, in the aligned reference slice's own coordinates.
    pub base_index: i64,
    /// The aligned reference, from this read's `MD` tag — see
    /// [`reference_from_md`] for why it is not sliced from the FASTA.
    pub ref_seq: Vec<u8>,
}

/// The aligned reference, rebuilt from the read's `MD` tag.
///
/// **Not a FASTA slice, and the difference is not cosmetic.** The corpus this
/// model was trained on takes its reference from `pysam`'s
/// `get_reference_sequence()`, which reconstructs the aligned span from `MD` +
/// the query rather than reading a FASTA. The two disagree wherever the FASTA
/// carries an ambiguity code: every reference in the shipped tRNA panel holds
/// exactly one `N`, and because levels are looked up per 9-mer, one `N` makes
/// **nine** consecutive k-mers unknown — so `extract_levels` leaves a nine-base
/// run of zeros that the corpus does not have. The banded DP hits that flat
/// region and walks a different path from there on, which moved boundaries by
/// a sample throughout the read and cost 169 of 256 chunks their bit-exactness
/// (rnabioco/escapepod-rs#306). Reading the FASTA instead scores every read and
/// errors on none of them, which is why this is spelled out rather than left to
/// whoever next sees a `reference: &str` parameter and assumes it is the source.
///
/// Mismatched positions are lowercased, as `pysam` does. Nothing downstream is
/// case-sensitive — [`escapepod_signal::resquiggle::extract_levels`] uppercases
/// and maps `U -> T` — but the corpus stores the sequence with that case, so
/// keeping it makes the two directly comparable.
///
/// `MD` walks *reference* positions: a run length means that many bases match
/// the query, a letter is the reference base at a mismatch, and `^` introduces
/// deleted reference bases that the query never carried.
pub fn reference_from_md(md: &str, query: &[u8], cigar: &[CigarOp]) -> Option<Vec<u8>> {
    // Reference-consuming positions, seeded with the query base where the two
    // are aligned and left blank across deletions for `MD` to fill.
    let mut refbuf: Vec<u8> = Vec::new();
    let mut q = 0usize;
    for op in cigar {
        let len = op.len as usize;
        match op.kind {
            CigarKind::Match | CigarKind::SequenceMatch | CigarKind::SequenceMismatch => {
                for _ in 0..len {
                    refbuf.push(*query.get(q)?);
                    q += 1;
                }
            }
            CigarKind::Insertion | CigarKind::SoftClip => q += len,
            CigarKind::Deletion | CigarKind::Skip => refbuf.extend(std::iter::repeat_n(b'N', len)),
            CigarKind::HardClip | CigarKind::Pad => {}
        }
    }

    let mut i = 0usize;
    let mut it = md.bytes().peekable();
    while let Some(c) = it.next() {
        if c.is_ascii_digit() {
            let mut n = (c - b'0') as usize;
            while let Some(d) = it.peek().copied().filter(u8::is_ascii_digit) {
                n = n * 10 + (d - b'0') as usize;
                it.next();
            }
            i += n;
        } else if c == b'^' {
            // Deleted reference bases, which `refbuf` is holding space for.
            while let Some(d) = it.peek().copied().filter(u8::is_ascii_alphabetic) {
                *refbuf.get_mut(i)? = d;
                i += 1;
                it.next();
            }
        } else if c.is_ascii_alphabetic() {
            *refbuf.get_mut(i)? = c.to_ascii_lowercase();
            i += 1;
        }
    }
    (i == refbuf.len()).then_some(refbuf)
}

/// Result of scanning one BAM for the windowed variant.
#[derive(Debug, Default)]
pub struct WaveformScan {
    /// One entry per read, best alignment (highest mapq) wins.
    pub anchored: HashMap<Uuid, WaveformRead>,
    pub records_scanned: u64,
    pub skips: HashMap<SkipReason, u64>,
}

/// A BAM CIGAR op as an [`escapepod_signal`] one.
fn cigar_kind(kind: Kind) -> CigarKind {
    match kind {
        Kind::Match => CigarKind::Match,
        Kind::Insertion => CigarKind::Insertion,
        Kind::Deletion => CigarKind::Deletion,
        Kind::Skip => CigarKind::Skip,
        Kind::SoftClip => CigarKind::SoftClip,
        Kind::HardClip => CigarKind::HardClip,
        Kind::Pad => CigarKind::Pad,
        Kind::SequenceMatch => CigarKind::SequenceMatch,
        Kind::SequenceMismatch => CigarKind::SequenceMismatch,
    }
}

/// Anchor one BAM record on the reference base the bundle's motif offset names.
fn scan_record(
    record: &RecordBuf,
    ref_name: &str,
    geometry: &HashMap<String, RefGeometry>,
    motif_offset: usize,
    min_mapq: u8,
) -> std::result::Result<WaveformRead, SkipReason> {
    let flags = record.flags();
    if flags.is_unmapped()
        || flags.is_reverse_complemented()
        || flags.is_secondary()
        || flags.is_supplementary()
    {
        return Err(SkipReason::Filtered);
    }
    let mapq = record.mapping_quality().map(u8::from).unwrap_or(255);
    if mapq < min_mapq {
        return Err(SkipReason::LowMapq);
    }
    let Some(g) = geometry.get(ref_name) else {
        return Err(SkipReason::NoGeometry);
    };
    let Some(start) = record.alignment_start() else {
        return Err(SkipReason::Filtered);
    };
    let ref_start = start.get() - 1; // 0-based

    let mut cigar = Vec::new();
    let mut ref_span = 0usize;
    for op in record.cigar().as_ref() {
        let kind = cigar_kind(op.kind());
        if kind.consumes_reference() {
            ref_span += op.len();
        }
        cigar.push(CigarOp::new(kind, op.len() as u32));
    }
    let ref_end = ref_start + ref_span;

    // The reference base the window is anchored on. `motif_start` is the
    // motif's own origin, which is *not* where the common arm begins — the
    // arm is what makes the motif unique, and a model is free to anchor
    // earlier inside it (this one does, at +2 against the arm's +3).
    let anchor_ref = g.motif_start + motif_offset;
    if anchor_ref < ref_start || anchor_ref >= ref_end {
        return Err(SkipReason::Unanchored);
    }

    let Some((stride, moves, ns, ts)) = move_table(record) else {
        return Err(SkipReason::NoTags);
    };
    let Some(read_id) = record
        .name()
        .and_then(|n| std::str::from_utf8(n.as_ref()).ok())
        .and_then(|s| escapepod_signal::parse_uuid_flexible(s).ok())
    else {
        return Err(SkipReason::BadName);
    };

    // The reference this read is scored against comes from its own `MD` tag,
    // never from the FASTA — see `reference_from_md`. Refused rather than
    // fallen back on, because the fallback is the defect: it scores every read
    // and errors on none of them while feeding the DP a nine-base run of zero
    // levels wherever the FASTA carries an `N`.
    let md =
        crate::bam_tags::string_tag(record, Tag::new(b'M', b'D')).ok_or(SkipReason::NoMdTag)?;
    let query: Vec<u8> = record.sequence().as_ref().to_vec();
    let ref_seq = reference_from_md(&md, &query, &cigar).ok_or(SkipReason::NoMdTag)?;
    if ref_seq.len() != ref_span {
        return Err(SkipReason::NoMdTag);
    }

    Ok(WaveformRead {
        read_id,
        ref_seq,
        reference: ref_name.to_string(),
        mapq,
        ns,
        ts,
        stride: stride as u32,
        moves,
        cigar,
        ref_start,
        ref_end,
        base_index: (anchor_ref - ref_start) as i64,
    })
}

/// `(stride, moves, ns, ts)` from a record's `mv`/`ns`/`ts` tags.
fn move_table(record: &RecordBuf) -> Option<(usize, Vec<u8>, i64, i64)> {
    use noodles_sam::alignment::record::data::field::Tag;
    let (stride, moves) = crate::bam_tags::parse_mv_tag(record).ok()?;
    let ns = crate::bam_tags::int_tag(record, Tag::new(b'n', b's'))?;
    let ts = crate::bam_tags::int_tag(record, Tag::new(b't', b's')).unwrap_or(0);
    Some((stride, moves, ns, ts))
}

/// Scan an aligned BAM into reference-anchored reads.
///
/// The batching and the dedup are [`crate::scan_bam`]'s, for the same reasons:
/// anchoring is per-record CPU and decoding is a single cursor, so records are
/// decoded serially and anchored in parallel, and the fold back is serial and
/// in file order so the best-mapq dedup is deterministic.
pub fn scan_bam(
    bam_path: &std::path::Path,
    geometry: &HashMap<String, RefGeometry>,
    motif_offset: usize,
    min_mapq: u8,
) -> Result<WaveformScan> {
    let file = std::fs::File::open(bam_path)
        .with_context(|| format!("cannot open BAM {}", bam_path.display()))?;
    let decoder =
        bgzf::io::MultithreadedReader::with_worker_count(crate::pipeline::bgzf_workers(), file);
    let mut reader = bam::io::Reader::from(decoder);
    let header = reader.read_header()?;
    let ref_names: Vec<String> = header
        .reference_sequences()
        .keys()
        .map(|k| k.to_string())
        .collect();

    let mut scan = WaveformScan::default();
    const BATCH: usize = 8192;
    let mut batch: Vec<RecordBuf> = vec![RecordBuf::default(); BATCH];
    let mut outcomes: Vec<std::result::Result<WaveformRead, SkipReason>> =
        Vec::with_capacity(BATCH);
    loop {
        let mut n = 0;
        while n < BATCH {
            if reader.read_record_buf(&header, &mut batch[n])? == 0 {
                break;
            }
            n += 1;
        }
        if n == 0 {
            break;
        }
        scan.records_scanned += n as u64;

        outcomes.clear();
        batch[..n]
            .par_iter()
            .map(|record| {
                match record
                    .reference_sequence_id()
                    .and_then(|id| ref_names.get(id))
                {
                    Some(name) => scan_record(record, name, geometry, motif_offset, min_mapq),
                    None => Err(SkipReason::Filtered),
                }
            })
            .collect_into_vec(&mut outcomes);

        for outcome in outcomes.drain(..) {
            match outcome {
                Ok(read) => match scan.anchored.entry(read.read_id) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        if read.mapq > e.get().mapq {
                            e.insert(read);
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(read);
                    }
                },
                Err(reason) => *scan.skips.entry(reason).or_default() += 1,
            }
        }
        if n < BATCH {
            break;
        }
    }
    Ok(scan)
}

/// Assemble one read's chunk, or `None` if the anchor does not resolve.
///
/// Takes the k-mer levels rather than the whole bundle, for the reason
/// [`crate::recipe::FeatureRecipe`] exists: the corpus builder that computes
/// these same tensors has no weights to hand over, and forcing it to invent a
/// bundle is how a second, divergent definition gets written.
///
/// Split out from [`classify_reads`] because it is the half that has to match
/// the corpus builder *bit for bit* and needs no weights to check: the parity
/// tests compare these three tensors against the corpus's own arrays, which is
/// the only way to tell an assembly error from a graph error.
pub fn assemble_chunk(
    kmer: Option<&KmerLevels>,
    spec: &WaveformSpec,
    read: &WaveformRead,
    raw: &[i16],
) -> Option<Chunk> {
    let levels = kmer.map(|k| LevelModel {
        table: &k.map,
        kmer_len: k.k,
        center_idx: k.center_idx,
    });
    let cfg = ProcessConfig {
        reverse_signal: spec.reverse_signal,
        normalization: spec.normalization,
        levels,
        refine: spec.refine,
    };
    // NOT a FASTA slice: the sequence the model was trained against is the
    // read's own `MD` reconstruction, carried on the read since the scan. The
    // FASTA is still what the *motif search* reads to place the anchor
    // (`preprocessing.motif_reference: "fasta"`), but that happens in
    // `scan_bam`, so this function no longer needs one at all.
    let sequence = read.ref_seq.as_slice();
    let processed = process_read(
        ReadInputs {
            raw,
            moves: &read.moves,
            stride: read.stride,
            trim: read.ts,
            num_samples: read.ns as u64,
        },
        Anchor::Reference {
            sequence,
            cigar: &read.cigar,
        },
        &cfg,
    )?;
    let rows = read_rows(&processed, &spec.chunk);
    cut_chunk(&processed, &rows, &spec.chunk, read.base_index)
}

/// Whether "no chunk" is a refusal the bundle asked for, or simply a read
/// this runtime could not score. The two are the same event and different
/// reports; see [`NoCallReason::NoChunk`].
#[cfg(feature = "waveform-onnx")]
fn no_chunk_reason(bundle: &ChargingBundle) -> NoCallReason {
    let declared = bundle
        .abstain
        .as_ref()
        .is_some_and(|a| a.kind == AbstainRule::NoChunk);
    if declared {
        NoCallReason::Abstained(AbstainRule::NoChunk)
    } else {
        NoCallReason::NoChunk
    }
}

#[cfg(feature = "waveform-onnx")]
fn no_call(read: &WaveformRead, reason: NoCallReason) -> NoCall {
    NoCall {
        read_id: read.read_id,
        reference: read.reference.clone(),
        reason,
    }
}

/// One BCE logit, of whichever class the bundle says, turned into the
/// reported call. `cl` is always `P(classes[1])`, so a positive class of 0
/// inverts here and nowhere else.
#[cfg(feature = "waveform-onnx")]
fn call_from_logit(read: &WaveformRead, spec: &WaveformSpec, logit: f64) -> ReadCall {
    let s = 1.0 / (1.0 + (-logit).exp());
    let p = if spec.positive_class == 1 { s } else { 1.0 - s };
    ReadCall {
        read_id: read.read_id,
        reference: read.reference.clone(),
        p,
        cl: crate::cl_from_probability(p),
    }
}

#[cfg(feature = "waveform-onnx")]
fn record_no_call(stats: &mut ClassifyStats, n: NoCall) {
    match n.reason {
        NoCallReason::NoSignal => stats.no_signal += 1,
        NoCallReason::NsMismatch => stats.ns_mismatch += 1,
        NoCallReason::NoChunk => stats.no_chunk += 1,
        NoCallReason::Abstained(_) => stats.abstained += 1,
    }
    stats.no_calls.push(n);
}

/// Anchored reads, in POD5 storage order.
///
/// The reads arrive in BAM order through a `HashMap`, and reading them that
/// way walks the whole POD5 pseudo-randomly. See [`Pod5Index::storage_key`].
#[cfg(feature = "waveform-onnx")]
fn sorted_reads<'a>(
    anchored: &'a HashMap<Uuid, WaveformRead>,
    pod5: &Pod5Index,
) -> Vec<&'a WaveformRead> {
    let mut reads: Vec<&WaveformRead> = anchored.values().collect();
    reads.sort_by_cached_key(|r| pod5.storage_key(&r.read_id));
    reads
}

/// Classify every anchored read with signal, in parallel, on the CPU scorer.
///
/// Returns calls sorted by read id (deterministic output order) plus the
/// no-call tallies — the same contract as [`crate::classify_reads`], so a
/// caller only has to pick the scan.
///
/// Takes no reference FASTA: each read carries the aligned reference it is
/// scored against, rebuilt from its own `MD` tag during the scan (see
/// [`reference_from_md`]). The FASTA is still read, once, to place the motif —
/// that happens in [`scan_bam`] via the geometry.
///
/// Gated on the graph runtime; the scan and [`assemble_chunk`] above are not,
/// so the assembly can be tested — and a corpus built — without one.
#[cfg(feature = "waveform-onnx")]
pub fn classify_reads(
    bundle: &ChargingBundle,
    anchored: &HashMap<Uuid, WaveformRead>,
    pod5: &Pod5Index,
) -> Result<(Vec<ReadCall>, ClassifyStats)> {
    let spec = bundle.waveform_spec()?;
    let net = bundle.waveform_net()?;
    let extractors = pod5.extractors()?;
    let no_chunk = no_chunk_reason(bundle);

    enum Outcome {
        Call(ReadCall),
        None(NoCall),
    }
    let reads = sorted_reads(anchored, pod5);

    let outcomes: Vec<Outcome> = reads
        .par_iter()
        .map(|read| {
            let Some(info) = pod5.reads().get(&read.read_id) else {
                return Ok(Outcome::None(no_call(read, NoCallReason::NoSignal)));
            };
            let raw = signal_adc(info, &extractors)?;
            if raw.len() as i64 != read.ns {
                return Ok(Outcome::None(no_call(read, NoCallReason::NsMismatch)));
            }
            let Some(chunk) = assemble_chunk(bundle.kmer.as_ref(), spec, read, &raw) else {
                return Ok(Outcome::None(no_call(read, no_chunk)));
            };
            let logit = net.logit(&chunk, spec)?;
            Ok(Outcome::Call(call_from_logit(read, spec, logit)))
        })
        .collect::<Result<_>>()?;

    let mut stats = ClassifyStats::default();
    let mut calls = Vec::with_capacity(outcomes.len());
    for o in outcomes {
        match o {
            Outcome::Call(c) => calls.push(c),
            Outcome::None(n) => record_no_call(&mut stats, n),
        }
    }
    calls.sort_by_key(|c| c.read_id);
    stats.no_calls.sort_by_key(|n| n.read_id);
    Ok((calls, stats))
}

/// Classify every anchored read on the GPU-batched scorer, falling back to
/// the CPU scorer for whatever does not fill a full batch.
///
/// Same contract as [`classify_reads`] (sorted calls, the same no-call
/// tallies), and shares its per-read preparation (signal fetch, chunk
/// assembly) — but not its parallel *scoring*: a tract-cuda plan's batch is
/// fixed at compile time, so reads are collected into groups of exactly
/// `gpu.batch_size()` and scored with one sequential GPU call per group,
/// rather than one call per read under `rayon`.
///
/// Reads are processed in memory-bounded superbatches rather than all at
/// once: at this bundle's geometry a [`Chunk`] is tens of KB, so holding
/// every chunk of a production run in memory at once would be tens of GB.
/// `SUPERBATCH` matches `commands/demux/detect.rs`'s `GPU_BLOCK`, the same
/// shape of bound for the same reason.
///
/// The last, short group of each superbatch (and the whole run, if it has
/// fewer reads than one GPU batch) is scored on the CPU rather than padded
/// through a full GPU call — a tract-cuda plan's cost is per call, not per
/// valid row, so padding a mostly-empty batch would cost the same as a full
/// one for a handful of reads. This is what makes "GPU is never worse than
/// CPU" true; see [`crate::waveform_net_gpu`]'s module doc for the numbers.
///
/// # Pipelining (#351)
///
/// A prior version prepared a whole superbatch on CPU (`par_iter` over all
/// 16,384 reads), fully serially, before issuing a single GPU call — so the
/// GPU sat idle for the entire prep phase and the CPU sat idle for the
/// entire scoring phase, alternating in bursts. Confirmed live on a
/// production node (`nvidia-smi dmon`, 4-core `--threads 4` job): 331 of 381
/// one-second samples flat at `sm=0`, a 13.1% duty cycle. This version
/// overlaps the two: a dedicated GPU consumer thread scores each group as it
/// becomes ready while the producer (this function's caller thread, using
/// `rayon`'s pool the same as before) prepares the next one, connected by a
/// bounded channel — the same producer/CPU-pool-thread ->
/// bounded-channel -> single-GPU-consumer-thread shape as
/// `commands/demux/detect.rs`'s GPU path, just with the channel carrying
/// exactly-`batch`-sized groups instead of whole superbatches (a tract-cuda
/// plan's batch dimension is fixed, unlike `detect.rs`'s CNN which accepts
/// one grouped call per length bucket).
///
/// Reads are prepared `prep_chunk` at a time — a size independent of
/// `batch`, see its doc below — so a group can be handed to the GPU as soon
/// as it fills, but a `carry` buffer folds the occasional no-call read
/// across those raw sub-chunks. Which reads land in which GPU group, and in
/// what order, is unaffected by any of this: this is a scheduling change,
/// not a grouping change, and the two are bit-identical by construction —
/// verified on the same real dataset at every point of the tuning sweep
/// below (BAM and TSV output, byte-for-byte).
///
/// End to end on that dataset (55,446 anchored reads, real POD5 + aligned
/// BAM + `charging_tcn_sup6_rna004@v0.1.0`, A30, `--threads 4`): 364.7s
/// fully-serial -> 203.3s pipelined at the tuned defaults below, ~1.8x. Most
/// of that came from `groups_in_flight` and `prep_chunk`, not from the
/// overlap alone — see their doc comments for the sweep that found them; the
/// first cut of this fix (both at their most conservative plausible values)
/// only reached 315.2s.
#[cfg(feature = "cuda")]
pub fn classify_reads_gpu(
    bundle: &ChargingBundle,
    anchored: &HashMap<Uuid, WaveformRead>,
    pod5: &Pod5Index,
    gpu: &crate::waveform_net_gpu::WaveformNetGpu,
) -> Result<(Vec<ReadCall>, ClassifyStats)> {
    let spec = bundle.waveform_spec()?;
    let cpu_fallback = bundle.waveform_net()?;
    let extractors = pod5.extractors()?;
    let no_chunk = no_chunk_reason(bundle);
    let reads = sorted_reads(anchored, pod5);
    let batch = gpu.batch_size();

    enum Prepared<'r> {
        Ready(&'r WaveformRead, Chunk),
        None(NoCall),
    }

    const SUPERBATCH: usize = 16_384;
    // Groups in flight through the channel. `detect.rs`'s GPU_BLOCK channel
    // (a different pipeline entirely — see the doc above) uses depth 2, and
    // that was this function's first guess too; it measured badly here. At
    // depth 2 the producer blocks on `tx.send` as soon as one group is
    // queued and the consumer hasn't drained it yet, and on a 4-core node
    // that block is *idle* rather than useful work — CPU never rose above
    // ~194% of the 4 allocated cores, and wall time barely improved on the
    // fully-serial version it replaced (315s vs 365s on a real 55k-read,
    // 60k-BAM-record production sample, `charging_tcn_sup6_rna004@v0.1.0`,
    // A30, `--threads 4`). Depth swept 2/8/16/32 at the default prep chunk
    // (below) plateaus and then regresses past 16: 315s / 242s / 222s / 240s,
    // CPU 194% / 256% / 278% / 257%. 16 is the plateau, not a guess.
    let groups_in_flight: usize = std::env::var("ESCAPEPOD_WAVEFORM_GPU_GROUPS_IN_FLIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(16);
    // How many reads' worth of prep `rayon` load-balances per synchronous
    // `par_iter` call, independent of `batch` (the GPU's own fixed contract).
    // Tying this 1:1 to `batch` — the first version's choice — was a second
    // mistake compounding the channel-depth one: every sync point produced
    // *exactly* one GPU group, so the finest possible handoff granularity
    // *and* the coarsest possible rayon load-balancing window landed on the
    // same number for no reason but convenience. Widening it (same sweep,
    // `groups_in_flight` pinned at 16) kept helping well past `batch` itself:
    // 128/256/512/1024 reads/chunk measured 222s / 212s / 206s / 203s wall,
    // 278% / 289% / 297% / 303% CPU — before flattening at 1024 (`chunk`
    // above `batch` still yields one GPU group as soon as `batch` valid
    // reads accumulate in `carry`; it does not wait for the whole chunk to
    // become one bigger GPU call). 8x batch sits on that plateau without
    // guessing how far past it diminishing returns would go for a
    // differently-shaped bundle. All numbers bit-identical to the
    // fully-serial baseline at every point on the sweep (see the module doc).
    let prep_chunk: usize = std::env::var("ESCAPEPOD_WAVEFORM_GPU_PREP_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(batch.saturating_mul(8));

    type Group<'r> = Vec<(&'r WaveformRead, Chunk)>;
    let (tx, rx) = std::sync::mpsc::sync_channel::<Group>(groups_in_flight);

    let mut stats = ClassifyStats::default();
    let mut calls: Vec<ReadCall> = Vec::with_capacity(reads.len());

    let gpu_calls = std::thread::scope(|scope| -> Result<Vec<ReadCall>> {
        // The only thread that calls `gpu.logits()`, so GPU calls stay
        // sequential exactly as before — only now overlapped with the next
        // group's CPU prep instead of blocking it.
        let gpu_handle = scope.spawn(move || -> Result<Vec<ReadCall>> {
            let mut out = Vec::new();
            while let Ok(group) = rx.recv() {
                let refs: Vec<&Chunk> = group.iter().map(|(_, c)| c).collect();
                let logits = gpu.logits(&refs, spec)?;
                for ((read, _), logit) in group.iter().zip(logits) {
                    out.push(call_from_logit(read, spec, logit));
                }
            }
            Ok(out)
        });

        'outer: for slice in reads.chunks(SUPERBATCH) {
            // Valid preps not yet folded into a full group, in the same
            // order they would have landed in `ready` if the whole
            // superbatch were prepared up front — so which reads share a
            // GPU group is unaffected by streaming them in smaller pieces.
            let mut carry: Vec<(&WaveformRead, Chunk)> = Vec::with_capacity(batch);

            for raw in slice.chunks(prep_chunk) {
                let prepared: Vec<Prepared> = raw
                    .par_iter()
                    .map(|&read| {
                        let Some(info) = pod5.reads().get(&read.read_id) else {
                            return Ok(Prepared::None(no_call(read, NoCallReason::NoSignal)));
                        };
                        let raw = signal_adc(info, &extractors)?;
                        if raw.len() as i64 != read.ns {
                            return Ok(Prepared::None(no_call(read, NoCallReason::NsMismatch)));
                        }
                        let Some(chunk) = assemble_chunk(bundle.kmer.as_ref(), spec, read, &raw)
                        else {
                            return Ok(Prepared::None(no_call(read, no_chunk)));
                        };
                        Ok(Prepared::Ready(read, chunk))
                    })
                    .collect::<Result<_>>()?;

                for p in prepared {
                    match p {
                        Prepared::Ready(read, chunk) => carry.push((read, chunk)),
                        Prepared::None(n) => record_no_call(&mut stats, n),
                    }
                }

                while carry.len() >= batch {
                    let group: Group = carry.drain(0..batch).collect();
                    if tx.send(group).is_err() {
                        // The GPU thread died; stop producing and surface
                        // its error at `join` below instead of this one.
                        break 'outer;
                    }
                }
            }

            // Fewer than `batch` valid reads left at the end of this
            // superbatch: score them on the CPU rather than pad a GPU call —
            // same rule as before, see the doc comment above.
            for (read, chunk) in carry.drain(..) {
                let logit = cpu_fallback.logit(&chunk, spec)?;
                calls.push(call_from_logit(read, spec, logit));
            }
        }

        drop(tx);
        gpu_handle
            .join()
            .map_err(|_| anyhow::anyhow!("GPU scoring thread panicked"))?
    })?;

    calls.extend(gpu_calls);
    calls.sort_by_key(|c| c.read_id);
    stats.no_calls.sort_by_key(|n| n.read_id);
    Ok((calls, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(kind: CigarKind, len: u32) -> CigarOp {
        CigarOp::new(kind, len)
    }

    fn md(md: &str, query: &str, cigar: &[CigarOp]) -> String {
        String::from_utf8(reference_from_md(md, query.as_bytes(), cigar).expect("reconstructs"))
            .unwrap()
    }

    #[test]
    fn every_base_matching_returns_the_query() {
        assert_eq!(
            md("10", "ACGTACGTAC", &[op(CigarKind::Match, 10)]),
            "ACGTACGTAC"
        );
    }

    #[test]
    fn a_mismatch_takes_the_reference_base_from_md_and_lowercases_it() {
        // MD names the REFERENCE base at a mismatch; the query has T there.
        assert_eq!(
            md("4G5", "ACGTTCGTAC", &[op(CigarKind::Match, 10)]),
            "ACGTgCGTAC"
        );
    }

    #[test]
    fn a_deletion_restores_bases_the_query_never_carried() {
        let cigar = [
            op(CigarKind::Match, 4),
            op(CigarKind::Deletion, 2),
            op(CigarKind::Match, 6),
        ];
        let got = md("4^GG6", "ACGTACGTAC", &cigar);
        assert_eq!(got, "ACGTGGACGTAC");
        assert_eq!(got.len(), 12, "reference spans the deletion");
    }

    #[test]
    fn insertions_and_soft_clips_consume_query_without_reference() {
        let ins = [
            op(CigarKind::Match, 4),
            op(CigarKind::Insertion, 2),
            op(CigarKind::Match, 4),
        ];
        assert_eq!(md("8", "ACGTAACGTA", &ins), "ACGTCGTA");
        let clip = [op(CigarKind::SoftClip, 2), op(CigarKind::Match, 8)];
        assert_eq!(md("8", "TTACGTACGT", &clip), "ACGTACGT");
    }

    #[test]
    fn a_truncated_md_is_refused_rather_than_half_applied() {
        // Covers fewer reference positions than the CIGAR does.
        assert!(reference_from_md("4", b"ACGTACGTAC", &[op(CigarKind::Match, 10)]).is_none());
    }

    /// The defect this whole path exists to avoid (rnabioco/escapepod-rs#306).
    ///
    /// The shipped tRNA references each carry exactly one `N`. Slicing the
    /// FASTA keeps it; `MD` names the base the aligner actually matched. That
    /// is not a cosmetic difference: levels are looked up per 9-mer, so a
    /// single `N` makes **nine** consecutive k-mers unknown and
    /// `extract_levels` leaves nine zeros the training corpus does not have —
    /// which moved refined boundaries throughout the read and cost 169 of 256
    /// chunks their bit-exactness. The fixture reference has no `N` at all,
    /// which is exactly why the counted golden never caught it.
    #[test]
    fn md_resolves_an_ambiguity_code_the_fasta_would_keep() {
        let fasta = "ACGTACNTACGT";
        let query = "ACGTACGTACGT";
        // The alignment recorded a concrete reference base where this FASTA
        // carries `N` -- observed on all 256 reads of the parity corpus, e.g.
        // `md CctGgcGgGGC` against `fasta CCTGGNGGGGC`. MD names that base, so
        // the reconstruction resolves the ambiguity the FASTA keeps.
        let got = md("6C5", query, &[op(CigarKind::Match, 12)]);
        assert_eq!(got, "ACGTACcTACGT");
        assert_ne!(
            got.to_ascii_uppercase(),
            fasta.to_ascii_uppercase(),
            "if these ever agree the regression is not being exercised"
        );

        // And the blast radius, against ONE table: the resolved sequence gets
        // levels, the FASTA slice gets a run of zeros. A real k-mer table holds
        // no `N`-bearing k-mer, so every window touching the `N` misses.
        let resolved = got.to_ascii_uppercase();
        let table: std::collections::HashMap<String, f64> = (0..=(resolved.len() - 9))
            .map(|i| (resolved[i..i + 9].to_string(), (i + 1) as f64))
            .collect();

        let with_n = escapepod_signal::resquiggle::extract_levels(fasta, &table, 9, Some(4));
        assert!(
            with_n.iter().all(|&v| v == 0.0),
            "one N makes every overlapping 9-mer unknown, so the DP sees a flat run: {with_n:?}"
        );

        let resolved_levels =
            escapepod_signal::resquiggle::extract_levels(&resolved, &table, 9, Some(4));
        assert_eq!(
            resolved_levels.iter().filter(|&&v| v != 0.0).count(),
            resolved.len() - 9 + 1,
            "every full window scores once the ambiguity is resolved"
        );
    }
}

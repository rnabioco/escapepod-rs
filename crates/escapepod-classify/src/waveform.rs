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

    Ok(WaveformRead {
        read_id,
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
    let decoder = bgzf::io::MultithreadedReader::new(file);
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
    reference: &[u8],
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
    let sequence = reference.get(read.ref_start..read.ref_end)?;
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

/// Classify every anchored read with signal, in parallel.
///
/// Returns calls sorted by read id (deterministic output order) plus the
/// no-call tallies — the same contract as [`crate::classify_reads`], so a
/// caller only has to pick the scan.
///
/// Gated on the graph runtime; the scan and [`assemble_chunk`] above are not,
/// so the assembly can be tested — and a corpus built — without one.
#[cfg(feature = "waveform-onnx")]
pub fn classify_reads(
    bundle: &ChargingBundle,
    anchored: &HashMap<Uuid, WaveformRead>,
    references: &HashMap<String, String>,
    pod5: &Pod5Index,
) -> Result<(Vec<ReadCall>, ClassifyStats)> {
    let spec = bundle.waveform_spec()?;
    let net = bundle.waveform_net()?;
    let extractors = pod5.extractors()?;
    // Whether "no chunk" is a refusal the bundle asked for, or simply a read
    // this runtime could not score. The two are the same event and different
    // reports; see `NoCallReason::NoChunk`.
    let declared_no_chunk = bundle
        .abstain
        .as_ref()
        .is_some_and(|a| a.kind == AbstainRule::NoChunk);
    let no_chunk_reason = if declared_no_chunk {
        NoCallReason::Abstained(AbstainRule::NoChunk)
    } else {
        NoCallReason::NoChunk
    };

    enum Outcome {
        Call(ReadCall),
        None(NoCall),
    }
    let reads: Vec<&WaveformRead> = anchored.values().collect();
    let no_call = |read: &WaveformRead, reason| {
        Outcome::None(NoCall {
            read_id: read.read_id,
            reference: read.reference.clone(),
            reason,
        })
    };

    let outcomes: Vec<Outcome> = reads
        .par_iter()
        .map(|read| {
            let Some(info) = pod5.reads().get(&read.read_id) else {
                return Ok(no_call(read, NoCallReason::NoSignal));
            };
            let raw = signal_adc(info, &extractors)?;
            if raw.len() as i64 != read.ns {
                return Ok(no_call(read, NoCallReason::NsMismatch));
            }
            // Unreachable given the scan — a read only anchors against a
            // reference the geometry was built from, and both come from the
            // same FASTA — but a missing sequence must not be a panic in a
            // rayon worker, where it would take the whole batch with it.
            let Some(reference) = references.get(&read.reference) else {
                return Ok(no_call(read, no_chunk_reason));
            };
            let Some(chunk) =
                assemble_chunk(bundle.kmer.as_ref(), spec, read, reference.as_bytes(), &raw)
            else {
                return Ok(no_call(read, no_chunk_reason));
            };
            let logit = net.logit(&chunk, spec)?;
            // One BCE logit, of whichever class the bundle says. `cl` is
            // always `P(classes[1])`, so a positive class of 0 inverts here
            // and nowhere else.
            let s = 1.0 / (1.0 + (-logit).exp());
            let p = if spec.positive_class == 1 { s } else { 1.0 - s };
            Ok(Outcome::Call(ReadCall {
                read_id: read.read_id,
                reference: read.reference.clone(),
                p,
                cl: crate::cl_from_probability(p),
            }))
        })
        .collect::<Result<_>>()?;

    let mut stats = ClassifyStats::default();
    let mut calls = Vec::with_capacity(outcomes.len());
    for o in outcomes {
        match o {
            Outcome::Call(c) => calls.push(c),
            Outcome::None(n) => {
                match n.reason {
                    NoCallReason::NoSignal => stats.no_signal += 1,
                    NoCallReason::NsMismatch => stats.ns_mismatch += 1,
                    NoCallReason::NoChunk => stats.no_chunk += 1,
                    NoCallReason::Abstained(_) => stats.abstained += 1,
                }
                stats.no_calls.push(n);
            }
        }
    }
    calls.sort_by_key(|c| c.read_id);
    stats.no_calls.sort_by_key(|n| n.read_id);
    Ok((calls, stats))
}

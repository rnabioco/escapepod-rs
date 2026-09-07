// SPDX-License-Identifier: MIT

//! End-to-end classification pipeline: BAM scan → POD5 index → calls.
//!
//! One implementation of the orchestration the `escpod classify`
//! command, the parity tests, and any future binding all need: scan an
//! aligned BAM into anchored reads and orientation votes, index the POD5
//! set for the reads that anchored, then compute features and classify in
//! parallel.
//! Keeping it here (rather than in the CLI) means the golden-parity tests
//! exercise the very code the command runs.

use crate::anchor::{self, AnchoredRead, Orientation, OrientationVotes, ScanOutcome, SkipReason};
use crate::bundle::{Abstain, AbstainRule, ChargingBundle, ChargingScorer};
use crate::features;
use crate::geometry::RefGeometry;
use crate::recipe::FeatureRecipe;
use anyhow::{Context, Result};
use escapepod_demux::GbmPredictor;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use noodles_bam as bam;
use noodles_bgzf as bgzf;
use noodles_sam::alignment::RecordBuf;

/// Result of scanning one BAM: deduped anchored reads, orientation votes,
/// and bookkeeping for the caller's report.
#[derive(Debug, Default)]
pub struct BamScan {
    /// One entry per read, best alignment (highest mapq) wins.
    pub anchored: HashMap<Uuid, AnchoredRead>,
    pub votes: OrientationVotes,
    pub records_scanned: u64,
    pub skips: HashMap<SkipReason, u64>,
}

/// BGZF inflate workers for a scan that anchors records on the rayon pool.
///
/// `MultithreadedReader::new` is ONE worker, whatever the name suggests. The
/// scan decodes on one cursor and anchors in parallel, so the reader gets
/// half the pool: enough that inflate keeps ahead of the anchoring, without
/// doubling the thread count on a shared node.
pub(crate) fn bgzf_workers() -> std::num::NonZero<usize> {
    std::num::NonZero::new(rayon::current_num_threads().div_ceil(2).max(1)).expect("at least one")
}

/// Scan an aligned BAM into anchored reads and orientation votes.
///
/// Every record runs through [`anchor::scan_record`]; records whose
/// reference is absent from the header resolve as
/// [`SkipReason::Filtered`]. Multiple alignments of one read keep the
/// highest-mapq record (the reference implementation's dedup).
pub fn scan_bam(
    bam_path: &Path,
    geometry: &HashMap<String, RefGeometry>,
    offsets: &[i32],
    min_mapq: u8,
) -> Result<BamScan> {
    let file = std::fs::File::open(bam_path)
        .with_context(|| format!("cannot open BAM {}", bam_path.display()))?;
    let decoder = bgzf::io::MultithreadedReader::with_worker_count(bgzf_workers(), file);
    let mut reader = bam::io::Reader::from(decoder);
    let header = reader.read_header()?;
    let ref_names: Vec<String> = header
        .reference_sequences()
        .keys()
        .map(|k| k.to_string())
        .collect();

    let mut scan = BamScan::default();

    // Anchoring is ~66% of this scan (9.2 us/record against 2.9 us to decode
    // one) and is pure per-record CPU, so records are decoded into a batch and
    // anchored in parallel. Decoding stays serial -- BGZF is already
    // multithreaded underneath, and the reader is a single cursor.
    //
    // The fold back into `scan` is serial and in batch order, so the dedup
    // (best mapq wins) and the orientation vote see records in file order
    // exactly as they did before: same result, deterministically.
    const BATCH: usize = 8192;
    let mut batch: Vec<RecordBuf> = vec![RecordBuf::default(); BATCH];
    let mut outcomes: Vec<ScanOutcome> = Vec::with_capacity(BATCH);
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
                    Some(ref_name) => {
                        anchor::scan_record(record, ref_name, geometry, offsets, min_mapq)
                    }
                    // A reference the header does not name cannot be placed.
                    None => ScanOutcome::Skip(SkipReason::Filtered),
                }
            })
            .collect_into_vec(&mut outcomes);

        for outcome in outcomes.drain(..) {
            match outcome {
                ScanOutcome::Anchored(read) => {
                    scan.votes.add(&read);
                    match scan.anchored.entry(read.read_id) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if read.mapq > e.get().mapq {
                                e.insert(*read);
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(*read);
                        }
                    }
                }
                ScanOutcome::Skip(reason) => {
                    *scan.skips.entry(reason).or_default() += 1;
                }
            }
        }
        if n < BATCH {
            break;
        }
    }
    Ok(scan)
}

/// Per-read signal lookup info from the POD5 index.
#[derive(Debug, Clone)]
pub struct Pod5ReadInfo {
    pub reader_idx: usize,
    pub calibration_scale: f32,
    pub calibration_offset: f32,
    pub signal_rows: Vec<u64>,
}

/// Say, once, that some of this run's POD5 inputs have no `.p5s` sidecar.
///
/// Not an error and not a fallback: [`Pod5Index::build`] gets an index either
/// way, because a reader with no sidecar builds one in memory. What differs is
/// that the in-memory one is discarded at exit, so every run over the same
/// file pays for it again — and *nothing else distinguishes the two runs but
/// the clock*. The right answer arrives either way, which is exactly why the
/// missing sidecar went unnoticed fourteen times over (escapepod-rs#334). This
/// line is the symptom it otherwise does not have.
fn warn_unindexed(paths: &[&Path]) {
    if let Some(message) = unindexed_hint(paths) {
        tracing::warn!("{message}");
    }
}

/// The wording of [`warn_unindexed`], split out so a test can hold it to the
/// two things it must always carry: the remedy (`escpod index`) and the file
/// to run it on. `None` when every input is already indexed.
fn unindexed_hint(paths: &[&Path]) -> Option<String> {
    const SHOWN: usize = 3;
    let (first, rest) = paths.split_first()?;
    if rest.is_empty() {
        let p = first.display();
        return Some(format!(
            "no `.p5s` read index for {p} — building one in memory for this run, and \
             every later run over it will build it again. `escpod index {p}` writes it \
             once, beside the POD5."
        ));
    }
    let mut listed: Vec<String> = paths
        .iter()
        .take(SHOWN)
        .map(|p| p.display().to_string())
        .collect();
    if paths.len() > SHOWN {
        listed.push(format!("and {} more", paths.len() - SHOWN));
    }
    Some(format!(
        "no `.p5s` read index for {} of this run's POD5 inputs ({}) — building them in \
         memory for this run, and every later run over them will build them again. \
         `escpod index` on those files writes them once, beside each POD5.",
        paths.len(),
        listed.join(", "),
    ))
}

/// An index over one or more POD5 files, restricted to wanted read ids.
pub struct Pod5Index {
    readers: Vec<escapepod_signal::Reader>,
    reads: HashMap<Uuid, Pod5ReadInfo>,
}

impl Pod5Index {
    /// Index `paths`, keeping only reads in `wanted`.
    ///
    /// Goes through each reader's read index — the `.p5s` sidecar when there
    /// is one, an in-memory build when there is not — rather than decoding
    /// every reads batch and discarding the rows nobody asked for. A charging
    /// run selects its reads from a BAM, so `wanted` is typically a small
    /// fraction of a production POD5 (20 k of 12.1 M in escapepod-rs#334), and
    /// the lookup is projected to the four columns a signal fetch needs
    /// against the 22 a full row carries.
    ///
    /// Nothing about *which* reads are indexed changes: an id `wanted` names
    /// and the file holds is found either way, and one it does not hold is
    /// absent either way.
    pub fn build(paths: &[PathBuf], wanted: &HashSet<Uuid>) -> Result<Self> {
        let mut readers = Vec::with_capacity(paths.len());
        for path in paths {
            readers.push(escapepod_signal::Reader::open(path)?);
        }
        // Ahead of the lookups, not after them: a line about how long this run
        // is going to take is only useful while the run is still ahead of you.
        // Opening a reader is an mmap and a footer parse; the index — the part
        // a sidecar saves — is not built until the first lookup below.
        let unindexed: Vec<&Path> = paths
            .iter()
            .zip(&readers)
            .filter(|(_, reader)| !reader.has_sidecar_index())
            .map(|(path, _)| path.as_path())
            .collect();
        warn_unindexed(&unindexed);

        let mut reads = HashMap::new();
        for (reader_idx, reader) in readers.iter().enumerate() {
            for found in reader.find_signal_rows_with_calibration_by_ids(wanted)? {
                reads.insert(
                    found.read_id,
                    Pod5ReadInfo {
                        reader_idx,
                        calibration_scale: found.calibration_scale,
                        calibration_offset: found.calibration_offset,
                        signal_rows: found.signal_rows,
                    },
                );
            }
        }
        Ok(Self { readers, reads })
    }

    /// Indexed reads (those of `wanted` that have signal).
    pub fn reads(&self) -> &HashMap<Uuid, Pod5ReadInfo> {
        &self.reads
    }

    /// Where a read sits in the POD5 set: its file, then its first signal row.
    ///
    /// Sort a selection on this before reading it and each worker's next read
    /// is forward of its last, which is what the kernel's readahead and the
    /// filesystem's prefetch can serve — the same reason `demux` streams its
    /// input (escapepod-rs#72).
    ///
    /// Nothing else supplies that order. The reads a charging run scores
    /// arrive from a BAM, which is `SO:coordinate` — sorted by reference
    /// position, and against a tRNA reference that groups reads by *identity*,
    /// which has nothing to do with when a molecule was sequenced. So BAM
    /// order is not merely uncorrelated with POD5 layout, it is structured
    /// against it, and a caller cannot fix that: the layout is knowable only
    /// from the POD5 (escapepod-rs#334). Order is not selection — sorting
    /// changes which reads are read *when*, never which reads are read.
    ///
    /// `None` — a read this index does not hold — sorts first and costs
    /// nothing to visit.
    pub fn storage_key(&self, read_id: &Uuid) -> Option<(usize, u64)> {
        self.reads
            .get(read_id)
            .map(|i| (i.reader_idx, i.signal_rows.first().copied().unwrap_or(0)))
    }

    pub fn n_files(&self) -> usize {
        self.readers.len()
    }

    /// One signal extractor per file, for parallel on-demand extraction.
    pub fn extractors(&self) -> Result<Vec<escapepod_signal::SignalExtractor<'_>>> {
        Ok(self
            .readers
            .iter()
            .map(|r| r.signal_extractor())
            .collect::<escapepod_signal::Result<_>>()?)
    }
}

/// Extract one read's raw ADC samples, uncalibrated.
///
/// What a median/MAD-normalising model wants. The calibration is affine and
/// positive, so the normalisation divides it straight back out — running it
/// first would buy nothing and cost a rounding difference against the corpus
/// builder, which reads the POD5 integers.
pub fn signal_adc(
    info: &Pod5ReadInfo,
    extractors: &[escapepod_signal::SignalExtractor<'_>],
) -> Result<Vec<i16>> {
    Ok(extractors[info.reader_idx].get_signal(&info.signal_rows)?)
}

/// Extract one read's calibrated picoamp signal:
/// `pA = (adc + offset) * scale`, in `f32`.
pub fn signal_pa(
    info: &Pod5ReadInfo,
    extractors: &[escapepod_signal::SignalExtractor<'_>],
) -> Result<Vec<f32>> {
    let raw = extractors[info.reader_idx].get_signal(&info.signal_rows)?;
    Ok(raw
        .iter()
        .map(|&adc| (adc as f32 + info.calibration_offset) * info.calibration_scale)
        .collect())
}

/// The canonical `offsets × FEAT_STATS` feature grid for one read: spans
/// resolved in the run's frame, expected levels z-scored (when the recipe
/// carries a k-mer table), per-base stats over the calibrated signal.
///
/// Takes a [`FeatureRecipe`], not a bundle: the three things that define the
/// feature space are all it reads, and the corpus builder that computes these
/// same features has no weights to hand it. `bundle.recipe()` produces one
/// for the inference path.
pub fn feature_grid(
    recipe: &FeatureRecipe<'_>,
    read: &AnchoredRead,
    orientation: Orientation,
    sig_pa: &[f32],
) -> Vec<f32> {
    let coords = anchor::finalize(read, orientation, recipe.offsets, recipe.span_mode);
    feature_grid_at(recipe, read, &coords, sig_pa)
}

/// [`feature_grid`] for a caller that already resolved the read's coords.
///
/// Callers that also want the window or the coord columns (the corpus
/// builder) would otherwise run [`anchor::finalize`] twice — or, worse, keep
/// their own copy of the k-mer/residual half of the grid, which is how this
/// pipeline came to have two feature definitions once already.
pub fn feature_grid_at(
    recipe: &FeatureRecipe<'_>,
    read: &AnchoredRead,
    coords: &crate::JunctionCoords,
    sig_pa: &[f32],
) -> Vec<f32> {
    // The SAME positions the spans came from. Under the counting anchor
    // `read.qf` is the aligner's answer, which is not what the offsets
    // resolved to -- using it here leaves dwell/mean/std right and the
    // residual silently wrong.
    let qf = anchor::query_positions(read, recipe.offsets, recipe.span_mode);
    let expected = recipe.kmer.map(|k| match k.packed() {
        Some(table) => {
            features::expected_levels_z_packed(&read.seq, table, k.center_idx, &qf, read.nb)
        }
        None => features::expected_levels_z(&read.seq, &k.map, k.k, k.center_idx, &qf, read.nb),
    });
    features::junction_features(sig_pa, coords, expected.as_deref())
}

/// One classified read.
#[derive(Debug, Clone)]
pub struct ReadCall {
    pub read_id: Uuid,
    pub reference: String,
    /// `P(classes[1])`.
    pub p: f64,
    /// `round(p * 255)`.
    pub cl: u8,
}

/// Why an anchored read has no call.
///
/// Named per read rather than tallied, because a drop that is only a count is
/// a drop nobody can chase: the same 12% that
/// `rnabioco/aa-tRNA-seq-pipeline#110` had to infer from the difference
/// between two rows of a QC table, because remora reports no per-read reason.
/// Every read that anchors leaves this pipeline either with a probability or
/// with one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoCallReason {
    /// Excluded by the bundle's abstain rule, which one — see [`abstains`].
    ///
    /// Carries the rule rather than a bare "abstained" so the reason column
    /// names the **population**, not the mechanism that caught it. What the
    /// rule catches is a real class of molecule, not a degraded tRNA read:
    /// measured over a 1.06M-read run (`aligner_arm_depth == 0`, 0.85% of
    /// scoreable reads), the alignment stops exactly at the junction with a
    /// median 81-101 nt of unaligned sequence after it, at *higher* mapq than
    /// the reads that were called, and that sequence is
    ///
    /// | 3' tail                              | share |
    /// |--------------------------------------|-------|
    /// | reverse complement of the common arm | 51.8% |
    /// | other                                | 42.5% |
    /// | poly(A)                              |  4.2% |
    /// | the arm, present but unaligned       |  1.4% |
    ///
    /// The common partner oligo is the revcomp of the arm, so the plurality
    /// are reads of the **wrong strand of the duplex**. They are not reads the
    /// model scores badly; they are reads of something else, and lumping them
    /// under "could not classify" hides a population worth counting on its own.
    Abstained(crate::bundle::AbstainRule),
    /// No signal in the POD5 set (dorado read splitting mints child ids that
    /// are not in the file).
    NoSignal,
    /// Signal length disagreed with the `ns` tag, so the move-table frame
    /// would put every span in the wrong place.
    NsMismatch,
    /// The windowed variant could not cut a chunk at the anchor: the read's
    /// alignment does not reach it, or the map it resolves to covers no
    /// signal.
    ///
    /// Distinct from [`Self::Abstained`] carrying the same condition. When the
    /// bundle *declares* `no chunk, no call` this is a refusal it asked for and
    /// is reported as such; when it declares nothing, it is simply a read this
    /// runtime could not score, and conflating the two would report an abstain
    /// rate for a bundle that never named one.
    NoChunk,
}

impl NoCallReason {
    /// Stable token for the `reason` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Abstained(AbstainRule::NoAlignedArm) => "no_aligned_arm",
            Self::Abstained(AbstainRule::NoChunk) | Self::NoChunk => "no_chunk",
            Self::NoSignal => "no_signal",
            Self::NsMismatch => "ns_mismatch",
        }
    }
}

/// An anchored read that got no probability, and why.
#[derive(Debug, Clone)]
pub struct NoCall {
    pub read_id: Uuid,
    pub reference: String,
    pub reason: NoCallReason,
}

/// Reads that anchored but could not be classified.
#[derive(Debug, Default, Clone)]
pub struct ClassifyStats {
    /// Every unscored read, with its reason — the attributable form of the
    /// counters below.
    pub no_calls: Vec<NoCall>,
    /// No signal in the POD5 set (e.g. dorado read-splitting children).
    pub no_signal: u64,
    /// Signal length disagreed with the `ns` tag (split/trimmed reads);
    /// the move-table frame would put every span in the wrong place.
    pub ns_mismatch: u64,
    /// Excluded by the bundle's own abstain rule: anchored, with signal, and
    /// deliberately not scored.
    ///
    /// Distinct from the two above, which are reads the runtime *could not*
    /// score. This one is a refusal, and its rate belongs beside any charging
    /// fraction computed from the calls — arm resolvability is correlated with
    /// the label, so a fraction over called reads alone is biased.
    pub abstained: u64,
    /// The windowed variant could not cut a chunk, and the bundle named no
    /// rule that says so — see [`NoCallReason::NoChunk`].
    pub no_chunk: u64,
}

/// The bundle's scorer, made ready to run: `P(classes[1])` from the flat
/// column vector, whichever model the bundle carries.
///
/// The two arms take the same input and differ only in what runs on it, so
/// this is the entire model-specific part of the pipeline — everything above
/// (anchoring, spans, features, column selection) is shared verbatim.
enum Scorer<'a> {
    Gbm(GbmPredictor<'a>),
    #[cfg(feature = "fnn-onnx")]
    FeatureNn(&'a crate::fnn::FeatureNet),
}

impl Scorer<'_> {
    fn new(bundle: &ChargingBundle) -> Result<Scorer<'_>> {
        Ok(match &bundle.scorer {
            ChargingScorer::Gbm(g) => Scorer::Gbm(GbmPredictor::new(g)),
            #[cfg(feature = "fnn-onnx")]
            ChargingScorer::FeatureNn(net) => Scorer::FeatureNn(net),
            // The windowed variant has its own scan and its own assembly; it
            // does not reach this pipeline. Naming it here rather than
            // wildcarding is what makes a future variant a compile error.
            #[cfg(feature = "waveform-onnx")]
            ChargingScorer::Waveform(_) => anyhow::bail!(
                "this bundle scores a signal window; use `waveform::classify_reads`, \
                 which anchors in reference coordinates and assembles its own tensors"
            ),
        })
    }

    fn p_positive(&self, features: &[f64]) -> Result<f64> {
        match self {
            Self::Gbm(p) => {
                let (probs, _) = p
                    .predict(features)
                    .map_err(|e| anyhow::anyhow!("GBM predict failed: {e}"))?;
                Ok(probs[1])
            }
            #[cfg(feature = "fnn-onnx")]
            Self::FeatureNn(net) => Ok(net.predict(features)?[1]),
        }
    }

    /// How many reads to hand [`Self::p_positive_batch`] at a time. One for
    /// a scorer that gains nothing from a group.
    fn preferred_batch(&self) -> usize {
        match self {
            Self::Gbm(_) => 1,
            #[cfg(feature = "fnn-onnx")]
            Self::FeatureNn(net) => net.preferred_batch(),
        }
    }

    /// [`Self::p_positive`] for a group of reads, in order.
    fn p_positive_batch(&self, features: &[&[f64]]) -> Result<Vec<f64>> {
        match self {
            Self::Gbm(_) => features.iter().map(|f| self.p_positive(f)).collect(),
            #[cfg(feature = "fnn-onnx")]
            Self::FeatureNn(net) => Ok(net.predict_batch(features)?.iter().map(|p| p[1]).collect()),
        }
    }
}

/// Does the bundle's abstain rule exclude this read?
///
/// Separate from [`classify_reads`] so the decision can be tested on coords
/// directly: the rule fires on a population no small fixture is guaranteed to
/// contain (reads the aligner could not place a single arm base on), and a
/// test that silently never fires would be worse than none. Its rate on a real
/// corpus is what the CLI reports.
pub fn abstained_by(
    abstain: Option<&Abstain>,
    coords: &crate::JunctionCoords,
) -> Option<AbstainRule> {
    match abstain.map(|a| a.kind) {
        // The aligner reached no arm base at all. Note this is NOT "the window
        // was short": under the counting anchor the read still has arm
        // features, walked along the query. See [`NoCallReason::Abstained`]
        // for what these reads turn out to be.
        Some(AbstainRule::NoAlignedArm) if coords.aligner_arm_depth == 0 => {
            Some(AbstainRule::NoAlignedArm)
        }
        _ => None,
    }
}

/// Classify every anchored read with signal, in parallel.
///
/// Returns calls sorted by read id (deterministic output order) plus the
/// skip tallies.
pub fn classify_reads(
    bundle: &ChargingBundle,
    anchored: &HashMap<Uuid, AnchoredRead>,
    pod5: &Pod5Index,
    orientation: Orientation,
) -> Result<(Vec<ReadCall>, ClassifyStats)> {
    let extractors = pod5.extractors()?;
    let predictor = Scorer::new(bundle)?;
    let recipe = bundle.recipe()?;
    let mut reads: Vec<&AnchoredRead> = anchored.values().collect();
    // Walk each POD5 forward. `anchored` is a HashMap, so its order is the
    // hash's — and behind that, the BAM's — and each worker's next read would
    // be a random seek into a file that is a mmap over shared storage. Ordered
    // by [`Pod5Index::storage_key`], a rayon chunk is a contiguous forward
    // sweep instead.
    reads.sort_by_cached_key(|r| pod5.storage_key(&r.read_id));

    enum Outcome {
        Call(ReadCall),
        None(NoCall),
    }
    let no_call = |read: &AnchoredRead, reason| {
        Outcome::None(NoCall {
            read_id: read.read_id,
            reference: read.reference.clone(),
            reason,
        })
    };
    // Reads go to the scorer in groups: the native BiLSTM kernel streams its
    // recurrent weights from L2 once per timestep and scores a group in
    // lockstep, so the group shares that traffic. The per-read work before
    // scoring — signal, coordinates, the abstain rule, the feature grid — is
    // unchanged and still per read; only the reads a chunk actually scores are
    // handed over together, in order.
    //
    // A chunk is several of the scorer's groups, not one. With chunks of
    // exactly one group the single-thread saving was a quarter of what it is
    // now: the prep between chunks evicted the 295 KB of weights, so every
    // group re-fetched them, and a chunk that lost one read to the abstain
    // rule scored a narrower group (a sixth of them did). Eight groups per
    // chunk amortise the reload over the chunk and leave one tail per chunk
    // instead of one per group.
    let batch = predictor.preferred_batch().max(1);
    let chunk_reads = batch * if batch > 1 { 8 } else { 1 };
    let outcomes: Vec<Outcome> = reads
        .par_chunks(chunk_reads)
        .map(|chunk| -> Result<Vec<Outcome>> {
            let mut outs: Vec<Option<Outcome>> = (0..chunk.len()).map(|_| None).collect();
            let mut to_score: Vec<(usize, Vec<f64>)> = Vec::with_capacity(chunk.len());
            for (i, read) in chunk.iter().enumerate() {
                let Some(info) = pod5.reads().get(&read.read_id) else {
                    outs[i] = Some(no_call(read, NoCallReason::NoSignal));
                    continue;
                };
                let sig_pa = signal_pa(info, &extractors)?;
                if sig_pa.len() as i64 != read.ns {
                    outs[i] = Some(no_call(read, NoCallReason::NsMismatch));
                    continue;
                }
                // Resolved once and reused: the abstain rule reads the same
                // coords the features are taken from, so the two cannot
                // disagree about what the aligner reached.
                let coords = anchor::finalize(read, orientation, recipe.offsets, recipe.span_mode);
                if let Some(rule) = abstained_by(bundle.abstain.as_ref(), &coords) {
                    outs[i] = Some(no_call(read, NoCallReason::Abstained(rule)));
                    continue;
                }
                let grid = feature_grid_at(&recipe, read, &coords, &sig_pa);
                to_score.push((i, bundle.select_columns(&grid)?));
            }
            if !to_score.is_empty() {
                let cols: Vec<&[f64]> = to_score.iter().map(|(_, c)| c.as_slice()).collect();
                let ps = predictor.p_positive_batch(&cols)?;
                for ((i, _), p) in to_score.iter().zip(ps) {
                    let read = chunk[*i];
                    outs[*i] = Some(Outcome::Call(ReadCall {
                        read_id: read.read_id,
                        reference: read.reference.clone(),
                        p,
                        cl: crate::cl_from_probability(p),
                    }));
                }
            }
            Ok(outs
                .into_iter()
                .map(|o| o.expect("every read in the group has an outcome"))
                .collect())
        })
        .collect::<Result<Vec<Vec<Outcome>>>>()?
        .into_iter()
        .flatten()
        .collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The hint has one job: name the remedy and the file. A rewording that
    /// drops either is the fifteenth occurrence waiting to happen.
    #[test]
    fn the_unindexed_hint_names_the_command_and_the_file() {
        assert!(unindexed_hint(&[]).is_none());

        let one = unindexed_hint(&[Path::new("/data/run/reads.pod5")]).unwrap();
        assert!(one.contains("escpod index /data/run/reads.pod5"), "{one}");
        assert!(one.contains(".p5s"), "{one}");

        let paths: Vec<PathBuf> = (0..5)
            .map(|i| PathBuf::from(format!("f{i}.pod5")))
            .collect();
        let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let many = unindexed_hint(&refs).unwrap();
        assert!(many.contains("escpod index"), "{many}");
        assert!(many.contains("5 of this run's POD5 inputs"), "{many}");
        assert!(many.contains("f0.pod5"), "{many}");
        // Listed, then truncated — a directory of hundreds must not print one
        // line per file.
        assert!(many.contains("and 2 more"), "{many}");
        assert!(!many.contains("f4.pod5"), "{many}");
    }
}

// SPDX-License-Identifier: MIT

//! End-to-end classification pipeline: BAM scan → POD5 index → calls.
//!
//! One implementation of the orchestration the `escpod signal classify`
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
    let decoder = bgzf::io::MultithreadedReader::new(file);
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

/// An index over one or more POD5 files, restricted to wanted read ids.
pub struct Pod5Index {
    readers: Vec<escapepod_signal::Reader>,
    reads: HashMap<Uuid, Pod5ReadInfo>,
}

impl Pod5Index {
    /// Index `paths`, keeping only reads in `wanted`.
    pub fn build(paths: &[PathBuf], wanted: &HashSet<Uuid>) -> Result<Self> {
        let mut reads = HashMap::new();
        let mut readers = Vec::with_capacity(paths.len());
        for (reader_idx, path) in paths.iter().enumerate() {
            let reader = escapepod_signal::Reader::open(path)?;
            for batch_result in reader.read_batches()? {
                let batch = batch_result?;
                let view = escapepod_signal::ReadsBatchView::new(&batch, false)?;
                for row in 0..view.num_rows() {
                    let read = view.read(row)?;
                    if wanted.contains(&read.read_id) {
                        reads.insert(
                            read.read_id,
                            Pod5ReadInfo {
                                reader_idx,
                                calibration_scale: read.calibration_scale,
                                calibration_offset: read.calibration_offset,
                                signal_rows: read.signal_rows,
                            },
                        );
                    }
                }
            }
            readers.push(reader);
        }
        Ok(Self { readers, reads })
    }

    /// Indexed reads (those of `wanted` that have signal).
    pub fn reads(&self) -> &HashMap<Uuid, Pod5ReadInfo> {
        &self.reads
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
    let expected = recipe
        .kmer
        .map(|k| features::expected_levels_z(&read.seq, &k.map, k.k, k.center_idx, &qf, read.nb));
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
}

impl NoCallReason {
    /// Stable token for the `reason` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Abstained(AbstainRule::NoAlignedArm) => "no_aligned_arm",
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
    fn new(bundle: &ChargingBundle) -> Scorer<'_> {
        match &bundle.scorer {
            ChargingScorer::Gbm(g) => Scorer::Gbm(GbmPredictor::new(g)),
            #[cfg(feature = "fnn-onnx")]
            ChargingScorer::FeatureNn(net) => Scorer::FeatureNn(net),
        }
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
    let predictor = Scorer::new(bundle);
    let recipe = bundle.recipe();
    let reads: Vec<&AnchoredRead> = anchored.values().collect();

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
    let outcomes: Vec<Outcome> = reads
        .par_iter()
        .map(|read| {
            let Some(info) = pod5.reads().get(&read.read_id) else {
                return Ok(no_call(read, NoCallReason::NoSignal));
            };
            let sig_pa = signal_pa(info, &extractors)?;
            if sig_pa.len() as i64 != read.ns {
                return Ok(no_call(read, NoCallReason::NsMismatch));
            }
            // Resolved once and reused: the abstain rule reads the same coords
            // the features are taken from, so the two cannot disagree about
            // what the aligner reached.
            let coords = anchor::finalize(read, orientation, recipe.offsets, recipe.span_mode);
            if let Some(rule) = abstained_by(bundle.abstain.as_ref(), &coords) {
                return Ok(no_call(read, NoCallReason::Abstained(rule)));
            }
            let grid = feature_grid_at(&recipe, read, &coords, &sig_pa);
            let features = bundle.select_columns(&grid);
            let p = predictor.p_positive(&features)?;
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

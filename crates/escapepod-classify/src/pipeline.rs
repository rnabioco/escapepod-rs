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
use crate::bundle::ChargingBundle;
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

/// Reads that anchored but could not be classified.
#[derive(Debug, Default, Clone, Copy)]
pub struct ClassifyStats {
    /// No signal in the POD5 set (e.g. dorado read-splitting children).
    pub no_signal: u64,
    /// Signal length disagreed with the `ns` tag (split/trimmed reads);
    /// the move-table frame would put every span in the wrong place.
    pub ns_mismatch: u64,
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
    let predictor = GbmPredictor::new(&bundle.gbm);
    let recipe = bundle.recipe();
    let reads: Vec<&AnchoredRead> = anchored.values().collect();

    enum Outcome {
        Call(ReadCall),
        NoSignal,
        NsMismatch,
    }
    let outcomes: Vec<Outcome> = reads
        .par_iter()
        .map(|read| {
            let Some(info) = pod5.reads().get(&read.read_id) else {
                return Ok(Outcome::NoSignal);
            };
            let sig_pa = signal_pa(info, &extractors)?;
            if sig_pa.len() as i64 != read.ns {
                return Ok(Outcome::NsMismatch);
            }
            let grid = feature_grid(&recipe, read, orientation, &sig_pa);
            let features = bundle.select_columns(&grid);
            let (probs, _) = predictor
                .predict(&features)
                .map_err(|e| anyhow::anyhow!("GBM predict failed: {e}"))?;
            let p = probs[1];
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
            Outcome::NoSignal => stats.no_signal += 1,
            Outcome::NsMismatch => stats.ns_mismatch += 1,
        }
    }
    calls.sort_by_key(|c| c.read_id);
    Ok((calls, stats))
}

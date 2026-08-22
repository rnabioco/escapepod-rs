//! Motif-anchored per-read signal windows and per-offset statistics.
//!
//! Given an aligned BAM and its POD5, this locates a reference motif in each
//! read, resolves the move-table frame the run was basecalled in, and returns
//! two things per read: the raw signal window around the anchor (masked where
//! the caller says not to look) and summary statistics for a set of
//! base offsets either side of it, optionally against a k-mer level model.
//!
//! None of that is specific to one assay -- it is what any per-base signal
//! model over aligned reads needs. The tRNA aminoacylation classifier is the
//! current caller; the offsets, the mask boundary rule and the k-mer table are
//! all its recipe, supplied here rather than assumed.
//!
//! The point of doing it here is parallelism. The work is independent per read
//! and entirely numeric, so it runs across cores with the GIL released; the
//! equivalent NumPy loop is one core and was ~37% interpreter dispatch.
//!
//! State lives on [`AnchoredReads`] rather than crossing the boundary: each
//! anchored read carries a basecalled sequence and a seq->sig map, and at
//! millions of reads materialising those as Python objects costs tens of
//! gigabytes on its own. Python holds a handle and pulls batches.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use escapepod_classify::{
    AnchoredRead, BaseJustify, FeatureRecipe, KmerLevels, MaskSource, Orientation, Pod5Index,
    SpanMode, finalize, junction_positions, resolve_orientation, scan_bam, signal_window,
};

fn value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn mask_source_name(m: MaskSource) -> &'static str {
    match m {
        MaskSource::Exact => "exact",
        MaskSource::Counted => "counted",
        MaskSource::ArmFallback => "arm_fallback",
        MaskSource::JunctionFallback => "junction_fallback",
    }
}

/// One read's extracted row, produced off the GIL.
struct Row {
    idx: usize,
    window: Vec<f32>,
    features: Vec<f32>,
    coords: escapepod_classify::JunctionCoords,
    mask_source: &'static str,
    /// Which rule placed the junction: "exact", "flank_interp" or
    /// "backfill". Carried per read because the flank rule fires ~30x
    /// more often on modified reads, so its footprint is itself
    /// class-correlated and a corpus needs to be auditable by it.
    anchor_source: &'static str,
}

/// A scanned BAM, ready to extract windows and statistics in parallel.
#[pyclass]
pub struct AnchoredReads {
    anchored: HashMap<uuid::Uuid, AnchoredRead>,
    order: Vec<uuid::Uuid>,
    orientation: Orientation,
    offsets: Vec<i32>,
    mode: SpanMode,
    pod5: Option<Pod5Index>,
    kmer: Option<KmerLevels>,
    n_records: usize,
    records_scanned: u64,
    skips: HashMap<escapepod_classify::SkipReason, u64>,
    votes: (usize, usize),
}

#[pymethods]
impl AnchoredReads {
    /// Scan `bam_path` against `ref_fasta` and resolve the run's frame.
    ///
    /// `count_arm_bases > 0` selects the counting anchor; `0` the aligner's
    /// mapping. This is the feature-space choice and has no default here on
    /// purpose — the caller's recipe knows it, this function cannot guess it.
    #[new]
    #[pyo3(signature = (
        bam_path, ref_fasta, offsets, count_arm_bases,
        motif = "CCAGGC", motif_offset = 3,
        common_arm = "GGCTTCTTCTTGCTCTT",
        min_mapq = 1, min_orientation_votes = 50, orientation = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bam_path: PathBuf,
        ref_fasta: PathBuf,
        offsets: Vec<i32>,
        count_arm_bases: u32,
        motif: &str,
        motif_offset: usize,
        common_arm: &str,
        min_mapq: u8,
        min_orientation_votes: usize,
        orientation: Option<&str>,
    ) -> PyResult<Self> {
        if offsets.is_empty() {
            return Err(PyValueError::new_err("offsets must not be empty"));
        }
        let geometry =
            junction_positions(&ref_fasta, motif, motif_offset, common_arm).map_err(value_err)?;
        let scan = scan_bam(&bam_path, &geometry, &offsets, min_mapq).map_err(value_err)?;

        // Getting the frame wrong silently mirrors every window, so it is
        // voted on rather than assumed; an override exists only for batches
        // too small to vote.
        let orientation = match orientation {
            Some("time") => Orientation::Time,
            Some("reversed") => Orientation::Reversed,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "orientation must be time|reversed, got {other:?}"
                )));
            }
            None => resolve_orientation(&scan.votes, min_orientation_votes).map_err(value_err)?,
        };

        let mut order: Vec<uuid::Uuid> = scan.anchored.keys().copied().collect();
        order.sort_unstable();
        Ok(Self {
            n_records: scan.anchored.len(),
            records_scanned: scan.records_scanned,
            skips: scan.skips.clone(),
            votes: (scan.votes.time, scan.votes.reversed),
            anchored: scan.anchored,
            order,
            orientation,
            offsets,
            mode: SpanMode::from_arm_bases(count_arm_bases),
            pod5: None,
            kmer: None,
        })
    }

    /// `"time"` or `"reversed"`.
    #[getter]
    fn orientation(&self) -> &'static str {
        match self.orientation {
            Orientation::Time => "time",
            Orientation::Reversed => "reversed",
        }
    }

    /// Anchored read ids, deduplicated (best mapq wins) and sorted.
    #[getter]
    fn read_ids(&self) -> Vec<String> {
        self.order.iter().map(|u| u.to_string()).collect()
    }

    /// Per-read junction coordinates, WITHOUT touching POD5.
    ///
    /// Everything returned here was decided by the BAM scan that already ran in
    /// `new`: the anchor, the mask boundary, the spans. `extract` reports the
    /// same coordinates but has to pull signal to do it, and on a real corpus
    /// that is ~136 GB of POD5 for ~15.9M reads -- I/O-bound, measured at 4.18
    /// of 48 allocated cores. Asking a question about GEOMETRY should not cost
    /// that.
    ///
    /// It is what makes an incremental re-extract possible. Changing the anchor
    /// rule moves ~1% of reads; this says which ones from the BAM alone, so the
    /// signal pass can be restricted to them and spliced into the corpus that
    /// already exists.
    ///
    /// Every value is a numpy array, so the result saves as-is:
    ///
    /// ```python
    /// np.savez(path, **reads.coords())
    /// ```
    ///
    /// `read_id` is a flat ASCII buffer, `.view("S36")` for the UUIDs -- 15.9M
    /// Python strings would be ~1.3 GB of PyObject to build and throw away.
    /// `anchor_source` and `mask_source` are indices into the module constants
    /// `ANCHOR_SOURCES` and `MASK_SOURCES`, so an npz round-trip keeps them
    /// interpretable without carrying the strings per read.
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        struct Row {
            id: [u8; 36],
            anchor: u8,
            mask: u8,
            junction_sig: i64,
            common_start_sig: i64,
            cca_a_sig: i64,
            junction_dwell: i64,
            cca_a_dwell: i64,
            arm_depth: i32,
        }

        let rows: Vec<Row> = py.detach(|| {
            self.order
                .par_iter()
                .filter_map(|id| {
                    let read = self.anchored.get(id)?;
                    let c = finalize(read, self.orientation, &self.offsets, self.mode);
                    let mut buf = [0u8; 36];
                    let mut enc = uuid::Uuid::encode_buffer();
                    buf.copy_from_slice(id.as_hyphenated().encode_lower(&mut enc).as_bytes());
                    Some(Row {
                        id: buf,
                        anchor: read.anchor_source as u8,
                        mask: c.mask_source as u8,
                        junction_sig: c.junction_sig,
                        common_start_sig: c.common_start_sig,
                        cca_a_sig: c.cca_a_sig,
                        junction_dwell: c.junction_dwell,
                        cca_a_dwell: c.cca_a_dwell,
                        arm_depth: c.arm_resolved_depth,
                    })
                })
                .collect()
        });

        let mut ids = Vec::with_capacity(rows.len() * 36);
        for r in &rows {
            ids.extend_from_slice(&r.id);
        }

        let out = PyDict::new(py);
        out.set_item("read_id", PyArray1::from_vec(py, ids))?;
        let col_u8 = |f: fn(&Row) -> u8| PyArray1::from_vec(py, rows.iter().map(f).collect());
        let col_i64 = |f: fn(&Row) -> i64| PyArray1::from_vec(py, rows.iter().map(f).collect());
        out.set_item("anchor_source", col_u8(|r| r.anchor))?;
        out.set_item("mask_source", col_u8(|r| r.mask))?;
        out.set_item("junction_sig", col_i64(|r| r.junction_sig))?;
        out.set_item("common_start_sig", col_i64(|r| r.common_start_sig))?;
        out.set_item("cca_a_sig", col_i64(|r| r.cca_a_sig))?;
        out.set_item("junction_dwell", col_i64(|r| r.junction_dwell))?;
        out.set_item("cca_a_dwell", col_i64(|r| r.cca_a_dwell))?;
        out.set_item(
            "arm_resolved_depth",
            PyArray1::from_vec(py, rows.iter().map(|r| r.arm_depth).collect::<Vec<i32>>()),
        )?;
        Ok(out)
    }

    #[getter]
    fn n_anchored(&self) -> usize {
        self.n_records
    }

    /// Total BAM records read, before any filtering.
    #[getter]
    fn records_scanned(&self) -> u64 {
        self.records_scanned
    }

    /// Why records were dropped, keyed by reason.
    ///
    /// The corpus builder's retention audit compares these run to run: a
    /// rejection rate that differs by class is a label-correlated filter,
    /// which is how two separate bugs got into this pipeline.
    #[getter]
    fn skips(&self) -> HashMap<&'static str, u64> {
        self.skips
            .iter()
            .map(|(reason, n)| (reason.as_str(), *n))
            .collect()
    }

    /// Move-table frame votes as `(time, reversed)`.
    #[getter]
    fn orientation_votes(&self) -> (usize, usize) {
        self.votes
    }

    /// Index POD5 files, keeping only reads this scan anchored.
    ///
    /// Accepts `.pod5` files or directories of them; a run is usually split
    /// across many files and making the caller glob is a papercut.
    fn index_pod5(&mut self, paths: Vec<PathBuf>) -> PyResult<usize> {
        let mut files = Vec::new();
        for p in paths {
            if p.is_dir() {
                let mut found: Vec<PathBuf> = std::fs::read_dir(&p)
                    .map_err(value_err)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|f| f.extension().is_some_and(|e| e == "pod5"))
                    .collect();
                if found.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "no .pod5 files in {}",
                        p.display()
                    )));
                }
                // Deterministic order: the index is keyed by read id, but a
                // stable file order keeps any per-file diagnostics comparable.
                found.sort();
                files.extend(found);
            } else {
                files.push(p);
            }
        }
        let wanted: HashSet<uuid::Uuid> = self.anchored.keys().copied().collect();
        let idx = Pod5Index::build(&files, &wanted).map_err(value_err)?;
        let n = idx.reads().len();
        self.pod5 = Some(idx);
        Ok(n)
    }

    /// Reorder `read_ids` into POD5 storage order.
    ///
    /// `extract` already sorts each batch this way, but that only helps WITHIN
    /// a batch: a caller that shuffles ids and then slices them into batches
    /// makes every batch touch every file, so a run gets swept once per batch
    /// rather than once. On an 8M-read run in 250k batches that is ~32 passes
    /// over the whole POD5 set, which on a network filesystem is the entire
    /// cost of extraction.
    ///
    /// Select randomly, then order by storage, then batch. Ids with no signal
    /// are dropped -- they cannot be extracted anyway.
    fn storage_order(&self, read_ids: Vec<String>) -> PyResult<Vec<String>> {
        let idx = self
            .pod5
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call index_pod5 first"))?;
        let mut keyed: Vec<(usize, u64, String)> = read_ids
            .into_iter()
            .filter_map(|s| {
                let id = s.parse::<uuid::Uuid>().ok()?;
                let info = idx.reads().get(&id)?;
                Some((
                    info.reader_idx,
                    info.signal_rows.first().copied().unwrap_or(0),
                    s,
                ))
            })
            .collect();
        keyed.sort_unstable_by_key(|(f, r, _)| (*f, *r));
        Ok(keyed.into_iter().map(|(_, _, s)| s).collect())
    }

    /// Read ids that anchored **and** have signal, in sorted order.
    #[getter]
    fn read_ids_with_signal(&self) -> PyResult<Vec<String>> {
        let idx = self
            .pod5
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call index_pod5 first"))?;
        Ok(self
            .order
            .iter()
            .filter(|u| idx.reads().contains_key(u))
            .map(|u| u.to_string())
            .collect())
    }

    /// Load the k-mer level table the `resid` statistic is defined against.
    ///
    /// Returns the table's `k`. `center_idx` is the position within the k-mer
    /// the level belongs to, defaulting to `k / 2` -- what the bundle recipe
    /// and the training path both use. Note this is **not**
    /// `KmerTable::extract_levels`, which centres on the empirically dominant
    /// base (3 on the RNA004 9-mer table, against a midpoint of 4). A one-base
    /// shift moves every residual, so the override exists and is explicit.
    #[pyo3(signature = (path, center_idx = None))]
    fn load_kmer_table(&mut self, path: PathBuf, center_idx: Option<usize>) -> PyResult<usize> {
        let (map, k) = escapepod_signal::resquiggle::load_kmer_table(&path).map_err(value_err)?;
        let center_idx = center_idx.unwrap_or(k / 2);
        if center_idx >= k {
            return Err(PyValueError::new_err(format!(
                "center_idx {center_idx} outside a {k}-mer"
            )));
        }
        self.kmer = Some(KmerLevels { map, k, center_idx });
        Ok(k)
    }

    /// Extract windows and per-offset features for a batch of reads.
    ///
    /// Returns a dict of column arrays. `X` is `(n, left + right)` raw pA with
    /// `NaN` for padding and for everything earlier in time than the
    /// common-arm start; `F` is `(n, len(offsets) * 4)` in offsets-outer,
    /// (dwell, mean, std, resid) order. Both come from
    /// `escapepod_classify` ([`signal_window`], `feature_grid_at`) — the
    /// window and the grid are model contract, so this layer only marshals.
    ///
    /// Reads with no signal, or with fewer than `right` real samples falling
    /// inside the window, are dropped — `read_id` says which survived rather
    /// than the caller having to infer it from a shorter array.
    #[pyo3(signature = (read_ids, left, right, base_justify = "start"))]
    fn extract<'py>(
        &self,
        py: Python<'py>,
        read_ids: Vec<String>,
        left: i64,
        right: i64,
        base_justify: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let justify: BaseJustify = base_justify.parse().map_err(value_err)?;
        if left < 0 || right <= 0 {
            return Err(PyValueError::new_err("left >= 0 and right > 0 required"));
        }
        let idx = self
            .pod5
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call index_pod5 first"))?;
        let extractors = idx.extractors().map_err(value_err)?;

        let ids: Vec<uuid::Uuid> = read_ids
            .iter()
            .map(|s| s.parse::<uuid::Uuid>().map_err(value_err))
            .collect::<PyResult<_>>()?;

        let w = (left + right) as usize;
        let n_feat = self.offsets.len() * escapepod_classify::FEAT_STATS.len();
        // The feature space, borrowed: offsets, span mode, k-mer levels. The
        // grid itself is computed by escapepod-classify, so the corpus this
        // builds and the model that scores it cannot drift apart.
        let recipe = FeatureRecipe::new(&self.offsets, self.mode, self.kmer.as_ref());

        // Per-read work is independent and entirely numeric, so it runs off
        // the GIL. Rows carry their input index so the output keeps the
        // caller's order regardless of completion order.
        // Visit reads in STORAGE order, not the caller's. POD5 signal lives in
        // row-indexed chunks per file, and a batch sorted by read id is
        // scattered across all of them -- 60k reads that way spent ~99% of
        // their time in signal IO. Sorting by (file, first row) turns that
        // into a near-sequential sweep; each row carries its input index, so
        // the output still comes back in the caller's order.
        let mut work: Vec<(usize, uuid::Uuid)> = ids.iter().copied().enumerate().collect();
        work.sort_unstable_by_key(|(_, id)| {
            idx.reads()
                .get(id)
                .map(|i| (i.reader_idx, i.signal_rows.first().copied().unwrap_or(0)))
                .unwrap_or((usize::MAX, u64::MAX))
        });

        let rows: Vec<Row> = py.detach(|| {
            work.par_iter()
                .filter_map(|&(i, id)| {
                    let read = self.anchored.get(&id)?;
                    let info = idx.reads().get(&id)?;
                    let sig = escapepod_classify::signal_pa(info, &extractors).ok()?;
                    // A split or trimmed read whose signal disagrees with its
                    // `ns` tag would put every span in the wrong place.
                    if sig.len() as i64 != read.ns {
                        return None;
                    }
                    let coords = finalize(read, self.orientation, recipe.offsets, recipe.span_mode);
                    // Anchoring, padding and the mask boundary are model
                    // contract, not binding convenience -- they live in
                    // escapepod-classify so the Rust inference path reproduces
                    // exactly the window the corpus was cut with.
                    let window = signal_window(&sig, &coords, left, right, justify)?;
                    let features =
                        escapepod_classify::feature_grid_at(&recipe, read, &coords, &sig);

                    Some(Row {
                        idx: i,
                        window,
                        features,
                        mask_source: mask_source_name(coords.mask_source),
                        anchor_source: read.anchor_source.name(),
                        coords,
                    })
                })
                .collect()
        });

        let mut rows = rows;
        rows.sort_unstable_by_key(|r| r.idx);

        let n = rows.len();
        let mut x = Vec::with_capacity(n * w);
        let mut f = Vec::with_capacity(n * n_feat);
        // Column vectors, one per corpus metadata field. The caller writes a
        // row per read, so everything it would otherwise recompute from the
        // read travels back with the features.
        let col = |_n: usize| Vec::<i64>::with_capacity(n);
        let (mut js, mut cs) = (col(n), col(n));
        let (mut cca_sig, mut cca_dwell, mut j_dwell) = (col(n), col(n), col(n));
        let (mut arm_depth, mut aln_depth) = (col(n), col(n));
        let (mut polya, mut body) = (col(n), col(n));
        let (mut ns, mut ts, mut mapq) = (col(n), col(n), col(n));
        let mut asrc: Vec<&'static str> = Vec::with_capacity(n);
        let (mut kept, mut msrc, mut refs) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for r in &rows {
            x.extend_from_slice(&r.window);
            f.extend_from_slice(&r.features);
            let c = &r.coords;
            js.push(c.junction_sig);
            cs.push(c.common_start_sig);
            cca_sig.push(c.cca_a_sig);
            cca_dwell.push(c.cca_a_dwell);
            j_dwell.push(c.junction_dwell);
            arm_depth.push(c.arm_resolved_depth as i64);
            aln_depth.push(c.aligner_arm_depth as i64);
            polya.push(c.polya_mid_sig);
            body.push(c.body_mid_sig);
            msrc.push(r.mask_source);
            asrc.push(r.anchor_source);
            let id = ids[r.idx];
            kept.push(id.to_string());
            let read = &self.anchored[&id];
            refs.push(read.reference.clone());
            mapq.push(read.mapq as i64);
            ns.push(read.ns);
            ts.push(read.ts);
        }

        let out = PyDict::new(py);
        out.set_item("read_id", kept)?;
        out.set_item("reference", refs)?;
        out.set_item("mask_source", msrc)?;
        out.set_item("anchor_source", asrc)?;
        out.set_item("X", PyArray1::from_vec(py, x).reshape([n, w])?)?;
        out.set_item("F", PyArray1::from_vec(py, f).reshape([n, n_feat])?)?;
        for (name, v) in [
            ("junction_sig", js),
            ("common_start_sig", cs),
            ("cca_a_sig", cca_sig),
            ("cca_a_dwell", cca_dwell),
            ("junction_dwell", j_dwell),
            ("arm_resolved_depth", arm_depth),
            ("aligner_arm_depth", aln_depth),
            ("polya_mid_sig", polya),
            ("body_mid_sig", body),
            ("mapq", mapq),
            ("ns", ns),
            ("ts", ts),
        ] {
            out.set_item(name, PyArray1::from_vec(py, v))?;
        }
        Ok(out)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnchoredReads>()?;
    // Capability marker. A caller that needs the flank-anchored junction must
    // be able to tell whether THIS binary has it, and a version string cannot
    // answer that -- it can be patched, backported, or built from a dirty
    // tree. Exporting the vocabulary the extractor emits makes the question
    // decidable without a BAM in hand. Consumed by
    // `escapepod_models.charging.rs_reports_anchor_source`.
    m.add("ANCHOR_SOURCES", vec!["exact", "flank_interp", "backfill"])?;
    // Both lists are ORDERED: `coords` emits an index into them rather
    // than the string, so a reader of a saved npz needs them to decode
    // its `anchor_source` / `mask_source` columns.
    m.add(
        "MASK_SOURCES",
        vec!["exact", "counted", "arm_fallback", "junction_fallback"],
    )?;
    Ok(())
}

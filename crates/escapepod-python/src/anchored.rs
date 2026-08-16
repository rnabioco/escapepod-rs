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
    AnchoredRead, MaskSource, Orientation, Pod5Index, SpanMode, finalize, junction_positions,
    query_positions, resolve_orientation, scan_bam,
};

fn value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Where the window sits inside the junction base's own signal span.
///
/// A fixed offset from the base's START makes the window cover a different
/// number of *bases* on a fast read than a slow one, which encodes
/// translocation rate — the nuisance variable that tracks flowcell.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BaseJustify {
    Start,
    Center,
    End,
}

impl BaseJustify {
    fn parse(s: &str) -> PyResult<Self> {
        match s {
            "start" => Ok(Self::Start),
            "center" => Ok(Self::Center),
            "end" => Ok(Self::End),
            other => Err(PyValueError::new_err(format!(
                "base_justify must be start|center|end, got {other:?}"
            ))),
        }
    }
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
    junction_sig: i64,
    common_start_sig: i64,
    mask_source: &'static str,
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
    kmer: Option<(HashMap<String, f64>, usize, usize)>,
    n_records: usize,
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

    #[getter]
    fn n_anchored(&self) -> usize {
        self.n_records
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
        let (levels, k) =
            escapepod_signal::resquiggle::load_kmer_table(&path).map_err(value_err)?;
        let centre = center_idx.unwrap_or(k / 2);
        if centre >= k {
            return Err(PyValueError::new_err(format!(
                "center_idx {centre} outside a {k}-mer"
            )));
        }
        self.kmer = Some((levels, k, centre));
        Ok(k)
    }

    /// Extract windows and per-offset features for a batch of reads.
    ///
    /// Returns a dict of column arrays. `X` is `(n, left + right)` raw pA with
    /// `NaN` for padding and for everything earlier in time than the
    /// common-arm start; `F` is `(n, len(offsets) * 4)` in offsets-outer,
    /// (dwell, mean, std, resid) order.
    ///
    /// Reads with no signal, or whose window would not hold `right` samples
    /// after the anchor, are dropped — `read_id` says which survived rather
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
        let justify = BaseJustify::parse(base_justify)?;
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
                    let coords = finalize(read, self.orientation, &self.offsets, self.mode);

                    let mut anchor = coords.junction_sig;
                    let dwell = read_junction_dwell(read, self.orientation);
                    if dwell > 0 {
                        anchor += match justify {
                            BaseJustify::Start => 0,
                            BaseJustify::Center => dwell / 2,
                            BaseJustify::End => dwell,
                        };
                    }

                    let (lo, hi) = (anchor - left, anchor + right);
                    let (src_lo, src_hi) = (lo.max(0), hi.min(sig.len() as i64));
                    if src_hi - src_lo < right {
                        return None;
                    }

                    let mut window = vec![f32::NAN; w];
                    let dst = (src_lo - lo) as usize;
                    window[dst..dst + (src_hi - src_lo) as usize]
                        .copy_from_slice(&sig[src_lo as usize..src_hi as usize]);
                    // Mask everything earlier than the common arm: poly(A) and
                    // the divergent 13-mer, which differ between libraries.
                    let cut = coords.common_start_sig - lo;
                    if cut > 0 {
                        let end = (cut as usize).min(w);
                        window[..end].fill(f32::NAN);
                    }

                    let expected = self.kmer.as_ref().map(|(levels, k, centre)| {
                        // The SAME positions the spans came from — under the
                        // counting anchor `read.qf` is the aligner's answer,
                        // and using it here leaves dwell/mean/std right while
                        // the residual is silently wrong.
                        let qf = query_positions(read, &self.offsets, self.mode);
                        escapepod_classify::expected_levels_z(
                            &read.seq, levels, *k, *centre, &qf, read.nb,
                        )
                    });
                    let features =
                        escapepod_classify::junction_features(&sig, &coords, expected.as_deref());

                    Some(Row {
                        idx: i,
                        window,
                        features,
                        junction_sig: coords.junction_sig,
                        common_start_sig: coords.common_start_sig,
                        mask_source: mask_source_name(coords.mask_source),
                    })
                })
                .collect()
        });

        let mut rows = rows;
        rows.sort_unstable_by_key(|r| r.idx);

        let n = rows.len();
        let mut x = Vec::with_capacity(n * w);
        let mut f = Vec::with_capacity(n * n_feat);
        let (mut js, mut cs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut kept, mut msrc, mut refs, mut mapq) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for r in &rows {
            x.extend_from_slice(&r.window);
            f.extend_from_slice(&r.features);
            js.push(r.junction_sig);
            cs.push(r.common_start_sig);
            msrc.push(r.mask_source);
            let id = ids[r.idx];
            kept.push(id.to_string());
            let read = &self.anchored[&id];
            refs.push(read.reference.clone());
            mapq.push(read.mapq as i64);
        }

        let out = PyDict::new(py);
        out.set_item("read_id", kept)?;
        out.set_item("reference", refs)?;
        out.set_item("mask_source", msrc)?;
        out.set_item("X", PyArray1::from_vec(py, x).reshape([n, w])?)?;
        out.set_item("F", PyArray1::from_vec(py, f).reshape([n, n_feat])?)?;
        out.set_item("junction_sig", PyArray1::from_vec(py, js))?;
        out.set_item("common_start_sig", PyArray1::from_vec(py, cs))?;
        out.set_item("mapq", PyArray1::from_vec(py, mapq))?;
        Ok(out)
    }
}

/// Samples assigned to the junction base, in the run's frame.
fn read_junction_dwell(read: &AnchoredRead, orientation: Orientation) -> i64 {
    let s2s = &read.seq_to_sig;
    let i = read.q_junction;
    if i + 1 >= s2s.len() {
        return 0;
    }
    let (a, b) = (s2s[i], s2s[i + 1]);
    let (a, b) = if orientation == Orientation::Reversed {
        (read.ns - b, read.ns - a)
    } else {
        (a, b)
    };
    (b - a).max(0)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnchoredReads>()?;
    Ok(())
}

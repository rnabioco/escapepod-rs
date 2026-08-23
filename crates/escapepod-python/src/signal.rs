//! Python bindings for `escapepod-signal` algorithms: normalization, kmer
//! level tables, and signal-to-sequence refinement (resquiggle).

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::path::PathBuf;

use escapepod_signal::features::{
    MedianConvention, Normalization, SpanBounds, SpanConfig, SpanFill, SpanScratch, SpanStatsOut,
    span_stats,
};
use escapepod_signal::resquiggle::{KmerTable, RefineAlgo, RefineSettings, refine_signal_map};
use escapepod_signal::segmentation;
use rayon::prelude::*;

/// Map an `anyhow`/`Display` error into a Python `ValueError`.
fn value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Median-MAD normalize a float32 signal (1.4826 Gaussian scale factor,
/// graceful fallback on a constant signal).
#[pyfunction]
fn mad_normalize<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let out = segmentation::mad_normalize_robust(signal.as_slice()?);
    Ok(PyArray1::from_vec(py, out))
}

/// Normalize a raw int16 (DAC) signal to float32 via median-MAD.
#[pyfunction]
fn normalize_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, i16>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let out = segmentation::normalize_signal(signal.as_slice()?);
    Ok(PyArray1::from_vec(py, out))
}

/// A kmer level table loaded from a `kmer\tlevel` file (gzip supported).
#[pyclass(name = "KmerTable")]
pub struct PyKmerTable {
    inner: KmerTable,
}

#[pymethods]
impl PyKmerTable {
    /// Load a kmer table from a tab-delimited `kmer\tlevel` file (`.gz` ok).
    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        KmerTable::from_file(&path)
            .map(|inner| Self { inner })
            .map_err(value_err)
    }

    /// Kmer length.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// Expected level for a single kmer.
    fn get(&self, kmer: &str) -> PyResult<f32> {
        self.inner.get(kmer.as_bytes()).map_err(value_err)
    }

    /// Position within the kmer that this table assigns its level to.
    ///
    /// Empirical, and not necessarily the midpoint: on the RNA004 9-mer table
    /// it is 3 while `k / 2` is 4.
    #[getter]
    fn dominant_base(&self) -> usize {
        self.inner.dominant_base()
    }

    /// Per-base expected levels for a sequence, centred on
    /// [`dominant_base`](Self::dominant_base).
    fn extract_levels<'py>(
        &self,
        py: Python<'py>,
        seq: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let levels = self
            .inner
            .extract_levels(seq.as_bytes())
            .map_err(value_err)?;
        Ok(PyArray1::from_vec(py, levels))
    }

    /// Per-base expected levels with an explicit position within the kmer.
    ///
    /// Callers inheriting leech's convention want `k / 2`; this crate's
    /// resquiggle wants `dominant_base`. Those differ by one on RNA004, which
    /// shifts every predicted level, so neither is assumed.
    fn extract_levels_at<'py>(
        &self,
        py: Python<'py>,
        seq: &str,
        centre: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let levels = self
            .inner
            .extract_levels_at(seq.as_bytes(), centre)
            .map_err(value_err)?;
        Ok(PyArray1::from_vec(py, levels))
    }
}

fn normalization(mad_floor: Option<f32>) -> Normalization {
    match mad_floor {
        Some(f) => Normalization::MedianMad { mad_floor: f },
        None => Normalization::None,
    }
}

/// Assemble the Rust-side [`SpanConfig`] from the keyword knobs.
///
/// `fill = None` is `SpanFill::Nan`; any float is `SpanFill::Value`, of which
/// `0.0` is the `SpanFill::Zero` a model-feeding caller wants. The two string
/// knobs are spelled out rather than boolean because they are policies, not
/// switches, and a wrong one has to be visible in the traceback.
fn span_config(
    mad_floor: Option<f32>,
    fill: Option<f32>,
    bounds: &str,
    median_convention: &str,
) -> PyResult<SpanConfig> {
    let bounds = match bounds {
        "skip" => SpanBounds::Skip,
        "clamp" => SpanBounds::Clamp,
        other => {
            return Err(PyValueError::new_err(format!(
                "bounds must be 'skip' or 'clamp', got {other:?}"
            )));
        }
    };
    let median = match median_convention {
        "select" => MedianConvention::SelectTotalCmp,
        "sort" => MedianConvention::SortPartialCmp,
        other => {
            return Err(PyValueError::new_err(format!(
                "median_convention must be 'select' or 'sort', got {other:?}"
            )));
        }
    };
    Ok(SpanConfig {
        norm: normalization(mad_floor),
        fill: fill.map_or(SpanFill::Nan, SpanFill::Value),
        bounds,
        median,
    })
}

/// Per-span `(dwell, mean, sd)` for one read.
///
/// `spans` is `(n, 2)` of `[start, end)` signal indices; a span that does not
/// resolve comes back as `fill` in every output. `mad_floor` selects per-read
/// median/MAD normalisation with that flat-read fallback threshold; omit it to
/// summarise the signal as given.
///
/// The optional knobs, all keyword-only and all defaulting to the historical
/// behaviour:
///
/// - `median=True` / `range=True` append a fourth and fifth output array, in
///   that order. Neither is computed unless asked for -- the median needs its
///   own pass and a select over each span, which a caller wanting only
///   `dwell`/`mean`/`sd` should not pay for.
/// - `fill` is the value written for an unresolved span: `None` (the default)
///   means `NaN`, and any float is used verbatim -- pass `0.0` when the arrays
///   feed a network that a `NaN` would poison.
/// - `bounds` is `"skip"` (default; a span that is negative or runs past the
///   end does not resolve at all) or `"clamp"` (intersect it with the signal
///   and summarise what survives, with `dwell` reporting the *clamped* length).
/// - `median_convention` is `"select"` (default; `select_nth_unstable` with
///   `total_cmp`, the convention every other median in escapepod-signal uses)
///   or `"sort"` (a full sort with `partial_cmp`, reproducing `numpy.median`
///   exactly, `NaN` propagation included). Both average the two middle
///   elements on an even-length span.
#[pyfunction]
#[pyo3(signature = (
    signal,
    spans,
    mad_floor=None,
    *,
    median=false,
    range=false,
    fill=None,
    bounds="skip",
    median_convention="select",
))]
#[allow(clippy::too_many_arguments)]
fn span_statistics<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    spans: PyReadonlyArray2<'py, i64>,
    mad_floor: Option<f32>,
    median: bool,
    range: bool,
    fill: Option<f32>,
    bounds: &str,
    median_convention: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let cfg = span_config(mad_floor, fill, bounds, median_convention)?;
    let sig = signal.as_slice()?;
    let sp = spans_as_pairs(&spans)?;
    let n = sp.len();
    let (mut d, mut m, mut s) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let mut med = vec![0.0f32; if median { n } else { 0 }];
    let mut rng = vec![0.0f32; if range { n } else { 0 }];
    let mut scratch = SpanScratch::default();
    let mut out = SpanStatsOut::new(&mut d, &mut m, &mut s);
    if median {
        out = out.with_median(&mut med);
    }
    if range {
        out = out.with_range(&mut rng);
    }
    span_stats(sig, sp, cfg, &mut scratch, out);

    let mut cols = vec![
        PyArray1::from_vec(py, d),
        PyArray1::from_vec(py, m),
        PyArray1::from_vec(py, s),
    ];
    if median {
        cols.push(PyArray1::from_vec(py, med));
    }
    if range {
        cols.push(PyArray1::from_vec(py, rng));
    }
    PyTuple::new(py, cols)
}

/// Reinterpret an `(n, 2)` i64 array as `&[[i64; 2]]` without copying.
fn spans_as_pairs<'a>(spans: &'a PyReadonlyArray2<'a, i64>) -> PyResult<&'a [[i64; 2]]> {
    let dims = spans.shape();
    if dims.len() != 2 || dims[1] != 2 {
        return Err(PyValueError::new_err("spans must have shape (n, 2)"));
    }
    let flat = spans.as_slice()?;
    // Safe: [[i64; 2]] has the same layout as a contiguous i64 pair sequence,
    // and as_slice() already guaranteed C-contiguity.
    Ok(unsafe { std::slice::from_raw_parts(flat.as_ptr().cast::<[i64; 2]>(), dims[0]) })
}

/// [`span_statistics`] over many reads at once, in parallel, GIL released.
///
/// The batch is laid out flat so nothing is copied: `signal` is every read's
/// samples concatenated, `read_offsets` is the `n_reads + 1` boundaries into
/// it, and `spans` is `(n_reads * spans_per_read, 2)` with indices **relative
/// to each read's own start**. Returns three `(n_reads, spans_per_read)`
/// arrays.
///
/// This is the shape that makes per-read feature extraction worth doing in
/// Rust: the work is embarrassingly parallel and entirely numeric, so it scales
/// with cores instead of being serialised behind the interpreter.
#[pyfunction]
#[pyo3(signature = (
    signal,
    read_offsets,
    spans,
    spans_per_read,
    mad_floor=None,
    *,
    median=false,
    range=false,
    fill=None,
    bounds="skip",
    median_convention="select",
))]
#[allow(clippy::too_many_arguments)]
fn span_statistics_batch<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    read_offsets: PyReadonlyArray1<'py, i64>,
    spans: PyReadonlyArray2<'py, i64>,
    spans_per_read: usize,
    mad_floor: Option<f32>,
    median: bool,
    range: bool,
    fill: Option<f32>,
    bounds: &str,
    median_convention: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let cfg = span_config(mad_floor, fill, bounds, median_convention)?;
    let sig = signal.as_slice()?;
    let offs = read_offsets.as_slice()?;
    let sp = spans_as_pairs(&spans)?;
    if offs.is_empty() {
        return Err(PyValueError::new_err("read_offsets must not be empty"));
    }
    let n_reads = offs.len() - 1;
    if spans_per_read == 0 {
        return Err(PyValueError::new_err("spans_per_read must be > 0"));
    }
    if sp.len() != n_reads * spans_per_read {
        return Err(PyValueError::new_err(format!(
            "spans has {} rows, expected n_reads * spans_per_read = {}",
            sp.len(),
            n_reads * spans_per_read
        )));
    }
    for w in offs.windows(2) {
        if w[0] < 0 || w[1] < w[0] || w[1] as usize > sig.len() {
            return Err(PyValueError::new_err(
                "read_offsets must be non-decreasing and within signal",
            ));
        }
    }

    let total = n_reads * spans_per_read;
    let (mut d, mut m, mut s) = (
        vec![0.0f32; total],
        vec![0.0f32; total],
        vec![0.0f32; total],
    );
    let mut med = vec![0.0f32; if median { total } else { 0 }];
    let mut rng = vec![0.0f32; if range { total } else { 0 }];

    // An unrequested optional output is `None` for every read rather than a
    // zero-length buffer, so the rayon zip below keeps one arm per read either
    // way and nothing is allocated for it.
    fn per_read(buf: &mut [f32], on: bool, k: usize, n_reads: usize) -> Vec<Option<&mut [f32]>> {
        if on {
            buf.chunks_mut(k).map(Some).collect()
        } else {
            (0..n_reads).map(|_| None).collect()
        }
    }
    let mut med_rows = per_read(&mut med, median, spans_per_read, n_reads);
    let mut rng_rows = per_read(&mut rng, range, spans_per_read, n_reads);

    py.detach(|| {
        d.par_chunks_mut(spans_per_read)
            .zip(m.par_chunks_mut(spans_per_read))
            .zip(s.par_chunks_mut(spans_per_read))
            .zip(med_rows.par_iter_mut())
            .zip(rng_rows.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((((dw, mn), sd), md), rg))| {
                let read = &sig[offs[i] as usize..offs[i + 1] as usize];
                let rs = &sp[i * spans_per_read..(i + 1) * spans_per_read];
                // Scratch is per-task, not shared: rayon may run any number of
                // these concurrently.
                let mut scratch = SpanScratch::default();
                let mut out = SpanStatsOut::new(dw, mn, sd);
                if let Some(row) = md.take() {
                    out = out.with_median(row);
                }
                if let Some(row) = rg.take() {
                    out = out.with_range(row);
                }
                span_stats(read, rs, cfg, &mut scratch, out);
            });
    });
    drop((med_rows, rng_rows));

    let reshape = |v: Vec<f32>| -> PyResult<Bound<'py, PyArray2<f32>>> {
        PyArray1::from_vec(py, v).reshape([n_reads, spans_per_read])
    };
    let mut cols = vec![reshape(d)?, reshape(m)?, reshape(s)?];
    if median {
        cols.push(reshape(med)?);
    }
    if range {
        cols.push(reshape(rng)?);
    }
    PyTuple::new(py, cols)
}

/// Refine a signal-to-sequence boundary map against a level model.
///
/// Uses `RefineSettings::move_table_refinement` — escapepod's named preset for
/// refining a basecaller move table (fixed banding, least-squares rough rescale
/// over the 0.05–0.95 quantiles clipped 10 bases, Theil-Sen inter-iteration
/// rescale over at most 200 points, asymmetric dwell penalty at weight 0.5 with
/// a per-read target). Rust callers that want the same refinement construct the
/// same preset rather than transcribing its fields, because a transcription of
/// this block previously drifted in `dwell_target` and the two paths refined
/// the same reads to different boundaries.
///
/// The dwell target is resolved **per read** from the median dwell of the input
/// `seq_to_signal_map`, so it tracks the chemistry and translocation rate of
/// the data instead of a constant. Pass `dwell_target`/`dwell_weight` only to
/// override the preset; `None` (the default) uses it.
///
/// `signal` must already be normalized. Returns
/// `(refined_seq_to_signal_map, scale, shift, drift)`.
///
/// The rescale parameters are returned **for inspection**; whether to apply
/// them is the caller's decision, and escapepod does not apply them to the
/// returned map. They would be applied as
/// `(signal[i] - shift - drift*i) / scale`. Be aware of how the fit can fail:
/// a per-read affine fit estimated over a near-constant stretch of signal (a
/// 3' adapter, a homopolymer) is weakly identified, and in practice produces
/// wild or negative scales — observed values ranged from 15 to 1084 on tRNA
/// reads, sign flips included. Downstream consumers that refine over such a
/// region discard these values and keep their own normalization.
#[pyfunction]
#[pyo3(
    name = "refine_signal_map",
    signature = (
        signal,
        seq_to_signal_map,
        expected_levels,
        half_bandwidth = 5,
        scale_iters = 2,
        dwell_target = None,
        dwell_weight = None,
        seed = None,
    )
)]
#[allow(clippy::too_many_arguments)]
fn py_refine_signal_map<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    seq_to_signal_map: Vec<usize>,
    expected_levels: PyReadonlyArray1<'py, f32>,
    half_bandwidth: usize,
    scale_iters: usize,
    dwell_target: Option<f32>,
    dwell_weight: Option<f32>,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, f32, f32, f32)> {
    let mut settings = RefineSettings::move_table_refinement(half_bandwidth, scale_iters, seed);
    if dwell_target.is_some() || dwell_weight.is_some() {
        settings.refinement_algo = RefineAlgo::DwellPenalty {
            target: dwell_target.unwrap_or(RefineAlgo::PER_READ_DWELL_TARGET),
            weight: dwell_weight.unwrap_or(RefineSettings::MOVE_TABLE_DWELL_WEIGHT),
        };
    }

    let result = refine_signal_map(
        &settings,
        signal.as_slice()?,
        &seq_to_signal_map,
        expected_levels.as_slice()?,
        1.0,
        0.0,
    )
    .map_err(value_err)?;

    let refined: Vec<i64> = result.seq_to_signal_map.iter().map(|&v| v as i64).collect();
    Ok((
        PyArray1::from_vec(py, refined),
        result.scale,
        result.shift,
        result.drift,
    ))
}

/// Register the signal-processing bindings on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKmerTable>()?;
    m.add_function(wrap_pyfunction!(mad_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_signal, m)?)?;
    m.add_function(wrap_pyfunction!(py_refine_signal_map, m)?)?;
    m.add_function(wrap_pyfunction!(span_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(span_statistics_batch, m)?)?;
    Ok(())
}

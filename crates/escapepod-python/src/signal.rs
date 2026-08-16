//! Python bindings for `escapepod-signal` algorithms: normalization, kmer
//! level tables, and signal-to-sequence refinement (resquiggle).

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use escapepod_signal::features::{Normalization, SpanScratch, SpanStatsOut, span_stats};
use escapepod_signal::resquiggle::{
    BandingAlgo, KmerTable, RefineAlgo, RefineSettings, RescaleAlgo, RescaleFilterParams,
    RoughRescaleAlgo, refine_signal_map,
};
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

/// `(dwell, mean, sd)`, one entry per span.
type SpanTriple1<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);
/// `(dwell, mean, sd)`, `(n_reads, spans_per_read)` each.
type SpanTriple2<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
);

fn normalization(mad_floor: Option<f32>) -> Normalization {
    match mad_floor {
        Some(f) => Normalization::MedianMad { mad_floor: f },
        None => Normalization::None,
    }
}

/// Per-span `(dwell, mean, sd)` for one read.
///
/// `spans` is `(n, 2)` of `[start, end)` signal indices; invalid or unresolved
/// spans come back `NaN` in all three outputs. `mad_floor` selects per-read
/// median/MAD normalisation with that flat-read fallback threshold; omit it to
/// summarise the signal as given.
#[pyfunction]
#[pyo3(signature = (signal, spans, mad_floor=None))]
fn span_statistics<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    spans: PyReadonlyArray2<'py, i64>,
    mad_floor: Option<f32>,
) -> PyResult<SpanTriple1<'py>> {
    let sig = signal.as_slice()?;
    let sp = spans_as_pairs(&spans)?;
    let n = sp.len();
    let (mut d, mut m, mut s) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let mut scratch = SpanScratch::default();
    span_stats(
        sig,
        sp,
        normalization(mad_floor),
        &mut scratch,
        SpanStatsOut {
            dwell: &mut d,
            mean: &mut m,
            sd: &mut s,
        },
    );
    Ok((
        PyArray1::from_vec(py, d),
        PyArray1::from_vec(py, m),
        PyArray1::from_vec(py, s),
    ))
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
#[pyo3(signature = (signal, read_offsets, spans, spans_per_read, mad_floor=None))]
fn span_statistics_batch<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    read_offsets: PyReadonlyArray1<'py, i64>,
    spans: PyReadonlyArray2<'py, i64>,
    spans_per_read: usize,
    mad_floor: Option<f32>,
) -> PyResult<SpanTriple2<'py>> {
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
    let norm = normalization(mad_floor);

    py.detach(|| {
        d.par_chunks_mut(spans_per_read)
            .zip(m.par_chunks_mut(spans_per_read))
            .zip(s.par_chunks_mut(spans_per_read))
            .enumerate()
            .for_each(|(i, ((dw, mn), sd))| {
                let read = &sig[offs[i] as usize..offs[i + 1] as usize];
                let rs = &sp[i * spans_per_read..(i + 1) * spans_per_read];
                // Scratch is per-task, not shared: rayon may run any number of
                // these concurrently.
                let mut scratch = SpanScratch::default();
                span_stats(
                    read,
                    rs,
                    norm,
                    &mut scratch,
                    SpanStatsOut {
                        dwell: dw,
                        mean: mn,
                        sd,
                    },
                );
            });
    });

    let reshape = |v: Vec<f32>| -> PyResult<Bound<'py, PyArray2<f32>>> {
        PyArray1::from_vec(py, v).reshape([n_reads, spans_per_read])
    };
    Ok((reshape(d)?, reshape(m)?, reshape(s)?))
}

/// Refine a signal-to-sequence boundary map against a level model.
///
/// Uses leech's refinement configuration (fixed banding, least-squares rough
/// rescale over the 0.05–0.95 quantiles clipped 10 bases, Theil-Sen
/// inter-iteration rescale, asymmetric dwell penalty) so the Python path
/// matches leech_core's Rust path bit-for-bit.
///
/// `signal` must already be normalized. Returns
/// `(refined_seq_to_signal_map, scale, shift, drift)`; apply the rescale as
/// `(signal[i] - shift - drift*i) / scale` to recover the level-matched signal.
#[pyfunction]
#[pyo3(
    name = "refine_signal_map",
    signature = (
        signal,
        seq_to_signal_map,
        expected_levels,
        half_bandwidth = 5,
        scale_iters = 2,
        dwell_target = 4.0,
        dwell_weight = 0.5,
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
    dwell_target: f32,
    dwell_weight: f32,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, f32, f32, f32)> {
    let settings = RefineSettings {
        refinement_algo: RefineAlgo::DwellPenalty {
            target: dwell_target,
            weight: dwell_weight,
        },
        n_refinement_iters: scale_iters,
        half_bandwidth,
        adjust_band_min_size: 2,
        rescale_algo: RescaleAlgo::TheilSen {
            filter: RescaleFilterParams::default(),
            max_points: 200,
            seed,
        },
        rough_rescale_algo: RoughRescaleAlgo::LeastSquares {
            quantiles: RoughRescaleAlgo::default_quantiles(),
            clip_bases: 10,
            use_base_center: true,
        },
        normalize_levels: false,
        banding_algo: BandingAlgo::Fixed,
    };

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

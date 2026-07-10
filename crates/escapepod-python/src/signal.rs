//! Python bindings for `escapepod-signal` algorithms: normalization, kmer
//! level tables, and signal-to-sequence refinement (resquiggle).

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use escapepod_signal::resquiggle::{
    BandingAlgo, KmerTable, RefineAlgo, RefineSettings, RescaleAlgo, RescaleFilterParams,
    RoughRescaleAlgo, refine_signal_map,
};
use escapepod_signal::segmentation;

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

    /// Per-base expected levels for a sequence.
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
    Ok(())
}

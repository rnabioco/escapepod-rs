use std::sync::Arc;

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::run_info::RunInfo;

/// Calibration parameters for converting ADC signal to picoamps.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Calibration {
    #[pyo3(get)]
    offset: f32,
    #[pyo3(get)]
    scale: f32,
}

#[pymethods]
impl Calibration {
    fn __repr__(&self) -> String {
        format!("Calibration(offset={}, scale={})", self.offset, self.scale)
    }
}

/// A single read from a POD5 file.
///
/// Signal data is loaded lazily on first access to `signal` or `signal_pa`.
#[pyclass(frozen)]
pub struct Read {
    data: escapepod::ReadData,
    reader: Arc<escapepod::Reader>,
    cached_run_info: RunInfo,
    cached_sample_rate: u16,
}

impl Read {
    pub fn new(
        data: escapepod::ReadData,
        reader: Arc<escapepod::Reader>,
        run_info: RunInfo,
        sample_rate: u16,
    ) -> Self {
        Self {
            data,
            reader,
            cached_run_info: run_info,
            cached_sample_rate: sample_rate,
        }
    }
}

#[pymethods]
impl Read {
    /// Unique read identifier as a string.
    #[getter]
    fn read_id(&self) -> String {
        self.data.read_id.to_string()
    }

    /// Sequential read number within the run.
    #[getter]
    fn read_number(&self) -> u32 {
        self.data.read_number
    }

    /// Start sample number (absolute position in acquisition).
    #[getter]
    fn start_sample(&self) -> u64 {
        self.data.start_sample
    }

    /// Channel number (1-indexed).
    #[getter]
    fn channel(&self) -> u16 {
        self.data.channel
    }

    /// Well number (typically 1-4).
    #[getter]
    fn well(&self) -> u8 {
        self.data.well
    }

    /// Pore type string.
    #[getter]
    fn pore_type(&self) -> &str {
        &self.data.pore_type
    }

    /// Calibration parameters (offset and scale).
    #[getter]
    fn calibration(&self) -> Calibration {
        Calibration {
            offset: self.data.calibration_offset,
            scale: self.data.calibration_scale,
        }
    }

    /// Calibration offset for ADC to pA conversion.
    #[getter]
    fn calibration_offset(&self) -> f32 {
        self.data.calibration_offset
    }

    /// Calibration scale for ADC to pA conversion.
    #[getter]
    fn calibration_scale(&self) -> f32 {
        self.data.calibration_scale
    }

    /// Median current before the read started.
    #[getter]
    fn median_before(&self) -> f32 {
        self.data.median_before
    }

    /// Reason the read ended (as a string).
    #[getter]
    fn end_reason(&self) -> &str {
        self.data.end_reason.as_str()
    }

    /// Whether the end reason was forced.
    #[getter]
    fn end_reason_forced(&self) -> bool {
        self.data.end_reason_forced
    }

    /// Total number of signal samples.
    #[getter]
    fn num_samples(&self) -> u64 {
        self.data.num_samples
    }

    /// Number of MinKNOW events.
    #[getter]
    fn num_minknow_events(&self) -> u64 {
        self.data.num_minknow_events
    }

    /// Sample rate from the associated run info (Hz).
    #[getter]
    fn sample_rate(&self) -> u16 {
        self.cached_sample_rate
    }

    /// Run information for this read.
    #[getter]
    fn run_info(&self) -> RunInfo {
        self.cached_run_info.clone()
    }

    /// Raw signal as a numpy int16 array.
    ///
    /// The GIL is released during decompression for better concurrency.
    #[getter]
    fn signal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let signal_rows = self.data.signal_rows.clone();
        let reader = self.reader.clone();

        let samples = py
            .allow_threads(move || reader.get_signal(&signal_rows))
            .map_err(|e| crate::to_py_err(&e))?;

        Ok(PyArray1::from_vec(py, samples))
    }

    /// Calibrated signal in picoamps as a numpy float32 array.
    ///
    /// Computed as: (raw_signal + offset) * scale
    /// The GIL is released during decompression.
    #[getter]
    fn signal_pa<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let signal_rows = self.data.signal_rows.clone();
        let reader = self.reader.clone();
        let offset = self.data.calibration_offset;
        let scale = self.data.calibration_scale;

        let samples = py
            .allow_threads(move || reader.get_signal(&signal_rows))
            .map_err(|e| crate::to_py_err(&e))?;

        let pa: Vec<f32> = samples
            .iter()
            .map(|&s| (s as f32 + offset) * scale)
            .collect();

        Ok(PyArray1::from_vec(py, pa))
    }

    fn __repr__(&self) -> String {
        format!(
            "Read(read_id='{}', channel={}, num_samples={})",
            self.data.read_id, self.data.channel, self.data.num_samples
        )
    }
}

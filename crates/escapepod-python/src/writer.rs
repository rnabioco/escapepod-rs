use std::collections::HashMap;
use std::path::PathBuf;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::read_data::PyRunInfo;

/// Writer for POD5 files.
///
/// Can be used as a context manager:
///
///     with Writer("output.pod5") as writer:
///         ri_idx = writer.add_run_info(run_info)
///         writer.add_read(read_data_dict, signal)
#[pyclass(name = "Writer")]
pub struct PyWriter {
    inner: Option<escapepod_signal::Writer>,
}

#[pymethods]
impl PyWriter {
    /// Create a new POD5 file for writing.
    ///
    /// Parameters
    /// ----------
    /// path : str or PathLike
    ///     Output file path.
    /// max_signal_chunk_size : int, optional
    ///     Maximum samples per signal chunk (default: 102400).
    /// signal_batch_size : int, optional
    ///     Signal chunks per batch (default: 100).
    /// read_batch_size : int, optional
    ///     Reads per batch (default: 1000).
    /// compress_signal : bool, optional
    ///     Whether to VBZ-compress signal (default: True).
    /// software : str, optional
    ///     Software name for metadata.
    #[new]
    #[pyo3(signature = (
        path,
        max_signal_chunk_size=None,
        signal_batch_size=None,
        read_batch_size=None,
        compress_signal=None,
        software=None,
    ))]
    fn new(
        path: PathBuf,
        max_signal_chunk_size: Option<u32>,
        signal_batch_size: Option<u32>,
        read_batch_size: Option<u32>,
        compress_signal: Option<bool>,
        software: Option<String>,
    ) -> PyResult<Self> {
        let mut opts = escapepod_signal::WriterOptions::default();
        if let Some(v) = max_signal_chunk_size {
            opts.max_signal_chunk_size = v;
        }
        if let Some(v) = signal_batch_size {
            opts.signal_batch_size = v;
        }
        if let Some(v) = read_batch_size {
            opts.read_batch_size = v;
        }
        if let Some(v) = compress_signal {
            opts.compress_signal = v;
        }
        if let Some(v) = software {
            opts.software = v;
        }

        let writer = escapepod_signal::Writer::create(&path, opts).map_err(to_py_err)?;
        Ok(Self {
            inner: Some(writer),
        })
    }

    /// Add run information. Returns the run info index.
    fn add_run_info(&mut self, run_info: &PyRunInfo) -> PyResult<u32> {
        let writer = self.writer_mut()?;
        writer
            .add_run_info(run_info.inner.clone())
            .map_err(to_py_err)
    }

    /// Add a read with its raw ADC signal (int16 numpy array).
    ///
    /// Parameters
    /// ----------
    /// read_id : str
    ///     UUID of the read.
    /// read_number : int
    ///     Read number.
    /// start_sample : int
    ///     Start sample index.
    /// channel : int
    ///     Channel number.
    /// well : int
    ///     Well number.
    /// pore_type : str
    ///     Pore type name.
    /// calibration_offset : float
    ///     Calibration offset.
    /// calibration_scale : float
    ///     Calibration scale.
    /// median_before : float
    ///     Median signal level before the read.
    /// end_reason : str
    ///     Why the read ended (e.g. "signal_positive").
    /// end_reason_forced : bool
    ///     Whether the end was forced.
    /// run_info_index : int
    ///     Index of the run info (from add_run_info).
    /// num_minknow_events : int
    ///     Number of MinKNOW events.
    /// signal : numpy.ndarray[int16]
    ///     Raw ADC signal data.
    /// num_samples : int, optional
    ///     Number of samples (default: len(signal)).
    /// tracked_scaling_scale, tracked_scaling_shift : float, optional
    ///     Tracked scaling parameters (defaults: 1.0, 0.0).
    /// predicted_scaling_scale, predicted_scaling_shift : float, optional
    ///     Predicted scaling parameters (defaults: 1.0, 0.0).
    /// num_reads_since_mux_change : int, optional
    ///     Reads since last mux change (default: 0).
    /// time_since_mux_change : float, optional
    ///     Seconds since last mux change (default: 0.0).
    /// open_pore_level : float, optional
    ///     Estimated open pore current (default: 0.0).
    /// expected_open_pore_level : float, optional
    ///     Expected open pore current level for this read (POD5 V5; default: 0.0).
    /// selected_read_level : float, optional
    ///     Selected pore level for this read (POD5 V5; default: 0.0).
    #[pyo3(signature = (
        read_id,
        read_number,
        start_sample,
        channel,
        well,
        pore_type,
        calibration_offset,
        calibration_scale,
        median_before,
        end_reason,
        end_reason_forced,
        run_info_index,
        num_minknow_events,
        signal,
        num_samples=None,
        tracked_scaling_scale=1.0,
        tracked_scaling_shift=0.0,
        predicted_scaling_scale=1.0,
        predicted_scaling_shift=0.0,
        num_reads_since_mux_change=0,
        time_since_mux_change=0.0,
        open_pore_level=0.0,
        expected_open_pore_level=0.0,
        selected_read_level=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_read(
        &mut self,
        read_id: &str,
        read_number: u32,
        start_sample: u64,
        channel: u16,
        well: u8,
        pore_type: &str,
        calibration_offset: f32,
        calibration_scale: f32,
        median_before: f32,
        end_reason: &str,
        end_reason_forced: bool,
        run_info_index: u32,
        num_minknow_events: u64,
        signal: &Bound<'_, PyArray1<i16>>,
        num_samples: Option<u64>,
        tracked_scaling_scale: f32,
        tracked_scaling_shift: f32,
        predicted_scaling_scale: f32,
        predicted_scaling_shift: f32,
        num_reads_since_mux_change: u32,
        time_since_mux_change: f32,
        open_pore_level: f32,
        expected_open_pore_level: f32,
        selected_read_level: f32,
    ) -> PyResult<()> {
        let uuid = escapepod_signal::utils::parse_uuid_flexible(read_id)
            .map_err(|e| to_py_err(escapepod_signal::Error::InvalidUuid(e.to_string())))?;

        let parsed_end_reason = parse_end_reason(end_reason)?;

        let signal_vec: Vec<i16> = signal.to_vec()?;
        let sample_count = num_samples.unwrap_or(signal_vec.len() as u64);

        let read = escapepod_signal::ReadData {
            read_id: uuid,
            read_number,
            start_sample,
            channel,
            well,
            pore_type: pore_type.into(),
            calibration_offset,
            calibration_scale,
            median_before,
            end_reason: parsed_end_reason,
            end_reason_forced,
            run_info_index,
            num_minknow_events,
            tracked_scaling_scale,
            tracked_scaling_shift,
            predicted_scaling_scale,
            predicted_scaling_shift,
            num_reads_since_mux_change,
            time_since_mux_change,
            num_samples: sample_count,
            open_pore_level,
            expected_open_pore_level,
            selected_read_level,
            signal_rows: Vec::new(), // Writer populates this
        };

        let writer = self.writer_mut()?;
        writer.add_read(read, &signal_vec).map_err(to_py_err)
    }

    /// Add a read from an existing ReadData object and signal array.
    fn add_read_data(
        &mut self,
        read: &crate::read_data::PyReadData,
        signal: &Bound<'_, PyArray1<i16>>,
    ) -> PyResult<()> {
        let signal_vec: Vec<i16> = signal.to_vec()?;
        let writer = self.writer_mut()?;
        writer
            .add_read(read.inner.clone(), &signal_vec)
            .map_err(to_py_err)
    }

    /// Add many reads at once from parallel lists of ReadData and signals.
    ///
    /// Equivalent to calling `add_read_data` for each pair, but in a single
    /// call (mirrors `pod5.Writer.add_reads`). `reads` and `signals` must be
    /// the same length.
    fn add_reads(
        &mut self,
        reads: Vec<PyRef<'_, crate::read_data::PyReadData>>,
        signals: Vec<Bound<'_, PyArray1<i16>>>,
    ) -> PyResult<()> {
        if reads.len() != signals.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reads and signals must be the same length ({} != {})",
                reads.len(),
                signals.len()
            )));
        }
        // Materialize signals before touching the writer so a bad array errors
        // out before any read is committed.
        let signal_vecs: Vec<Vec<i16>> = signals
            .iter()
            .map(|s| s.to_vec())
            .collect::<Result<_, _>>()?;

        let writer = self.writer_mut()?;
        for (read, signal) in reads.iter().zip(&signal_vecs) {
            writer
                .add_read(read.inner.clone(), signal)
                .map_err(to_py_err)?;
        }
        Ok(())
    }

    /// Finalize and close the POD5 file.
    ///
    /// The output appears at its destination only once this succeeds.
    fn close(&mut self) -> PyResult<()> {
        if let Some(writer) = self.inner.take() {
            writer.finish().map_err(to_py_err)?;
        }
        Ok(())
    }

    /// Discard the file being written without finalizing it.
    ///
    /// Nothing is left at the destination, and any file that was already
    /// there is untouched.
    fn abort(&mut self) -> PyResult<()> {
        if let Some(writer) = self.inner.take() {
            writer.abort().map_err(to_py_err)?;
        }
        Ok(())
    }

    // -- Context manager protocol ------------------------------------------

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[allow(unused_variables)]
    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_val: Option<&Bound<'_, PyAny>>,
        exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        // Leaving the block via an exception means the caller never finished
        // describing the file, so committing a half-populated archive would be
        // wrong. Staging makes discarding it possible.
        if exc_type.is_some() {
            self.abort()?;
        } else {
            self.close()?;
        }
        Ok(false) // Don't suppress exceptions
    }
}

impl PyWriter {
    fn writer_mut(&mut self) -> PyResult<&mut escapepod_signal::Writer> {
        self.inner
            .as_mut()
            .ok_or_else(|| to_py_err(escapepod_signal::Error::WriterFinalized))
    }
}

/// Best-effort finalize on garbage collection.
///
/// Writer holds an open file handle but its footer is only written by
/// `finish()` (called from `close()` / `__exit__`). If the user forgets
/// the context manager and the wrapper is GC'd, drop the inner Writer
/// through `finish()` so the file ends up readable, and emit a
/// `ResourceWarning` so the omission is at least visible.
///
/// `finish()` runs before we touch Python so the file is finalized even
/// if interpreter shutdown is racing us; the warning is best-effort.
impl Drop for PyWriter {
    fn drop(&mut self) {
        let Some(writer) = self.inner.take() else {
            return;
        };
        let finish_result = writer.finish();

        // Don't attach during interpreter shutdown — Py_IsInitialized() is
        // false then and Python::attach would deadlock or panic.
        if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
            return;
        }

        Python::attach(|py| {
            let msg = if finish_result.is_ok() {
                c"escapepod.Writer was not explicitly closed; finalized on garbage collection. Use a `with` block or call .close()."
            } else {
                c"escapepod.Writer was not explicitly closed and finalization failed; no output file was written."
            };
            if let Err(e) = PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyResourceWarning>(),
                msg,
                1,
            ) {
                e.write_unraisable(py, None);
            }
        });
    }
}

/// Parse an end-reason string and reject unknown values.
///
/// `EndReason::from_str` is infallible and silently maps unknowns to
/// `Unknown`, which would let callers write misspelled metadata without
/// noticing. Here we validate against the known set and raise a
/// `ValueError` for anything else.
pub(crate) fn parse_end_reason(s: &str) -> PyResult<escapepod_signal::EndReason> {
    const VALID: &[&str] = &[
        "unknown",
        "mux_change",
        "unblock_mux_change",
        "data_service_unblock_mux_change",
        "signal_positive",
        "signal_negative",
        "api_request",
        "device_data_error",
        "analysis_config_change",
        "paused",
    ];
    if VALID.contains(&s) {
        Ok(s.parse().unwrap_or_default())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid end_reason '{s}'. Valid values: {VALID:?}"
        )))
    }
}

/// Construct a RunInfo from keyword arguments (Python-constructable).
///
/// This allows creating new RunInfo objects for use with the Writer.
#[pyfunction]
#[pyo3(signature = (
    acquisition_id,
    acquisition_start_time=0,
    adc_max=2047,
    adc_min=-2048,
    experiment_name="",
    flow_cell_id="",
    flow_cell_product_code="",
    protocol_name="",
    protocol_run_id="",
    protocol_start_time=0,
    sample_id="",
    sample_rate=4000,
    sequencing_kit="",
    sequencer_position="",
    sequencer_position_type="",
    software="",
    system_name="",
    system_type="",
    context_tags=None,
    tracking_id=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn create_run_info(
    acquisition_id: &str,
    acquisition_start_time: i64,
    adc_max: i16,
    adc_min: i16,
    experiment_name: &str,
    flow_cell_id: &str,
    flow_cell_product_code: &str,
    protocol_name: &str,
    protocol_run_id: &str,
    protocol_start_time: i64,
    sample_id: &str,
    sample_rate: u16,
    sequencing_kit: &str,
    sequencer_position: &str,
    sequencer_position_type: &str,
    software: &str,
    system_name: &str,
    system_type: &str,
    context_tags: Option<HashMap<String, String>>,
    tracking_id: Option<HashMap<String, String>>,
) -> PyRunInfo {
    PyRunInfo {
        inner: escapepod_signal::RunInfoData {
            acquisition_id: acquisition_id.to_string(),
            acquisition_start_time,
            adc_max,
            adc_min,
            experiment_name: experiment_name.to_string(),
            flow_cell_id: flow_cell_id.to_string(),
            flow_cell_product_code: flow_cell_product_code.to_string(),
            protocol_name: protocol_name.to_string(),
            protocol_run_id: protocol_run_id.to_string(),
            protocol_start_time,
            sample_id: sample_id.to_string(),
            sample_rate,
            sequencing_kit: sequencing_kit.to_string(),
            sequencer_position: sequencer_position.to_string(),
            sequencer_position_type: sequencer_position_type.to_string(),
            software: software.to_string(),
            system_name: system_name.to_string(),
            system_type: system_type.to_string(),
            context_tags: context_tags.unwrap_or_default(),
            tracking_id: tracking_id.unwrap_or_default(),
        },
    }
}

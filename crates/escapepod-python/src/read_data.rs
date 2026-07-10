use std::collections::HashMap;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::to_py_err;
use crate::reader::adc_to_pa;
use crate::writer::parse_end_reason;

/// A single read's metadata from a POD5 file.
#[pyclass(name = "ReadData", frozen)]
pub struct PyReadData {
    pub(crate) inner: escapepod_signal::ReadData,
}

#[pymethods]
impl PyReadData {
    /// Construct a ReadData from scratch.
    ///
    /// All fields except `read_id` have defaults matching the underlying
    /// `ReadData::default()`. Use this to build reads for `Writer.add_read_data`
    /// without going through the 15+ kwargs of `Writer.add_read`.
    ///
    /// `signal_rows` is informational — when passing the result to
    /// `Writer.add_read_data`, the writer assigns its own signal rows.
    #[new]
    #[pyo3(signature = (
        read_id,
        read_number = 0,
        start_sample = 0,
        channel = 0,
        well = 0,
        pore_type = "not_set",
        calibration_offset = 0.0,
        calibration_scale = 1.0,
        median_before = 0.0,
        end_reason = "unknown",
        end_reason_forced = false,
        run_info_index = 0,
        num_minknow_events = 0,
        tracked_scaling_scale = 1.0,
        tracked_scaling_shift = 0.0,
        predicted_scaling_scale = 1.0,
        predicted_scaling_shift = 0.0,
        num_reads_since_mux_change = 0,
        time_since_mux_change = 0.0,
        num_samples = 0,
        open_pore_level = 0.0,
        expected_open_pore_level = 0.0,
        selected_read_level = 0.0,
        signal_rows = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        tracked_scaling_scale: f32,
        tracked_scaling_shift: f32,
        predicted_scaling_scale: f32,
        predicted_scaling_shift: f32,
        num_reads_since_mux_change: u32,
        time_since_mux_change: f32,
        num_samples: u64,
        open_pore_level: f32,
        expected_open_pore_level: f32,
        selected_read_level: f32,
        signal_rows: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        let uuid = escapepod_signal::utils::parse_uuid_flexible(read_id)
            .map_err(|e| to_py_err(escapepod_signal::Error::InvalidUuid(e.to_string())))?;
        let end_reason = parse_end_reason(end_reason)?;
        Ok(Self {
            inner: escapepod_signal::ReadData {
                read_id: uuid,
                read_number,
                start_sample,
                channel,
                well,
                pore_type: pore_type.into(),
                calibration_offset,
                calibration_scale,
                median_before,
                end_reason,
                end_reason_forced,
                run_info_index,
                num_minknow_events,
                tracked_scaling_scale,
                tracked_scaling_shift,
                predicted_scaling_scale,
                predicted_scaling_shift,
                num_reads_since_mux_change,
                time_since_mux_change,
                num_samples,
                open_pore_level,
                expected_open_pore_level,
                selected_read_level,
                signal_rows: signal_rows.unwrap_or_default(),
            },
        })
    }

    #[getter]
    fn read_id(&self) -> String {
        self.inner.read_id.to_string()
    }

    #[getter]
    fn read_number(&self) -> u32 {
        self.inner.read_number
    }

    #[getter]
    fn start_sample(&self) -> u64 {
        self.inner.start_sample
    }

    #[getter]
    fn channel(&self) -> u16 {
        self.inner.channel
    }

    #[getter]
    fn well(&self) -> u8 {
        self.inner.well
    }

    #[getter]
    fn pore_type(&self) -> &str {
        self.inner.pore_type.as_str()
    }

    #[getter]
    fn calibration_offset(&self) -> f32 {
        self.inner.calibration_offset
    }

    #[getter]
    fn calibration_scale(&self) -> f32 {
        self.inner.calibration_scale
    }

    #[getter]
    fn median_before(&self) -> f32 {
        self.inner.median_before
    }

    #[getter]
    fn end_reason(&self) -> &str {
        self.inner.end_reason.as_str()
    }

    #[getter]
    fn end_reason_forced(&self) -> bool {
        self.inner.end_reason_forced
    }

    #[getter]
    fn run_info_index(&self) -> u32 {
        self.inner.run_info_index
    }

    #[getter]
    fn num_minknow_events(&self) -> u64 {
        self.inner.num_minknow_events
    }

    #[getter]
    fn tracked_scaling_scale(&self) -> f32 {
        self.inner.tracked_scaling_scale
    }

    #[getter]
    fn tracked_scaling_shift(&self) -> f32 {
        self.inner.tracked_scaling_shift
    }

    #[getter]
    fn predicted_scaling_scale(&self) -> f32 {
        self.inner.predicted_scaling_scale
    }

    #[getter]
    fn predicted_scaling_shift(&self) -> f32 {
        self.inner.predicted_scaling_shift
    }

    #[getter]
    fn num_reads_since_mux_change(&self) -> u32 {
        self.inner.num_reads_since_mux_change
    }

    #[getter]
    fn time_since_mux_change(&self) -> f32 {
        self.inner.time_since_mux_change
    }

    #[getter]
    fn num_samples(&self) -> u64 {
        self.inner.num_samples
    }

    #[getter]
    fn open_pore_level(&self) -> f32 {
        self.inner.open_pore_level
    }

    #[getter]
    fn expected_open_pore_level(&self) -> f32 {
        self.inner.expected_open_pore_level
    }

    #[getter]
    fn selected_read_level(&self) -> f32 {
        self.inner.selected_read_level
    }

    #[getter]
    fn signal_rows(&self) -> Vec<u64> {
        self.inner.signal_rows.clone()
    }

    /// Calibrate an int16 ADC signal array to picoamperes using this read's
    /// calibration, returning a float32 numpy array.
    ///
    /// Matches `pod5.ReadRecord.calibrate_signal_array`; applies
    /// `pA = (ADC + calibration_offset) * calibration_scale`.
    fn calibrate_signal_array<'py>(
        &self,
        py: Python<'py>,
        signal_adc: &Bound<'py, PyArray1<i16>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let raw: Vec<i16> = signal_adc.to_vec()?;
        let pa = adc_to_pa(
            &raw,
            self.inner.calibration_offset,
            self.inner.calibration_scale,
        );
        Ok(PyArray1::from_vec(py, pa))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner.read_id == other.inner.read_id
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.read_id.as_bytes().hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Run information metadata from a POD5 file.
#[pyclass(name = "RunInfo", frozen)]
pub struct PyRunInfo {
    pub(crate) inner: escapepod_signal::RunInfoData,
}

#[pymethods]
impl PyRunInfo {
    /// Construct a RunInfo from scratch.
    ///
    /// Equivalent to `escapepod.create_run_info(...)` — both are kept so
    /// existing call sites continue to work.
    #[new]
    #[pyo3(signature = (
        acquisition_id,
        acquisition_start_time = 0,
        adc_max = 2047,
        adc_min = -2048,
        experiment_name = "",
        flow_cell_id = "",
        flow_cell_product_code = "",
        protocol_name = "",
        protocol_run_id = "",
        protocol_start_time = 0,
        sample_id = "",
        sample_rate = 4000,
        sequencing_kit = "",
        sequencer_position = "",
        sequencer_position_type = "",
        software = "",
        system_name = "",
        system_type = "",
        context_tags = None,
        tracking_id = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> Self {
        Self {
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

    #[getter]
    fn acquisition_id(&self) -> &str {
        &self.inner.acquisition_id
    }

    #[getter]
    fn acquisition_start_time(&self) -> i64 {
        self.inner.acquisition_start_time
    }

    #[getter]
    fn adc_max(&self) -> i16 {
        self.inner.adc_max
    }

    #[getter]
    fn adc_min(&self) -> i16 {
        self.inner.adc_min
    }

    #[getter]
    fn experiment_name(&self) -> &str {
        &self.inner.experiment_name
    }

    #[getter]
    fn flow_cell_id(&self) -> &str {
        &self.inner.flow_cell_id
    }

    #[getter]
    fn flow_cell_product_code(&self) -> &str {
        &self.inner.flow_cell_product_code
    }

    #[getter]
    fn protocol_name(&self) -> &str {
        &self.inner.protocol_name
    }

    #[getter]
    fn protocol_run_id(&self) -> &str {
        &self.inner.protocol_run_id
    }

    #[getter]
    fn protocol_start_time(&self) -> i64 {
        self.inner.protocol_start_time
    }

    #[getter]
    fn sample_id(&self) -> &str {
        &self.inner.sample_id
    }

    #[getter]
    fn sample_rate(&self) -> u16 {
        self.inner.sample_rate
    }

    #[getter]
    fn sequencing_kit(&self) -> &str {
        &self.inner.sequencing_kit
    }

    #[getter]
    fn sequencer_position(&self) -> &str {
        &self.inner.sequencer_position
    }

    #[getter]
    fn sequencer_position_type(&self) -> &str {
        &self.inner.sequencer_position_type
    }

    #[getter]
    fn software(&self) -> &str {
        &self.inner.software
    }

    #[getter]
    fn system_name(&self) -> &str {
        &self.inner.system_name
    }

    #[getter]
    fn system_type(&self) -> &str {
        &self.inner.system_type
    }

    #[getter]
    fn context_tags(&self) -> HashMap<String, String> {
        self.inner.context_tags.clone()
    }

    #[getter]
    fn tracking_id(&self) -> HashMap<String, String> {
        self.inner.tracking_id.clone()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Build a column-oriented dict of read metadata suitable for constructing a
/// pandas/polars DataFrame (`pd.DataFrame(reader.to_dict())`).
///
/// Scalar metadata fields only — signal is fetched separately. Column names
/// mirror `ReadData`'s properties so the frame matches the object surface.
/// Accumulates read metadata into a column dict, one column at a time.
///
/// `num` emits a numpy array; `strs` emits a Python list (object dtype).
/// pandas and polars wrap a numpy array as a column without re-boxing, whereas a
/// Python list of scalars forces a per-element parse on DataFrame construction —
/// the dominant cost for wide metadata frames (see benchmarks #98). Only 3 of
/// ~23 columns are strings, so the numeric fast path covers the bulk.
struct ColumnSet<'py, 'r> {
    py: Python<'py>,
    dict: Bound<'py, PyDict>,
    reads: &'r [&'r escapepod_signal::ReadData],
}

impl<'py, 'r> ColumnSet<'py, 'r> {
    fn num<T: numpy::Element>(
        &self,
        name: &str,
        get: impl Fn(&escapepod_signal::ReadData) -> T,
    ) -> PyResult<()> {
        let col: Vec<T> = self.reads.iter().map(|&r| get(r)).collect();
        self.dict.set_item(name, PyArray1::from_vec(self.py, col))
    }

    fn strs<'a, S: pyo3::IntoPyObject<'py>>(
        &self,
        name: &str,
        get: impl Fn(&'a escapepod_signal::ReadData) -> S,
    ) -> PyResult<()>
    where
        'r: 'a,
    {
        let col: Vec<S> = self.reads.iter().map(|&r| get(r)).collect();
        self.dict.set_item(name, col)
    }
}

pub(crate) fn reads_to_columns<'py>(
    py: Python<'py>,
    reads: &[&escapepod_signal::ReadData],
) -> PyResult<Bound<'py, PyDict>> {
    let c = ColumnSet {
        py,
        dict: PyDict::new(py),
        reads,
    };

    c.strs("read_id", |r| r.read_id.to_string())?;
    c.num("read_number", |r| r.read_number)?;
    c.num("start_sample", |r| r.start_sample)?;
    c.num("channel", |r| r.channel)?;
    c.num("well", |r| r.well)?;
    c.strs("pore_type", |r| r.pore_type.as_str())?;
    c.num("calibration_offset", |r| r.calibration_offset)?;
    c.num("calibration_scale", |r| r.calibration_scale)?;
    c.num("median_before", |r| r.median_before)?;
    c.strs("end_reason", |r| r.end_reason.as_str())?;
    c.num("end_reason_forced", |r| r.end_reason_forced)?;
    c.num("run_info_index", |r| r.run_info_index)?;
    c.num("num_minknow_events", |r| r.num_minknow_events)?;
    c.num("num_samples", |r| r.num_samples)?;
    c.num("tracked_scaling_scale", |r| r.tracked_scaling_scale)?;
    c.num("tracked_scaling_shift", |r| r.tracked_scaling_shift)?;
    c.num("predicted_scaling_scale", |r| r.predicted_scaling_scale)?;
    c.num("predicted_scaling_shift", |r| r.predicted_scaling_shift)?;
    c.num("num_reads_since_mux_change", |r| {
        r.num_reads_since_mux_change
    })?;
    c.num("time_since_mux_change", |r| r.time_since_mux_change)?;
    c.num("open_pore_level", |r| r.open_pore_level)?;
    c.num("expected_open_pore_level", |r| r.expected_open_pore_level)?;
    c.num("selected_read_level", |r| r.selected_read_level)?;

    Ok(c.dict)
}

/// Wrap a column dict in a `pandas.DataFrame`, importing pandas lazily so the
/// dependency is only required by callers that ask for a frame.
pub(crate) fn columns_to_pandas<'py>(
    py: Python<'py>,
    reads: &[&escapepod_signal::ReadData],
) -> PyResult<Bound<'py, PyAny>> {
    let cols = reads_to_columns(py, reads)?;
    let pandas = py.import("pandas").map_err(|_| {
        pyo3::exceptions::PyImportError::new_err(
            "pandas is required for to_pandas(); install pandas or use to_dict()",
        )
    })?;
    pandas.call_method1("DataFrame", (cols,))
}

/// Wrap a column dict in a `polars.DataFrame`, importing polars lazily.
pub(crate) fn columns_to_polars<'py>(
    py: Python<'py>,
    reads: &[&escapepod_signal::ReadData],
) -> PyResult<Bound<'py, PyAny>> {
    let cols = reads_to_columns(py, reads)?;
    let polars = py.import("polars").map_err(|_| {
        pyo3::exceptions::PyImportError::new_err(
            "polars is required for to_polars(); install polars or use to_dict()",
        )
    })?;
    polars.call_method1("DataFrame", (cols,))
}

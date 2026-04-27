use std::collections::HashMap;

use pyo3::prelude::*;

/// A single read's metadata from a POD5 file.
#[pyclass(name = "ReadData", frozen)]
pub struct PyReadData {
    pub(crate) inner: escapepod_signal::ReadData,
}

#[pymethods]
impl PyReadData {
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
    fn signal_rows(&self) -> Vec<u64> {
        self.inner.signal_rows.clone()
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

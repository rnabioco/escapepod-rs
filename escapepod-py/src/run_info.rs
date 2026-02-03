use std::collections::HashMap;

use pyo3::prelude::*;

/// Run information metadata from a POD5 file.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct RunInfo {
    inner: escapepod::RunInfoData,
}

impl RunInfo {
    pub fn new(data: escapepod::RunInfoData) -> Self {
        Self { inner: data }
    }
}

#[pymethods]
impl RunInfo {
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
    fn context_tags(&self) -> HashMap<String, String> {
        self.inner.context_tags.clone()
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
    fn tracking_id(&self) -> HashMap<String, String> {
        self.inner.tracking_id.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RunInfo(acquisition_id='{}', flow_cell_id='{}', sample_rate={})",
            self.inner.acquisition_id, self.inner.flow_cell_id, self.inner.sample_rate
        )
    }
}

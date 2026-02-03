use std::collections::HashSet;
use std::sync::Arc;

use pyo3::prelude::*;

use crate::read::Read;
use crate::run_info::RunInfo;

/// A reader for POD5 files.
///
/// Provides read-only access to reads, signal data, and run metadata.
/// Can be used as a context manager.
#[pyclass]
pub struct Reader {
    inner: Arc<escapepod::Reader>,
    num_reads: usize,
}

#[pymethods]
impl Reader {
    /// Open a POD5 file for reading.
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let reader =
            escapepod::Reader::open(path).map_err(|e| crate::to_py_err(&e))?;
        let num_reads = reader.read_count().map_err(|e| crate::to_py_err(&e))?;
        Ok(Self {
            inner: Arc::new(reader),
            num_reads,
        })
    }

    /// Number of reads in the file.
    #[getter]
    fn num_reads(&self) -> usize {
        self.num_reads
    }

    /// Number of read batches in the file.
    #[getter]
    fn num_read_batches(&self) -> PyResult<usize> {
        self.inner
            .read_batch_count()
            .map_err(|e| crate::to_py_err(&e))
    }

    /// File identifier (UUID string).
    #[getter]
    fn file_identifier(&self) -> &str {
        self.inner.file_identifier()
    }

    /// Software that wrote this file.
    #[getter]
    fn writing_software(&self) -> &str {
        self.inner.software()
    }

    /// POD5 format version.
    #[getter]
    fn pod5_version(&self) -> &str {
        self.inner.pod5_version()
    }

    /// List of run info entries.
    #[getter]
    fn run_infos(&self) -> Vec<RunInfo> {
        self.inner
            .run_infos()
            .iter()
            .map(|ri| RunInfo::new(ri.clone()))
            .collect()
    }

    /// Get all reads, optionally filtered by a list of read IDs.
    ///
    /// Parameters
    /// ----------
    /// selection : list of str, optional
    ///     If provided, only return reads whose read_id is in this list.
    ///
    /// Returns
    /// -------
    /// list of Read
    #[pyo3(signature = (*, selection=None))]
    fn reads(&self, selection: Option<Vec<String>>) -> PyResult<Vec<Read>> {
        let filter_set: Option<HashSet<uuid::Uuid>> = selection
            .map(|ids| {
                ids.iter()
                    .map(|s| {
                        uuid::Uuid::parse_str(s).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid UUID '{s}': {e}"
                            ))
                        })
                    })
                    .collect::<PyResult<HashSet<_>>>()
            })
            .transpose()?;

        let iter = self
            .inner
            .reads()
            .map_err(|e| crate::to_py_err(&e))?;

        let run_infos: Vec<RunInfo> = self
            .inner
            .run_infos()
            .iter()
            .map(|ri| RunInfo::new(ri.clone()))
            .collect();

        let run_info_rates: Vec<u16> = self
            .inner
            .run_infos()
            .iter()
            .map(|ri| ri.sample_rate)
            .collect();

        let mut reads = Vec::new();
        for read_result in iter {
            let read_data = read_result.map_err(|e| crate::to_py_err(&e))?;

            if let Some(ref filter) = filter_set {
                if !filter.contains(&read_data.read_id) {
                    continue;
                }
            }

            let ri_idx = read_data.run_info_index as usize;
            let run_info = run_infos
                .get(ri_idx)
                .cloned()
                .unwrap_or_else(|| RunInfo::new(escapepod::RunInfoData::default()));
            let sample_rate = run_info_rates.get(ri_idx).copied().unwrap_or(0);

            reads.push(Read::new(
                read_data,
                self.inner.clone(),
                run_info,
                sample_rate,
            ));
        }

        Ok(reads)
    }

    /// Get all read IDs without decompressing signal data.
    ///
    /// Returns
    /// -------
    /// list of str
    fn read_ids(&self) -> PyResult<Vec<String>> {
        let ids = self
            .inner
            .read_ids()
            .map_err(|e| crate::to_py_err(&e))?;
        Ok(ids.iter().map(|id| id.to_string()).collect())
    }

    fn __len__(&self) -> usize {
        self.num_reads
    }

    fn __repr__(&self) -> String {
        format!(
            "Reader(num_reads={}, pod5_version='{}')",
            self.num_reads,
            self.inner.pod5_version()
        )
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> bool {
        false
    }
}

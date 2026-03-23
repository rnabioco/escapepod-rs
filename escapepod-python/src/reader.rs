use std::collections::HashSet;
use std::path::PathBuf;

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::read_data::{PyReadData, PyRunInfo};

/// Reader for POD5 files.
///
/// Provides access to read metadata and signal data with optimized
/// lookup paths for single and batch read retrieval.
#[pyclass(name = "Reader")]
pub struct PyReader {
    inner: escapepod::Reader,
    path: PathBuf,
}

#[pymethods]
impl PyReader {
    /// Open a POD5 file for reading.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let path = PathBuf::from(path);
        let reader = escapepod::Reader::open(&path).map_err(to_py_err)?;
        Ok(Self {
            inner: reader,
            path,
        })
    }

    /// Number of reads in the file.
    fn read_count(&self) -> PyResult<usize> {
        self.inner.read_count().map_err(to_py_err)
    }

    /// Number of read batches in the file.
    fn read_batch_count(&self) -> PyResult<usize> {
        self.inner.read_batch_count().map_err(to_py_err)
    }

    /// Get all run info records.
    fn run_infos(&self) -> Vec<PyRunInfo> {
        self.inner
            .run_infos()
            .iter()
            .map(|ri| PyRunInfo { inner: ri.clone() })
            .collect()
    }

    /// Get all read IDs as strings (fast column-projected scan).
    fn read_ids(&self) -> PyResult<Vec<String>> {
        let ids = self.inner.read_ids().map_err(to_py_err)?;
        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    /// Get all reads (materializes the full read list).
    fn reads(&self) -> PyResult<Vec<PyReadData>> {
        let mut result = Vec::new();
        for read_result in self.inner.reads().map_err(to_py_err)? {
            let inner = read_result.map_err(to_py_err)?;
            result.push(PyReadData { inner });
        }
        Ok(result)
    }

    /// Look up a single read by UUID string.
    ///
    /// Uses the ReadIndex for O(log n) lookup when a .p5i sidecar exists.
    fn get_read(&self, read_id: &str) -> PyResult<PyReadData> {
        let uuid = escapepod::utils::parse_uuid_flexible(read_id)
            .map_err(|e| to_py_err(escapepod::Error::InvalidUuid(e.to_string())))?;

        let index = self.inner.read_index().map_err(to_py_err)?;
        let (batch_idx, row_idx) = index
            .get(&uuid)
            .ok_or_else(|| to_py_err(escapepod::Error::ReadNotFound(uuid)))?;

        let batch = self.inner.read_batch(batch_idx).map_err(to_py_err)?;
        let inner = escapepod::Reader::read_from_batch(&batch, row_idx).map_err(to_py_err)?;
        Ok(PyReadData { inner })
    }

    /// Look up multiple reads by UUID strings.
    ///
    /// Uses indexed batch-skipping when a .p5i sidecar exists,
    /// otherwise falls back to a single-pass scan with early exit.
    fn get_reads(&self, read_ids: Vec<String>) -> PyResult<Vec<PyReadData>> {
        let target_ids: HashSet<escapepod::Uuid> = read_ids
            .iter()
            .map(|s| {
                escapepod::utils::parse_uuid_flexible(s)
                    .map_err(|e| to_py_err(escapepod::Error::InvalidUuid(e.to_string())))
            })
            .collect::<PyResult<_>>()?;

        let reads = self.inner.reads_by_ids(&target_ids).map_err(to_py_err)?;
        Ok(reads
            .into_iter()
            .map(|inner| PyReadData { inner })
            .collect())
    }

    /// Get raw ADC signal for a read as a numpy int16 array.
    ///
    /// Releases the GIL during VBZ decompression.
    fn get_signal<'py>(
        &self,
        py: Python<'py>,
        read: &PyReadData,
    ) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let signal_rows = read.inner.signal_rows.clone();
        let signal = py.detach(|| self.inner.get_signal(&signal_rows).map_err(to_py_err))?;
        Ok(PyArray1::from_vec(py, signal))
    }

    /// Get calibrated signal in picoamperes as a numpy float32 array.
    ///
    /// Applies: pA = (ADC + calibration_offset) * calibration_scale
    fn get_signal_pa<'py>(
        &self,
        py: Python<'py>,
        read: &PyReadData,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let signal_rows = read.inner.signal_rows.clone();
        let offset = read.inner.calibration_offset;
        let scale = read.inner.calibration_scale;

        let raw = py.detach(|| self.inner.get_signal(&signal_rows).map_err(to_py_err))?;

        let pa: Vec<f32> = raw
            .iter()
            .map(|&adc| (f32::from(adc) + offset) * scale)
            .collect();
        Ok(PyArray1::from_vec(py, pa))
    }

    /// Get raw ADC signal for multiple reads in parallel.
    ///
    /// Returns a list of (read_id, signal) tuples. Uses rayon for
    /// parallel VBZ decompression. Releases the GIL during decompression.
    fn get_signals<'py>(
        &self,
        py: Python<'py>,
        reads: Vec<PyRef<'_, PyReadData>>,
    ) -> PyResult<Vec<(String, Bound<'py, PyArray1<i16>>)>> {
        let inputs: Vec<(String, Vec<u64>)> = reads
            .iter()
            .map(|r| (r.inner.read_id.to_string(), r.inner.signal_rows.clone()))
            .collect();

        let results = py.detach(|| self.inner.get_signal_bulk(&inputs).map_err(to_py_err))?;

        results
            .into_iter()
            .map(|(id, sig)| Ok((id, PyArray1::from_vec(py, sig))))
            .collect()
    }

    /// Check if a .p5i sidecar index exists for this file.
    fn has_index(&self) -> bool {
        let mut p5i = self.path.as_os_str().to_owned();
        p5i.push(".p5i");
        PathBuf::from(p5i).exists()
    }

    /// Build and write a .p5i sidecar index for fast UUID lookups.
    ///
    /// Returns the number of reads indexed.
    fn build_index(&self) -> PyResult<usize> {
        let mut p5i = self.path.as_os_str().to_owned();
        p5i.push(".p5i");
        self.inner
            .build_and_write_index(PathBuf::from(p5i))
            .map_err(to_py_err)
    }

    /// Advise the OS to prefetch signal data into memory.
    fn prefetch_signal(&self) {
        self.inner.prefetch_signal();
    }
}

use std::collections::HashSet;
use std::path::PathBuf;

use escapepod_signal::ReadsBatchView;
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::read_data::{PyReadData, PyRunInfo};

/// Convert raw ADC samples to picoamperes: `(adc + offset) * scale`.
///
/// Uses `mul_add` so LLVM can emit FMA on AVX2+ and keep the loop tight.
pub(crate) fn adc_to_pa(raw: &[i16], offset: f32, scale: f32) -> Vec<f32> {
    let bias = offset * scale;
    raw.iter()
        .map(|&adc| f32::from(adc).mul_add(scale, bias))
        .collect()
}

/// Reader for POD5 files.
///
/// Provides access to read metadata and signal data with optimized
/// lookup paths for single and batch read retrieval.
///
/// Can be used as a context manager:
///
///     with Reader("reads.pod5") as reader:
///         reads = reader.get_reads(ids)
#[pyclass(name = "Reader")]
pub struct PyReader {
    inner: escapepod_signal::Reader,
    path: PathBuf,
}

#[pymethods]
impl PyReader {
    /// Open a POD5 file for reading.
    ///
    /// Accepts a string path or any os.PathLike object (e.g. pathlib.Path).
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let reader = escapepod_signal::Reader::open(&path).map_err(to_py_err)?;
        Ok(Self {
            inner: reader,
            path,
        })
    }

    // -- File metadata properties ------------------------------------------

    /// File path this reader was opened from.
    #[getter]
    fn path(&self) -> String {
        self.path.display().to_string()
    }

    /// File identifier string from the POD5 footer.
    #[getter]
    fn file_identifier(&self) -> &str {
        self.inner.file_identifier()
    }

    /// Software that wrote this POD5 file (e.g. "MinKNOW 5.x").
    #[getter]
    fn software(&self) -> &str {
        self.inner.software()
    }

    /// POD5 format version string.
    #[getter]
    fn pod5_version(&self) -> &str {
        self.inner.pod5_version()
    }

    // -- Read/batch counts -------------------------------------------------

    /// Number of reads in the file.
    #[getter]
    fn read_count(&self) -> PyResult<usize> {
        self.inner.read_count().map_err(to_py_err)
    }

    /// Number of read batches in the file.
    #[getter]
    fn read_batch_count(&self) -> PyResult<usize> {
        self.inner.read_batch_count().map_err(to_py_err)
    }

    /// Total number of signal rows across all batches.
    #[getter]
    fn signal_row_count(&self) -> PyResult<u64> {
        self.inner.signal_row_count().map_err(to_py_err)
    }

    // -- Run info ----------------------------------------------------------

    /// Get all run info records.
    #[getter]
    fn run_infos(&self) -> Vec<PyRunInfo> {
        self.inner
            .run_infos()
            .iter()
            .map(|ri| PyRunInfo { inner: ri.clone() })
            .collect()
    }

    // -- Read access -------------------------------------------------------

    /// Get all read IDs as strings (fast column-projected scan).
    fn read_ids(&self) -> PyResult<Vec<String>> {
        let ids = self.inner.read_ids().map_err(to_py_err)?;
        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    /// Get reads from the file, optionally filtered by read IDs.
    ///
    /// Parameters
    /// ----------
    /// selection : list[str], optional
    ///     Read IDs to retrieve. If None, returns all reads.
    /// missing_ok : bool, optional
    ///     If False (default), raise KeyError when any requested ID is absent.
    ///     If True, silently skip missing IDs. Ignored when selection is None.
    ///
    /// Returns
    /// -------
    /// list[ReadData]
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn reads(&self, selection: Option<Vec<String>>, missing_ok: bool) -> PyResult<Vec<PyReadData>> {
        Ok(self
            .collect_inner(selection, missing_ok)?
            .into_iter()
            .map(|inner| PyReadData { inner })
            .collect())
    }

    /// Read metadata as a column-oriented dict (one list per field).
    ///
    /// Construct a DataFrame with `pd.DataFrame(reader.to_dict())` (pandas) or
    /// `pl.DataFrame(reader.to_dict())` (polars) without a hard dependency on
    /// either. Signal is not included; fetch it via `get_signal`.
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn to_dict<'py>(
        &self,
        py: Python<'py>,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.build_columns(py, selection, missing_ok)
    }

    /// Read metadata as a `pandas.DataFrame` (pandas imported lazily).
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn to_pandas<'py>(
        &self,
        py: Python<'py>,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cols = self.build_columns(py, selection, missing_ok)?;
        crate::read_data::dict_to_pandas(py, cols)
    }

    /// Read metadata as a `polars.DataFrame` (polars imported lazily).
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn to_polars<'py>(
        &self,
        py: Python<'py>,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cols = self.build_columns(py, selection, missing_ok)?;
        crate::read_data::dict_to_polars(py, cols)
    }

    /// Look up a single read by UUID string.
    ///
    /// Uses the ReadIndex for O(log n) lookup when a .p5s sidecar exists.
    fn get_read(&self, read_id: &str) -> PyResult<PyReadData> {
        let uuid = escapepod_signal::utils::parse_uuid_flexible(read_id)
            .map_err(|e| to_py_err(escapepod_signal::Error::InvalidUuid(e.to_string())))?;

        let index = self.inner.read_index().map_err(to_py_err)?;
        let (batch_idx, row_idx) = index
            .get(&uuid)
            .ok_or_else(|| to_py_err(escapepod_signal::Error::ReadNotFound(uuid)))?;

        let batch = self.inner.read_batch(batch_idx).map_err(to_py_err)?;
        let inner =
            escapepod_signal::Reader::read_from_batch(&batch, row_idx).map_err(to_py_err)?;
        Ok(PyReadData { inner })
    }

    /// Look up multiple reads by UUID strings.
    ///
    /// Uses indexed batch-skipping when a .p5s sidecar exists,
    /// otherwise falls back to a single-pass scan with early exit.
    ///
    /// If `missing_ok` is False (default), raises KeyError when any requested
    /// ID is absent; set it True to silently skip missing IDs.
    #[pyo3(signature = (read_ids, missing_ok=false))]
    fn get_reads(&self, read_ids: Vec<String>, missing_ok: bool) -> PyResult<Vec<PyReadData>> {
        Ok(self
            .collect_inner(Some(read_ids), missing_ok)?
            .into_iter()
            .map(|inner| PyReadData { inner })
            .collect())
    }

    // -- Signal access -----------------------------------------------------

    /// Get raw ADC signal for a read as a numpy int16 array.
    ///
    /// With `max_samples`, decodes at most that many leading samples — the
    /// same array `get_signal(read)[:max_samples]` would give, without paying
    /// to decode the tail. A read shorter than `max_samples` comes back whole.
    ///
    /// Releases the GIL during VBZ decompression.
    #[pyo3(signature = (read, max_samples=None))]
    fn get_signal<'py>(
        &self,
        py: Python<'py>,
        read: &PyReadData,
        max_samples: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let signal_rows = read.inner.signal_rows.clone();
        let take = max_samples.unwrap_or(usize::MAX);
        let signal = py.detach(|| {
            self.inner
                .get_signal_prefix(&signal_rows, take)
                .map_err(to_py_err)
        })?;
        Ok(PyArray1::from_vec(py, signal))
    }

    /// Get calibrated signal in picoamperes as a numpy float32 array.
    ///
    /// Applies: pA = (ADC + calibration_offset) * calibration_scale
    ///
    /// `max_samples` truncates as in `get_signal`; calibration is applied to
    /// the returned prefix.
    #[pyo3(signature = (read, max_samples=None))]
    fn get_signal_pa<'py>(
        &self,
        py: Python<'py>,
        read: &PyReadData,
        max_samples: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let signal_rows = read.inner.signal_rows.clone();
        let offset = read.inner.calibration_offset;
        let scale = read.inner.calibration_scale;
        let take = max_samples.unwrap_or(usize::MAX);

        let raw = py.detach(|| {
            self.inner
                .get_signal_prefix(&signal_rows, take)
                .map_err(to_py_err)
        })?;
        Ok(PyArray1::from_vec(py, adc_to_pa(&raw, offset, scale)))
    }

    /// Get raw ADC signal for multiple reads in parallel.
    ///
    /// Returns a list of (read_id, signal) tuples. Uses rayon for
    /// parallel VBZ decompression. Releases the GIL during decompression.
    /// `max_samples` truncates each read as in `get_signal`.
    #[pyo3(signature = (reads, max_samples=None))]
    fn get_signals<'py>(
        &self,
        py: Python<'py>,
        reads: Vec<PyRef<'_, PyReadData>>,
        max_samples: Option<usize>,
    ) -> PyResult<Vec<(String, Bound<'py, PyArray1<i16>>)>> {
        let inputs: Vec<(String, Vec<u64>)> = reads
            .iter()
            .map(|r| (r.inner.read_id.to_string(), r.inner.signal_rows.clone()))
            .collect();
        let take = max_samples.unwrap_or(usize::MAX);

        let results = py.detach(|| {
            self.inner
                .get_signal_bulk_prefix(&inputs, take)
                .map_err(to_py_err)
        })?;

        results
            .into_iter()
            .map(|(id, sig)| Ok((id, PyArray1::from_vec(py, sig))))
            .collect()
    }

    /// Get calibrated pA signal for multiple reads in parallel.
    ///
    /// Returns a list of (read_id, signal_pa) tuples. Uses rayon for
    /// parallel VBZ decompression, then applies per-read calibration.
    /// `max_samples` truncates each read as in `get_signal`.
    #[pyo3(signature = (reads, max_samples=None))]
    fn get_signals_pa<'py>(
        &self,
        py: Python<'py>,
        reads: Vec<PyRef<'_, PyReadData>>,
        max_samples: Option<usize>,
    ) -> PyResult<Vec<(String, Bound<'py, PyArray1<f32>>)>> {
        let inputs: Vec<(String, Vec<u64>)> = reads
            .iter()
            .map(|r| (r.inner.read_id.to_string(), r.inner.signal_rows.clone()))
            .collect();
        let cal: Vec<(f32, f32)> = reads
            .iter()
            .map(|r| (r.inner.calibration_offset, r.inner.calibration_scale))
            .collect();
        let take = max_samples.unwrap_or(usize::MAX);

        // get_signal_bulk preserves input order, so we can zip with `cal` directly.
        let raw_results = py.detach(|| {
            self.inner
                .get_signal_bulk_prefix(&inputs, take)
                .map_err(to_py_err)
        })?;

        raw_results
            .into_iter()
            .zip(cal)
            .map(|((id, raw_signal), (offset, scale))| {
                Ok((
                    id,
                    PyArray1::from_vec(py, adc_to_pa(&raw_signal, offset, scale)),
                ))
            })
            .collect()
    }

    /// Number of stored (VBZ-compressed) signal bytes for a read.
    ///
    /// Matches `pod5.ReadRecord.byte_count`. Sums the compressed chunk sizes
    /// without decompressing.
    fn byte_count(&self, py: Python<'_>, read: &PyReadData) -> PyResult<usize> {
        let signal_rows = read.inner.signal_rows.clone();
        py.detach(|| {
            let chunks = self
                .inner
                .get_compressed_signal_for_rows(&signal_rows)
                .map_err(to_py_err)?;
            Ok(chunks.iter().map(|c| c.data.len()).sum())
        })
    }

    // -- Sidecar management ------------------------------------------------

    /// Check if a .p5s sidecar exists for this file.
    #[getter]
    fn has_index(&self) -> bool {
        escapepod_signal::pod5::sidecar::sidecar_path(&self.path).exists()
    }

    /// Build and write the .p5s sidecar read index for fast UUID lookups.
    ///
    /// Annotations already in the sidecar are preserved.
    /// Returns the number of reads indexed.
    fn build_index(&self) -> PyResult<usize> {
        self.inner
            .build_and_write_index(escapepod_signal::pod5::sidecar::sidecar_path(&self.path))
            .map_err(to_py_err)
    }

    /// The annotation names available in the .p5s sidecar, in name order.
    ///
    /// Empty when no sidecar exists. A sidecar that does not match this
    /// POD5 (stale or copied from another file) raises instead.
    fn annotation_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let path = self.path.clone();
        py.detach(|| {
            let identity = self.inner.sidecar_identity().map_err(to_py_err)?;
            let p5s = escapepod_signal::pod5::sidecar::sidecar_path(&path);
            let sidecar = escapepod_signal::pod5::sidecar::read_sidecar_file(&p5s, &identity)
                .map_err(to_py_err)?;
            Ok(sidecar
                .map(|s| {
                    s.annotation_names()
                        .into_iter()
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default())
        })
    }

    /// The experimental design recorded in the .p5s sidecar, or None.
    ///
    /// Returns a dict with ``key_columns`` (annotation names forming the
    /// lookup key), ``value_columns`` (experimental variables, each also
    /// materialized as a derived annotation), and ``rows`` (list of dicts,
    /// one per design row).
    fn design<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let path = self.path.clone();
        let design = py.detach(|| -> PyResult<_> {
            let identity = self.inner.sidecar_identity().map_err(to_py_err)?;
            let p5s = escapepod_signal::pod5::sidecar::sidecar_path(&path);
            let sidecar = escapepod_signal::pod5::sidecar::read_sidecar_file(&p5s, &identity)
                .map_err(to_py_err)?;
            Ok(sidecar.and_then(|s| s.design().cloned()))
        })?;
        let Some(design) = design else {
            return Ok(None);
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("key_columns", design.key_columns.clone())?;
        dict.set_item("value_columns", design.value_columns.clone())?;
        let rows: Vec<std::collections::HashMap<String, String>> = design
            .rows
            .iter()
            .map(|row| {
                design
                    .key_columns
                    .iter()
                    .chain(&design.value_columns)
                    .cloned()
                    .zip(row.iter().cloned())
                    .collect()
            })
            .collect();
        dict.set_item("rows", rows)?;
        Ok(Some(dict))
    }

    /// Read an annotation from the .p5s sidecar as a dict of
    /// ``read_id -> label`` (read IDs as standard UUID strings).
    ///
    /// With ``name=None`` the sidecar must contain exactly one annotation.
    /// Unassigned reads are absent from the dict. Raises if no sidecar
    /// exists, if it does not match this POD5, or if the annotation is
    /// missing.
    #[pyo3(signature = (name=None))]
    fn annotation(
        &self,
        py: Python<'_>,
        name: Option<&str>,
    ) -> PyResult<std::collections::HashMap<String, String>> {
        let path = self.path.clone();
        py.detach(|| {
            let annotation =
                escapepod_signal::operations::read_annotation(&path, name).map_err(to_py_err)?;
            Ok(annotation
                .iter()
                .map(|(uuid, label)| (uuid.to_string(), label.to_string()))
                .collect())
        })
    }

    /// Advise the OS to prefetch signal data into memory.
    fn prefetch_signal(&self) {
        self.inner.prefetch_signal();
    }

    // -- Context manager protocol ------------------------------------------

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        // A context manager signals a long-lived reader with likely repeated
        // random access, so warm the in-memory read-id index: subsequent
        // reads(selection=…) then take the O(k) indexed path instead of
        // re-scanning the reads table each call (#97). Gated by size to
        // honor the ~32 B/read memory cost on very large files; best-effort,
        // since the index is a pure optimization and must never make entering
        // the context manager fail.
        let py = slf.py();
        let n = slf.inner.read_count().unwrap_or(usize::MAX);
        if n <= crate::autoindex_max() {
            let inner = &slf.inner;
            py.detach(|| {
                let _ = inner.read_index();
            });
        }
        slf
    }

    #[allow(unused_variables)]
    fn __exit__(
        &self,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_val: Option<&Bound<'_, PyAny>>,
        exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        // mmap-based, no cleanup needed; return false to not suppress exceptions
        false
    }

    // -- Display -----------------------------------------------------------

    fn __repr__(&self) -> PyResult<String> {
        let n = self.inner.read_count().map_err(to_py_err)?;
        Ok(format!("Reader('{}', reads={})", self.path.display(), n))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner.read_count().map_err(to_py_err)
    }

    /// Iterate over all reads in the file.
    ///
    /// Yields ReadData objects one at a time without materializing
    /// the full list. Useful for large files.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyReadIterator> {
        let num_batches = slf.inner.read_batch_count().map_err(to_py_err)?;
        Ok(PyReadIterator {
            reader: slf.into(),
            num_batches,
            batch_idx: 0,
            current: Vec::new().into_iter(),
        })
    }
}

impl PyReader {
    /// Build a metadata column dict, shared by `to_dict`/`to_pandas`/`to_polars`.
    ///
    /// With no selection this takes the fast columnar path (`read_columns`),
    /// filling numeric columns by bulk slice copy straight from the Arrow buffers
    /// and skipping per-read `ReadData` materialization. A selection falls back to
    /// the random-access `collect_inner` path, where the read count is small and
    /// per-read `ReadData` cost is not the bottleneck.
    fn build_columns<'py>(
        &self,
        py: Python<'py>,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        match selection {
            None => {
                let cols = self.inner.read_columns().map_err(to_py_err)?;
                crate::read_data::columns_to_dict(py, cols)
            }
            Some(sel) => {
                let reads = self.collect_inner(Some(sel), missing_ok)?;
                let refs: Vec<&escapepod_signal::ReadData> = reads.iter().collect();
                crate::read_data::reads_to_columns(py, &refs)
            }
        }
    }

    /// Collect read metadata, optionally filtered by a selection of IDs.
    ///
    /// Shared backing for `reads`, `get_reads`, `to_dict`, and `to_pandas`.
    fn collect_inner(
        &self,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Vec<escapepod_signal::ReadData>> {
        match selection {
            Some(read_ids) => {
                let target_ids: HashSet<escapepod_signal::Uuid> = read_ids
                    .iter()
                    .map(|s| {
                        escapepod_signal::utils::parse_uuid_flexible(s).map_err(|e| {
                            to_py_err(escapepod_signal::Error::InvalidUuid(e.to_string()))
                        })
                    })
                    .collect::<PyResult<_>>()?;

                let reads = self.inner.reads_by_ids(&target_ids).map_err(to_py_err)?;
                if !missing_ok && reads.len() != target_ids.len() {
                    let missing = target_ids.len() - reads.len();
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "{missing} of {} requested read id(s) not found \
                         (pass missing_ok=True to ignore)",
                        target_ids.len()
                    )));
                }
                Ok(reads)
            }
            // Resolve columns once per batch via `ReadsBatchView` rather than
            // once per row (`reads()`'s per-row `column_by_name` lookups).
            None => self.inner.collect_all_reads().map_err(to_py_err),
        }
    }
}

/// Iterator over reads in a POD5 file (Python protocol).
///
/// Streams one batch at a time: each batch's rows are decoded once (columns
/// resolved once per batch via `ReadsBatchView`, not per row) and yielded
/// lazily, so a whole-file iteration never materializes more than one batch of
/// `ReadData` at once.
#[pyclass]
struct PyReadIterator {
    reader: Py<PyReader>,
    num_batches: usize,
    batch_idx: usize,
    current: std::vec::IntoIter<escapepod_signal::ReadData>,
}

#[pymethods]
impl PyReadIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyReadData>> {
        loop {
            if let Some(inner) = self.current.next() {
                return Ok(Some(PyReadData { inner }));
            }

            // Current batch drained — decode the next one.
            if self.batch_idx >= self.num_batches {
                return Ok(None);
            }

            let batch = self
                .reader
                .borrow(py)
                .inner
                .read_batch(self.batch_idx)
                .map_err(to_py_err)?;
            self.batch_idx += 1;

            let view = ReadsBatchView::new(&batch, false).map_err(to_py_err)?;
            let mut rows = Vec::with_capacity(view.num_rows());
            for row in 0..view.num_rows() {
                rows.push(view.read(row).map_err(to_py_err)?);
            }
            self.current = rows.into_iter();
        }
    }
}

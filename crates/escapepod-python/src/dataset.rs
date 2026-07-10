use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use escapepod_signal::RecordBatch;
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::read_data::{PyReadData, PyRunInfo};
use crate::reader::adc_to_pa;

/// Recursively collect POD5 files under `path` into `out`.
///
/// A path that is an explicit file is included regardless of `suffix`
/// (the user named it directly); directories are scanned for entries whose
/// file name ends with `suffix`, descending into subdirectories only when
/// `recursive` is set.
fn collect_pod5(
    path: &Path,
    recursive: bool,
    suffix: &str,
    out: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let p = entry?.path();
            if p.is_dir() {
                if recursive {
                    collect_pod5(&p, recursive, suffix, out)?;
                }
            } else if p
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(suffix))
            {
                out.push(p);
            }
        }
    } else {
        // Explicit file path — include it even if it doesn't match the pattern.
        out.push(path.to_path_buf());
    }
    Ok(())
}

/// Reader over a collection of POD5 files as one logical dataset.
///
/// Accepts a single file, a directory (scanned for `*.pod5`), or a list
/// mixing files and directories, and presents the reads across every file
/// as a single stream — the escapepod analogue of `pod5.DatasetReader`.
///
/// Can be used as a context manager:
///
///     with DatasetReader("run_dir/") as ds:
///         for read in ds:
///             print(read.read_id)
#[pyclass(name = "DatasetReader")]
pub struct PyDatasetReader {
    readers: Vec<(PathBuf, escapepod_signal::Reader)>,
    /// Lazily built map from read UUID to the index of its owning reader,
    /// used to route signal lookups to the file that holds the read.
    id_index: OnceLock<HashMap<escapepod_signal::Uuid, usize>>,
}

impl PyDatasetReader {
    /// Reader index → owning file for a given read, built once and cached.
    fn id_index(&self) -> PyResult<&HashMap<escapepod_signal::Uuid, usize>> {
        if let Some(map) = self.id_index.get() {
            return Ok(map);
        }
        let mut map = HashMap::new();
        for (i, (_, reader)) in self.readers.iter().enumerate() {
            for id in reader.read_ids().map_err(to_py_err)? {
                map.insert(id, i);
            }
        }
        // Ignore the error case: another thread won the race and set it first,
        // which is fine — either map is equivalent.
        let _ = self.id_index.set(map);
        Ok(self.id_index.get().unwrap())
    }

    /// Look up the reader that owns `read`, erroring if it belongs to no file
    /// in this dataset.
    fn owning_reader(&self, read: &PyReadData) -> PyResult<&escapepod_signal::Reader> {
        let idx = self.id_index()?.get(&read.inner.read_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "read {} is not part of this dataset",
                read.inner.read_id
            ))
        })?;
        Ok(&self.readers[*idx].1)
    }
}

#[pymethods]
impl PyDatasetReader {
    /// Open a POD5 dataset.
    ///
    /// Parameters
    /// ----------
    /// path : str, PathLike, or list
    ///     A single POD5 file, a directory to scan, or a list mixing files
    ///     and directories.
    /// recursive : bool, optional
    ///     Descend into subdirectories when scanning a directory (default: True).
    /// pattern : str, optional
    ///     Glob-style suffix used to match files inside directories
    ///     (default: "*.pod5"). Explicitly listed files are always included.
    #[new]
    #[pyo3(signature = (path, recursive=true, pattern="*.pod5"))]
    fn new(path: &Bound<'_, PyAny>, recursive: bool, pattern: &str) -> PyResult<Self> {
        // Accept a single path (str / PathLike) or a list of them.
        let roots: Vec<PathBuf> = if let Ok(single) = path.extract::<PathBuf>() {
            vec![single]
        } else {
            path.extract::<Vec<PathBuf>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "path must be a str, PathLike, or a list of them",
                )
            })?
        };

        let suffix = pattern.trim_start_matches('*');
        let mut files = Vec::new();
        for root in &roots {
            collect_pod5(root, recursive, suffix, &mut files).map_err(|e| {
                to_py_err(escapepod_signal::Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("scanning {}: {e}", root.display()),
                )))
            })?;
        }
        files.sort();
        files.dedup();

        if files.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "no POD5 files found matching '{pattern}' in: {roots:?}"
            )));
        }

        let mut readers = Vec::with_capacity(files.len());
        for file in files {
            let reader = escapepod_signal::Reader::open(&file).map_err(to_py_err)?;
            readers.push((file, reader));
        }

        Ok(Self {
            readers,
            id_index: OnceLock::new(),
        })
    }

    /// Paths of the POD5 files in this dataset, in sorted order.
    #[getter]
    fn paths(&self) -> Vec<String> {
        self.readers
            .iter()
            .map(|(p, _)| p.display().to_string())
            .collect()
    }

    /// Number of POD5 files in the dataset.
    #[getter]
    fn file_count(&self) -> usize {
        self.readers.len()
    }

    /// Total number of reads across all files.
    #[getter]
    fn read_count(&self) -> PyResult<usize> {
        let mut total = 0;
        for (_, reader) in &self.readers {
            total += reader.read_count().map_err(to_py_err)?;
        }
        Ok(total)
    }

    /// All run info records across the dataset, deduplicated by acquisition id.
    #[getter]
    fn run_infos(&self) -> Vec<PyRunInfo> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (_, reader) in &self.readers {
            for ri in reader.run_infos() {
                if seen.insert(ri.acquisition_id.clone()) {
                    out.push(PyRunInfo { inner: ri.clone() });
                }
            }
        }
        out
    }

    /// All read IDs across the dataset as strings.
    fn read_ids(&self) -> PyResult<Vec<String>> {
        let mut out = Vec::new();
        for (_, reader) in &self.readers {
            for id in reader.read_ids().map_err(to_py_err)? {
                out.push(id.to_string());
            }
        }
        Ok(out)
    }

    /// Reads across the dataset, optionally filtered by a selection of IDs.
    ///
    /// Parameters
    /// ----------
    /// selection : list[str], optional
    ///     Read IDs to retrieve. If None, returns all reads in file order.
    /// missing_ok : bool, optional
    ///     If False (default), raise KeyError when any requested ID is not
    ///     present in the dataset. If True, silently skip missing IDs.
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn reads(&self, selection: Option<Vec<String>>, missing_ok: bool) -> PyResult<Vec<PyReadData>> {
        Ok(self
            .collect_inner(selection, missing_ok)?
            .into_iter()
            .map(|inner| PyReadData { inner })
            .collect())
    }

    /// Read metadata across the dataset as a column-oriented dict.
    ///
    /// Feeds `pd.DataFrame(...)` or `pl.DataFrame(...)`; signal is not included.
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn to_dict<'py>(
        &self,
        py: Python<'py>,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let reads = self.collect_inner(selection, missing_ok)?;
        let refs: Vec<&escapepod_signal::ReadData> = reads.iter().collect();
        crate::read_data::reads_to_columns(py, &refs)
    }

    /// Read metadata across the dataset as a `pandas.DataFrame`.
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn to_pandas<'py>(
        &self,
        py: Python<'py>,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reads = self.collect_inner(selection, missing_ok)?;
        let refs: Vec<&escapepod_signal::ReadData> = reads.iter().collect();
        crate::read_data::columns_to_pandas(py, &refs)
    }

    /// Read metadata across the dataset as a `polars.DataFrame`.
    #[pyo3(signature = (selection=None, missing_ok=false))]
    fn to_polars<'py>(
        &self,
        py: Python<'py>,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reads = self.collect_inner(selection, missing_ok)?;
        let refs: Vec<&escapepod_signal::ReadData> = reads.iter().collect();
        crate::read_data::columns_to_polars(py, &refs)
    }

    /// Raw ADC signal for a read as a numpy int16 array.
    ///
    /// Routes to whichever file in the dataset owns the read.
    fn get_signal<'py>(
        &self,
        py: Python<'py>,
        read: &PyReadData,
    ) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let reader = self.owning_reader(read)?;
        let signal_rows = read.inner.signal_rows.clone();
        let signal = py.detach(|| reader.get_signal(&signal_rows).map_err(to_py_err))?;
        Ok(PyArray1::from_vec(py, signal))
    }

    /// Calibrated pA signal for a read as a numpy float32 array.
    fn get_signal_pa<'py>(
        &self,
        py: Python<'py>,
        read: &PyReadData,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let reader = self.owning_reader(read)?;
        let signal_rows = read.inner.signal_rows.clone();
        let offset = read.inner.calibration_offset;
        let scale = read.inner.calibration_scale;
        let raw = py.detach(|| reader.get_signal(&signal_rows).map_err(to_py_err))?;
        Ok(PyArray1::from_vec(py, adc_to_pa(&raw, offset, scale)))
    }

    /// Number of stored (VBZ-compressed) signal bytes for a read.
    ///
    /// Routes to whichever file in the dataset owns the read.
    fn byte_count(&self, py: Python<'_>, read: &PyReadData) -> PyResult<usize> {
        let reader = self.owning_reader(read)?;
        let signal_rows = read.inner.signal_rows.clone();
        py.detach(|| {
            let chunks = reader
                .get_compressed_signal_for_rows(&signal_rows)
                .map_err(to_py_err)?;
            Ok(chunks.iter().map(|c| c.data.len()).sum())
        })
    }

    /// Raw ADC signal for multiple reads, decoded per owning file in parallel.
    ///
    /// Returns a list of (read_id, signal) tuples in the input order.
    fn get_signals<'py>(
        &self,
        py: Python<'py>,
        reads: Vec<PyRef<'_, PyReadData>>,
    ) -> PyResult<Vec<(String, Bound<'py, PyArray1<i16>>)>> {
        let decoded = self.decode_bulk(py, &reads)?;
        Ok(decoded
            .into_iter()
            .map(|(id, sig)| (id, PyArray1::from_vec(py, sig)))
            .collect())
    }

    /// Calibrated pA signal for multiple reads, decoded per owning file.
    fn get_signals_pa<'py>(
        &self,
        py: Python<'py>,
        reads: Vec<PyRef<'_, PyReadData>>,
    ) -> PyResult<Vec<(String, Bound<'py, PyArray1<f32>>)>> {
        let cal: HashMap<String, (f32, f32)> = reads
            .iter()
            .map(|r| {
                (
                    r.inner.read_id.to_string(),
                    (r.inner.calibration_offset, r.inner.calibration_scale),
                )
            })
            .collect();
        let decoded = self.decode_bulk(py, &reads)?;
        Ok(decoded
            .into_iter()
            .map(|(id, raw)| {
                let (offset, scale) = cal[&id];
                (id, PyArray1::from_vec(py, adc_to_pa(&raw, offset, scale)))
            })
            .collect())
    }

    // -- Context manager / dunders -----------------------------------------

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[allow(unused_variables)]
    fn __exit__(
        &self,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_val: Option<&Bound<'_, PyAny>>,
        exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        // mmap-based, no cleanup needed; return false to not suppress exceptions.
        false
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DatasetReader(files={}, reads={})",
            self.readers.len(),
            self.read_count()?
        ))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.read_count()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyDatasetReadIterator> {
        Ok(PyDatasetReadIterator {
            dataset: slf.into(),
            reader_idx: 0,
            num_batches: 0,
            batch_idx: 0,
            current_batch: None,
            batch_row: 0,
            batch_num_rows: 0,
            started: false,
        })
    }
}

impl PyDatasetReader {
    /// Collect read metadata across all files, optionally filtered by IDs.
    ///
    /// Shared backing for `reads`, `to_dict`, `to_pandas`, and `to_polars`.
    fn collect_inner(
        &self,
        selection: Option<Vec<String>>,
        missing_ok: bool,
    ) -> PyResult<Vec<escapepod_signal::ReadData>> {
        match selection {
            None => {
                let mut out = Vec::new();
                for (_, reader) in &self.readers {
                    for read in reader.reads().map_err(to_py_err)? {
                        out.push(read.map_err(to_py_err)?);
                    }
                }
                Ok(out)
            }
            Some(ids) => {
                let target: HashSet<escapepod_signal::Uuid> = ids
                    .iter()
                    .map(|s| {
                        escapepod_signal::utils::parse_uuid_flexible(s).map_err(|e| {
                            to_py_err(escapepod_signal::Error::InvalidUuid(e.to_string()))
                        })
                    })
                    .collect::<PyResult<_>>()?;

                let mut out = Vec::new();
                let mut found = HashSet::new();
                for (_, reader) in &self.readers {
                    for read in reader.reads_by_ids(&target).map_err(to_py_err)? {
                        found.insert(read.read_id);
                        out.push(read);
                    }
                }

                if !missing_ok && found.len() != target.len() {
                    let missing = target.len() - found.len();
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "{missing} of {} requested read id(s) not found in dataset \
                         (pass missing_ok=True to ignore)",
                        target.len()
                    )));
                }
                Ok(out)
            }
        }
    }

    /// Group reads by owning file, decode each file's signals in one bulk call,
    /// then restore the caller's input order.
    fn decode_bulk(
        &self,
        py: Python<'_>,
        reads: &[PyRef<'_, PyReadData>],
    ) -> PyResult<Vec<(String, Vec<i16>)>> {
        let index = self.id_index()?;

        // Bucket (original position, id, signal_rows) by owning reader.
        let mut buckets: HashMap<usize, Vec<(usize, String, Vec<u64>)>> = HashMap::new();
        for (pos, r) in reads.iter().enumerate() {
            let idx = *index.get(&r.inner.read_id).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "read {} is not part of this dataset",
                    r.inner.read_id
                ))
            })?;
            buckets.entry(idx).or_default().push((
                pos,
                r.inner.read_id.to_string(),
                r.inner.signal_rows.clone(),
            ));
        }

        let mut ordered: Vec<Option<(String, Vec<i16>)>> = (0..reads.len()).map(|_| None).collect();
        py.detach(|| -> PyResult<()> {
            for (reader_idx, items) in &buckets {
                let inputs: Vec<(usize, Vec<u64>)> = items
                    .iter()
                    .map(|(pos, _, rows)| (*pos, rows.clone()))
                    .collect();
                let results = self.readers[*reader_idx]
                    .1
                    .get_signal_bulk(&inputs)
                    .map_err(to_py_err)?;
                // get_signal_bulk preserves input order, so zip back to (pos, id).
                for ((pos, id, _), (_, sig)) in items.iter().zip(results) {
                    ordered[*pos] = Some((id.clone(), sig));
                }
            }
            Ok(())
        })?;

        Ok(ordered.into_iter().map(|o| o.unwrap()).collect())
    }
}

/// Streaming iterator over reads across every file in a dataset.
#[pyclass]
pub struct PyDatasetReadIterator {
    dataset: Py<PyDatasetReader>,
    reader_idx: usize,
    num_batches: usize,
    batch_idx: usize,
    current_batch: Option<RecordBatch>,
    batch_row: usize,
    batch_num_rows: usize,
    started: bool,
}

#[pymethods]
impl PyDatasetReadIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyReadData>> {
        let dataset = self.dataset.borrow(py);
        loop {
            // Yield remaining rows in the current batch.
            if self.batch_row < self.batch_num_rows {
                let batch = self.current_batch.as_ref().unwrap();
                let row = self.batch_row;
                self.batch_row += 1;
                let inner =
                    escapepod_signal::Reader::read_from_batch(batch, row).map_err(to_py_err)?;
                return Ok(Some(PyReadData { inner }));
            }

            // Advance to the next batch, crossing into the next file as needed.
            loop {
                if self.reader_idx >= dataset.readers.len() {
                    return Ok(None);
                }
                let reader = &dataset.readers[self.reader_idx].1;
                if !self.started {
                    self.num_batches = reader.read_batch_count().map_err(to_py_err)?;
                    self.batch_idx = 0;
                    self.started = true;
                }
                if self.batch_idx < self.num_batches {
                    break;
                }
                // Current file exhausted; move to the next one.
                self.reader_idx += 1;
                self.started = false;
            }

            let reader = &dataset.readers[self.reader_idx].1;
            let batch = reader.read_batch(self.batch_idx).map_err(to_py_err)?;
            self.batch_num_rows = batch.num_rows();
            self.current_batch = Some(batch);
            self.batch_row = 0;
            self.batch_idx += 1;
        }
    }
}

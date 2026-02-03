use pyo3::prelude::*;

mod read;
mod reader;
mod run_info;

/// Convert an escapepod error to a Python exception.
pub(crate) fn to_py_err(err: &escapepod::Error) -> PyErr {
    match err {
        escapepod::Error::Io(e) => {
            pyo3::exceptions::PyIOError::new_err(e.to_string())
        }
        escapepod::Error::InvalidSignature => {
            pyo3::exceptions::PyValueError::new_err("Invalid POD5 signature")
        }
        escapepod::Error::ReadNotFound(id) => {
            pyo3::exceptions::PyKeyError::new_err(id.to_string())
        }
        other => {
            pyo3::exceptions::PyRuntimeError::new_err(other.to_string())
        }
    }
}

/// Open a POD5 file for reading.
///
/// Parameters
/// ----------
/// path : str
///     Path to the POD5 file.
///
/// Returns
/// -------
/// Reader
#[pyfunction]
fn open(path: &str) -> PyResult<reader::Reader> {
    reader::Reader::new(path)
}

/// Python bindings for the escapepod POD5 library.
#[pymodule]
fn _escapepod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<reader::Reader>()?;
    m.add_class::<read::Read>()?;
    m.add_class::<read::Calibration>()?;
    m.add_class::<run_info::RunInfo>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    Ok(())
}

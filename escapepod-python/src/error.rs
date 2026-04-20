use pyo3::create_exception;
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;

create_exception!(escapepod, Pod5Error, pyo3::exceptions::PyException);

pub fn to_py_err(e: escapepod_signal::Error) -> PyErr {
    match e {
        escapepod_signal::Error::Io(e) => PyIOError::new_err(e.to_string()),
        escapepod_signal::Error::ReadNotFound(uuid) => {
            PyValueError::new_err(format!("Read not found: {uuid}"))
        }
        escapepod_signal::Error::InvalidUuid(msg) => {
            PyValueError::new_err(format!("Invalid UUID: {msg}"))
        }
        escapepod_signal::Error::BatchIndexOutOfBounds { index, max } => {
            PyIndexError::new_err(format!("Batch index {index} out of bounds (max: {max})"))
        }
        other => Pod5Error::new_err(other.to_string()),
    }
}

use pyo3::prelude::*;

mod error;
mod read_data;
mod reader;

/// Python bindings for the escapepod POD5 library.
#[pymodule]
fn escapepod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<reader::PyReader>()?;
    m.add_class::<read_data::PyReadData>()?;
    m.add_class::<read_data::PyRunInfo>()?;
    m.add("Pod5Error", m.py().get_type::<error::Pod5Error>())?;
    Ok(())
}

use pyo3::prelude::*;

mod error;
mod read_data;
mod reader;
mod writer;

/// Python bindings for the escapepod POD5 library.
#[pymodule]
fn escapepod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<reader::PyReader>()?;
    m.add_class::<read_data::PyReadData>()?;
    m.add_class::<read_data::PyRunInfo>()?;
    m.add_class::<writer::PyWriter>()?;
    m.add_function(wrap_pyfunction!(writer::create_run_info, m)?)?;
    m.add("Pod5Error", m.py().get_type::<error::Pod5Error>())?;
    Ok(())
}

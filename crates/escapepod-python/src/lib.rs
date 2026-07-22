use pyo3::prelude::*;

mod dataset;
mod error;
mod read_data;
mod reader;
mod signal;
mod writer;

/// Max read count for which entering a reader as a context manager
/// auto-builds the in-memory read-id index. Above this, entry is a no-op
/// and random-access selection falls back to the scan path. Overridable
/// via the `ESCAPEPOD_AUTOINDEX_MAX` environment variable.
pub(crate) fn autoindex_max() -> usize {
    std::env::var("ESCAPEPOD_AUTOINDEX_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000_000)
}

/// Python bindings for the escapepod POD5 library.
#[pymodule]
fn escapepod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<reader::PyReader>()?;
    m.add_class::<dataset::PyDatasetReader>()?;
    m.add_class::<read_data::PyReadData>()?;
    m.add_class::<read_data::PyRunInfo>()?;
    m.add_class::<writer::PyWriter>()?;
    m.add_function(wrap_pyfunction!(writer::create_run_info, m)?)?;
    m.add("Pod5Error", m.py().get_type::<error::Pod5Error>())?;
    signal::register(m)?;
    Ok(())
}

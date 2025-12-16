//! POD5 file writer implementation.

mod file_writer;

pub use file_writer::{AsyncSignalWriter, MmapSignalWriter, PredefinedDictionaries, Writer, WriterOptions};

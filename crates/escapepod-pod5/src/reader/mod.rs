//! POD5 file reader implementation.

mod cache;
mod file_reader;
mod read_index;
mod read_iter;
mod signal_extractor;
#[cfg(test)]
mod v6_compat;

pub use cache::{ReaderCache, cached_reader, global_reader_cache};
pub use file_reader::{NonUniformSignalBatch, Reader, autoindex_max};
pub use read_index::ReadIndex;
pub use signal_extractor::SignalExtractor;

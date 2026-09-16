//! POD5 file reader implementation.

mod cache;
mod dataset;
mod file_reader;
mod read_index;
mod read_iter;
mod signal_extractor;
#[cfg(test)]
mod v6_compat;

pub use cache::{ReaderCache, cached_reader, global_reader_cache};
pub use dataset::{Dataset, DatasetCache, cached_dataset, global_dataset_cache};
pub use file_reader::{NonUniformSignalBatch, Reader, SignalCalibration, autoindex_max};
pub use read_index::ReadIndex;
pub use signal_extractor::SignalExtractor;

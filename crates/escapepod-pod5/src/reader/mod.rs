//! POD5 file reader implementation.

pub mod byte_source;
mod file_reader;
mod read_index;
mod read_iter;
#[cfg(feature = "remote")]
mod remote;
mod signal_extractor;

pub use byte_source::{ByteSource, MemorySource, MmapSource, is_remote_url};
pub use file_reader::Reader;
pub use read_index::ReadIndex;
#[cfg(feature = "remote")]
pub use remote::RemoteSource;
pub use signal_extractor::SignalExtractor;

//! POD5 file writer implementation.

pub mod atomic;
mod file_writer;

pub use atomic::{AtomicFile, Durability, abort_all_in_flight_writes};
pub use file_writer::{PredefinedDictionaries, Writer, WriterOptions};

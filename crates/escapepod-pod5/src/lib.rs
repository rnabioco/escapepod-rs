//! POD5 file format I/O for Oxford Nanopore sequencing data.
//!
//! This crate provides the format layer — reading and writing POD5 files,
//! VBZ signal compression, and block-level operations (merge, filter, repack,
//! subset). Signal-processing algorithms live in the `escapepod` crate.
//!
//! # Example
//!
//! ```no_run
//! use escapepod_pod5::Reader;
//!
//! let reader = Reader::open("example.pod5")?;
//! for read_result in reader.reads()? {
//!     let read = read_result?;
//!     println!("Read: {}", read.read_id);
//! }
//! # Ok::<(), escapepod_pod5::Error>(())
//! ```

// Internal modules - not part of public API
pub(crate) mod arrow_helpers;
pub(crate) mod arrow_ipc;
pub(crate) mod footer;
pub(crate) mod schema;

// Modules with some public re-exports (implementation details hidden)
mod fields;
#[allow(unused_imports, dead_code, clippy::all, unsafe_op_in_unsafe_fn)]
mod flatbuffers_gen;
pub mod utils;

// Public modules
pub mod compression;
pub mod error;
pub mod merge;
pub mod operations;
pub mod progress;
pub mod reader;
pub mod types;
pub mod writer;

// Re-export CLI-facing utilities
pub use fields::{
    ALL_FIELDS, DEFAULT_FIELDS, FieldError, determine_fields, get_field_value, write_field_value,
};
pub use utils::parse_uuid_flexible;
pub use utils::{Statistics, compute_n50, compute_statistics};

// Re-export commonly used types
pub use arrow_helpers::{ReadColumns, ReadsBatchView};
pub use error::{Error, Result};
pub use merge::{MergeOptions, MergePhase, MergeProgress, MergeResult, merge_files};
pub use operations::{RepackOptions, RepackResult, repack_files};
pub use progress::{Progress, ProgressCallback};
pub use reader::ReadIndex;
pub use reader::Reader;
pub use reader::SignalExtractor;
pub use types::{EndReason, PoreType, ReadData, RunInfoData, SignalType, Uuid};
pub use writer::{
    AtomicFile, Durability, PredefinedDictionaries, Writer, WriterOptions,
    abort_all_in_flight_writes,
};

// Re-export Arrow types needed for batch-level operations
pub use arrow::record_batch::RecordBatch;

use std::sync::Arc;

/// A compressed signal chunk for block-level copying.
/// Uses Arc to avoid expensive clones during signal lookups.
#[derive(Debug, Clone)]
pub struct CompressedSignalChunk {
    /// The read ID this chunk belongs to.
    pub read_id: Uuid,
    /// Number of samples in this chunk.
    pub samples: u32,
    /// Compressed signal data (VBZ format).
    pub data: Arc<[u8]>,
}

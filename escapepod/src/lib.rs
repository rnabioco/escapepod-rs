//! POD5 file format library for Oxford Nanopore sequencing data.
//!
//! This crate provides functionality for reading and writing POD5 files,
//! which store nanopore sequencing data including raw signal, read metadata,
//! and run information.
//!
//! # Example
//!
//! ```no_run
//! use escapepod::Reader;
//!
//! let reader = Reader::open("example.pod5")?;
//! for read_result in reader.reads()? {
//!     let read = read_result?;
//!     println!("Read: {}", read.read_id);
//! }
//! # Ok::<(), escapepod::Error>(())
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
mod utils;

// Public modules
pub mod compression;
pub mod demux;
pub mod dtw;
pub mod error;
pub mod merge;
pub mod operations;
pub mod reader;
pub mod resquiggle;
pub mod segmentation;
pub mod types;
pub mod writer;

// Re-export CLI-facing utilities
pub use fields::{ALL_FIELDS, DEFAULT_FIELDS, FieldError, determine_fields, get_field_value};
pub use utils::parse_uuid_flexible;
pub use utils::{Statistics, compute_n50, compute_statistics};

// Re-export commonly used types
pub use error::{Error, Result};
pub use merge::{MergeOptions, MergePhase, MergeProgress, MergeResult, merge_files};
pub use operations::{RepackOptions, RepackResult, repack_files};
pub use reader::ReadIndex;
pub use reader::Reader;
pub use reader::SignalExtractor;
pub use types::{EndReason, ReadData, RunInfoData, SignalType, Uuid};
pub use writer::{PredefinedDictionaries, Writer, WriterOptions};

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

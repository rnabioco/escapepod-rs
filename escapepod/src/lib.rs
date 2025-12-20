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
mod utils;

// Public modules
pub mod compression;
pub mod dtw;
pub mod error;
pub mod merge;
pub mod operations;
pub mod reader;
pub mod segmentation;
pub mod types;
pub mod writer;

// Re-export CLI-facing utilities
pub use fields::{determine_fields, get_field_value, FieldError, ALL_FIELDS, DEFAULT_FIELDS};
pub use utils::parse_uuid_flexible;
pub use utils::{compute_n50, compute_statistics, Statistics};

// Re-export commonly used types
pub use error::{Error, Result};
pub use merge::{merge_files, MergeOptions, MergeResult};
pub use operations::{repack_files, RepackOptions, RepackResult};
pub use reader::Reader;
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

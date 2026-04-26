//! Signal-processing algorithms for Oxford Nanopore POD5 data.
//!
//! This crate layers signal-processing algorithms (DTW, resquiggle,
//! segmentation) on top of the POD5 format I/O provided by
//! [`escapepod_pod5`]. Format types and operations are re-exported here so
//! consumers can depend on a single crate for both layers.
//!
//! Barcode demultiplexing has moved to the dedicated
//! [`escapepod-demux`](https://crates.io/crates/escapepod-demux) crate,
//! which depends on this one for DTW and fingerprint primitives.
//!
//! # Example
//!
//! ```no_run
//! use escapepod_signal::Reader;
//!
//! let reader = Reader::open("example.pod5")?;
//! for read_result in reader.reads()? {
//!     let read = read_result?;
//!     println!("Read: {}", read.read_id);
//! }
//! # Ok::<(), escapepod_signal::Error>(())
//! ```

// Signal-processing modules live in this crate.
pub mod dtw;
pub mod resquiggle;
pub mod segmentation;

// Format layer (POD5 I/O) lives in escapepod-pod5. Re-export its modules and
// types so downstream consumers can pull in both layers via this crate.
pub use escapepod_pod5 as pod5;

pub use escapepod_pod5::{
    compression, error, merge, operations, progress, reader, types, utils, writer,
};

pub use escapepod_pod5::{
    ALL_FIELDS, CompressedSignalChunk, DEFAULT_FIELDS, EndReason, Error, FieldError, MergeOptions,
    MergePhase, MergeProgress, MergeResult, PoreType, PredefinedDictionaries, Progress,
    ProgressCallback, ReadData, ReadIndex, Reader, ReadsBatchView, RecordBatch, RepackOptions,
    RepackResult, Result, RunInfoData, SignalExtractor, SignalType, Statistics, Uuid, Writer,
    WriterOptions, compute_n50, compute_statistics, determine_fields, get_field_value, merge_files,
    parse_uuid_flexible, repack_files, write_field_value,
};

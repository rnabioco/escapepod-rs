//! Signal-processing algorithms for Oxford Nanopore POD5 data.
//!
//! This crate layers signal-processing algorithms (DTW, barcode demultiplexing,
//! resquiggle, segmentation) on top of the POD5 format I/O provided by
//! [`escapepod_pod5`]. Format types and operations are re-exported here so
//! existing consumers can depend solely on `escapepod`.
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

// Signal-processing modules live in this crate.
pub mod demux;
pub mod dtw;
pub mod resquiggle;
pub mod segmentation;

// Format layer (POD5 I/O) lives in escapepod-pod5. Re-export its modules and
// types so consumers that depend on `escapepod` keep working unchanged.
pub use escapepod_pod5 as pod5;

pub use escapepod_pod5::{
    compression, error, merge, operations, progress, reader, types, utils, writer,
};

pub use escapepod_pod5::{
    ALL_FIELDS, CompressedSignalChunk, DEFAULT_FIELDS, EndReason, Error, FieldError, MergeOptions,
    MergePhase, MergeProgress, MergeResult, PredefinedDictionaries, Progress, ProgressCallback,
    ReadData, ReadIndex, Reader, RecordBatch, RepackOptions, RepackResult, Result, RunInfoData,
    SignalExtractor, SignalType, Statistics, Uuid, Writer, WriterOptions, compute_n50,
    compute_statistics, determine_fields, get_field_value, merge_files, parse_uuid_flexible,
    repack_files,
};

//! POD5 file format library for Oxford Nanopore sequencing data.
//!
//! This crate provides functionality for reading and writing POD5 files,
//! which store nanopore sequencing data including raw signal, read metadata,
//! and run information.
//!
//! # Example
//!
//! ```no_run
//! use podfive_core::Reader;
//!
//! let reader = Reader::open("example.pod5")?;
//! for read_result in reader.reads()? {
//!     let read = read_result?;
//!     println!("Read: {}", read.read_id);
//! }
//! # Ok::<(), podfive_core::Error>(())
//! ```

pub mod compression;
pub mod error;
pub mod footer;
pub mod reader;
pub mod schema;
pub mod types;
pub mod writer;

// Re-export commonly used types
pub use error::{Error, Result};
pub use reader::Reader;
pub use types::{EndReason, ReadData, RunInfoData, SignalType, Uuid};
pub use writer::{Writer, WriterOptions};

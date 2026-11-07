//! Signal-processing algorithms for Oxford Nanopore POD5 data.
//!
//! This crate layers signal-processing algorithms (DTW, resquiggle,
//! segmentation) on top of the POD5 format I/O provided by
//! [`escapepod_pod5`], which is re-exported as [`pod5`].
//!
//! Barcode demultiplexing has moved to the dedicated
//! [`escapepod-demux`](https://crates.io/crates/escapepod-demux) crate,
//! which depends on this one for DTW and fingerprint primitives.
//!
//! # Example
//!
//! ```no_run
//! use escapepod_signal::pod5::Reader;
//!
//! let reader = Reader::open("example.pod5")?;
//! for read_result in reader.reads()? {
//!     let read = read_result?;
//!     println!("Read: {}", read.read_id);
//! }
//! # Ok::<(), escapepod_signal::pod5::Error>(())
//! ```

// Signal-processing modules live in this crate.
pub mod chunk;
pub mod dtw;
pub mod fasta;
pub mod features;
pub mod lstm;
pub mod mapping;
pub mod resquiggle;
pub mod segmentation;
pub mod seq_encoding;
pub mod stats;

// Format layer (POD5 I/O) lives in escapepod-pod5. It is re-exported whole as
// `pod5` and nowhere else: `escapepod_signal::pod5::Reader`, not
// `escapepod_signal::Reader`.
pub use escapepod_pod5 as pod5;

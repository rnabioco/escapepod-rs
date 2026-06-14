//! Pure-Rust toolkit for Oxford Nanopore POD5 data.
//!
//! `escapepod` is an umbrella crate: it bundles the escapepod workspace layers
//! behind feature flags and re-exports each as a module, so you can depend on a
//! single crate and pull in exactly the layers you need.
//!
//! | Module                       | Crate                                                         | Feature       |
//! |------------------------------|---------------------------------------------------------------|---------------|
//! | [`pod5`]                     | [`escapepod-pod5`](https://crates.io/crates/escapepod-pod5)   | `pod5`        |
//! | [`signal`]                   | [`escapepod-signal`](https://crates.io/crates/escapepod-signal) | `signal`    |
//! | [`demux`]                    | [`escapepod-demux`](https://crates.io/crates/escapepod-demux) | `demux`       |
//!
//! The `pod5` and `signal` features are enabled by default. `demux` is opt-in
//! (it pulls in serde/linfa/tract-onnx); the `train`, `gpu`, and `cnn-detect`
//! features forward to the demux crate and each imply `demux`.
//!
//! ```toml
//! # Just the format + signal layers (default):
//! escapepod = "0.5"
//!
//! # Add barcode demultiplexing:
//! escapepod = { version = "0.5", features = ["demux"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use escapepod::signal::Reader;
//!
//! let reader = Reader::open("example.pod5")?;
//! for read_result in reader.reads()? {
//!     let read = read_result?;
//!     println!("Read: {}", read.read_id);
//! }
//! # Ok::<(), escapepod::signal::Error>(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

/// POD5 format I/O — reader, writer, VBZ compression, block operations.
#[cfg(feature = "pod5")]
pub use escapepod_pod5 as pod5;

/// Signal-processing algorithms — DTW, resquiggle, segmentation.
///
/// Re-exports the [`pod5`] format surface as well, so this is the single
/// entry point most consumers want.
#[cfg(feature = "signal")]
pub use escapepod_signal as signal;

/// WarpDemuX-compatible barcode demultiplexing — DTW + SVM, optional CNN
/// adapter detection and GPU acceleration.
#[cfg(feature = "demux")]
pub use escapepod_demux as demux;

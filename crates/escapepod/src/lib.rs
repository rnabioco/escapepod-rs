//! Pure-Rust toolkit for Oxford Nanopore POD5 data.
//!
//! The headline artifact of this crate is the **`escpod` command-line tool**:
//!
//! ```sh
//! cargo install escapepod   # installs the `escpod` binary
//! ```
//!
//! This library surface is secondary. It is an umbrella that re-exports the
//! escapepod workspace layers behind feature flags, so library consumers can
//! depend on a single crate and pull in exactly the layers they need without
//! the CLI's dependency tree:
//!
//! ```toml
//! # Library only — no clap/noodles/etc.:
//! escapepod = { version = "0.5", default-features = false, features = ["signal"] }
//! ```
//!
//! | Module       | Crate                                                            | Feature   |
//! |--------------|------------------------------------------------------------------|-----------|
//! | [`pod5`]     | [`escapepod-pod5`](https://crates.io/crates/escapepod-pod5)       | `pod5`    |
//! | [`signal`]   | [`escapepod-signal`](https://crates.io/crates/escapepod-signal)  | `signal`  |
//! | [`demux`]    | [`escapepod-demux`](https://crates.io/crates/escapepod-demux)    | `demux`   |
//!
//! The default `cli` feature enables `signal` (and `pod5` transitively).
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "signal")]
//! # fn main() -> Result<(), escapepod::signal::Error> {
//! use escapepod::signal::Reader;
//!
//! let reader = Reader::open("example.pod5")?;
//! for read_result in reader.reads()? {
//!     let read = read_result?;
//!     println!("Read: {}", read.read_id);
//! }
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "signal"))]
//! # fn main() {}
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

/// POD5 format I/O — reader, writer, VBZ compression, block operations.
#[cfg(feature = "pod5")]
pub use escapepod_pod5 as pod5;

/// Signal-processing algorithms — DTW, resquiggle, segmentation.
///
/// Re-exports the [`pod5`] format surface as well, so this is the single
/// entry point most library consumers want.
#[cfg(feature = "signal")]
pub use escapepod_signal as signal;

/// WarpDemuX-compatible barcode demultiplexing — DTW + SVM, optional CNN
/// adapter detection and GPU acceleration.
#[cfg(feature = "demux")]
pub use escapepod_demux as demux;

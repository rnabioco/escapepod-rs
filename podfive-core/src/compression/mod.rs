//! Signal compression and decompression (VBZ format).
//!
//! VBZ compression uses a two-stage pipeline:
//! 1. SVB16: StreamVByte encoding with delta and zigzag transforms
//! 2. ZSTD: General-purpose compression at level 1

pub mod svb16;
pub mod vbz;

pub use vbz::{compress_signal, decompress_signal};

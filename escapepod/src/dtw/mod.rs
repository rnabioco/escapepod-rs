//! Dynamic Time Warping (DTW) distance computation for barcode fingerprint comparison.
//!
//! This module provides DTW distance computation with Sakoe-Chiba band constraints,
//! parallel distance matrix computation, and kernel conversion for classification.
//!
//! Inspired by WarpDemuX for nanopore barcode demultiplexing.

mod dtw;
mod fingerprint;
mod kernel;

pub use dtw::{dtw_distance, dtw_distance_matrix, dtw_distance_matrix_blocked};
pub use fingerprint::{normalize_fingerprint, Fingerprint, NormMethod};
pub use kernel::{distance_to_kernel, distance_to_kernel_auto};

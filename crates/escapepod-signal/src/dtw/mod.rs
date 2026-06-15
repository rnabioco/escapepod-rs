//! Dynamic Time Warping (DTW) distance computation for barcode fingerprint comparison.
//!
//! This module provides DTW distance computation with Sakoe-Chiba band constraints,
//! parallel distance matrix computation, and kernel conversion for classification.
//!
//! Inspired by WarpDemuX for nanopore barcode demultiplexing.

mod distance;
mod fingerprint;
mod kernel;

#[cfg(feature = "gpu")]
pub mod cuda;

pub use distance::{
    DtwScratch, dtw_distance, dtw_distance_bounded, dtw_distance_bounded_penalty,
    dtw_distance_bounded_penalty_into, dtw_distance_matrix, dtw_distance_matrix_blocked,
    dtw_distance_penalty,
};
pub use fingerprint::{Fingerprint, NormMethod, normalize_fingerprint};
pub use kernel::{distance_to_kernel, distance_to_kernel_auto};

#[cfg(feature = "gpu")]
pub use cuda::{
    DTW_KERNEL_NAME, DTW_MODULE_NAME, GpuDtwContext, GpuDtwError, OVO_DECISION_KERNEL_NAME,
    RBF_KERNEL_NAME, SVM_MODULE_NAME, dtw_distance_matrix_gpu,
};

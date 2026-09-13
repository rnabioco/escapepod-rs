//! Dynamic Time Warping (DTW) distance computation for barcode fingerprint comparison.
//!
//! This module provides DTW distance computation with Sakoe-Chiba band constraints
//! and parallel distance matrix computation.
//!
//! Inspired by WarpDemuX for nanopore barcode demultiplexing.

mod distance;
mod fingerprint;

#[cfg(feature = "gpu")]
pub mod cuda;

pub use distance::{
    DTW_LANES, DtwBatchScratch, DtwScratch, dtw_distance, dtw_distance_bounded,
    dtw_distance_bounded_penalty, dtw_distance_bounded_penalty_into, dtw_distance_matrix,
    dtw_distances_batch, dtw_distances_batch_unconstrained, pack_training_blocks,
};
pub use fingerprint::{Fingerprint, NormMethod, normalize_fingerprint};

#[cfg(feature = "gpu")]
pub use cuda::{
    DTW_KERNEL_NAME, DTW_MODULE_NAME, GpuDtwContext, GpuDtwError, OVO_DECISION_KERNEL_NAME,
    RBF_KERNEL_NAME, SVM_MODULE_NAME, dtw_distance_matrix_gpu,
};

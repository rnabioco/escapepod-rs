//! Signal segmentation primitives for nanopore signal processing.
//!
//! This module provides algorithms for detecting boundaries and segments in nanopore
//! signal data, useful for adapter detection, poly(A) tail detection, and general
//! signal changepoint analysis.
//!
//! # Modules
//!
//! - [`llr`]: Log-Likelihood Ratio boundary detection (adapted from ADAPTed)
//! - [`ttest`]: Windowed t-test segmentation (adapted from WarpDemuX/Tombo)
//! - [`normalize`]: Signal normalization utilities (MAD normalization, downscaling)
//!
//! # Examples
//!
//! ## MAD Normalization
//!
//! ```
//! use escapepod::segmentation::mad_normalize;
//!
//! let signal = vec![100.0, 102.0, 98.0, 101.0, 99.0];
//! let normalized = mad_normalize(&signal);
//! ```
//!
//! ## LLR Adapter Detection
//!
//! ```
//! use escapepod::segmentation::{LlrTrace, detect_adapter};
//!
//! let signal = vec![120.0; 100]; // Your signal here
//! let (adapter_start, adapter_end) = detect_adapter(&signal, 10, 5);
//!
//! if adapter_end > 0 {
//!     println!("Adapter detected from {} to {}", adapter_start, adapter_end);
//! }
//! ```
//!
//! ## T-test Segmentation
//!
//! ```
//! use escapepod::segmentation::segment_signal;
//!
//! let signal = vec![50.0; 100]; // Your signal here
//! let segments = segment_signal(&signal, 10, 5, 15);
//!
//! for (start, end, mean) in segments {
//!     println!("Segment [{}, {}): mean = {:.2}", start, end, mean);
//! }
//! ```

pub mod consensus;
pub mod llr;
pub mod normalize;
pub mod ttest;

// Re-export main types and functions for convenience
pub use consensus::{
    segment_with_consensus, ConsensusConfig, ConsensusSegmentationResult,
    CONSENSUS_RNA004_130BPS_V1_0,
};
pub use llr::{detect_adapter, LlrTrace};
pub use normalize::{
    downscale, mad_normalize, mad_normalize_with_clipping, normalize_dwell_times,
    normalize_dwell_times_mad,
};
pub use ttest::{
    compute_segment_means, find_changepoints, segment_signal, segment_signal_with_dwell,
    windowed_ttest, SegmentationResult,
};

//! Barcode demultiplexing using WarpDemuX-style models.
//!
//! This module provides support for loading and using trained WarpDemuX models
//! to classify nanopore reads by barcode.
//!
//! ## Model Types
//!
//! Two model types are supported:
//!
//! - **WarpDemuxModel**: Legacy distance-based nearest-neighbor classifier
//! - **DtwSvmModel**: Full SVM classifier with probability output
//!
//! ## Workflow
//!
//! 1. Load a trained model from JSON
//! 2. Extract fingerprints from reads (using DTW fingerprinting)
//! 3. Classify reads and get probability distributions
//!
//! ## Example (Legacy Model)
//!
//! ```no_run
//! use escapepod_signal::demux::{load_model, classify_read};
//! use std::path::Path;
//!
//! let model = load_model(Path::new("model.json"))?;
//! let fingerprint = vec![0.1, 0.2, 0.3, 0.4, 0.5];
//! let result = classify_read(&model, &fingerprint);
//!
//! println!("Barcode: {}", result.barcode);
//! println!("Confidence: {:.3}", result.confidence);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example (SVM Model with Probabilities)
//!
//! ```no_run
//! use escapepod_signal::demux::{load_svm_model, classify_with_svm};
//! use std::path::Path;
//!
//! let model = load_svm_model(Path::new("svm_model.json"))?;
//! let fingerprint = vec![0.1, 0.2, 0.3, 0.4, 0.5];
//! let (probs, result) = classify_with_svm(&model, &fingerprint);
//!
//! println!("Barcode: {}", result.predicted_barcode);
//! println!("Confidence: {:.3}", result.confidence);
//! println!("Probabilities: {:?}", probs);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod classify;
mod model;
mod probability;
mod svm;

// Training module (feature-gated)
#[cfg(feature = "train")]
mod train;

// Legacy exports (distance-based classifier)
pub use classify::{ClassificationResult, classify_from_distances, classify_read};

#[cfg(feature = "gpu")]
pub use classify::{classify_reads_gpu, classify_reads_gpu_with_ctx};
pub use model::{KernelParams, WarpDemuxModel, load_model};

// New SVM exports
pub use model::{DtwSvmModel, load_svm_model};
pub use probability::{
    ProbabilityResult, confidence_margin, format_probability_columns, process_probabilities,
    softmax,
};
pub use svm::{SvmModel, SvmPredictor, classify_with_svm, compute_distances, distances_to_kernel};

#[cfg(feature = "gpu")]
pub use svm::{classify_with_svm_batch_gpu, classify_with_svm_batch_gpu_with_ctx};

// Training exports (feature-gated)
#[cfg(feature = "train")]
pub use train::*;

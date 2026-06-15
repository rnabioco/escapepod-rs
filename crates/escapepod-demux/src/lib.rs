//! WarpDemuX-compatible barcode demultiplexing for Oxford Nanopore POD5 data.
//!
//! This crate layers on top of [`escapepod-signal`] and packages the pieces
//! needed to demultiplex nanopore reads against a WarpDemuX-style model:
//!
//! - [`WarpDemuxModel`] and [`DtwSvmModel`] JSON loaders.
//! - Per-read DTW classifier ([`classify_read`]) and full SVM predictor
//!   ([`classify_with_svm`]) with Platt scaling + libsvm-style OvO coupling.
//! - Optional `train` feature: fit a `DtwSvmModel` from labeled fingerprints
//!   via linfa-svm ([`train_svm`] and friends).
//! - Optional `gpu` feature: batched GPU DTW matrix (routed through
//!   `escapepod-signal`'s CUDA kernel) for classify and training.
//! - Optional `cnn-detect` feature: port of ADAPTed's `BoundariesCNN` for
//!   adapter-end detection via tract-onnx ([`AdapterCnn`]).
//!
//! # Model workflow
//!
//! 1. Load a trained model from JSON.
//! 2. Extract fingerprints from reads (DTW fingerprinting — see
//!    `escapepod_signal::dtw`).
//! 3. Classify reads and read off probabilities / assignments.
//!
//! # Example — legacy distance-based model
//!
//! ```no_run
//! use escapepod_demux::{load_model, classify_read};
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
//! # Example — SVM model with probabilities
//!
//! ```no_run
//! use escapepod_demux::{load_svm_model, classify_with_svm};
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
mod fingerprint;
mod gbm;
mod model;
mod probability;
mod svm;

#[cfg(feature = "train")]
mod train;

#[cfg(feature = "cnn-detect")]
pub mod adapter_cnn;

#[cfg(feature = "cnn-detect")]
mod adapter_cnn_compute;

#[cfg(all(feature = "gpu", feature = "cnn-detect"))]
mod gpu_cnn;

pub use fingerprint::{
    BarcodeFingerprint, ReadBoundaries, ReadFingerprint, compute_consensus_fingerprint,
    compute_std_dev_fingerprint, extract_fingerprint_from_signal,
};

// Legacy distance-based classifier.
pub use classify::{ClassificationResult, classify_from_distances, classify_read};

#[cfg(feature = "gpu")]
pub use classify::{classify_reads_gpu, classify_reads_gpu_with_ctx};

pub use gbm::{GbmModel, GbmNode, GbmPredictor, GbmTree, load_gbm_model};
pub use model::{
    AnyModel, DtwSvmModel, KernelParams, WarpDemuxModel, load_any_model, load_model, load_svm_model,
};
pub use probability::{
    ProbabilityResult, confidence_margin, format_probability_columns, process_probabilities,
    softmax,
};
pub use svm::{
    SvmModel, SvmPredictor, SvmWorkspace, classify_with_svm, compute_distances, distances_to_kernel,
};

pub use svm::DEFAULT_GPU_CHUNK_CELLS;

#[cfg(feature = "gpu")]
pub use svm::{classify_with_svm_batch_gpu, classify_with_svm_batch_gpu_with_ctx};

#[cfg(feature = "train")]
pub use train::*;

#[cfg(feature = "cnn-detect")]
pub use adapter_cnn::{AdapterCnn, AdapterCnnConfig, AdapterCnnError};

#[cfg(feature = "cnn-detect")]
pub use adapter_cnn_compute::{CnnCompute, CnnComputeError, CnnWeights};

#[cfg(all(feature = "gpu", feature = "cnn-detect"))]
pub use gpu_cnn::{GpuCnn, GpuCnnError};

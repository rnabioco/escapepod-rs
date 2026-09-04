//! Barcode demultiplexing for Oxford Nanopore POD5 data.
//!
//! This crate layers on top of [`escapepod-signal`] and packages two
//! independent ways to assign a read to a barcode — fingerprint-and-classify
//! (WarpDemuX-compatible) and basecall-then-match (CTC-CRF) — plus the adapter
//! detection both need.
//!
//! Fingerprint-and-classify:
//!
//! - [`WarpDemuxModel`] and [`DtwSvmModel`] JSON loaders.
//! - Per-read DTW classifier ([`classify_read`]) and full SVM predictor
//!   ([`classify_with_svm`]) with Platt scaling + libsvm-style OvO coupling.
//! - Optional `train` feature: build a `DtwSvmModel` from labeled fingerprints
//!   ([`train_svm`] and friends). Today this is a labels-only stub that relies
//!   on kernel-weighted voting at predict time; see `train.rs`.
//! - Optional `gpu` feature: batched GPU DTW matrix (routed through
//!   `escapepod-signal`'s CUDA kernel) for classify and training.
//! - Optional `cnn-detect` feature: adapter-end detection by running an
//!   exported boundary-CNN ONNX graph through tract-onnx ([`AdapterCnn`]).
//!   This is CPU-only and architecture-agnostic (any `[B,1,L] -> [B,2,L]`
//!   graph); [`AdapterCnnGpu`] adds a CUDA path under `gpu`.
//!
//! Basecall-then-match ([`crf`]):
//!
//! - [`crf::lattice`]: the CTC-CRF decode — a port of bonito's `CTC_CRF`,
//!   which upstream exists only as CUDA kernels. Pure `f32` with no
//!   dependencies and no feature gate, plus runtime-dispatched AVX2 kernels.
//! - Optional `crf-decode` feature: the ONNX encoder through tract
//!   ([`crf::CrfEncoder`]); `gpu` runs it on onnxruntime + CUDA instead.
//! - [`crf::BarcodeRefs`]: matching a decoded sequence to a barcode reference
//!   by edit distance (wavefront alignment via `fqxv-align`), with the
//!   margin-to-second-best as the confidence.
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
pub mod crf;

#[cfg(feature = "gpu")]
pub mod cuda;
mod fingerprint;
mod gbm;
mod model;
#[cfg(feature = "gpu")]
mod ort_ep;
mod probability;
mod svm;

#[cfg(feature = "train")]
mod train;

#[cfg(feature = "cnn-detect")]
pub mod adapter_cnn;

#[cfg(feature = "gpu")]
pub mod adapter_cnn_gpu;

// Present whenever tract is: both ONNX features pull it, and every tract
// loader in the workspace (here and in escapepod-classify) runs its graph
// through this before optimizing.
#[cfg(any(feature = "cnn-detect", feature = "crf-decode"))]
pub mod onnx_rewrite;

pub use fingerprint::{
    BOUNDARY_PADDING_SAMPLES, BarcodeFingerprint, MAX_FINGERPRINT_WINDOW, ReadBoundaries,
    ReadFingerprint, compute_consensus_fingerprint, compute_std_dev_fingerprint,
    extract_fingerprint_from_signal,
};

/// Make onnxruntime CUDA execution-provider registration fatal for every session
/// built after this call, instead of silently falling back to the CPU provider.
///
/// Call once at startup when the GPU was *demanded* rather than preferred (the
/// CLI's `--device gpu`); see `src/ort_ep.rs` for why this is process-wide state
/// and not a loader parameter.
#[cfg(feature = "gpu")]
pub use ort_ep::require_cuda as require_cuda_ep;

// Legacy distance-based classifier.
pub use classify::{ClassificationResult, classify_from_distances, classify_read};

#[cfg(feature = "gpu")]
pub use classify::{classify_reads_gpu, classify_reads_gpu_with_ctx};

pub use gbm::{GbmHead, GbmModel, GbmNode, GbmPredictor, GbmTree, load_gbm_model};
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
pub use adapter_cnn::{AdapterCnn, AdapterCnnConfig, AdapterCnnError, PreppedWindow};

#[cfg(feature = "gpu")]
pub use adapter_cnn_gpu::AdapterCnnGpu;

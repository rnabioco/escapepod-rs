//! CTC-CRF barcode basecalling: ONNX encoder inference plus a native decode.
//!
//! The split follows what is and isn't expressible in ONNX. The **encoder**
//! (convolutions → LSTM stack → `LinearCRFEncoder`) is ordinary feed-forward
//! torch and exports cleanly; the **decode** is a sparse lattice forward-backward
//! that standard ONNX ops cannot express and that bonito itself only implements
//! as hand-written CUDA (`koi`). So the encoder ships as an ONNX file and the
//! decode lives here, in portable Rust with no GPU requirement.
//!
//! * [`lattice`] — the decode. Pure `f32` arithmetic, no dependencies, always
//!   compiled, so CI exercises it without needing a model file.
//! * [`encoder`] — the ONNX encoder through tract on the CPU (`crf-decode`).
//!
//! Both are architecture-agnostic in the same sense as [`crate::adapter_cnn`]:
//! the contract is a graph taking `[batch, 1, chunk]` standardised signal and
//! returning **time-major** `[chunk / stride, batch, n_score]` scores. Note the
//! axis order — the boundary CNN is batch-major and its loader hard-rejects
//! anything else, so the two cannot share a shape probe.

pub mod lattice;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(feature = "crf-decode")]
pub mod barcode;

#[cfg(feature = "crf-decode")]
pub mod encoder;

#[cfg(feature = "crf-gpu")]
pub mod encoder_gpu;

pub use lattice::{Backend, CrfDecodeError, CrfLayout, CrfScratch, decode, decode_with};

#[cfg(feature = "crf-decode")]
pub use barcode::{BarcodeError, BarcodeMatch, BarcodeRefs};

#[cfg(feature = "crf-decode")]
pub use encoder::{BarcodeEntry, BoundarySpec, CrfEncoder, CrfError, CrfMetadata, ModelIdent};

#[cfg(feature = "crf-gpu")]
pub use encoder_gpu::CrfEncoderGpu;

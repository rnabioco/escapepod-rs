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

/// Constrained partition functions over known references — the CRF's own
/// opinion of a barcode, as opposed to an edit distance to its decode.
/// Always compiled, like [`lattice`], so CI exercises it without a model file.
pub mod refchain;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(feature = "crf-decode")]
pub mod barcode;

#[cfg(feature = "crf-decode")]
pub mod encoder;

#[cfg(feature = "gpu")]
pub mod encoder_gpu;

/// Batched CUDA lattice decode. The CPU decode in [`lattice`] stays the
/// reference and the only always-compiled path; this is the same two passes
/// with the batch axis mapped onto the device.
#[cfg(feature = "gpu")]
pub mod lattice_gpu;

#[cfg(feature = "gpu")]
mod lattice_gpu_kernel;

pub use lattice::{
    Backend, CrfDecodeError, CrfLayout, CrfScratch, decode, decode_with, decode_with_refs,
    decode_with_refs_strided,
};
pub use refchain::{RefChainError, RefChains, ScoredDecode};

#[cfg(feature = "crf-decode")]
pub use barcode::{BarcodeError, BarcodeMatch, BarcodeRefs};

#[cfg(feature = "crf-decode")]
pub use encoder::{
    Anchor, BarcodeEntry, BoundarySpec, CrfEncoder, CrfError, CrfMetadata, ModelIdent,
};

/// The input tensor a bundle's pinned boundary CNN consumes
/// (`boundary.input` in the sidecar; escapepod-models writes it from the same
/// `DataConfig` that framed every training example).
///
/// This block exists because nothing declared it before: the geometry lived as
/// hardcoded constants on both sides of the repo boundary, they drifted, and
/// escpod fed the model variable-length truncated tensors it was never trained
/// on (#187). `AdapterCnnConfig::from_bundle_input` is the one consumer.
///
/// Defined here, outside the `crf-decode` gate, because the detector pin
/// plumbing has to name the type in every feature combination — the sidecar
/// parsing (`crf-decode`) and the CNN it configures (`cnn-detect`) are
/// independently selectable.
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
pub struct BoundaryInputSpec {
    /// Raw-signal window start, in samples.
    pub min_obs_adapter: usize,
    /// Raw-signal window end, in samples. Normalisation statistics and the
    /// model input both come from `signal[min_obs_adapter:max_obs_trace]`.
    pub max_obs_trace: usize,
    /// Mean-pool factor from raw samples to model positions.
    pub downscale_factor: usize,
    /// Fixed tensor length the model consumes, in downscaled positions.
    /// Derivable from the three fields above and declared anyway, so producer
    /// and consumer computing different lengths is a load error rather than a
    /// silent preference for one of them.
    pub input_len: usize,
    /// Value training right-pads short windows with (and substitutes for NaN).
    pub pad_value: f32,
}

#[cfg(feature = "gpu")]
pub use encoder_gpu::CrfEncoderGpu;

#[cfg(feature = "gpu")]
pub use lattice_gpu::CrfLatticeGpu;

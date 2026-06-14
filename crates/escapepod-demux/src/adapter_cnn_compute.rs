//! Self-contained BoundariesCNN forward pass (no ONNX runtime), for the
//! batched CPU reference and the GPU port.
//!
//! `adapter_cnn.rs` runs the CNN one read at a time through `tract-onnx`. For
//! GPU acceleration the win comes from batching many reads through the conv
//! stack at once, which means re-implementing the (small, fixed) network:
//!
//! ```text
//!   x:[1,L]
//!   Conv1d(1->64, k=7, stride=3, pad=3) + ReLU        -> [64, L0]
//!   Conv1d(64->64, k=7, stride=1, pad=3) + ReLU        -> [64, L0]
//!   Conv1d(64->64, k=7, stride=1, pad=3) + ReLU        -> [64, L0]
//!   ConvTranspose1d(64->2, k=7, stride=3, pad=3)       -> [2,  L6]   (= scores)
//! ```
//!
//! Weights come from the companion `.weights` blob produced by
//! `scripts/dump_adapter_cnn_weights.py` (raw little-endian f32, fixed order).
//! This module is the verified CPU reference; `gpu_cnn.rs` ports the same
//! arithmetic to cudarc kernels and is checked against it.

use std::io::Read;
use std::path::Path;

use thiserror::Error;

use crate::adapter_cnn::{AdapterCnnConfig, mean_pool, median_mad_normalize};

pub(crate) const K: usize = 7;
pub(crate) const C: usize = 64; // hidden channels

#[derive(Debug, Error)]
pub enum CnnComputeError {
    #[error("failed to read weights: {0}")]
    Io(#[from] std::io::Error),
    #[error("weights file has {got} f32, expected {expected}")]
    BadSize { got: usize, expected: usize },
    #[error("signal too short ({len} samples, need > {required})")]
    SignalTooShort { len: usize, required: usize },
}

/// BoundariesCNN weights, flat little-endian f32 in the order written by
/// `dump_adapter_cnn_weights.py`.
pub struct CnnWeights {
    pub(crate) w0: Vec<f32>, // [64,1,7]
    pub(crate) b0: Vec<f32>, // [64]
    pub(crate) w2: Vec<f32>, // [64,64,7]
    pub(crate) b2: Vec<f32>,
    pub(crate) w4: Vec<f32>, // [64,64,7]
    pub(crate) b4: Vec<f32>,
    pub(crate) w6: Vec<f32>, // [64,2,7]  (ConvTranspose: Cin=64, Cout=2, K=7)
    pub(crate) b6: Vec<f32>, // [2]
}

impl CnnWeights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CnnComputeError> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        let n = bytes.len() / 4;
        let mut floats = Vec::with_capacity(n);
        for c in bytes.chunks_exact(4) {
            floats.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        let expected = 64 * K + 64 + C * C * K + C + C * C * K + C + C * 2 * K + 2;
        if floats.len() != expected {
            return Err(CnnComputeError::BadSize {
                got: floats.len(),
                expected,
            });
        }
        let mut off = 0;
        let mut take = |len: usize| {
            let s = floats[off..off + len].to_vec();
            off += len;
            s
        };
        Ok(Self {
            w0: take(64 * K),
            b0: take(64),
            w2: take(C * C * K),
            b2: take(C),
            w4: take(C * C * K),
            b4: take(C),
            w6: take(C * 2 * K),
            b6: take(2),
        })
    }
}

/// Output length of a Conv1d given input length.
#[inline]
pub(crate) fn conv_out_len(lin: usize, k: usize, stride: usize, pad: usize) -> usize {
    (lin + 2 * pad - k) / stride + 1
}

/// Output length of a ConvTranspose1d given input length.
#[inline]
pub(crate) fn convt_out_len(lin: usize, k: usize, stride: usize, pad: usize) -> usize {
    (lin - 1) * stride + k - 2 * pad
}

/// Direct Conv1d on one read. `input` is `[cin, lin]` row-major; returns
/// `[cout, lout]` row-major. Zero-padded by `pad`; ReLU applied if `relu`.
#[allow(clippy::too_many_arguments)] // mirrors the conv signature; clearer than a struct
#[allow(clippy::needless_range_loop)] // k indexes weight and input together
fn conv1d(
    input: &[f32],
    cin: usize,
    lin: usize,
    w: &[f32],
    b: &[f32],
    cout: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) -> (Vec<f32>, usize) {
    let lout = conv_out_len(lin, K, stride, pad);
    let mut out = vec![0.0f32; cout * lout];
    for co in 0..cout {
        for o in 0..lout {
            let mut acc = b[co];
            let base = (o * stride) as isize - pad as isize;
            for ci in 0..cin {
                let wrow = &w[(co * cin + ci) * K..(co * cin + ci) * K + K];
                let irow = &input[ci * lin..ci * lin + lin];
                for k in 0..K {
                    let idx = base + k as isize;
                    if idx >= 0 && (idx as usize) < lin {
                        acc += irow[idx as usize] * wrow[k];
                    }
                }
            }
            out[co * lout + o] = if relu && acc < 0.0 { 0.0 } else { acc };
        }
    }
    (out, lout)
}

/// Direct ConvTranspose1d on one read (gather form). PyTorch weight layout is
/// `[cin, cout, k]`. `input` is `[cin, lin]`; returns `[cout, lout]`.
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d(
    input: &[f32],
    cin: usize,
    lin: usize,
    w: &[f32],
    b: &[f32],
    cout: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize) {
    let lout = convt_out_len(lin, K, stride, pad);
    let mut out = vec![0.0f32; cout * lout];
    for co in 0..cout {
        for o in 0..lout {
            let mut acc = b[co];
            // o = i*stride - pad + k  =>  i = (o + pad - k) / stride, integer & in range
            for k in 0..K {
                let num = o as isize + pad as isize - k as isize;
                if num < 0 || !(num as usize).is_multiple_of(stride) {
                    continue;
                }
                let i = (num as usize) / stride;
                if i >= lin {
                    continue;
                }
                for ci in 0..cin {
                    acc += input[ci * lin + i] * w[(ci * cout + co) * K + k];
                }
            }
            out[co * lout + o] = acc;
        }
    }
    (out, lout)
}

/// CPU BoundariesCNN. Mirrors [`crate::adapter_cnn::AdapterCnn`] but with our
/// own conv stack (the verified reference for the GPU port).
pub struct CnnCompute {
    weights: CnnWeights,
    config: AdapterCnnConfig,
}

impl CnnCompute {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CnnComputeError> {
        Self::load_with_config(path, AdapterCnnConfig::default())
    }

    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
    ) -> Result<Self, CnnComputeError> {
        Ok(Self {
            weights: CnnWeights::load(path)?,
            config,
        })
    }

    /// Run the conv stack on an already-prepared (pooled + normalized) read.
    /// Returns the `[2, lout]` score map row-major and `lout`.
    pub fn forward(&self, normalized: &[f32]) -> (Vec<f32>, usize) {
        let w = &self.weights;
        let lin = normalized.len();
        let (h0, l0) = conv1d(normalized, 1, lin, &w.w0, &w.b0, C, 3, 3, true);
        let (h2, _) = conv1d(&h0, C, l0, &w.w2, &w.b2, C, 1, 3, true);
        let (h4, l4) = conv1d(&h2, C, l0, &w.w4, &w.b4, C, 1, 3, true);
        conv_transpose1d(&h4, C, l4, &w.w6, &w.b6, 2, 3, 3)
    }

    /// Full detection on a calibrated-pA read — same contract as
    /// [`crate::adapter_cnn::AdapterCnn::detect_adapter_end`].
    pub fn detect_adapter_end(&self, signal_pa: &[f32]) -> Result<usize, CnnComputeError> {
        let cfg = self.config;
        if signal_pa.len() <= cfg.min_obs_adapter + cfg.downscale_factor {
            return Err(CnnComputeError::SignalTooShort {
                len: signal_pa.len(),
                required: cfg.min_obs_adapter + cfg.downscale_factor,
            });
        }
        let pooled = mean_pool(&signal_pa[cfg.min_obs_adapter..], cfg.downscale_factor);
        let normalized = median_mad_normalize(&pooled);
        let (scores, lout) = self.forward(&normalized);

        let search_end = ((cfg.max_obs_adapter.saturating_sub(cfg.min_obs_adapter))
            / cfg.downscale_factor)
            .min(lout);
        let mut best_idx = 0usize;
        let mut best = f32::NEG_INFINITY;
        // Channel 0 occupies rows [0, lout); argmax over the valid search range.
        for (k, &v) in scores.iter().take(search_end).enumerate() {
            if v > best {
                best = v;
                best_idx = k;
            }
        }
        Ok(if best_idx == 0 {
            0
        } else {
            best_idx * cfg.downscale_factor + cfg.min_obs_adapter
        })
    }
}

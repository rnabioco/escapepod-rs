//! CNN-based adapter-end detection, compatible with ADAPTed (`KleistLab/ADAPTed`).
//!
//! Only available with the `cnn-detect` feature. The caller supplies an
//! ONNX file at runtime (produced by
//! `scripts/export_adapter_cnn_to_onnx.py` from a local ADAPTed install).
//! The weights themselves ship under CC BY-NC 4.0; this crate redistributes
//! only the code that consumes them.
//!
//! Port of `adapted.detect.cnn.{prepare_data, cnn_predict, cnn_detect}`:
//!
//! * Slice raw calibrated signal from `[min_obs_adapter:]` (default 1000).
//! * Mean-pool by `downscale_factor` (default 10) — `efficient_average_pooling`
//!   with zero-padding when length isn't a multiple.
//! * Subtract median, divide by MAD. Replace any NaN with `-5.0`.
//! * Feed through the ONNX `BoundariesCNN` (3× Conv1d + 1× ConvTranspose1d).
//! * `adapter_end_pos = argmax(scores[0, 0, :search_len])` where
//!   `search_len = (max_obs_adapter - min_obs_adapter) / downscale_factor`.
//! * Return `adapter_end = adapter_end_pos * downscale_factor + min_obs_adapter`
//!   (or 0 if argmax landed at slot 0, matching ADAPTed).

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
use tract_onnx::prelude::*;

/// Parameters controlling the CNN detector. Defaults match ADAPTed's
/// `rna004_130bps@v0.2.4.toml` `[core]` block.
#[derive(Debug, Clone, Copy)]
pub struct AdapterCnnConfig {
    pub min_obs_adapter: usize,
    pub max_obs_adapter: usize,
    pub downscale_factor: usize,
}

impl Default for AdapterCnnConfig {
    fn default() -> Self {
        Self {
            min_obs_adapter: 1000,
            max_obs_adapter: 6500,
            downscale_factor: 10,
        }
    }
}

/// Errors from the CNN detector.
#[derive(Debug, Error)]
pub enum AdapterCnnError {
    #[error("failed to load ONNX model: {0}")]
    Load(String),
    #[error("failed to run inference: {0}")]
    Run(String),
    #[error("unexpected output shape (expected [1, 2, N], got {got:?})")]
    BadShape { got: Vec<usize> },
    #[error("signal too short ({len} samples, need at least {required})")]
    SignalTooShort { len: usize, required: usize },
}

type Plan = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

/// CNN adapter-end detector. Build once via [`AdapterCnn::load`] and reuse
/// across many reads; the ONNX plan is immutable so the handle is `Sync`.
pub struct AdapterCnn {
    plan: Arc<Plan>,
    config: AdapterCnnConfig,
}

impl AdapterCnn {
    /// Load an ADAPTed-exported ONNX model from disk with default config.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AdapterCnnError> {
        Self::load_with_config(path, AdapterCnnConfig::default())
    }

    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
    ) -> Result<Self, AdapterCnnError> {
        // Dynamic input shape: `[batch=1, channels=1, length=S]` where S depends
        // on the input signal. We re-compile per signal length cheaply by
        // leaving `length` symbolic in the ONNX graph and re-resolving for the
        // actual length at call time.
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            .into_optimized()
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            .into_runnable()
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?;
        Ok(Self {
            plan: Arc::new(model),
            config,
        })
    }

    pub fn config(&self) -> AdapterCnnConfig {
        self.config
    }

    /// Run adapter-end detection on a single calibrated-pA signal.
    ///
    /// Returns the adapter_end sample index (in the original, un-downscaled
    /// signal frame). Returns 0 when the CNN chose slot 0 of the score array,
    /// matching ADAPTed's semantics for "no adapter found".
    pub fn detect_adapter_end(&self, signal_pa: &[f32]) -> Result<usize, AdapterCnnError> {
        let cfg = self.config;
        if signal_pa.len() <= cfg.min_obs_adapter + cfg.downscale_factor {
            return Err(AdapterCnnError::SignalTooShort {
                len: signal_pa.len(),
                required: cfg.min_obs_adapter + cfg.downscale_factor,
            });
        }

        // 1. Slice + downscale (mean-pool) + median/MAD normalize.
        let tail = &signal_pa[cfg.min_obs_adapter..];
        let pooled = mean_pool(tail, cfg.downscale_factor);
        let normalized = median_mad_normalize(&pooled);

        // 2. Feed through the ONNX plan.
        let input = Tensor::from_shape(&[1, 1, normalized.len()], &normalized)
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let scores = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;

        let shape = scores.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 2 {
            return Err(AdapterCnnError::BadShape {
                got: shape.to_vec(),
            });
        }
        let length_out = shape[2];

        // 3. Argmax on adapter-end channel, restricted to the expected range.
        let search_end = ((cfg.max_obs_adapter.saturating_sub(cfg.min_obs_adapter))
            / cfg.downscale_factor)
            .min(length_out);
        let mut best_idx: usize = 0;
        let mut best_score = f32::NEG_INFINITY;
        for k in 0..search_end {
            let v = scores[[0, 0, k]];
            if v > best_score {
                best_score = v;
                best_idx = k;
            }
        }

        // 4. Scale back to the original signal frame. ADAPTed returns 0 when
        //    the argmax landed at the very first slot (treated as "no adapter").
        let adapter_end = if best_idx == 0 {
            0
        } else {
            best_idx * cfg.downscale_factor + cfg.min_obs_adapter
        };
        Ok(adapter_end)
    }
}

/// Non-overlapping mean pool. Last block is zero-padded to `factor` if the
/// input length isn't a multiple — matches ADAPTed's `efficient_average_pooling`
/// behaviour (extends with zeros, then averages).
pub(crate) fn mean_pool(signal: &[f32], factor: usize) -> Vec<f32> {
    debug_assert!(factor >= 1);
    let full = signal.len() / factor;
    let rem = signal.len() % factor;
    let n_out = full + if rem > 0 { 1 } else { 0 };
    let mut out = Vec::with_capacity(n_out);
    for b in 0..full {
        let start = b * factor;
        let sum: f32 = signal[start..start + factor].iter().sum();
        out.push(sum / factor as f32);
    }
    if rem > 0 {
        let start = full * factor;
        let sum: f32 = signal[start..].iter().sum();
        // Mean with zero-padded tail == sum / factor.
        out.push(sum / factor as f32);
    }
    out
}

/// In-place `(x - median) / MAD` where `MAD = median(|x - median|)`. Returns
/// a fresh vector. NaN-safe — any non-finite input is replaced with `-5.0`
/// (ADAPTed's `SCORE_EXCL`) after the transform.
pub(crate) fn median_mad_normalize(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    let med = median(signal);
    let mut dev: Vec<f32> = signal.iter().map(|&x| (x - med).abs()).collect();
    let mad = median(&dev);
    // Guard against degenerate MAD=0 (all-constant signal). 1.0 is arbitrary
    // but avoids producing inf/NaN; the CNN treats uniform signals as noise.
    let scale = if mad > 0.0 { mad } else { 1.0 };
    dev.clear();
    signal
        .iter()
        .map(|&x| {
            let v = (x - med) / scale;
            if v.is_finite() { v } else { -5.0 }
        })
        .collect()
}

/// Median via a copy-and-select. `O(n log n)` sort is fine for our sizes
/// (signal lengths of a few thousand post-downscale), and is NaN-tolerant
/// — NaNs sink to the end of the sort via `partial_cmp` + `Ordering::Equal`.
fn median(xs: &[f32]) -> f32 {
    let mut v: Vec<f32> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_exact_multiple() {
        let s: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let out = mean_pool(&s, 4);
        // Block means: [2.5, 6.5, 10.5]
        assert_eq!(out, vec![2.5, 6.5, 10.5]);
    }

    #[test]
    fn mean_pool_padded_tail() {
        let s: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 samples, factor 3
        let out = mean_pool(&s, 3);
        // [mean(1,2,3), mean(4,5,0)] = [2, 3]
        assert_eq!(out.len(), 2);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn normalize_constant_signal() {
        // Constant signal: median = 1.0, MAD = 0 (guarded to 1.0). Output = 0.
        let out = median_mad_normalize(&[1.0; 10]);
        for x in &out {
            assert!(x.abs() < 1e-6);
        }
    }

    #[test]
    fn normalize_nan_replaced() {
        let out = median_mad_normalize(&[1.0, 2.0, 3.0, f32::NAN, 5.0]);
        assert!(out.iter().all(|x| x.is_finite()));
        // NaN position got replaced with -5.0.
        assert_eq!(out[3], -5.0);
    }
}

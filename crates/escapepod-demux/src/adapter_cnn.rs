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
use tract_onnx::tract_core::model::TypedRunnableModel;

/// Parameters controlling the CNN detector. Defaults match ADAPTed's
/// `rna004_130bps@v0.2.4.toml` `[core]` block and escapepod-models' training
/// config.
#[derive(Debug, Clone, Copy)]
pub struct AdapterCnnConfig {
    pub min_obs_adapter: usize,
    pub max_obs_adapter: usize,
    pub downscale_factor: usize,
    /// End of the signal window the model was trained/normalized over
    /// (`signal[min_obs_adapter:max_obs_trace]`). Normalization statistics and
    /// the model input are taken from this bounded window — NOT the whole read.
    /// Critical for long reads (mRNA): normalizing over an entire multi-hundred-k
    /// transcript gives the CNN statistics it never saw in training. For reads
    /// shorter than this (tRNA) the window clamps to the read end, so bounding
    /// is a no-op. Matches escapepod-models `DataConfig.max_obs_trace`.
    pub max_obs_trace: usize,
}

impl Default for AdapterCnnConfig {
    fn default() -> Self {
        Self {
            min_obs_adapter: 1000,
            max_obs_adapter: 6500,
            downscale_factor: 10,
            max_obs_trace: 16000,
        }
    }
}

impl AdapterCnnConfig {
    /// Preprocess a raw signal into the model input (slice + mean-pool +
    /// median/MAD-normalize + truncate to the receptive-field-bounded cap), or
    /// `None` if the read is too short. Exposed so callers can prep many reads
    /// in parallel and then hand the prepped vectors to a batched detector
    /// (e.g. [`AdapterCnnGpu::detect_prepped`](crate::adapter_cnn_gpu::AdapterCnnGpu::detect_prepped)).
    pub fn prep(&self, signal_pa: &[f32]) -> Option<Vec<f32>> {
        prep_adapter_signal(signal_pa, self)
    }
}

/// Errors from the CNN detector.
#[derive(Debug, Clone, Error)]
pub enum AdapterCnnError {
    #[error("failed to load ONNX model: {0}")]
    Load(String),
    #[error("failed to run inference: {0}")]
    Run(String),
    #[error("unexpected output shape (expected [_, 2, N], got {got:?})")]
    BadShape { got: Vec<usize> },
    #[error("signal too short ({len} samples, need at least {required})")]
    SignalTooShort { len: usize, required: usize },
}

// tract 0.23 renamed the 3-parameter `SimplePlan<F, O, M>` to the 2-parameter
// `RunnableModel<F, O>` (aliased `TypedRunnableModel` for typed graphs), and
// `into_runnable()` now hands back an `Arc<_>` directly.
type Plan = TypedRunnableModel;

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
        let detector = Self {
            plan: model,
            config,
        };
        // Validate the `[_, 2, _]` output contract once at load, with a dummy
        // forward pass. A model with the wrong channel layout (e.g. the static
        // v1.0.0 adapter export) then fails *here* with a clear error, instead
        // of silently returning adapter_end=0 for every read downstream.
        detector.probe_output_contract()?;
        Ok(detector)
    }

    /// Run one dummy forward pass and assert the output is `[_, 2, _]`
    /// (per-position scores for the two boundary channels). Cheap; catches a
    /// mismatched ONNX model at load time rather than per-read.
    fn probe_output_contract(&self) -> Result<(), AdapterCnnError> {
        const PROBE_LEN: usize = 1024;
        let dummy = vec![0f32; PROBE_LEN];
        let input = Tensor::from_shape(&[1, 1, PROBE_LEN], &dummy)
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let scores = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let shape = scores.shape();
        if shape.len() != 3 || shape[1] != 2 {
            return Err(AdapterCnnError::BadShape {
                got: shape.to_vec(),
            });
        }
        Ok(())
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
        let normalized =
            prep_adapter_signal(signal_pa, &cfg).ok_or(AdapterCnnError::SignalTooShort {
                len: signal_pa.len(),
                required: cfg.min_obs_adapter + cfg.downscale_factor,
            })?;

        // Feed through the ONNX plan.
        let input = Tensor::from_shape(&[1, 1, normalized.len()], &normalized)
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let scores = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;

        let shape = scores.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 2 {
            return Err(AdapterCnnError::BadShape {
                got: shape.to_vec(),
            });
        }
        let length_out = shape[2];
        Ok(decode_adapter_end(
            &cfg,
            length_out,
            normalized.len(),
            |k| scores[[0, 0, k]],
        ))
    }

    /// Batched adapter-end detection over many signals in one forward pass.
    ///
    /// Returns one result per input signal, in the same order. Reads that are
    /// too short produce `Err(SignalTooShort)` and are excluded from the batch;
    /// a model-load/run/shape failure maps every otherwise-valid read to the
    /// corresponding `Err`.
    ///
    /// **Bit-exact with [`detect_adapter_end`]**: signals are grouped by exact
    /// prepped length and each group is run as an *unpadded* `[group, 1, len]`
    /// batch, so every read is fed at precisely its own length — identical to
    /// running it alone. (Cross-read zero-padding is *not* safe here: this
    /// model's conv boundary handling makes a padded short read score
    /// differently than when run solo.) Downscaling collapses ranges of sample
    /// counts onto one length, so groups stay sizable; the main use is the GPU
    /// backend, where a length group is one onnxruntime batch.
    pub fn detect_adapter_end_batch(
        &self,
        signals: &[&[f32]],
    ) -> Vec<Result<usize, AdapterCnnError>> {
        let cfg = self.config;
        let min_len = cfg.min_obs_adapter + cfg.downscale_factor;

        // 1. Prep each signal independently; record which are usable.
        let prepped: Vec<Option<Vec<f32>>> = signals
            .iter()
            .map(|&sig| prep_adapter_signal(sig, &cfg))
            .collect();

        let valid_idx: Vec<usize> = (0..signals.len())
            .filter(|&i| prepped[i].is_some())
            .collect();

        // Helper to build the all-error / too-short result vector.
        let short_err = |i: usize| AdapterCnnError::SignalTooShort {
            len: signals[i].len(),
            required: min_len,
        };
        if valid_idx.is_empty() {
            return (0..signals.len()).map(|i| Err(short_err(i))).collect();
        }

        // 2. Group by exact prepped length and run each group as an *unpadded*
        //    `[group, 1, len]` batch. Cross-read zero-padding is NOT safe for
        //    this model (its conv boundary handling makes a padded short read
        //    score differently than when run alone), so we never pad: every
        //    read is fed at exactly its own length, making batched results
        //    bit-identical to the per-read path. Downscaling collapses ranges
        //    of sample counts onto the same length, so groups stay sizable.
        let mut out: Vec<Result<usize, AdapterCnnError>> =
            (0..signals.len()).map(|i| Err(short_err(i))).collect();

        for (len, group) in group_by_len(&prepped, &valid_idx) {
            let run = (|| -> Result<Vec<usize>, AdapterCnnError> {
                let g = group.len();
                let data = pack_batch(&prepped, &group, len);
                let input = Tensor::from_shape(&[g, 1, len], &data)
                    .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
                let outputs = self
                    .plan
                    .run(tvec!(input.into()))
                    .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
                let scores = outputs[0]
                    .to_plain_array_view::<f32>()
                    .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
                let shape = scores.shape();
                if shape.len() != 3 || shape[0] != g || shape[1] != 2 {
                    return Err(AdapterCnnError::BadShape {
                        got: shape.to_vec(),
                    });
                }
                let length_out = shape[2];
                Ok((0..g)
                    .map(|row| decode_adapter_end(&cfg, length_out, len, |k| scores[[row, 0, k]]))
                    .collect())
            })();
            scatter_group(&mut out, &group, run);
        }
        out
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

/// Downscaled positions past the adapter search window that can still reach it
/// through the CNN's receptive field. The graph is local (`Conv`/`Add`/`Relu`),
/// so any model input beyond `search_window + this margin` cannot affect a
/// score *inside* the search window — feeding only that prefix is
/// output-preserving. Truncating to it also bounds conv work/memory for very
/// long reads AND collapses every read longer than the cap onto one common
/// length, which is what lets the GPU path batch them together. Default 256 is
/// comfortably above the `tcn_l4` receptive field (kernel 7 × dilations ≤ 8 ⇒
/// half-width ~90); override via `ESCAPEPOD_CNN_MARGIN` for a model with a
/// larger field (a too-small value silently shifts boundaries — validate with
/// the batch-parity test against a larger margin).
fn search_receptive_margin() -> usize {
    use std::sync::OnceLock;
    static MARGIN: OnceLock<usize> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        std::env::var("ESCAPEPOD_CNN_MARGIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256)
    })
}

/// Slice `[min_obs_adapter:]`, mean-pool by `downscale_factor`, then
/// median/MAD-normalize — the exact input the boundary CNN expects. Returns
/// `None` when the signal is too short to contain an adapter. Shared by the
/// CPU (tract) and GPU (onnxruntime) backends so their preprocessing is
/// bit-identical.
///
/// Normalization statistics are computed over the **full** tail (matching the
/// per-read reference exactly), then the model input is truncated to a prefix
/// covering the search window plus receptive-field margin — see
/// [`search_receptive_margin`]. This keeps results identical to feeding the
/// whole read while keeping batched conv tensors bounded.
pub(crate) fn prep_adapter_signal(signal_pa: &[f32], cfg: &AdapterCnnConfig) -> Option<Vec<f32>> {
    if signal_pa.len() <= cfg.min_obs_adapter + cfg.downscale_factor {
        return None;
    }
    // Bound the window to `[min_obs_adapter:max_obs_trace]` — the window the
    // model was normalized/trained over. Clamps to the read end for short reads
    // (tRNA: a no-op vs the whole tail), and stops at max_obs_trace for long
    // reads (mRNA: avoids normalizing over the entire transcript). Only this
    // prefix is ever needed, so callers may decode just up to `max_obs_trace`.
    let end = cfg
        .max_obs_trace
        .max(cfg.min_obs_adapter)
        .min(signal_pa.len());
    let tail = &signal_pa[cfg.min_obs_adapter..end];
    let mut normalized = median_mad_normalize(&mean_pool(tail, cfg.downscale_factor));
    let search_window =
        cfg.max_obs_adapter.saturating_sub(cfg.min_obs_adapter) / cfg.downscale_factor;
    let cap = search_window + search_receptive_margin();
    if normalized.len() > cap {
        normalized.truncate(cap);
    }
    Some(normalized)
}

/// Decode one read's adapter-end channel into a sample index in the original
/// (un-downscaled) signal frame. `score_at(k)` returns the channel-0 score at
/// downscaled position `k`; `length_out` is the model output length and
/// `valid_len` the read's own un-padded downscaled length (they're equal for
/// the single-read path; for a padded batch `valid_len <= length_out`, so the
/// clamp keeps the argmax out of the zero-padded tail). Argmax over the
/// expected adapter window; 0 when the peak is slot 0 ("no adapter"), matching
/// ADAPTed. Shared by both backends so decoding is bit-identical.
pub(crate) fn decode_adapter_end(
    cfg: &AdapterCnnConfig,
    length_out: usize,
    valid_len: usize,
    score_at: impl Fn(usize) -> f32,
) -> usize {
    let search_end = (cfg.max_obs_adapter.saturating_sub(cfg.min_obs_adapter)
        / cfg.downscale_factor)
        .min(length_out)
        .min(valid_len);
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for k in 0..search_end {
        let v = score_at(k);
        if v > best {
            best = v;
            best_idx = k;
        }
    }
    if best_idx == 0 {
        0
    } else {
        best_idx * cfg.downscale_factor + cfg.min_obs_adapter
    }
}

/// Group valid (non-`None`) prepped signals by their exact length, so each
/// group can be run as an unpadded batch. Returns `(len, original-indices)`.
/// Shared by the CPU (tract) and GPU (onnxruntime) batch paths.
pub(crate) fn group_by_len(
    prepped: &[Option<Vec<f32>>],
    valid_idx: &[usize],
) -> Vec<(usize, Vec<usize>)> {
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for &i in valid_idx {
        let len = prepped[i]
            .as_ref()
            .expect("valid_idx points at a prepped signal")
            .len();
        groups.entry(len).or_default().push(i);
    }
    groups.into_iter().collect()
}

/// Pack a set of same-length prepped signals into a row-major `[g, 1, len]` f32
/// batch buffer, gathered from `prepped` at `indices`. No padding — every row is
/// exactly `len`. Shared by the CPU (tract) and GPU (onnxruntime) batch paths so
/// the batch layout stays byte-identical between backends.
pub(crate) fn pack_batch(prepped: &[Option<Vec<f32>>], indices: &[usize], len: usize) -> Vec<f32> {
    let mut data = vec![0f32; indices.len() * len];
    for (row, &i) in indices.iter().enumerate() {
        data[row * len..(row + 1) * len].copy_from_slice(prepped[i].as_ref().unwrap());
    }
    data
}

/// Scatter a group/sub-batch's inference `result` into `out` at each read's
/// original index: `Ok` writes each read's decoded adapter-end in order; `Err`
/// clones the error to every read in the group. Shared by the CPU and GPU batch
/// paths (which resolve `indices` per whole-group and per sub-batch respectively).
pub(crate) fn scatter_group(
    out: &mut [Result<usize, AdapterCnnError>],
    indices: &[usize],
    result: Result<Vec<usize>, AdapterCnnError>,
) {
    match result {
        Ok(ends) => {
            for (&i, end) in indices.iter().zip(ends) {
                out[i] = Ok(end);
            }
        }
        Err(e) => {
            for &i in indices {
                out[i] = Err(e.clone());
            }
        }
    }
}

/// Median of the finite values. The finite-only filter matters here (NaN/inf
/// must not skew the adapter MAD); the shared O(n) helper returns 0.0 for the
/// resulting empty slice, matching the previous behavior.
fn median(xs: &[f32]) -> f32 {
    let mut v: Vec<f32> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    escapepod_signal::stats::median_via_select(&mut v)
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

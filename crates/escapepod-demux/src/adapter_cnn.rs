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
//! * Slice raw calibrated signal to `[min_obs_adapter:max_obs_trace]`
//!   (defaults 1000 and 16000).
//! * Mean-pool by `downscale_factor` (default 10) — `efficient_average_pooling`
//!   with zero-padding when length isn't a multiple.
//! * Subtract median, divide by MAD. Replace any NaN with `-5.0`.
//! * Right-pad to the fixed `input_len` (1500) with `SCORE_EXCL` = `-5.0`, the
//!   convention every training example was built with (#187). Normalisation
//!   statistics come from the real window, before padding — training's order.
//! * Feed through the ONNX graph (`[B,1,L] -> [B,2,L]`; ADAPTed's
//!   `BoundariesCNN` is 3× Conv1d + 1× ConvTranspose1d, escapepod-models'
//!   `adapter_rna004` is a 1-D U-Net — this module is agnostic to which).
//! * `adapter_end_pos = argmax(scores[0, 0, :search_len])` where
//!   `search_len = (max_obs_adapter - min_obs_adapter) / downscale_factor`.
//! * Return `adapter_end = adapter_end_pos * downscale_factor + min_obs_adapter`
//!   (or 0 if argmax landed at slot 0, matching ADAPTed).

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::framework::Framework;
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
    /// Value the window is right-padded to [`input_len`](Self::input_len) with,
    /// and the NaN substitute during normalisation ([`SCORE_EXCL`] unless a
    /// bundle declares otherwise). Padding with anything else — zero included —
    /// feeds the model a pattern it never saw.
    pub pad_value: f32,
}

impl Default for AdapterCnnConfig {
    fn default() -> Self {
        Self {
            min_obs_adapter: 1000,
            max_obs_adapter: 6500,
            downscale_factor: 10,
            max_obs_trace: 16000,
            pad_value: SCORE_EXCL,
        }
    }
}

impl AdapterCnnConfig {
    /// The model's fixed input length, in downscaled samples.
    ///
    /// Mirrors escapepod-models `DataConfig.input_len`
    /// (`(max_obs_trace - min_obs_adapter) // downscale_factor` = 1500), which is
    /// the length every training example was padded to. Derived rather than
    /// stored so it cannot drift from the three fields it depends on — though
    /// none of this is declared in the model manifest today, which is the
    /// underlying problem (#187).
    pub fn input_len(&self) -> usize {
        self.max_obs_trace.saturating_sub(self.min_obs_adapter) / self.downscale_factor.max(1)
    }

    /// Preprocess a raw signal into the model input (slice + mean-pool +
    /// median/MAD-normalize + pad to [`input_len`](Self::input_len) with
    /// `SCORE_EXCL`), or `None` if the read is too short. Exposed so callers can
    /// prep many reads
    /// in parallel and then hand the prepped vectors to a batched detector
    /// (e.g. [`AdapterCnnGpu::detect_prepped`](crate::adapter_cnn_gpu::AdapterCnnGpu::detect_prepped)).
    pub fn prep(&self, signal_pa: &[f32]) -> Option<PreppedWindow> {
        prep_adapter_signal(signal_pa, self)
    }

    /// Build from the contract a bundle declares (`boundary.input`), keeping
    /// the decode-policy fields the contract deliberately omits
    /// (`max_obs_adapter`, the argmax cap) at their defaults.
    ///
    /// Refuses a self-inconsistent block: the sidecar declares `input_len`
    /// redundantly so that producer and consumer computing different lengths is
    /// a load error here, not a silent preference for one of them.
    pub fn from_bundle_input(
        spec: &crate::crf::BoundaryInputSpec,
    ) -> Result<Self, AdapterCnnError> {
        let cfg = Self {
            min_obs_adapter: spec.min_obs_adapter,
            max_obs_trace: spec.max_obs_trace,
            downscale_factor: spec.downscale_factor.max(1),
            pad_value: spec.pad_value,
            ..Self::default()
        };
        if cfg.input_len() != spec.input_len {
            return Err(AdapterCnnError::ContractMismatch {
                declared: spec.input_len,
                derived: cfg.input_len(),
            });
        }
        Ok(cfg)
    }
}

/// One read's model input: the fixed-length tensor row, and how much of it is
/// real.
///
/// The two travel together because keeping them apart is what broke: #187 padded
/// every read to `input_len`, and the decode's clamp — which took a separate
/// `valid_len` argument — was then handed the *padded* length by all three call
/// sites. That silently widened the argmax over up to 540 positions of pure
/// `SCORE_EXCL` padding, and on 503 k real reads it put `adapter_end` past the
/// end of the read for **272 of them** (0.054%, all reads under 6500 samples;
/// 0.7.0 produced none). Making the length a field of the data removes the
/// opportunity to pass the wrong one.
#[derive(Debug, Clone, PartialEq)]
pub struct PreppedWindow {
    /// Model input row, always [`AdapterCnnConfig::input_len`] long.
    pub data: Vec<f32>,
    /// Leading positions of `data` that came from real signal, before padding.
    /// Never greater than `data.len()`.
    pub valid_len: usize,
}

impl PreppedWindow {
    /// Length of the tensor row handed to the graph.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
    #[error(
        "boundary input contract is self-inconsistent: declares input_len {declared} but \
         (max_obs_trace - min_obs_adapter) / downscale_factor = {derived}; refusing to guess \
         which one the model was trained with"
    )]
    ContractMismatch { declared: usize, derived: usize },
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
        // Since #187 every input is the one shape `[1, 1, input_len]`: prep
        // pads each read to `input_len`, and CPU inference is per read. Pin it,
        // and hoist the convolutions' padding out of the graph first.
        //
        // Both are load-time rewrites that tract then plans around, and each
        // was measured on its own (adapter_rna004, 119k reads, 32 threads):
        // the graph used to be optimized with `length` left symbolic — a
        // comment here said tract "re-resolves for the actual length at call
        // time", which stopped being a reason once the length stopped
        // varying — and pinning it alone took the run from 600 to 401
        // CPU-seconds. Hoisting the padding (nine zero-padded 1-D convs,
        // dilations 1-8) took it to 380: tract's im2col leaves its block-copy
        // path when `pads != 0`, the defect `onnx_rewrite::hoist_conv_padding`
        // documents on the charging network. Sorted classifications were
        // identical across all three loaders.
        let onnx = tract_onnx::onnx();
        let mut proto = onnx
            .proto_model_for_path(path)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?;
        crate::onnx_rewrite::hoist_conv_padding(&mut proto, 1);
        let model = onnx
            .model_for_proto_model(&proto)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            .with_input_fact(0, f32::fact([1, 1, config.input_len()]).into())
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
    /// mismatched ONNX model at load time rather than per-read. Probes at the
    /// pinned input length, the only shape the plan accepts.
    fn probe_output_contract(&self) -> Result<(), AdapterCnnError> {
        let probe_len = self.config.input_len();
        let dummy = vec![0f32; probe_len];
        let input = Tensor::from_shape(&[1, 1, probe_len], &dummy)
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
        self.run_window(&normalized)
    }

    /// One prepped window through the plan: the `[1, 1, input_len]` forward
    /// pass and the argmax decode.
    fn run_window(&self, window: &PreppedWindow) -> Result<usize, AdapterCnnError> {
        let cfg = self.config;
        let input = Tensor::from_shape(&[1, 1, window.len()], &window.data)
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
        Ok(decode_adapter_end(&cfg, length_out, window.valid_len, |k| {
            scores[[0, 0, k]]
        }))
    }

    /// Adapter-end detection over many signals, one result per input in the
    /// same order. Reads that are too short produce `Err(SignalTooShort)`.
    ///
    /// **Bit-exact with [`detect_adapter_end`]** by construction: each read runs
    /// through the same `[1, 1, input_len]` plan on its own. The plan is pinned
    /// to batch 1 at load — tract has no efficient batched convolution, and
    /// batching it here measured *slower* than per-read parallelism across
    /// reads — so there is no CPU batch to assemble. The batched call is the
    /// GPU backend's, where a block of prepped rows is one onnxruntime call
    /// ([`AdapterCnnGpu::detect_prepped`](crate::adapter_cnn_gpu::AdapterCnnGpu::detect_prepped));
    /// this method keeps the two backends' signatures aligned and is what
    /// `tests/adapter_cnn_batch_parity.rs` pins.
    pub fn detect_adapter_end_batch(
        &self,
        signals: &[&[f32]],
    ) -> Vec<Result<usize, AdapterCnnError>> {
        let cfg = self.config;
        let min_len = cfg.min_obs_adapter + cfg.downscale_factor;
        signals
            .iter()
            .map(|&sig| {
                let window =
                    prep_adapter_signal(sig, &cfg).ok_or(AdapterCnnError::SignalTooShort {
                        len: sig.len(),
                        required: min_len,
                    })?;
                self.run_window(&window)
            })
            .collect()
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
/// a fresh vector. NaN-safe — any non-finite input is replaced with `excl`
/// (the config's pad value; training's `np.nan_to_num(..., nan=cfg.score_excl)`)
/// after the transform.
pub(crate) fn median_mad_normalize(signal: &[f32], excl: f32) -> Vec<f32> {
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
            if v.is_finite() { v } else { excl }
        })
        .collect()
}

/// Value the training pipeline pads short windows with, and the same value it
/// substitutes for NaN (`adapted.detect.cnn.SCORE_EXCL`, escapepod-models
/// `config.py:17`). Padding with anything else — zero included — feeds the model
/// a pattern it never saw.
pub(crate) const SCORE_EXCL: f32 = -5.0;

/// Slice `[min_obs_adapter:max_obs_trace]`, mean-pool by `downscale_factor`,
/// median/MAD-normalize, then pad to the model's fixed input length with
/// [`SCORE_EXCL`] — the exact input the boundary CNN was trained on. Returns
/// `None` when the signal is too short to contain an adapter. Shared by the CPU
/// (tract) and GPU (onnxruntime) backends so their preprocessing is identical.
///
/// # Why fixed-length, and why this pad value (#187)
///
/// escapepod-models builds *every* training example through
/// `dataset.py::prepare_signal`, which ends:
///
/// ```python
/// out = np.full(cfg.input_len, cfg.score_excl, dtype=np.float32)
/// out[:min(len(norm), cfg.input_len)] = norm[:...]
/// ```
///
/// — a fixed `input_len = (max_obs_trace - min_obs_adapter) / downscale` = 1500,
/// right-padded with `score_excl = -5.0`. Its own inference path
/// (`barcode.py`) prepares inputs the same way.
///
/// This used to truncate to `search_window + ESCAPEPOD_CNN_MARGIN` (550 + 256 =
/// 806) instead, on the argument that a local graph cannot reach past it. Two
/// things were wrong with that. The receptive field is larger than the margin
/// assumed — `Conv k=7 s=5` then `k=7` at dilations 1,1,2,2,4,4,8,8 gives
/// `((1 + 6+6+12+12+24+24+48+48) - 1) * 5 + 7 = 907`, so half-width ~453, and an
/// output at the far edge of the search window depended on input the truncation
/// removed. And the model had never seen a 806-long input at all.
///
/// It was also expensive. Because the window clamps to the read end, every read
/// shorter than `max_obs_trace` got its own length: over a production run's 527k
/// reads, 680 distinct lengths, so a 65,536-read block issued one large call plus
/// ~679 of ~30 rows, each a new shape paying fresh cuDNN plan selection.
/// Detection measured 401 s of device time in a 425 s wall — 94% — against
/// 0.009 ms/read for the same kernel at a steady shape.
///
/// One fixed length fixes both: it is what the model was trained on, and it is
/// one shape for every read.
pub(crate) fn prep_adapter_signal(
    signal_pa: &[f32],
    cfg: &AdapterCnnConfig,
) -> Option<PreppedWindow> {
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
    let mut normalized =
        median_mad_normalize(&mean_pool(tail, cfg.downscale_factor), cfg.pad_value);
    // How much of the row is real, captured *before* padding. Training pads the
    // same way, so the model is in-distribution either way — but the decode must
    // not argmax into the pad, or a short read gets a boundary past its own end.
    let valid_len = normalized.len().min(cfg.input_len());
    // Normalisation statistics come from the true window above; only the tensor
    // handed to the graph is padded, which is the order training uses too.
    normalized.resize(cfg.input_len(), cfg.pad_value);
    Some(PreppedWindow {
        data: normalized,
        valid_len,
    })
}

/// Decode one read's adapter-end channel into a sample index in the original
/// (un-downscaled) signal frame. `score_at(k)` returns the channel-0 score at
/// downscaled position `k`; `length_out` is the model output length and
/// `valid_len` the read's own un-padded downscaled length. Argmax over the
/// expected adapter window; 0 when the peak is slot 0 ("no adapter"), matching
/// ADAPTed. Shared by both backends so decoding is bit-identical.
///
/// # Two deliberate departures from escapepod-models' own decode
///
/// `predict.py::decode_scores` argmaxes the model's full output and has no
/// "slot 0 means no adapter" rule; both extras here are ADAPTed's rna004
/// conventions, inherited because this decode is architecture-agnostic.
///
/// The search cap is the one with a measurable cost: `max_obs_adapter` = 6500
/// makes `search_end` 550 of the 1500 positions the model emits, so any adapter
/// the model would place past raw sample 6500 is unreachable. On
/// escapepod-models' move-table gold set that is **84 of 20,303 reads (0.41%)**
/// — p99 is 5706, so the prior holds for the bulk of the distribution, but it is
/// a real ceiling and not an artefact of the model.
///
/// `valid_len` is now always equal to the padded length (prep emits one fixed
/// length), so the clamp is inert; it is kept for the same reason
/// [`group_by_len`] is.
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
///
/// Since #187 [`prep_adapter_signal`] emits exactly `cfg.input_len()` for every
/// read, so in practice this always returns one group. It is kept rather than
/// replaced by "one batch of everything" because the alternative fails
/// *silently* if that invariant ever breaks: [`pack_batch`] would truncate or
/// zero-pad the mismatched rows and the reads would come back with plausible,
/// wrong boundaries. The cost is one `HashMap` pass per block, far below the
/// inference it guards.
pub(crate) fn group_by_len(
    prepped: &[Option<PreppedWindow>],
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

/// Pack prepped signals into a row-major `[g, 1, len]` f32 batch buffer,
/// gathered from `prepped` at `indices`.
///
/// Every row is `cfg.input_len()` long since #187, so the `min` below is not
/// reached and no row is zero-filled. It stays defensive rather than asserting
/// because a short row zero-filled at the tail is at least positionally sound —
/// [`prep_adapter_signal`]'s window always starts at `min_obs_adapter` and
/// [`decode_adapter_end`] searches forward from index 0, so padding the tail
/// leaves every boundary index where it was.
///
/// Note that zero is *not* the model's pad value ([`SCORE_EXCL`] is), which is
/// exactly why this must not become the padding path: `cnn_pad_probe` measured
/// this graph's output changing well before the padding starts. Prep pads to the
/// full length itself, with the right value.
///
/// Shared by the CPU (tract) and GPU (onnxruntime) batch paths so the layout
/// stays byte-identical between backends.
pub(crate) fn pack_batch(
    prepped: &[Option<PreppedWindow>],
    indices: &[usize],
    len: usize,
) -> Vec<f32> {
    let mut data = vec![0f32; indices.len() * len];
    for (row, &i) in indices.iter().enumerate() {
        let src = &prepped[i]
            .as_ref()
            .expect("indices point at prepped signals")
            .data;
        let n = src.len().min(len);
        data[row * len..row * len + n].copy_from_slice(&src[..n]);
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
        let out = median_mad_normalize(&[1.0; 10], SCORE_EXCL);
        for x in &out {
            assert!(x.abs() < 1e-6);
        }
    }

    #[test]
    fn normalize_nan_replaced() {
        let out = median_mad_normalize(&[1.0, 2.0, 3.0, f32::NAN, 5.0], SCORE_EXCL);
        assert!(out.iter().all(|x| x.is_finite()));
        // NaN position got replaced with -5.0.
        assert_eq!(out[3], -5.0);
    }

    /// A short read pads to `input_len`, but its real content does not, and the
    /// two lengths must not be confused.
    #[test]
    fn prep_reports_the_pre_padding_length() {
        let cfg = AdapterCnnConfig::default();
        let sig = vec![100.0f32; 2120];
        let w = prep_adapter_signal(&sig, &cfg).expect("2120 samples is long enough");

        // The tensor is always the model's fixed width...
        assert_eq!(w.data.len(), cfg.input_len());
        assert_eq!(w.data.len(), 1500);
        // ...but only (2120 - 1000) / 10 = 112 positions came from signal.
        assert_eq!(w.valid_len, 112);
        // Everything past that is the training pad value, not zero.
        assert!(w.data[w.valid_len..].iter().all(|&v| v == SCORE_EXCL));
    }

    /// The regression this guards is a shipped one, not a hypothetical.
    ///
    /// #187 padded every read to `input_len`, and all three `decode_adapter_end`
    /// call sites then passed the *padded* length as `valid_len`. `search_end`
    /// widened from the read's own 112 positions to the full 550, so the argmax
    /// could land in `SCORE_EXCL` padding and return a boundary past the end of
    /// the read. On 503,076 production reads that produced `adapter_end >
    /// num_samples` for **272** of them; 0.7.0 produced none.
    #[test]
    fn decode_never_returns_a_boundary_past_the_read() {
        let cfg = AdapterCnnConfig::default();
        let raw_len = 2120usize;
        let sig = vec![100.0f32; raw_len];
        let w = prep_adapter_signal(&sig, &cfg).expect("long enough");

        // Worst case: the model's strongest response is out in the padding,
        // exactly where the old code was free to follow it.
        let end = decode_adapter_end(&cfg, cfg.input_len(), w.valid_len, |k| {
            if k == 400 { 100.0 } else { 0.0 }
        });
        assert!(
            end <= raw_len,
            "adapter_end {end} is past the {raw_len}-sample read"
        );

        // And a peak inside the real region is still found, so the clamp did not
        // simply disable detection.
        let end = decode_adapter_end(&cfg, cfg.input_len(), w.valid_len, |k| {
            if k == 50 { 100.0 } else { 0.0 }
        });
        assert_eq!(end, 50 * cfg.downscale_factor + cfg.min_obs_adapter);
        assert!(end <= raw_len);
    }

    mod bundle_contract {
        use super::*;
        use crate::crf::BoundaryInputSpec;

        /// What escapepod-models ships for adapter_rna004 (its
        /// `DataConfig().input_contract()`).
        fn rna004_spec() -> BoundaryInputSpec {
            BoundaryInputSpec {
                min_obs_adapter: 1000,
                max_obs_trace: 16000,
                downscale_factor: 10,
                input_len: 1500,
                pad_value: -5.0,
            }
        }

        /// The shipped contract must reproduce the config the runtime assumed
        /// all along — if this breaks, either the fallback defaults or the
        /// producer changed, and the two sides no longer agree on the model.
        #[test]
        fn shipped_contract_reproduces_the_legacy_defaults() {
            let cfg = AdapterCnnConfig::from_bundle_input(&rna004_spec()).expect("consistent");
            let dflt = AdapterCnnConfig::default();
            assert_eq!(cfg.min_obs_adapter, dflt.min_obs_adapter);
            assert_eq!(cfg.max_obs_trace, dflt.max_obs_trace);
            assert_eq!(cfg.downscale_factor, dflt.downscale_factor);
            assert_eq!(cfg.pad_value, dflt.pad_value);
            assert_eq!(cfg.input_len(), 1500);
            // Decode policy is not the contract's to set.
            assert_eq!(cfg.max_obs_adapter, dflt.max_obs_adapter);
        }

        #[test]
        fn inconsistent_input_len_is_refused() {
            // 806 is the truncated length #187's bug fed the model.
            let spec = BoundaryInputSpec {
                input_len: 806,
                ..rna004_spec()
            };
            match AdapterCnnConfig::from_bundle_input(&spec) {
                Err(AdapterCnnError::ContractMismatch { declared, derived }) => {
                    assert_eq!(declared, 806);
                    assert_eq!(derived, 1500);
                }
                other => panic!("expected ContractMismatch, got {other:?}"),
            }
        }

        /// A declared pad value must reach both the padding and the NaN
        /// substitution — a contract that only half-applies is worse than none.
        #[test]
        fn declared_pad_value_reaches_the_tensor() {
            let spec = BoundaryInputSpec {
                pad_value: -7.0,
                ..rna004_spec()
            };
            let cfg = AdapterCnnConfig::from_bundle_input(&spec).expect("consistent");
            let mut sig = vec![100.0f32; 2120];
            sig[1500] = f32::NAN;
            let w = prep_adapter_signal(&sig, &cfg).expect("long enough");
            assert!(w.data[w.valid_len..].iter().all(|&v| v == -7.0));
            assert_eq!(
                w.data[(1500 - cfg.min_obs_adapter) / cfg.downscale_factor],
                -7.0
            );
        }
    }
}

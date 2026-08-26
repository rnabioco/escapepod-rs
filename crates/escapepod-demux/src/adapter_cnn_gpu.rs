//! GPU boundary-CNN inference via onnxruntime (CUDA execution provider).
//!
//! Opt-in via the `gpu` feature. Mirrors [`AdapterCnn`](crate::AdapterCnn)
//! but runs the ONNX graph through onnxruntime with the CUDA execution
//! provider, in batches. Preprocessing (`prep_adapter_signal`) and decoding
//! (`decode_adapter_end`) are the *same shared helpers* the CPU tract path
//! uses, so results match bit-for-bit modulo float reassociation across runtimes
//! (well below the argmax granularity in practice; a parity test guards it).
//!
//! `load-dynamic`: onnxruntime is dlopened at runtime rather than linked at
//! build time. Point `ORT_DYLIB_PATH` at a CUDA-enabled `libonnxruntime.so`
//! and ensure a CUDA device + cuDNN are visible. If the CUDA EP cannot be
//! initialized, onnxruntime falls back to CPU — which would be slow but
//! correct.

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use crate::adapter_cnn::{
    PreppedWindow, decode_adapter_end, group_by_len, pack_batch, prep_adapter_signal, scatter_group,
};
use crate::{AdapterCnnConfig, AdapterCnnError};

/// Resolve the starting cap on input elements (`rows × len`) per onnxruntime
/// call, scaled to the device's memory. Since #187 prep emits one fixed length
/// for every read, so this splits the single group — which is every valid read
/// in the block — into chunks of this size; a chunk that still OOMs is halved
/// and retried (`run_grouped`), so this is a *starting* guess, not a hard limit.
///
/// Resolution order: `ESCAPEPOD_CNN_GPU_BATCH_ELEMS` env override → scaled from
/// total VRAM (`total_bytes / BYTES_PER_ELEM`) → a fixed fallback. Conv
/// activations scale with `rows × len × channels`, so on a 24 GB device ~5k
/// rows at the then-806 prep length (~4.2M elems) fit but ~10k OOM (measured)
/// — i.e. ~24 GB / 5500 bytes-per-element. The elements-not-rows unit is what
/// makes that measurement still apply at today's 1500: the same element budget
/// simply yields proportionally fewer rows. Using total VRAM means an 80 GB A100/H100
/// gets ~3× larger batches automatically, while the halve-retry covers any
/// over-estimate (e.g. a model with more channels).
fn resolve_batch_elems() -> usize {
    /// Empirical peak device bytes per input element (`rows × len`) at the OOM
    /// boundary for the rna004 TCN — folds in channel count and the number of
    /// live conv activations, with headroom.
    const BYTES_PER_ELEM: usize = 5500;
    const FALLBACK: usize = 4_194_304;

    if let Some(n) = std::env::var("ESCAPEPOD_CNN_GPU_BATCH_ELEMS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    match cuda_total_mem_bytes() {
        Some(total) => (total / BYTES_PER_ELEM).clamp(2_000_000, 64_000_000),
        None => FALLBACK,
    }
}

/// Total memory (bytes) of the CUDA device the CUDA EP will use (ordinal 0,
/// after `CUDA_VISIBLE_DEVICES`). `None` if the driver/device can't be queried.
fn cuda_total_mem_bytes() -> Option<usize> {
    use cudarc::driver::result;
    // SAFETY: these are read-only CUDA driver queries. `cuInit` is idempotent
    // (ort also initializes the driver) and device-property queries need no
    // context; any failure just yields `None` (we fall back to a fixed cap).
    unsafe {
        result::init().ok()?;
        let device = result::device::get(0).ok()?;
        result::device::total_mem(device).ok()
    }
}

/// Batched boundary-CNN adapter-end detector backed by onnxruntime + CUDA.
///
/// `ort::Session::run` takes `&mut self`, so the session sits behind a `Mutex`:
/// callers share `&AdapterCnnGpu` across rayon workers (parallel decode/prep),
/// and the actual GPU `run` calls serialize on the lock — which is what we want
/// anyway, since there's one device.
pub struct AdapterCnnGpu {
    session: Mutex<Session>,
    config: AdapterCnnConfig,
    /// Starting per-call input-element cap, scaled to this device's VRAM at load.
    batch_elems: usize,
}

impl AdapterCnnGpu {
    /// Load an ONNX model with the default (ADAPTed/rna004) preprocessing config.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AdapterCnnError> {
        Self::load_with_config(path, AdapterCnnConfig::default())
    }

    /// Load with an explicit preprocessing config, registering the CUDA EP.
    ///
    /// Thread counts are *not* a parameter: this type is always a CUDA session,
    /// and a CUDA session wants no intra-op pool at all (see
    /// [`load_with_config_on_device`](Self::load_with_config_on_device)). Nor do
    /// they belong in [`AdapterCnnConfig`] — that struct is `Copy` and carries
    /// preprocessing semantics the CPU/GPU parity tests compare, so a scheduling
    /// knob has no business in it either.
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
    ) -> Result<Self, AdapterCnnError> {
        Self::load_with_config_on_device(path, config, 0)
    }

    /// [`load_with_config`](Self::load_with_config) pinned to a CUDA ordinal.
    ///
    /// Detection and the CRF encoder are comparable in device cost (measured 5.4 s
    /// vs 6.4 s over 40 k reads), so on one GPU they contend and the pipeline is
    /// bound by their sum. Given more than one device they can be given separate
    /// roles instead, making the ceiling the *larger* of the two rather than the
    /// total. This is what lets the caller place them.
    pub fn load_with_config_on_device(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
        device: i32,
    ) -> Result<Self, AdapterCnnError> {
        let builder = Session::builder()
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            // The graph runs on the device, so onnxruntime's intra-op pool has
            // nothing to compute — but left at its default it is
            // `available_parallelism()` threads wide, spawned *on top of*
            // rayon's, and it does not sit idle: profiling `demux detect
            // --method cnn --gpu` on 150 k reads, 15 pool threads accounted for
            // ~35% of all CPU samples in the process, next to 4% for the
            // preprocessing they were starving.
            //
            // Both settings are needed and neither works alone (measured warm,
            // three interleaved reps): the default 16 threads spinning ran
            // 7.34 s, 1 thread still spinning 7.42 s, 16 threads without
            // spinning 7.37 s, and **1 thread without spinning 6.93 s** at 274%
            // CPU against 340%. One thread is what removes the per-op fan-out
            // and join; disabling the spin is what stops that one thread
            // burning a core between calls. Output is bit-identical across all
            // four (150,001 of 150,001 boundaries), as are the fused pipeline's
            // classifications.
            //
            // A graph the CUDA EP cannot place entirely on the device would run
            // its leftovers single-threaded here. That is the right trade for
            // this path — reads are the parallel axis, and the caller already
            // has every core busy on decode and prep.
            .with_intra_threads(1)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            // Only meaningful under parallel execution mode, which we don't
            // enable; set it anyway so the bound holds if that ever changes.
            .with_inter_threads(1)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            .with_intra_op_spinning(false)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?;
        let session = builder
            .with_execution_providers(crate::ort_ep::cuda_providers_on(device))
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
            config,
            batch_elems: resolve_batch_elems(),
        })
    }

    /// Preprocessing config in effect.
    pub fn config(&self) -> AdapterCnnConfig {
        self.config
    }

    /// Batched adapter-end detection from raw signals. Preps each signal then
    /// delegates to [`Self::detect_prepped`]. Same bit-exact length-grouping as
    /// [`AdapterCnn::detect_adapter_end_batch`](crate::AdapterCnn::detect_adapter_end_batch).
    pub fn detect_adapter_end_batch(
        &self,
        signals: &[&[f32]],
    ) -> Vec<Result<usize, AdapterCnnError>> {
        let cfg = self.config;
        let prepped: Vec<Option<PreppedWindow>> = signals
            .iter()
            .map(|&s| prep_adapter_signal(s, &cfg))
            .collect();
        // Re-stamp too-short errors with the real input length (detect_prepped
        // only sees `None`, not the original signal).
        let mut out = self.detect_prepped(&prepped);
        for (i, r) in out.iter_mut().enumerate() {
            if matches!(r, Err(AdapterCnnError::SignalTooShort { .. })) {
                *r = Err(AdapterCnnError::SignalTooShort {
                    len: signals[i].len(),
                    required: cfg.min_obs_adapter + cfg.downscale_factor,
                });
            }
        }
        out
    }

    /// Batched detection over **already-prepped** signals (`None` = too short).
    /// Lets callers run [`AdapterCnnConfig::prep`](crate::AdapterCnnConfig::prep)
    /// in parallel on CPU producer threads and feed prepped blocks to the GPU,
    /// so the GPU thread only does grouping + inference + decode. Each exact
    /// length is one unpadded `[group, 1, len]` onnxruntime batch.
    pub fn detect_prepped(
        &self,
        prepped: &[Option<PreppedWindow>],
    ) -> Vec<Result<usize, AdapterCnnError>> {
        let valid_idx: Vec<usize> = (0..prepped.len())
            .filter(|&i| prepped[i].is_some())
            .collect();
        let mut out: Vec<Result<usize, AdapterCnnError>> = (0..prepped.len())
            .map(|_| {
                Err(AdapterCnnError::SignalTooShort {
                    len: 0,
                    required: 0,
                })
            })
            .collect();
        self.run_grouped(prepped, &valid_idx, &mut out);
        out
    }

    /// Run each exact-length group as unpadded onnxruntime batches, writing
    /// `Ok`/`Err` into `out` at each read's original index. `out` must already
    /// be sized to `prepped.len()` (with too-short defaults in place).
    ///
    /// A group is split into sub-batches of at most [`gpu_batch_elems`] input
    /// elements (rows × len). If a sub-batch still hits a GPU out-of-memory
    /// error (conv activations scale with the model's channel count, which the
    /// element cap can't know), it is halved and retried — so detection adapts
    /// to the device/model instead of silently failing those reads. Splitting is
    /// bit-identical: same length, no padding, the batch axis is independent.
    fn run_grouped(
        &self,
        prepped: &[Option<PreppedWindow>],
        valid_idx: &[usize],
        out: &mut [Result<usize, AdapterCnnError>],
    ) {
        for (len, group) in group_by_len(prepped, valid_idx) {
            let start_rows = (self.batch_elems / len.max(1)).max(1);
            // Work stack of `[lo, hi)` index ranges into `group`. On OOM a range
            // is split in half and pushed back, shrinking until it fits.
            let mut ranges: Vec<(usize, usize)> = (0..group.len())
                .step_by(start_rows)
                .map(|lo| (lo, (lo + start_rows).min(group.len())))
                .collect();
            while let Some((lo, hi)) = ranges.pop() {
                let sub = &group[lo..hi];
                match self.run_one(prepped, sub, len) {
                    // OOM on a splittable range: halve and retry. Splitting is
                    // bit-identical — same length, no padding, batch axis is
                    // independent.
                    Err(e) if hi - lo > 1 && is_out_of_memory(&e) => {
                        let mid = lo + (hi - lo) / 2;
                        ranges.push((mid, hi));
                        ranges.push((lo, mid));
                    }
                    // Success, or a terminal error → scatter into out.
                    result => scatter_group(out, sub, result),
                }
            }
        }
    }

    /// Raw channel-0 (adapter-end) scores for one prepped signal, run at tensor
    /// length `len` and zero-padded if the signal is shorter.
    ///
    /// Diagnostic. It exists to answer whether padding changes the model's output
    /// *before* the padding starts — which decides whether length bucketing is a
    /// wiring problem or a property of the graph, and no decoded `adapter_end`
    /// can distinguish those.
    pub fn scores_for_probe(
        &self,
        prepped: &[f32],
        len: usize,
    ) -> Result<Vec<f32>, AdapterCnnError> {
        let mut data = vec![0f32; len];
        let n = prepped.len().min(len);
        data[..n].copy_from_slice(&prepped[..n]);
        let input = Tensor::from_array(([1usize, 1, len], data))
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let mut session = self.session.lock().expect("ort session mutex poisoned");
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let (shape, scores) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        if shape.len() != 3 || shape[1] != 2 {
            return Err(AdapterCnnError::BadShape {
                got: shape.iter().map(|&d| d as usize).collect(),
            });
        }
        let length_out = shape[2] as usize;
        Ok(scores[..length_out].to_vec())
    }

    /// One onnxruntime call over `sub` reads (all of prepped length `len`),
    /// returning each read's adapter_end. Unpadded `[sub.len(), 1, len]`.
    fn run_one(
        &self,
        prepped: &[Option<PreppedWindow>],
        sub: &[usize],
        len: usize,
    ) -> Result<Vec<usize>, AdapterCnnError> {
        let cfg = self.config;
        let g = sub.len();
        let data = pack_batch(prepped, sub, len);
        let input = Tensor::from_array(([g, 1, len], data))
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let mut session = self.session.lock().expect("ort session mutex poisoned");
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        let (shape, scores) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
        // Expect row-major `[sub, 2, length_out]`.
        if shape.len() != 3 || shape[0] as usize != g || shape[1] != 2 {
            return Err(AdapterCnnError::BadShape {
                got: shape.iter().map(|&d| d as usize).collect(),
            });
        }
        let length_out = shape[2] as usize;
        Ok((0..g)
            .map(|row| {
                // Channel-0 (adapter_end) of row `row`. `valid_len` is this
                // read's own PRE-PADDING length, not the tensor width — see
                // `PreppedWindow`. Passing the padded length here is what let a
                // short read's argmax wander into 540 positions of SCORE_EXCL
                // and return a boundary past the end of the read.
                let valid_len = prepped[sub[row]]
                    .as_ref()
                    .expect("sub points at prepped signals")
                    .valid_len;
                let base = row * 2 * length_out;
                decode_adapter_end(&cfg, length_out, valid_len, |k| scores[base + k])
            })
            .collect())
    }
}

/// Heuristic: does this onnxruntime error look like a GPU allocation failure?
/// (CUDA EP surfaces OOM as a failed `Conv`/alloc with a BFCArena message.)
fn is_out_of_memory(e: &AdapterCnnError) -> bool {
    matches!(e, AdapterCnnError::Run(m)
        if m.contains("Failed to allocate") || m.contains("out of memory") || m.contains("CUDA_ERROR_OUT_OF_MEMORY"))
}

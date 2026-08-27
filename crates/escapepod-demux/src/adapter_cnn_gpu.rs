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

/// Sub-batches to aim for per device when splitting one length-group across a
/// pool.
///
/// Enough that a device which finishes early can take more work, few enough that
/// the per-call overhead the element budget exists to amortise is not paid over
/// and over. Four is deliberately modest: the reason a device runs slow here is
/// that it is *also* carrying encoder workers (see the fused pipeline's device
/// policy), which slows it by a fraction rather than stalling it, so the queue
/// only has to absorb a skew of that size.
const RANGES_PER_DEVICE: usize = 4;

/// Split a length-group of `group_len` reads into the `[lo, hi)` sub-batches a
/// pool of `n_dev` devices will draw from.
///
/// `start_rows` is the single-device chunk size — the element budget's answer.
/// Above one device it is capped so the stack holds [`RANGES_PER_DEVICE`] chunks
/// per device, because that budget alone leaves too few: at the rna004 geometry
/// a 24 GB card takes ~42 k rows of length 1500 against a 65 k-read block, so a
/// four-device pool would find two chunks and idle half its cards.
///
/// **`n_dev == 1` is exempt from the cap**, and that is the whole of the
/// bit-identity guarantee on [`AdapterCnnGpuPool`]: a one-device pool must batch
/// exactly the way no pool at all does. The caller short-circuits before
/// reaching here in that case, so this branch is belt and braces — but the
/// guarantee belongs in the function that decides the batching, not only in the
/// caller that currently happens to skip it.
///
/// Pure, and separated from the device loop so the covering properties below can
/// be tested on a machine with no GPU in it.
fn split_ranges(group_len: usize, start_rows: usize, n_dev: usize) -> Vec<(usize, usize)> {
    let start_rows = start_rows.max(1);
    let rows = if n_dev <= 1 {
        start_rows
    } else {
        start_rows.min(group_len.div_ceil(n_dev * RANGES_PER_DEVICE).max(1))
    };
    (0..group_len)
        .step_by(rows)
        .map(|lo| (lo, (lo + rows).min(group_len)))
        .collect()
}

/// One boundary-CNN session per CUDA device, fed from a single work queue.
///
/// # Why a pool rather than one session used harder
///
/// [`AdapterCnnGpu`] holds its `Session` behind a `Mutex`, so however many
/// threads call it the device work serialises — right with one card, a hard
/// ceiling with four. onnxruntime binds a session to one device at creation, so
/// a second device means a second session, and the graph is small enough
/// (~21 MB of weights) that holding one per card is not the constraint.
///
/// # The queue is shared on purpose
///
/// Devices pull ranges from one stack rather than being handed a partition. That
/// matters because the pipeline using this deliberately co-locates encoder
/// workers on the same cards: a device carrying more encode work simply takes
/// fewer detect ranges, with nothing to balance explicitly. A static split would
/// leave the busiest card holding the tail — the same argument the CRF encoder
/// pool makes for pulling from one channel.
///
/// # Sessions load concurrently, because the serial cost was measured
///
/// Each `commit_from_file` pays CUDA/cuDNN initialisation — seconds, not
/// milliseconds. Loading a four-device pool one card at a time was the largest
/// single cost this type added: over 150 k reads on 4x A30 the *pipeline* ran
/// 15.7 s -> 9.0 s (1.74x) while total wall only moved 18.5 s -> 15.3 s (1.21x),
/// because start-up had grown 2.8 s -> 6.3 s. More than half the win was going
/// into sequential session construction. Devices initialise independently, so
/// this is one `thread::scope` and the cost collapses to the slowest card's.
///
/// # Results across device counts: measured identical, not guaranteed identical
///
/// A pool splits a group into more, smaller onnxruntime calls than one device
/// would, and cuDNN picks its convolution algorithm from the batch shape, so
/// this had every reason to perturb a few boundaries — reordering reads on this
/// same path, which likewise repacks the batches, moved **7 of 503,076**. It
/// does not. Measured on 150 k reads of RNA004 through `demux detect --method
/// cnn`, 1 vs 2 vs 4 A30s: **0 of 150,000 boundaries differ**, and the fused
/// pipeline's barcode assignments agree on 150,001 of 150,001 rows. Each device
/// count also reproduces itself run to run.
///
/// Treat that as an observation about this model at these shapes, not a promise.
/// The two things that *are* structural:
///
/// * **A one-device pool is bit-identical to no pool at all.**
///   [`Self::detect_prepped`] delegates straight to
///   [`AdapterCnnGpu::detect_prepped`] and never re-chunks — see
///   [`split_ranges`] — so the overwhelmingly common case, and every regression
///   baseline taken on it, is untouched by this type existing.
/// * Reproducibility across counts rests on homogeneous cards. Which device
///   draws which range is a race, so it holds only because identical cards
///   running identical cuDNN pick identical kernels for identical shapes. A
///   mixed-model node (an A30 beside an L40) forfeits it, and so might a model
///   whose shapes land on a more batch-sensitive kernel.
pub struct AdapterCnnGpuPool {
    devices: Vec<AdapterCnnGpu>,
}

impl AdapterCnnGpuPool {
    /// Load one session per CUDA ordinal in `devices`, all at once.
    ///
    /// Ordinals index the *visible* devices, so they already honour
    /// `CUDA_VISIBLE_DEVICES`: under SLURM `--gres=gpu:1` the only valid list is
    /// `[0]` and the pool collapses onto the single-device path.
    ///
    /// One device is loaded inline rather than through the scope, so the common
    /// case adds no thread and no behaviour to explain.
    pub fn load_on_devices(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
        devices: &[i32],
    ) -> Result<Self, AdapterCnnError> {
        let Some((&first, rest)) = devices.split_first() else {
            return Err(AdapterCnnError::Load(
                "boundary-CNN pool needs at least one CUDA device".to_string(),
            ));
        };
        let path = path.as_ref();
        if rest.is_empty() {
            return Ok(Self {
                devices: vec![AdapterCnnGpu::load_with_config_on_device(
                    path, config, first,
                )?],
            });
        }
        let loaded: Vec<Result<AdapterCnnGpu, AdapterCnnError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = devices
                .iter()
                .map(|&d| {
                    scope.spawn(move || AdapterCnnGpu::load_with_config_on_device(path, config, d))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(AdapterCnnError::Load(
                            "boundary-CNN session loader thread panicked".to_string(),
                        ))
                    })
                })
                .collect()
        });
        // Collected after the join rather than with `?` inside it, so a failure
        // on one card still lets every other loader finish and release its
        // context instead of being torn down mid-`cuInit`.
        Ok(Self {
            devices: loaded.into_iter().collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Load one session on CUDA device 0 — the pool spelling of
    /// [`AdapterCnnGpu::load_with_config`].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
    ) -> Result<Self, AdapterCnnError> {
        Self::load_on_devices(path, config, &[0])
    }

    /// How many devices this pool spans.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Preprocessing config in effect. Every session was loaded with the same
    /// one, so device 0's answer is the pool's.
    pub fn config(&self) -> AdapterCnnConfig {
        self.devices[0].config()
    }

    /// Batched adapter-end detection from raw signals, prepping on the calling
    /// thread. Mirrors [`AdapterCnnGpu::detect_adapter_end_batch`].
    pub fn detect_adapter_end_batch(
        &self,
        signals: &[&[f32]],
    ) -> Vec<Result<usize, AdapterCnnError>> {
        let cfg = self.config();
        let prepped: Vec<Option<PreppedWindow>> = signals
            .iter()
            .map(|&s| prep_adapter_signal(s, &cfg))
            .collect();
        // Re-stamp too-short errors with the real input length, exactly as the
        // single-device path does — `detect_prepped` only ever sees `None`.
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

    /// Batched detection over already-prepped signals (`None` = too short),
    /// spread across every device in the pool.
    ///
    /// With one device this *is* [`AdapterCnnGpu::detect_prepped`] — same call,
    /// same batching, same bits.
    pub fn detect_prepped(
        &self,
        prepped: &[Option<PreppedWindow>],
    ) -> Vec<Result<usize, AdapterCnnError>> {
        if self.devices.len() == 1 {
            return self.devices[0].detect_prepped(prepped);
        }
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
        self.run_grouped_pooled(prepped, &valid_idx, &mut out);
        out
    }

    /// [`AdapterCnnGpu::run_grouped`]'s work stack, drained by every device at
    /// once instead of by one.
    ///
    /// The OOM halve-and-retry carries over unchanged, including its
    /// bit-exactness argument: splitting a range neither pads nor reorders, and
    /// the batch axis is independent. One difference is worth stating — a device
    /// that finds the stack empty stops, so when a *later* OOM pushes halves
    /// back, the device that hit it finishes them alone. That degrades a rare
    /// recovery path to single-device speed rather than spin-waiting every other
    /// card on work that will usually never arrive.
    fn run_grouped_pooled(
        &self,
        prepped: &[Option<PreppedWindow>],
        valid_idx: &[usize],
        out: &mut [Result<usize, AdapterCnnError>],
    ) {
        let n_dev = self.devices.len();
        let out = Mutex::new(out);
        for (len, group) in group_by_len(prepped, valid_idx) {
            // The single-device chunk size is the starting point, then capped so
            // there is work for every device to draw several times over. Without
            // the cap a full block is ~2 chunks at the rna004 geometry (a 24 GB
            // card takes ~42 k rows of length 1500, against a 65 k-read block),
            // so two of four devices would sit idle.
            let start_rows = (self.devices[0].batch_elems / len.max(1)).max(1);
            let ranges: Mutex<Vec<(usize, usize)>> =
                Mutex::new(split_ranges(group.len(), start_rows, n_dev));
            let (ranges, out, group) = (&ranges, &out, &group);
            std::thread::scope(|scope| {
                for dev in &self.devices {
                    scope.spawn(move || {
                        loop {
                            let next = ranges.lock().expect("cnn pool range stack poisoned").pop();
                            let Some((lo, hi)) = next else { break };
                            let sub = &group[lo..hi];
                            match dev.run_one(prepped, sub, len) {
                                // Same halve-and-retry as `run_grouped`; the
                                // halves go back on the shared stack, so any
                                // device still drawing can pick them up.
                                Err(e) if hi - lo > 1 && is_out_of_memory(&e) => {
                                    let mid = lo + (hi - lo) / 2;
                                    let mut r =
                                        ranges.lock().expect("cnn pool range stack poisoned");
                                    r.push((mid, hi));
                                    r.push((lo, mid));
                                }
                                // Success or a terminal error. The lock is held
                                // only for the scatter, never across a device
                                // call, so it is uncontended in practice.
                                result => {
                                    let mut o = out.lock().expect("cnn pool output mutex poisoned");
                                    scatter_group(&mut o, sub, result);
                                }
                            }
                        }
                    });
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every read lands in exactly one sub-batch, and the sub-batches are in
    /// ascending order with no gap. Detection scatters by original index, so a
    /// gap is a read silently left at its `SignalTooShort` default and an
    /// overlap is two devices racing to write the same slot.
    fn assert_covers(ranges: &[(usize, usize)], group_len: usize) {
        let mut sorted = ranges.to_vec();
        sorted.sort_unstable();
        let mut next = 0;
        for &(lo, hi) in &sorted {
            assert_eq!(lo, next, "gap or overlap at {lo} in {sorted:?}");
            assert!(lo < hi, "empty range {lo}..{hi}");
            next = hi;
        }
        assert_eq!(next, group_len, "ranges stop short of the group");
    }

    #[test]
    fn ranges_cover_the_group_exactly() {
        for &group_len in &[1usize, 7, 4096, 65_536, 65_537] {
            for &n_dev in &[1usize, 2, 3, 4, 8] {
                for &start_rows in &[1usize, 512, 42_666, 1_000_000] {
                    let r = split_ranges(group_len, start_rows, n_dev);
                    assert_covers(&r, group_len);
                }
            }
        }
    }

    /// The pool must not hand a device *fewer* rows per call than the element
    /// budget already refused — splitting is for balance, never for growing the
    /// batch past what VRAM was sized for.
    #[test]
    fn never_exceeds_the_element_budget() {
        for &n_dev in &[1usize, 2, 4, 8] {
            for &start_rows in &[1usize, 64, 512, 42_666] {
                for &(lo, hi) in &split_ranges(65_536, start_rows, n_dev) {
                    assert!(hi - lo <= start_rows.max(1), "{lo}..{hi} > {start_rows}");
                }
            }
        }
    }

    /// The balance property this cap exists for: with a realistic block and
    /// element budget, every device gets several chunks rather than two of four
    /// finding the stack empty — and one device still batches exactly the way
    /// the un-pooled path does.
    #[test]
    fn a_full_block_gives_every_device_work() {
        // 65 k-read block, ~42 k rows per call on a 24 GB A30 at length 1500.
        const GROUP: usize = 65_536;
        const START_ROWS: usize = 42_666;
        // One device: the element budget alone, uncapped — the two chunks
        // `run_grouped` would make.
        assert_eq!(
            split_ranges(GROUP, START_ROWS, 1),
            vec![(0, START_ROWS), (START_ROWS, GROUP)]
        );
        for &n_dev in &[2usize, 4] {
            let r = split_ranges(GROUP, START_ROWS, n_dev);
            assert!(
                r.len() >= n_dev * RANGES_PER_DEVICE - 1,
                "{n_dev} devices got only {} chunks",
                r.len()
            );
        }
    }

    /// A group smaller than the device count still runs — every read is covered,
    /// and the devices that find nothing simply stop.
    #[test]
    fn tiny_groups_do_not_produce_empty_ranges() {
        let r = split_ranges(3, 42_666, 8);
        assert_covers(&r, 3);
        assert_eq!(r.len(), 3);
    }

    /// A pool needs at least one device; asking for none is a caller bug that
    /// must not reach `self.devices[0]`.
    #[test]
    fn empty_device_list_is_rejected() {
        // `unwrap_err` is out — a pool holds `Session`s and so is not `Debug`.
        let Err(err) =
            AdapterCnnGpuPool::load_on_devices("/nonexistent.onnx", Default::default(), &[])
        else {
            panic!("an empty device list must not produce a pool");
        };
        assert!(
            matches!(&err, AdapterCnnError::Load(m) if m.contains("at least one CUDA device")),
            "{err}"
        );
    }
}

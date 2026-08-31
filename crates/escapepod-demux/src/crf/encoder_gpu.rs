//! GPU CTC-CRF encoder inference via onnxruntime (CUDA execution provider).
//!
//! Opt-in via `gpu`. Runs the *same* ONNX graph as [`CrfEncoder`] and hands
//! the scores to the *same* [`lattice`](super::lattice) decode, so the only
//! thing that changes is where the LSTM stack executes.
//!
//! # Where the decode runs
//!
//! Measured per read on the RNA004 barcode export (rna partition):
//!
//! ```text
//! tract encoder        13.9 ms
//! decode, scalar       12.14 ms
//! decode, AVX2          1.92 ms
//! decode, AVX-512       1.19 ms
//! ```
//!
//! (`cargo bench --bench crf_decode` reproduces the three decode rows.)
//!
//! The decode started out as *half* the CPU cost, so moving only the encoder to
//! the GPU leaves it as essentially the entire remaining runtime — which is why
//! the vector kernels are a prerequisite for this path rather than a nicety.
//!
//! It now goes to the device too, via [`lattice_gpu`](super::lattice_gpu), and
//! that is the default whenever the kernels load. **This reverses an earlier
//! note here**, which read: "the lattice decode is sequential in time with a
//! 256-wide inner dimension, so it is a poor fit for the device compared to the
//! encoder's dense matmuls."
//!
//! That was right about decoding *one read* and wrong about this pipeline. A
//! device batch holds hundreds of independent reads, so a timestep is
//! `batch * n_states` lanes, not 256; only the `t_len` sweep is sequential, and
//! it is sequential on the CPU too. Profiling the encoder-only GPU path settled
//! it: with the encoder on the device the AVX-512 decode was **66% of all
//! remaining CPU cycles**, and the pipeline stopped scaling with cores entirely.
//! bonito reaches the same conclusion — its own forward/backward scores come
//! from koi's `ctc.{fwd,bwd}_scores_cu_sparse` CUDA kernels.
//!
//! The CPU decode remains the reference implementation, the only always-compiled
//! one, and the automatic fallback; `ESCAPEPOD_CRF_GPU_DECODE=0` forces it.
//!
//! `load-dynamic`: onnxruntime is dlopened at runtime, not linked at build
//! time. Point `ORT_DYLIB_PATH` at a CUDA-enabled `libonnxruntime.so` with a
//! visible CUDA device. If the CUDA EP cannot initialise, onnxruntime silently
//! falls back to CPU — slow, but correct.
//!
//! # What this graph is, and why the obvious accelerations do not apply
//!
//! 519,880 parameters, opset 17: three 1-D convolutions with SiLU (the last
//! `k=31, stride=10`), then **five unidirectional LSTM layers of `hidden_size`
//! 96** (direction alternated by a negative-step `Slice`, bonito's reverse
//! trick), then `MatMul(96->1024)`, `tanh`, `*5.0`, and a `Pad` with 2.0 that
//! expands 1024 to the 1280 blank-per-state width. Scores are therefore
//! `5*tanh(...)`, **bounded to [-5, 5]** — no dynamic range problem anywhere.
//!
//! Per-op device time at batch 512 says where the money is:
//!
//! ```text
//! LSTM    389.2 ms   89.8%
//! Reshape/Pad 15.1    3.5%
//! Conv      9.6       2.2%
//! MatMul    8.0       1.8%
//! ```
//!
//! **~90% of the work is an LSTM stack at `hidden = 96` over 300 timesteps x 5
//! layers = 1500 *sequential* steps.** Each recurrent GEMM is a thin
//! `[rows,96] x [96,384]`, so the stack is launch-latency and bandwidth bound,
//! not FLOP bound. That single fact decides the two accelerations people reach
//! for first:
//!
//! * **fp16 — measured and rejected.** ORT's CUDA EP cannot convert an fp32
//!   graph at runtime, so this means shipping a converted model. Done and
//!   benchmarked: only **1.18x** at batch 512 end-to-end, because the LSTM is
//!   89.8% of the work and gets just 1.23x. The cost is not worth arguing about:
//!   over 40 k real reads it changed **38 barcode calls (0.095%)**, and they are
//!   *unfilterable* — 32 of the 38 had margin > 5 on both sides and several were
//!   exact (`dist = 0`) matches to a **different** barcode, so no `--min-margin`
//!   removes them. The mechanism is chaotic amplification of 1-ulp gate
//!   decisions through 1500 sequential steps (max |delta| grows 0.0095 -> 2.30
//!   layer by layer while the *mean* stays at fp16 eps), not saturation — so
//!   loss scaling cannot fix it and bf16 would be worse. Keeping the LSTMs in
//!   fp32 caps the whole exercise at 1.03x.
//! * **TF32 is already on** via `CUDA::default()` and is worth 1.35x. See
//!   `crate::ort_ep` — do not turn it off.
//!
//! TensorRT is the one that attacks the actual bottleneck: it absorbs the whole
//! graph (all five LSTMs, no CUDA-EP fallback) and preserves the zero-copy
//! output binding, measuring 1.50x at batch 1024. But it decays with batch size
//! — 1.28x at 512, **1.01x at 256** — and the pipeline runs 512 or less, while
//! `libnvinfer` is ~860 MB and installed nowhere on this cluster. Left unbuilt
//! deliberately; `ort::ep::TensorRT` behind a feature is the seam if that
//! changes, and it needs an explicit optimization profile or every new batch
//! size triggers a ~15 s engine rebuild.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

use super::encoder::{CrfError, CrfMetadata};
use super::lattice::{Backend, CrfLayout, CrfScratch, decode_with, decode_with_refs_strided};
use super::refchain::{RefChains, ScoredDecode};

/// A cheap identity for a reference panel, so the device scan tables can be
/// cached without assuming the caller never changes panels.
///
/// Built from the chain geometry and every reference's terminal cell — a panel
/// that differs in any reference differs in a final, since that is the cell its
/// last base lands on. Reused tables for the wrong panel would score reads
/// confidently against references they were never compared to, which nothing
/// downstream could detect, so this errs toward rebuilding.
#[cfg(feature = "gpu")]
fn panel_fingerprint(chains: &RefChains) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    chains.fingerprint_parts().hash(&mut h);
    h.finish()
}

/// Rows one device may have in flight across *all* encoders sharing it.
///
/// 1024 is two 512-row batches, the configuration the pipeline was tuned at on a
/// 24 GB A30. It is a budget rather than a per-worker constant because every
/// worker on a device allocates its LSTM activations from the same VRAM: a
/// sweep at `ESCAPEPOD_CRF_GPU_WORKERS=4` on one A30 asked for 4x512 rows, hit
/// 49 allocation failures, and then died with a generic `CudaCall` error on a
/// `Reshape` — the halve-and-retry recognised every one of those 49 but could
/// not save the run, because by then the context was wedged. Prevention is the
/// only fix available; retry is the backstop for what prevention cannot size.
pub const DEVICE_ROW_BUDGET: usize = 1024;

/// Workers per device assumed before [`CrfEncoderGpu::share_device_with`] says
/// otherwise. Two is what the fused pipeline runs, and assuming it keeps the
/// out-of-the-box batch at the 512 rows every caller used before the budget
/// existed.
///
/// # Two is measured, not assumed — do not "simplify" it to one
///
/// Total worker-seconds inside encode+decode grow linearly with the worker
/// count at flat wall (2/4/6/8 workers on one A30: 28.2 / 28.2 / 26.1 / 31.0 s),
/// which reads like pure contention and suggests the second worker is
/// overhead. It is not. Against one worker on the same card, interleaved, the
/// fused pipeline over 100 k reads (#297):
///
/// ```text
/// workers=1   41.9 s / 48.3 s   GPU 19% / 17%
/// workers=2   35.3 s / 40.0 s   GPU 25% / 21%
/// ```
///
/// The second worker is worth ~16% and lifts device utilisation, because what
/// it overlaps is the *other* worker's per-call setup rather than adding
/// parallel device compute. Past two the row budget splits faster than the
/// overlap pays for itself, which is the flat sweep above.
pub const DEFAULT_WORKERS_PER_DEVICE: usize = 2;

/// Reads per onnxruntime call before splitting, for one encoder that shares its
/// device with `workers_on_device - 1` others.
///
/// LSTM activations scale with `rows * t_len * features * layers`, which this
/// cannot see, so the result is still a starting guess: a batch that hits a
/// device out-of-memory error is halved and retried, exactly as the boundary-CNN
/// GPU path does.
///
/// `ESCAPEPOD_CRF_GPU_BATCH_ROWS` overrides it and is **per worker, not per
/// device** — raising it with several workers per device is how you reproduce
/// the failure above.
fn resolve_batch_rows(workers_on_device: usize) -> usize {
    /// Below this, splitting costs more in per-call overhead than the memory is
    /// worth; a device that cannot host this many rows needs fewer workers.
    const MIN_ROWS: usize = 64;
    std::env::var("ESCAPEPOD_CRF_GPU_BATCH_ROWS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| (DEVICE_ROW_BUDGET / workers_on_device.max(1)).max(MIN_ROWS))
}

/// Batched CTC-CRF basecaller backed by onnxruntime + CUDA.
///
/// `ort::Session::run` takes `&mut self`, so the session sits behind a `Mutex`:
/// callers share `&CrfEncoderGpu` across rayon workers and the GPU calls
/// serialise on the lock, which is what we want with one device anyway.
pub struct CrfEncoderGpu {
    session: Mutex<Session>,
    meta: CrfMetadata,
    layout: CrfLayout,
    alphabet: Vec<u8>,
    /// Rows per device call. Atomic because the encoder is shared behind `&` by
    /// the time the pipeline knows how many workers will contend for its device
    /// — see [`Self::share_device_with`]. Written once at start-up, read per
    /// batch, so `Relaxed` is all the ordering this needs.
    batch_rows: std::sync::atomic::AtomicUsize,
    /// Batched lattice decode on the device. `None` falls back to the rayon CPU
    /// decode — correct either way, but the CPU decode is ~66% of this path's
    /// host cost, so which one is running is worth reporting rather than
    /// discovering from a benchmark.
    lattice: Option<super::lattice_gpu::CrfLatticeGpu>,
    /// The reference panel's scan tables, resident on the device, with the
    /// fingerprint of the panel they were built from.
    ///
    /// A run scores one panel, so this is uploaded on the first scoring batch
    /// and reused. It is keyed rather than assumed: a caller that switches
    /// panels mid-encoder gets a rebuild instead of another panel's chains,
    /// which would score confidently against the wrong references. `None` after
    /// an attempt means the panel does not fit the kernel and the CPU scan runs.
    ///
    /// `Arc` so the guard is released before the encode rather than held across
    /// it. Each worker owns its own encoder today, so the lock is uncontended
    /// either way — but a shared encoder would then serialize every worker on a
    /// multi-second GPU call, and nothing in the type would say so.
    ref_scan: Mutex<Option<(u64, Arc<super::lattice_gpu::RefScanDev>)>>,
    /// Why [`Self::lattice`] is `None`, when it is.
    decode_fallback: Option<String>,
    /// The graph's output name, resolved once so the IoBinding does not have to
    /// re-read session metadata per batch.
    output_name: String,
    /// The CUDA ordinal this session and its lattice context live on. The
    /// zero-copy binding's `MemoryInfo` must name this device and not device 0:
    /// onnxruntime matches the bound output's allocator against the session's
    /// own, and a mismatch fails the run outright with "Failed to find allocator
    /// for device".
    device: i32,
    /// Bind the encoder's output to CUDA memory and decode it in place, instead
    /// of letting onnxruntime copy it to the host for us to upload again.
    /// Requires [`Self::lattice`]; `ESCAPEPOD_CRF_GPU_ZEROCOPY=0` disables it.
    zero_copy: bool,
    /// Why zero-copy stood down at load, when it did. Surfaced like
    /// [`Self::decode_fallback`] rather than logged: this crate has no `tracing`
    /// dependency, and the CLI is the layer that decides how loud to be.
    zero_copy_fallback: Option<String>,
}

impl CrfEncoderGpu {
    /// Load an export directory containing `metadata.json` and its ONNX graph.
    ///
    /// The session's intra-op pool is fixed at one non-spinning thread — see
    /// [`load_on_device`](Self::load_on_device). It used to be a parameter
    /// threaded down from the CLI's `--threads`, which is the opposite of what
    /// helps: the work is on the device, and a wide pool only takes cores from
    /// the prep that feeds it.
    pub fn load_bundle(dir: impl AsRef<Path>) -> Result<Self, CrfError> {
        Self::load_bundle_on_device(dir, 0)
    }

    /// [`load_bundle`](Self::load_bundle) pinned to a CUDA device ordinal.
    ///
    /// One of these per encoder worker is what lets the pipeline run several
    /// encoders at once — on one device, so each worker's per-call setup
    /// overlaps another's device work, or spread across several. The session and
    /// its lattice context both land on `device`, so a worker never moves scores
    /// between devices.
    pub fn load_bundle_on_device(dir: impl AsRef<Path>, device: i32) -> Result<Self, CrfError> {
        let dir = dir.as_ref();
        let meta = CrfMetadata::load(dir.join("metadata.json"))?;
        let onnx = dir.join(&meta.onnx);
        Self::load_on_device(onnx, meta, device)
    }

    /// Load an ONNX graph with an already-parsed sidecar.
    pub fn load(onnx: impl AsRef<Path>, meta: CrfMetadata) -> Result<Self, CrfError> {
        Self::load_on_device(onnx, meta, 0)
    }

    /// [`load`](Self::load) pinned to a CUDA device ordinal.
    pub fn load_on_device(
        onnx: impl AsRef<Path>,
        meta: CrfMetadata,
        device: i32,
    ) -> Result<Self, CrfError> {
        // Same reason as `CrfEncoder::load`: a hand-built sidecar must not
        // reach the device unchecked.
        meta.validate()?;
        let layout = meta.layout()?;
        let alphabet = meta.alphabet_bytes();

        // One non-spinning intra-op thread, exactly as `AdapterCnnGpu::load`
        // does and for the same measured reason (#239/#240) — that fix landed
        // on the CNN session and was never carried to this one.
        //
        // The graph runs on the device, so the intra-op pool has almost nothing
        // to compute; left at its default it is `available_parallelism()` wide
        // *and spinning*, on top of rayon's pool. This path made that worse than
        // the CNN one ever was: the pool width came from the CLI's `--threads`,
        // and the fused pipeline builds one session per encoder worker, so
        // `--threads 32` with 2 workers meant 64 spinning threads on a 32-core
        // allocation — cores the decode and prep feeding this encoder needed,
        // which is how a GPU ends up underfed by its own inference sessions.
        // Scaling `--threads` up scaled the spin up with it.
        //
        // #240's numbers, measured on the CNN session (warm, three interleaved
        // reps): 16 threads spinning 7.34 s, 1 thread spinning 7.42 s, 16
        // without spinning 7.37 s, **1 without spinning 6.93 s** at 274% CPU
        // against 340%. Both settings are needed; neither works alone.
        let builder = Session::builder()
            .map_err(|e| CrfError::Load(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| CrfError::Load(e.to_string()))?
            // Only meaningful under parallel execution mode, which we don't
            // enable; set it anyway so the bound holds if that ever changes.
            .with_inter_threads(1)
            .map_err(|e| CrfError::Load(e.to_string()))?
            .with_intra_op_spinning(false)
            .map_err(|e| CrfError::Load(e.to_string()))?;
        let session = builder
            .with_execution_providers(crate::ort_ep::cuda_providers_on(device))
            .map_err(|e| CrfError::Load(e.to_string()))?
            .commit_from_file(onnx)
            .map_err(|e| CrfError::Load(e.to_string()))?;

        // The lattice decode is the larger half of this path's host cost, so it
        // goes to the device too unless it cannot. `ESCAPEPOD_CRF_GPU_DECODE=0`
        // forces the CPU decode, which is what A/B measurements and any future
        // bisect want.
        let (lattice, decode_fallback) =
            if std::env::var("ESCAPEPOD_CRF_GPU_DECODE").as_deref() == Ok("0") {
                (
                    None,
                    Some("disabled by ESCAPEPOD_CRF_GPU_DECODE=0".to_string()),
                )
            } else {
                match super::lattice_gpu::CrfLatticeGpu::new_on_device(layout, device as usize) {
                    Ok(l) => (Some(l), None),
                    Err(e) => (None, Some(e.to_string())),
                }
            };

        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| CrfError::Load("encoder graph declares no outputs".to_string()))?;
        // Only meaningful with the device decode: with the CPU decode the scores
        // have to reach the host anyway. Note this is a *request*: `lattice` only
        // proves the CUDA driver and NVRTC work, which says nothing about whether
        // onnxruntime registered its CUDA EP. The probe below settles that.
        let zero_copy =
            lattice.is_some() && std::env::var("ESCAPEPOD_CRF_GPU_ZEROCOPY").as_deref() != Ok("0");

        let mut encoder = Self {
            session: Mutex::new(session),
            meta,
            layout,
            alphabet,
            // Assume the device is shared the way the fused pipeline shares it
            // (two workers), which `share_device_with` then confirms or corrects
            // once the worker layout is known.
            //
            // NOT `resolve_batch_rows(1)`. A lone encoder really could take the
            // whole budget, but callers that never call `share_device_with` —
            // `demux basecall --gpu`, the examples — would then silently jump
            // from the 512 rows they used before this became a budget to 1024,
            // doubling per-call device memory on hardware that had not changed.
            // A default that only holds for the caller that overrides it anyway
            // is the wrong default.
            batch_rows: std::sync::atomic::AtomicUsize::new(resolve_batch_rows(
                DEFAULT_WORKERS_PER_DEVICE,
            )),
            lattice,
            ref_scan: Mutex::new(None),
            decode_fallback,
            output_name,
            device,
            zero_copy,
            zero_copy_fallback: None,
        };
        // Same reasoning as the CPU loader: catch a batch-major (boundary-CNN)
        // export here rather than after decoding noise for every read.
        encoder.encode_batch(&[vec![0f32; encoder.meta.signal.chunk]])?;
        encoder.probe_zero_copy();
        Ok(encoder)
    }

    pub fn metadata(&self) -> &CrfMetadata {
        &self.meta
    }

    /// Override the decode's boundary margin. See
    /// [`CrfMetadata::set_boundary_margin`].
    pub fn set_boundary_margin(&mut self, margin: usize) {
        self.meta.set_boundary_margin(margin);
    }

    /// Override how far the window may be clamped. See
    /// [`CrfMetadata::clamp_max_shift`].
    pub fn set_clamp_max_shift(&mut self, shift: usize) {
        self.meta.set_clamp_max_shift(shift);
    }

    pub fn layout(&self) -> &CrfLayout {
        &self.layout
    }

    /// Whether the lattice decode runs on the device.
    pub fn gpu_decode_active(&self) -> bool {
        self.lattice.is_some()
    }

    /// Rows this encoder sends to the device per call.
    pub fn batch_rows(&self) -> usize {
        self.batch_rows.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Tell this encoder that `workers` encoders (including itself) share its
    /// device, so it takes only its slice of [`DEVICE_ROW_BUDGET`].
    ///
    /// Call before the first batch. Workers do not allocate from a common pool —
    /// each holds its own LSTM activations in the same VRAM — so without this,
    /// asking for more workers multiplies device memory rather than dividing the
    /// work, and the run dies once that exceeds the card. An explicit
    /// `ESCAPEPOD_CRF_GPU_BATCH_ROWS` still wins: it is stated per worker, and
    /// overriding it is how you deliberately trade one against the other.
    pub fn share_device_with(&self, workers: usize) {
        self.batch_rows.store(
            resolve_batch_rows(workers),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Whether the encoder's scores are decoded in place in device memory,
    /// rather than copied to the host and uploaded again.
    pub fn zero_copy_active(&self) -> bool {
        self.zero_copy
    }

    /// Why zero-copy stood down at load, when it did. The caller should surface
    /// this: the run is still correct, but slower than the log line above it
    /// would otherwise imply.
    pub fn zero_copy_fallback_reason(&self) -> Option<&str> {
        self.zero_copy_fallback.as_deref()
    }

    /// Settle at load time whether the encoder's output really lands in device
    /// memory, and quietly stand down if it does not.
    ///
    /// `zero_copy` is requested from `lattice.is_some()`, which proves only that
    /// the CUDA *driver* and libnvrtc are usable. Whether **onnxruntime**
    /// registered its CUDA EP is a separate question, and the module header
    /// documents the answer being "no" as a survivable outcome: "If the CUDA EP
    /// cannot initialise, onnxruntime silently falls back to CPU — slow, but
    /// correct." Without this probe that stopped being true. A CUDA-capable node
    /// with a CPU-only `libonnxruntime` on `ORT_DYLIB_PATH` — the dangling-path
    /// and missing-libcudnn traps this repo has already been bitten by — would
    /// load cleanly, announce "decoded in place on the device", then fail on the
    /// first real batch. That is not an OOM, so the halve-and-retry does not
    /// catch it, and it lands after the per-barcode POD5 files are part-written.
    ///
    /// One batch-1 run through the binding, at load, converts that into the
    /// documented slow-but-correct degradation plus a warning.
    fn probe_zero_copy(&mut self) {
        if !self.zero_copy {
            return;
        }
        let Some(lattice) = self.lattice.as_ref() else {
            return;
        };
        let probe = vec![0f32; self.meta.signal.chunk];
        let rows: Vec<&[f32]> = vec![probe.as_slice()];
        if let Err(e) = self.run_zero_copy(&rows, lattice, None) {
            self.zero_copy = false;
            self.zero_copy_fallback = Some(e.to_string());
        }
    }

    /// Why the decode fell back to the CPU, when it did. The caller is expected
    /// to surface this: a CPU decode is correct but roughly three times slower
    /// end to end, and that is not something a run should discover silently.
    pub fn decode_fallback_reason(&self) -> Option<&str> {
        self.decode_fallback.as_deref()
    }

    /// Run the encoder over a batch of standardised windows.
    ///
    /// Returns one `t_len * n_score` score buffer per input, already
    /// de-interleaved out of the time-major `[T, batch, n_score]` output into
    /// the per-read `[t][dest][edge]` layout the decode wants.
    ///
    /// Every read's scores are held at once — `t_len * n_score` floats is 1 MB
    /// for the RNA004 geometry, so this is 1 MB *per input*. Prefer
    /// [`basecall_batch`](Self::basecall_batch), which decodes each device batch
    /// before encoding the next and so never holds more than `batch_rows` of
    /// them.
    pub fn encode_batch(&self, prepped: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, CrfError> {
        let mut out = Vec::with_capacity(prepped.len());
        for chunk in prepped.chunks(self.batch_rows()) {
            let rows: Vec<&[f32]> = chunk.iter().map(Vec::as_slice).collect();
            out.extend(self.run_batch(&rows)?);
        }
        Ok(out)
    }

    /// One onnxruntime call, halving and retrying on device out-of-memory.
    ///
    /// LSTM activations scale with the model's hidden width and layer count,
    /// which `batch_rows` cannot know, so the starting batch is a guess and
    /// this is what adapts it to the device. Splitting is exact: every window
    /// is the same length, nothing is padded, and the batch axis is
    /// independent, so a split batch and a whole one give identical scores.
    fn run_batch(&self, rows: &[&[f32]]) -> Result<Vec<Vec<f32>>, CrfError> {
        match self.try_run_batch(rows) {
            Err(CrfError::Run(msg)) if is_oom(&msg) && rows.len() > 1 => {
                let half = rows.len().div_ceil(2);
                let mut first = self.run_batch(&rows[..half])?;
                first.extend(self.run_batch(&rows[half..])?);
                Ok(first)
            }
            other => other,
        }
    }

    fn try_run_batch(&self, rows: &[&[f32]]) -> Result<Vec<Vec<f32>>, CrfError> {
        self.run_raw(rows, |data, t_len, batch, n_score| {
            Ok(split_time_major(data, t_len, batch, n_score))
        })
    }

    /// Build the `[batch, 1, chunk]` input, run one onnxruntime call, check the
    /// output shape, and hand the raw time-major buffer to `consume`.
    ///
    /// The session lock is held across `consume` because the output tensor
    /// borrows the session's run context. Both callers drive the encoder from a
    /// single thread — the fused pipeline's GPU thread and `demux basecall`'s
    /// batch loop — so nothing contends on it.
    fn run_raw<R>(
        &self,
        rows: &[&[f32]],
        consume: impl FnOnce(&[f32], usize, usize, usize) -> Result<R, CrfError>,
    ) -> Result<R, CrfError> {
        let chunk = self.meta.signal.chunk;
        let batch = rows.len();
        let mut flat = Vec::with_capacity(batch * chunk);
        for r in rows {
            if r.len() != chunk {
                return Err(CrfError::Run(format!(
                    "prepped window has {} samples, expected {chunk}",
                    r.len()
                )));
            }
            flat.extend_from_slice(r);
        }

        let input = Tensor::from_array(([batch, 1, chunk], flat))
            .map_err(|e| CrfError::Run(e.to_string()))?;
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs = session
            .run(ort::inputs!["signal" => input])
            .map_err(|e| CrfError::Run(e.to_string()))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| CrfError::Run(e.to_string()))?;

        let t_len = self.meta.t_len();
        let n_score = self.layout.n_score;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if dims != [t_len, batch, n_score] {
            return Err(CrfError::BadShape {
                got: dims,
                t: t_len,
                batch,
                n_score,
            });
        }

        consume(data, t_len, batch, n_score)
    }

    /// [`run_and_decode`](Self::run_and_decode) for one call, no OOM retry.
    ///
    /// Gathers each read out of the time-major buffer and decodes it in the
    /// *same* parallel pass, into a buffer the worker reuses across reads.
    /// Materialising the transpose first (as [`split_time_major`] does) costs a
    /// 1.5 MB allocation per read and pushes the whole 786 MB batch out to DRAM
    /// and back; gathering straight into the decode's input keeps it in cache
    /// and leaves one buffer per rayon worker instead of one per read.
    fn try_run_and_decode(
        &self,
        rows: &[&[f32]],
        backend: Backend,
    ) -> Result<Vec<String>, CrfError> {
        // The device decode reads onnxruntime's time-major output as it stands,
        // so nothing is de-interleaved on the host at all — neither the batch
        // axis nor the per-timestep [dest][edge] order.
        if let Some(lattice) = &self.lattice {
            if self.zero_copy {
                return self.run_zero_copy(rows, lattice, None);
            }
            return self.run_raw(rows, |data, t_len, _batch, _n_score| {
                lattice.decode_time_major(data, t_len, &self.alphabet)
            });
        }
        self.run_raw(rows, |data, t_len, batch, n_score| {
            (0..batch)
                .into_par_iter()
                .map_init(
                    || {
                        (
                            CrfScratch::new(),
                            Vec::<f32>::with_capacity(t_len * n_score),
                        )
                    },
                    |(scratch, buf), b| {
                        buf.clear();
                        for t in 0..t_len {
                            let off = (t * batch + b) * n_score;
                            buf.extend_from_slice(&data[off..off + n_score]);
                        }
                        decode_with(
                            &self.layout,
                            &self.alphabet,
                            buf.as_slice(),
                            t_len,
                            scratch,
                            backend,
                        )
                        .map_err(|e| CrfError::Decode(e.to_string()))
                    },
                )
                .collect()
        })
    }

    /// Encode with the output bound to CUDA memory and decode it where it lies.
    ///
    /// The scores are the pipeline's largest object by far — `t_len * n_score`
    /// floats, 1.5 MB per read at the RNA004 geometry. Letting onnxruntime copy
    /// them to the host so the decode can upload them again costs two PCIe
    /// crossings per read and nothing else; once both the encoder and the decode
    /// were on the device that was the binding constraint. Binding the output to
    /// device memory removes both crossings, and only `t_len` bytes of decoded
    /// path per read ever come back.
    ///
    /// The input still crosses host→device, but it is `chunk` floats per read
    /// (12 KB), two orders of magnitude smaller than the scores.
    fn run_zero_copy(
        &self,
        rows: &[&[f32]],
        lattice: &super::lattice_gpu::CrfLatticeGpu,
        scan: Option<(
            &super::lattice_gpu::RefScanDev,
            &mut Vec<f32>,
            &mut Vec<f32>,
        )>,
    ) -> Result<Vec<String>, CrfError> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
        use ort::value::{DynTensorValueType, ValueType};

        let chunk = self.meta.signal.chunk;
        let batch = rows.len();
        let mut flat = Vec::with_capacity(batch * chunk);
        for r in rows {
            if r.len() != chunk {
                return Err(CrfError::Run(format!(
                    "prepped window has {} samples, expected {chunk}",
                    r.len()
                )));
            }
            flat.extend_from_slice(r);
        }
        let input = Tensor::from_array(([batch, 1, chunk], flat))
            .map_err(|e| CrfError::Run(e.to_string()))?;
        let run = |e: ort::Error| CrfError::Run(e.to_string());

        // This worker's own ordinal, not 0. onnxruntime resolves the bound
        // output's allocator against the session's device, so naming the wrong
        // one fails the run outright with "Failed to find allocator for device"
        // — which is what a second worker placed on device 1 hit while this said
        // 0. The lattice context is on the same ordinal, so the decode reads the
        // scores from the device that produced them.
        let mem = MemoryInfo::new(
            AllocationDevice::CUDA,
            self.device,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(run)?;

        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut binding = session.create_binding().map_err(run)?;
        binding.bind_input("signal", &input).map_err(run)?;
        binding
            .bind_output_to_device(self.output_name.as_str(), &mem)
            .map_err(run)?;
        let outputs = session.run_binding(&binding).map_err(run)?;
        // onnxruntime's execution provider has its own stream; the decode runs on
        // the lattice context's. Without this the kernels could read the buffer
        // while the encoder is still filling it.
        binding.synchronize_outputs().map_err(run)?;

        let tensor = outputs[0]
            .downcast_ref::<DynTensorValueType>()
            .map_err(|e| CrfError::Run(format!("encoder output is not a tensor: {e}")))?;

        // If the EP declined the binding and produced a host tensor, `data_ptr`
        // is a host pointer and handing it to a kernel would read garbage — or
        // fault. Refuse rather than produce silent nonsense.
        let device = tensor.memory_info().allocation_device();
        if device != AllocationDevice::CUDA {
            return Err(CrfError::Run(format!(
                "encoder output landed on {device:?}, not CUDA, so it cannot be decoded \
                 in place. Set ESCAPEPOD_CRF_GPU_ZEROCOPY=0 to fall back to the copying path."
            )));
        }

        let t_len = self.meta.t_len();
        let n_score = self.layout.n_score;
        let dims: Vec<usize> = match tensor.dtype() {
            ValueType::Tensor { shape, .. } => shape.iter().map(|&d| d as usize).collect(),
            other => {
                return Err(CrfError::Run(format!(
                    "encoder output has non-tensor type {other:?}"
                )));
            }
        };
        if dims != [t_len, batch, n_score] {
            return Err(CrfError::BadShape {
                got: dims,
                t: t_len,
                batch,
                n_score,
            });
        }

        let ptr = tensor.data_ptr() as u64;
        // SAFETY: `ptr` is onnxruntime's own CUDA allocation for this output —
        // residency and shape are both checked immediately above, so it is
        // `t_len * batch * n_score` f32 on device 0, the device the lattice
        // context binds. `synchronize_outputs` has retired the producing stream.
        // `outputs` owns the allocation and is alive across the call, and nothing
        // else aliases it while the session lock is held. The decode overwrites
        // it in place with the log-posteriors, which is sound because we are its
        // only reader and onnxruntime fully rewrites the buffer on the next run.
        match scan {
            // SAFETY: as above. The scan reads the same buffer before pass 1
            // overwrites it and introduces no aliasing of its own.
            Some((tables, logp, mean)) => unsafe {
                lattice.decode_device_time_major_with_refs(
                    ptr,
                    batch,
                    t_len,
                    &self.alphabet,
                    tables,
                    logp,
                    mean,
                )
            },
            None => unsafe { lattice.decode_device_time_major(ptr, batch, t_len, &self.alphabet) },
        }
    }

    /// Encode one batch on the device and decode it on the CPU, halving and
    /// retrying on device out-of-memory exactly as [`run_batch`](Self::run_batch)
    /// does — the batch axis is independent, so a split batch decodes to the
    /// same sequences as a whole one.
    fn run_and_decode(&self, rows: &[&[f32]], backend: Backend) -> Result<Vec<String>, CrfError> {
        match self.try_run_and_decode(rows, backend) {
            // `Run` is an encode OOM, `Decode` a device-decode OOM — the GPU
            // decode cannot split a time-major batch itself, so halving here is
            // what handles it, and it shrinks the encode at the same time.
            Err(CrfError::Run(msg) | CrfError::Decode(msg)) if is_oom(&msg) && rows.len() > 1 => {
                let half = rows.len().div_ceil(2);
                let mut first = self.run_and_decode(&rows[..half], backend)?;
                first.extend(self.run_and_decode(&rows[half..], backend)?);
                Ok(first)
            }
            other => other,
        }
    }

    /// Encode on the GPU, then decode on the CPU across rayon workers.
    ///
    /// `prepped[i] == None` (a read with no usable window) yields `None` and
    /// never reaches the device.
    ///
    /// Encode and decode alternate one device batch at a time rather than
    /// encoding everything first. `batch_rows` bounds only the *device*-side
    /// activations; the scores coming back are 1 MB per read, so holding a
    /// whole caller batch would retain gigabytes of host memory for an Arrow
    /// batch of a few thousand reads. Interleaving caps that at `batch_rows`
    /// reads' worth regardless of how many reads are handed in.
    /// Build the constrained lattices for a reference panel — see
    /// [`CrfEncoder::ref_chains`](super::encoder::CrfEncoder::ref_chains).
    pub fn ref_chains(&self, seqs: &[&[u8]]) -> Result<RefChains, CrfError> {
        Ok(RefChains::build(&self.layout, &self.alphabet, seqs)?)
    }

    /// [`Self::basecall_batch`], additionally scoring every reference in
    /// `chains` against each read (`log P(reference | signal)`).
    ///
    /// Fully on the device when a CUDA lattice is available: the scan is its
    /// own kernel, running between the two decode passes because it needs the
    /// raw scores that pass 1 overwrites (#241). Only `n_refs` floats per read
    /// come back.
    ///
    /// It falls back to the host decode when the panel does not fit the
    /// kernel's shared memory, and on a CPU-lattice build. That path copies
    /// the whole score tensor out and cost +57% wall and +5.5 cores when it
    /// was the only path there was (#297) — worth avoiding, still correct.
    pub fn basecall_batch_with_refs(
        &self,
        prepped: &[Option<Vec<f32>>],
        chains: &RefChains,
    ) -> Result<Vec<Option<ScoredDecode>>, CrfError> {
        let valid: Vec<usize> = (0..prepped.len())
            .filter(|&i| prepped[i].is_some())
            .collect();

        let backend = Backend::best_for(&self.layout);
        let mut out = vec![None; prepped.len()];

        for idx in valid.chunks(self.batch_rows()) {
            let rows: Vec<&[f32]> = idx
                .iter()
                .map(|&i| prepped[i].as_deref().unwrap())
                .collect();
            for (&i, scored) in idx
                .iter()
                .zip(self.run_and_decode_with_refs(&rows, backend, chains)?)
            {
                out[i] = Some(scored);
            }
        }
        Ok(out)
    }

    /// The scoring counterpart of [`Self::run_and_decode`], including its
    /// halve-on-OOM retry — an encode OOM is a property of the batch size, not
    /// of what the host does with the scores afterwards.
    fn run_and_decode_with_refs(
        &self,
        rows: &[&[f32]],
        backend: Backend,
        chains: &RefChains,
    ) -> Result<Vec<ScoredDecode>, CrfError> {
        match self.try_run_and_decode_with_refs(rows, backend, chains) {
            Err(CrfError::Run(msg) | CrfError::Decode(msg)) if is_oom(&msg) && rows.len() > 1 => {
                let half = rows.len().div_ceil(2);
                let mut first = self.run_and_decode_with_refs(&rows[..half], backend, chains)?;
                first.extend(self.run_and_decode_with_refs(&rows[half..], backend, chains)?);
                Ok(first)
            }
            other => other,
        }
    }

    fn try_run_and_decode_with_refs(
        &self,
        rows: &[&[f32]],
        backend: Backend,
        chains: &RefChains,
    ) -> Result<Vec<ScoredDecode>, CrfError> {
        // The device path, when there is one. Without it `--ref-scores` copies
        // the whole score tensor to the host and runs the CPU decode: measured
        // at +57% wall and +5.5 cores with the card idle (#297). With it, only
        // `batch * n_refs` floats plus one path score per read come back.
        if let Some(lattice) = &self.lattice {
            let tables = {
                let mut cache = self
                    .ref_scan
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let want = panel_fingerprint(chains);
                if cache.as_ref().is_none_or(|(fp, _)| *fp != want) {
                    // A panel too wide for the kernel's shared memory fails
                    // here, which is not fatal — it means the CPU scan below.
                    *cache = lattice
                        .upload_ref_chains(chains)
                        .ok()
                        .map(|t| (want, Arc::new(t)));
                }
                cache.as_ref().map(|(_, t)| Arc::clone(t))
            };
            if let Some(tables) = tables {
                let tables = tables.as_ref();
                let (mut logp, mut mean) = (Vec::new(), Vec::new());
                let seqs = if self.zero_copy {
                    self.run_zero_copy(rows, lattice, Some((tables, &mut logp, &mut mean)))?
                } else {
                    self.run_raw(rows, |data, t_len, _batch, _n_score| {
                        lattice.decode_time_major_with_refs(
                            data,
                            t_len,
                            &self.alphabet,
                            tables,
                            &mut logp,
                            &mut mean,
                        )
                    })?
                };
                let n_refs = tables.n_refs();
                let bad = |what: &str| CrfError::Decode(format!("reference scan returned {what}"));
                return seqs
                    .into_iter()
                    .enumerate()
                    .map(|(b, sequence)| {
                        Ok(ScoredDecode {
                            sequence,
                            ref_logp: logp
                                .get(b * n_refs..(b + 1) * n_refs)
                                .ok_or_else(|| bad("fewer score rows than reads"))?
                                .to_vec(),
                            mean_logpost: *mean
                                .get(b)
                                .ok_or_else(|| bad("fewer path scores than reads"))?,
                        })
                    })
                    .collect();
            }
        }

        self.run_raw(rows, |data, t_len, batch, n_score| {
            (0..batch)
                .into_par_iter()
                .map_init(CrfScratch::new, |scratch, b| {
                    // Decode straight out of the time-major buffer. Read `b`'s
                    // rows live at `(t * batch + b) * n_score` and are each
                    // contiguous, so the decode only needs the stride between
                    // them — it copies every row into its own scratch anyway.
                    // Gathering them into a private buffer first cost 1.5 MB
                    // read plus 1.5 MB written per read, ~1.5 GB per 512-read
                    // call, to change a stride and nothing else (#297).
                    let mut ref_logp = Vec::with_capacity(chains.len());
                    let sequence = decode_with_refs_strided(
                        &self.layout,
                        &self.alphabet,
                        &data[b * n_score..],
                        t_len,
                        batch * n_score,
                        scratch,
                        backend,
                        chains,
                        &mut ref_logp,
                    )
                    .map_err(|e| CrfError::Decode(e.to_string()))?;
                    Ok(ScoredDecode {
                        sequence,
                        ref_logp,
                        mean_logpost: scratch.path_score() / t_len.max(1) as f32,
                    })
                })
                .collect()
        })
    }

    pub fn basecall_batch(
        &self,
        prepped: &[Option<Vec<f32>>],
    ) -> Result<Vec<Option<String>>, CrfError> {
        let valid: Vec<usize> = (0..prepped.len())
            .filter(|&i| prepped[i].is_some())
            .collect();

        let backend = Backend::best_for(&self.layout);
        let mut out = vec![None; prepped.len()];

        for idx in valid.chunks(self.batch_rows()) {
            // Borrowed, not cloned: `prepped` outlives the call, and these
            // windows are `chunk` floats each on the way to a device copy.
            let rows: Vec<&[f32]> = idx
                .iter()
                .map(|&i| prepped[i].as_deref().unwrap())
                .collect();
            for (&i, seq) in idx.iter().zip(self.run_and_decode(&rows, backend)?) {
                out[i] = Some(seq);
            }
        }
        Ok(out)
    }
}

/// Split a time-major `[t_len, batch, n_score]` buffer into one contiguous
/// score buffer per read.
///
/// This is the only genuinely new index arithmetic on the GPU path — the CPU
/// path pins batch to 1, where time-major and per-read layout coincide, so it
/// never exercises the stride. Getting it wrong would interleave reads rather
/// than fail, producing plausible sequences attributed to the wrong read IDs,
/// which is why it is a free function with its own test rather than a closure
/// buried in the onnxruntime call.
///
/// Parallel because this transpose is *large*, not because it is clever: at the
/// RNA004 geometry one 512-read device batch is 786 MB in and 786 MB out, plus
/// one 1.5 MB allocation per read. Serially that made the encoder thread, not
/// the device, the pipeline's bottleneck — the fused GPU producer refused to
/// scale past ~700 reads/s on any thread count while the CPU-encoder path kept
/// scaling to 1150. `into_par_iter()` over the batch axis is order-preserving,
/// so the caller's `idx` alignment is unaffected.
fn split_time_major(data: &[f32], t_len: usize, batch: usize, n_score: usize) -> Vec<Vec<f32>> {
    debug_assert_eq!(data.len(), t_len * batch * n_score);
    (0..batch)
        .into_par_iter()
        .map(|b| {
            let mut per_read = Vec::with_capacity(t_len * n_score);
            for t in 0..t_len {
                let off = (t * batch + b) * n_score;
                per_read.extend_from_slice(&data[off..off + n_score]);
            }
            per_read
        })
        .collect()
}

/// onnxruntime and the CUDA driver both surface device OOM as a message rather
/// than a typed error, and they word it differently.
///
/// The BFC-arena wording is the one that matters in practice and was missing:
/// onnxruntime reports its own allocator failures as
/// `"Failed to allocate memory for requested buffer of size N"`, with no
/// "out of memory" anywhere in it. Because this returned `false` for that, the
/// halve-and-retry above never fired and a batch that merely needed splitting
/// killed the run instead — reproduced at `ESCAPEPOD_CRF_GPU_BATCH_ROWS=2048`
/// on an A30, where the zero-copy path keeps the scores resident and so reaches
/// the arena limit sooner than the copying path does.
fn is_oom(msg: &str) -> bool {
    // Shared with the lattice decode so the two cannot drift; `_` is folded to
    // ` ` there so the driver's CUDA_ERROR_OUT_OF_MEMORY matches too.
    super::lattice_gpu::is_oom(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oom_detection_covers_the_shapes_ort_reports() {
        assert!(is_oom("CUDA failure 2: out of memory"));
        assert!(is_oom("cudaErrorMemoryAllocation"));
        // The BFC-arena wording, verbatim from a real failure at
        // ESCAPEPOD_CRF_GPU_BATCH_ROWS=2048. Note it never says "out of memory":
        // missing this turned a batch that only needed halving into a dead run.
        assert!(is_oom(
            "Non-zero status code returned while running Pad node. Name:'/9/Pad' \
             Status Message: bfc_arena.cc:358 void* \
             onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) \
             Failed to allocate memory for requested buffer of size 5551104000"
        ));
        assert!(!is_oom("invalid input shape"));
        assert!(!is_oom("unexpected output shape: expected time-major"));
    }

    /// Each read must come back with exactly the values the encoder emitted
    /// for it, in timestep order — not its neighbour's.
    #[test]
    fn split_time_major_recovers_each_read() {
        let (t_len, batch, n_score) = (7, 5, 3);
        // Encode (t, b, c) into the value so a mis-strided read is obvious.
        let data: Vec<f32> = (0..t_len * batch * n_score)
            .map(|i| {
                let (t, b, c) = (i / (batch * n_score), (i / n_score) % batch, i % n_score);
                (t * 10_000 + b * 100 + c) as f32
            })
            .collect();

        let split = split_time_major(&data, t_len, batch, n_score);
        assert_eq!(split.len(), batch);
        for (b, read) in split.iter().enumerate() {
            assert_eq!(read.len(), t_len * n_score);
            for t in 0..t_len {
                for c in 0..n_score {
                    assert_eq!(
                        read[t * n_score + c],
                        (t * 10_000 + b * 100 + c) as f32,
                        "read {b}, t {t}, score {c}"
                    );
                }
            }
        }
    }

    /// Batch 1 is the case the CPU path takes, where time-major and per-read
    /// layout are the same buffer. Pinning it means the two paths cannot drift.
    #[test]
    fn split_time_major_is_the_identity_for_batch_one() {
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let split = split_time_major(&data, 10, 1, 4);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0], data);
    }

    #[test]
    fn batch_rows_honours_the_env_override() {
        // Default when unset/garbage; the env var is process-global so this
        // only asserts the parse rules, not a specific ambient value.
        assert!(resolve_batch_rows(1) > 0);
    }

    /// The whole point of the budget: N workers on one device must not ask it
    /// for N times the memory. Skipped when the env override is set, since that
    /// is defined to win.
    #[test]
    fn workers_on_a_device_divide_its_row_budget() {
        if std::env::var("ESCAPEPOD_CRF_GPU_BATCH_ROWS").is_ok() {
            return;
        }
        let one = resolve_batch_rows(1);
        assert_eq!(one, DEVICE_ROW_BUDGET);
        // The out-of-the-box default assumes the fused layout, so a caller that
        // never calls `share_device_with` keeps the 512 rows it had before this
        // was a budget rather than silently doubling.
        assert_eq!(resolve_batch_rows(DEFAULT_WORKERS_PER_DEVICE), 512);
        assert_eq!(resolve_batch_rows(2), one / 2);
        assert_eq!(resolve_batch_rows(4), one / 4);
        // Total in flight stays within the budget as workers scale.
        for w in [1usize, 2, 3, 4, 8] {
            assert!(
                resolve_batch_rows(w) * w <= DEVICE_ROW_BUDGET.max(64 * w),
                "{w} workers exceed the device budget"
            );
        }
        // ...but never split so far that per-call overhead dominates.
        assert_eq!(resolve_batch_rows(1024), 64);
    }
}

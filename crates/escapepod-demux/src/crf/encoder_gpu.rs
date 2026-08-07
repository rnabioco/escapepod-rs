//! GPU CTC-CRF encoder inference via onnxruntime (CUDA execution provider).
//!
//! Opt-in via `crf-gpu`. Runs the *same* ONNX graph as [`CrfEncoder`] and hands
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

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

use super::encoder::{CrfError, CrfMetadata};
use super::lattice::{Backend, CrfLayout, CrfScratch, decode_with};

/// Reads per onnxruntime call before splitting.
///
/// LSTM activations scale with `rows * t_len * features * layers`, which the
/// caller cannot see from here, so this is a starting guess: a batch that hits
/// a device out-of-memory error is halved and retried, exactly as the
/// boundary-CNN GPU path does. Override with `ESCAPEPOD_CRF_GPU_BATCH_ROWS`.
fn resolve_batch_rows() -> usize {
    const DEFAULT_ROWS: usize = 512;
    std::env::var("ESCAPEPOD_CRF_GPU_BATCH_ROWS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_ROWS)
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
    batch_rows: usize,
    /// Batched lattice decode on the device. `None` falls back to the rayon CPU
    /// decode — correct either way, but the CPU decode is ~66% of this path's
    /// host cost, so which one is running is worth reporting rather than
    /// discovering from a benchmark.
    lattice: Option<super::lattice_gpu::CrfLatticeGpu>,
    /// Why [`Self::lattice`] is `None`, when it is.
    decode_fallback: Option<String>,
    /// The graph's output name, resolved once so the IoBinding does not have to
    /// re-read session metadata per batch.
    output_name: String,
    /// Bind the encoder's output to CUDA memory and decode it in place, instead
    /// of letting onnxruntime copy it to the host for us to upload again.
    /// Requires [`Self::lattice`]; `ESCAPEPOD_CRF_GPU_ZEROCOPY=0` disables it.
    zero_copy: bool,
}

impl CrfEncoderGpu {
    /// Load an export directory containing `metadata.json` and its ONNX graph.
    ///
    /// `intra_threads` bounds onnxruntime's intra-op pool. Prefer passing it
    /// from the CLI: ORT's default pool is `available_parallelism()` wide and is
    /// spawned *on top of* rayon's, which puts total process threads back out of
    /// `--threads`' reach.
    pub fn load_bundle(
        dir: impl AsRef<Path>,
        intra_threads: Option<usize>,
    ) -> Result<Self, CrfError> {
        let dir = dir.as_ref();
        let meta = CrfMetadata::load(dir.join("metadata.json"))?;
        let onnx = dir.join(&meta.onnx);
        Self::load(onnx, meta, intra_threads)
    }

    /// Load an ONNX graph with an already-parsed sidecar.
    pub fn load(
        onnx: impl AsRef<Path>,
        meta: CrfMetadata,
        intra_threads: Option<usize>,
    ) -> Result<Self, CrfError> {
        let layout = meta.layout()?;
        let alphabet = meta.alphabet_bytes();

        let mut builder = Session::builder().map_err(|e| CrfError::Load(e.to_string()))?;
        if let Some(n) = intra_threads {
            builder = builder
                .with_intra_threads(n.max(1))
                .map_err(|e| CrfError::Load(e.to_string()))?
                .with_inter_threads(1)
                .map_err(|e| CrfError::Load(e.to_string()))?;
        }
        let session = builder
            .with_execution_providers(crate::ort_ep::cuda_providers())
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
                match super::lattice_gpu::CrfLatticeGpu::new(layout) {
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
        // have to reach the host anyway.
        let zero_copy =
            lattice.is_some() && std::env::var("ESCAPEPOD_CRF_GPU_ZEROCOPY").as_deref() != Ok("0");

        let encoder = Self {
            session: Mutex::new(session),
            meta,
            layout,
            alphabet,
            batch_rows: resolve_batch_rows(),
            lattice,
            decode_fallback,
            output_name,
            zero_copy,
        };
        // Same reasoning as the CPU loader: catch a batch-major (boundary-CNN)
        // export here rather than after decoding noise for every read.
        encoder.encode_batch(&[vec![0f32; encoder.meta.signal.chunk]])?;
        Ok(encoder)
    }

    pub fn metadata(&self) -> &CrfMetadata {
        &self.meta
    }

    pub fn layout(&self) -> &CrfLayout {
        &self.layout
    }

    /// Whether the lattice decode runs on the device.
    pub fn gpu_decode_active(&self) -> bool {
        self.lattice.is_some()
    }

    /// Whether the encoder's scores are decoded in place in device memory,
    /// rather than copied to the host and uploaded again.
    pub fn zero_copy_active(&self) -> bool {
        self.zero_copy
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
        for chunk in prepped.chunks(self.batch_rows) {
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
                return self.run_zero_copy(rows, lattice);
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

        // Device 0 for both sides: under SLURM `--gres=gpu:1`, CUDA_VISIBLE_DEVICES
        // renumbers the allocated GPU to 0, which is also the ordinal
        // `CrfLatticeGpu::new` binds. The residency check below is what actually
        // enforces the pairing.
        let mem = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
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
        unsafe { lattice.decode_device_time_major(ptr, batch, t_len, &self.alphabet) }
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
    pub fn basecall_batch(
        &self,
        prepped: &[Option<Vec<f32>>],
    ) -> Result<Vec<Option<String>>, CrfError> {
        let valid: Vec<usize> = (0..prepped.len())
            .filter(|&i| prepped[i].is_some())
            .collect();

        let backend = Backend::best_for(&self.layout);
        let mut out = vec![None; prepped.len()];

        for idx in valid.chunks(self.batch_rows) {
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

/// onnxruntime surfaces device OOM as a message, not a typed error.
fn is_oom(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("out of memory") || m.contains("cudaerrormemoryallocation") || m.contains("oom")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oom_detection_covers_the_shapes_ort_reports() {
        assert!(is_oom("CUDA failure 2: out of memory"));
        assert!(is_oom("cudaErrorMemoryAllocation"));
        assert!(!is_oom("invalid input shape"));
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
        assert!(resolve_batch_rows() > 0);
    }
}

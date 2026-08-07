//! GPU CTC-CRF encoder inference via onnxruntime (CUDA execution provider).
//!
//! Opt-in via `crf-gpu`. Runs the *same* ONNX graph as [`CrfEncoder`] and hands
//! the scores to the *same* [`lattice`](super::lattice) decode, so the only
//! thing that changes is where the LSTM stack executes.
//!
//! # Why the decode stays on the CPU
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
//! With them, the decode is small enough that overlapping it across rayon
//! workers while the GPU runs the next batch keeps the device fed.
//!
//! That split also mirrors what the boundary-CNN work concluded: the lattice
//! decode is sequential in time with a 256-wide inner dimension, so it is a poor
//! fit for the device compared to the encoder's dense matmuls.
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

        let encoder = Self {
            session: Mutex::new(session),
            meta,
            layout,
            alphabet,
            batch_rows: resolve_batch_rows(),
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

    /// Encode one batch on the device and decode it on the CPU, halving and
    /// retrying on device out-of-memory exactly as [`run_batch`](Self::run_batch)
    /// does — the batch axis is independent, so a split batch decodes to the
    /// same sequences as a whole one.
    fn run_and_decode(&self, rows: &[&[f32]], backend: Backend) -> Result<Vec<String>, CrfError> {
        match self.try_run_and_decode(rows, backend) {
            Err(CrfError::Run(msg)) if is_oom(&msg) && rows.len() > 1 => {
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

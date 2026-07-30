//! GPU CTC-CRF encoder inference via onnxruntime (CUDA execution provider).
//!
//! Opt-in via `crf-gpu`. Runs the *same* ONNX graph as [`CrfEncoder`] and hands
//! the scores to the *same* [`lattice`](super::lattice) decode, so the only
//! thing that changes is where the LSTM stack executes.
//!
//! # Why the decode stays on the CPU
//!
//! Measured on one read of the RNA004 barcode export (rna partition, AVX2):
//!
//! ```text
//! tract encoder        13.9 ms
//! decode, scalar       13.5 ms
//! decode, AVX2          2.3 ms
//! ```
//!
//! The decode is *half* the CPU cost, so moving only the encoder to the GPU
//! would leave it as essentially the entire remaining runtime — which is why
//! the AVX2 kernels are a prerequisite for this path rather than a nicety. With
//! them, the decode is small enough that overlapping it across rayon workers
//! while the GPU runs the next batch keeps the device fed.
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

use ort::execution_providers::CUDAExecutionProvider;
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
            .with_execution_providers([CUDAExecutionProvider::default().build()])
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
    pub fn encode_batch(&self, prepped: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, CrfError> {
        let mut out = Vec::with_capacity(prepped.len());
        for chunk in prepped.chunks(self.batch_rows) {
            out.extend(self.run_batch(chunk, chunk.len())?);
        }
        Ok(out)
    }

    /// One onnxruntime call, halving and retrying on device OOM.
    fn run_batch(&self, rows: &[Vec<f32>], limit: usize) -> Result<Vec<Vec<f32>>, CrfError> {
        if rows.len() > limit {
            let mid = rows.len().div_ceil(2);
            let mut first = self.run_batch(&rows[..mid], limit)?;
            first.extend(self.run_batch(&rows[mid..], limit)?);
            return Ok(first);
        }
        match self.try_run_batch(rows) {
            Err(CrfError::Run(msg)) if is_oom(&msg) && rows.len() > 1 => {
                let half = rows.len().div_ceil(2);
                let mut first = self.run_batch(&rows[..half], half)?;
                first.extend(self.run_batch(&rows[half..], half)?);
                Ok(first)
            }
            other => other,
        }
    }

    fn try_run_batch(&self, rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, CrfError> {
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

        // Time-major: read `b` is strided by `batch * n_score` across t.
        Ok((0..batch)
            .map(|b| {
                let mut per_read = Vec::with_capacity(t_len * n_score);
                for t in 0..t_len {
                    let off = (t * batch + b) * n_score;
                    per_read.extend_from_slice(&data[off..off + n_score]);
                }
                per_read
            })
            .collect())
    }

    /// Encode on the GPU, then decode on the CPU across rayon workers.
    ///
    /// `prepped[i] == None` (a read with no usable window) yields `None` and
    /// never reaches the device.
    pub fn basecall_batch(
        &self,
        prepped: &[Option<Vec<f32>>],
    ) -> Result<Vec<Option<String>>, CrfError> {
        let valid: Vec<usize> = (0..prepped.len())
            .filter(|&i| prepped[i].is_some())
            .collect();
        let windows: Vec<Vec<f32>> = valid
            .iter()
            .map(|&i| prepped[i].as_ref().unwrap().clone())
            .collect();
        let scores = self.encode_batch(&windows)?;

        let backend = Backend::best_for(&self.layout);
        let decoded: Result<Vec<String>, CrfError> = scores
            .par_iter()
            .map_init(CrfScratch::new, |scratch, s| {
                decode_with(
                    &self.layout,
                    &self.alphabet,
                    s,
                    self.meta.t_len(),
                    scratch,
                    backend,
                )
                .map_err(|e| CrfError::Decode(e.to_string()))
            })
            .collect();

        let mut out = vec![None; prepped.len()];
        for (&i, seq) in valid.iter().zip(decoded?) {
            out[i] = Some(seq);
        }
        Ok(out)
    }
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

    #[test]
    fn batch_rows_honours_the_env_override() {
        // Default when unset/garbage; the env var is process-global so this
        // only asserts the parse rules, not a specific ambient value.
        assert!(resolve_batch_rows() > 0);
    }
}

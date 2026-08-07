//! GPU CTC-CRF lattice decode — the batched counterpart to [`super::lattice`].
//!
//! Same two passes, same edge indexing, same tie-breaking; the difference is
//! that reads are decoded as one batch instead of one per rayon worker. See
//! [`super::lattice_gpu_kernel`] for why batching is what makes this a good GPU
//! fit when decoding a *single* read is not.
//!
//! Compiles its kernels with NVRTC at load time, exactly like
//! `escapepod_signal::dtw::cuda` — the CUDA driver and libnvrtc are needed at
//! run time, no toolkit at build time.
//!
//! The contract is the SIMD backends' contract: **same sequence**, not
//! bit-identical. `same_sequence_as_cpu` in the tests pins it against the scalar
//! reference over synthetic lattices.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use super::encoder::CrfError;
use super::lattice::CrfLayout;
use super::lattice_gpu_kernel as k;

/// Threads per block for every kernel here.
///
/// Must be a power of two — the block reductions in the posterior and Viterbi
/// kernels halve the stride — and must equal `n_states` for the forward and
/// backward sweeps, which map one thread to one state. Every bonito CRF has
/// `n_states == 256`; a model with a different state count is rejected at load
/// rather than silently mis-launched.
const THREADS: usize = 256;

/// Widest `n_edges` the kernels size their per-thread accumulators for
/// (`CRF_MAX_EDGES`).
const MAX_EDGES: usize = 8;

/// How a score buffer packs its `(read, timestep)` rows.
///
/// The kernels address rows by stride rather than assuming a packing, so the
/// encoder's output can be decoded exactly as onnxruntime produced it. That
/// matters more than it sounds: de-interleaving `[T][batch]` into `[batch][T]`
/// on the host is 1.5 MB per read of pure memory movement, which is what the
/// CPU decode path had to do before it could start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreOrder {
    /// `[batch][t_len][n_score]`.
    BatchMajor,
    /// `[t_len][batch][n_score]` — onnxruntime's own time-major output.
    TimeMajor,
}

impl ScoreOrder {
    /// `(stride_b, stride_t)` in units of `n_score`-wide rows.
    fn strides(self, batch: usize, t_len: usize) -> (i32, i32) {
        match self {
            Self::BatchMajor => (t_len as i32, 1),
            Self::TimeMajor => (1, batch as i32),
        }
    }
}

/// A CUDA device with the lattice kernels loaded.
///
/// Build once and reuse: NVRTC compilation and module load are amortised across
/// every batch.
pub struct CrfLatticeGpu {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    layout: CrfLayout,
    /// Held so the device's primary context outlives the stream and module.
    _ctx: Arc<CudaContext>,
}

impl CrfLatticeGpu {
    /// Initialise on CUDA device 0 and compile the kernels.
    pub fn new(layout: CrfLayout) -> Result<Self, CrfError> {
        Self::new_on_device(layout, 0)
    }

    /// Initialise on a specific CUDA device ordinal.
    pub fn new_on_device(layout: CrfLayout, ordinal: usize) -> Result<Self, CrfError> {
        if layout.n_states != THREADS {
            return Err(CrfError::Load(format!(
                "GPU lattice decode needs n_states == {THREADS}, got {}. This model's \
                 geometry ({}^{}) is valid but unsupported on the GPU path; it decodes \
                 correctly on the CPU.",
                layout.n_states, layout.n_base, layout.state_len
            )));
        }
        if layout.n_edges > MAX_EDGES {
            return Err(CrfError::Load(format!(
                "GPU lattice decode supports at most {MAX_EDGES} edges, got {}",
                layout.n_edges
            )));
        }
        let ctx =
            CudaContext::new(ordinal).map_err(|e| CrfError::Load(format!("CUDA init: {e}")))?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(k::KERNEL_SRC)
            .map_err(|e| CrfError::Load(format!("CRF lattice kernel compilation: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| CrfError::Load(format!("CRF lattice module load: {e}")))?;
        Ok(Self {
            stream,
            module,
            layout,
            _ctx: ctx,
        })
    }

    pub fn layout(&self) -> &CrfLayout {
        &self.layout
    }

    fn func(&self, name: &'static str) -> Result<CudaFunction, CrfError> {
        self.module
            .load_function(name)
            .map_err(|e| CrfError::Decode(format!("kernel `{name}`: {e}")))
    }

    /// Decode a `[batch][t_len][n_score]` buffer.
    ///
    /// Halves and retries on device out-of-memory: the batch axis is
    /// independent, so a split batch decodes to the same sequences.
    pub fn decode_batch(
        &self,
        scores: &[f32],
        t_len: usize,
        alphabet: &[u8],
    ) -> Result<Vec<String>, CrfError> {
        self.decode(scores, t_len, alphabet, ScoreOrder::BatchMajor)
    }

    /// Decode onnxruntime's time-major `[t_len][batch][n_score]` output as it
    /// stands, with no host-side de-interleave.
    ///
    /// Unlike [`decode_batch`](Self::decode_batch) this cannot split itself on
    /// device out-of-memory — a sub-batch is strided rather than contiguous
    /// here — so the OOM is returned for the caller's own batch halving to
    /// handle. `CrfEncoderGpu::run_and_decode` already does exactly that, and
    /// halving there shrinks the encode as well as the decode.
    pub fn decode_time_major(
        &self,
        scores: &[f32],
        t_len: usize,
        alphabet: &[u8],
    ) -> Result<Vec<String>, CrfError> {
        self.decode(scores, t_len, alphabet, ScoreOrder::TimeMajor)
    }

    fn decode(
        &self,
        scores: &[f32],
        t_len: usize,
        alphabet: &[u8],
        order: ScoreOrder,
    ) -> Result<Vec<String>, CrfError> {
        let n_score = self.layout.n_score;
        if alphabet.len() != self.layout.n_edges {
            return Err(CrfError::Decode(format!(
                "alphabet has {} symbols, expected {}",
                alphabet.len(),
                self.layout.n_edges
            )));
        }
        let per_read = t_len
            .checked_mul(n_score)
            .ok_or_else(|| CrfError::Decode("t_len * n_score overflows".to_string()))?;
        if per_read == 0 || !scores.len().is_multiple_of(per_read) {
            return Err(CrfError::Decode(format!(
                "score buffer is {} floats, not a multiple of t_len * n_score = {per_read}",
                scores.len()
            )));
        }
        let batch = scores.len() / per_read;
        if batch == 0 {
            return Ok(Vec::new());
        }
        self.decode_chunked(scores, batch, t_len, alphabet, per_read, order)
    }

    fn decode_chunked(
        &self,
        scores: &[f32],
        batch: usize,
        t_len: usize,
        alphabet: &[u8],
        per_read: usize,
        order: ScoreOrder,
    ) -> Result<Vec<String>, CrfError> {
        match self.decode_one(scores, batch, t_len, alphabet, order) {
            // Only batch-major buffers can be split here; see decode_time_major.
            Err(CrfError::Decode(msg))
                if is_oom(&msg) && batch > 1 && order == ScoreOrder::BatchMajor =>
            {
                let half = batch.div_ceil(2);
                let mut first = self.decode_chunked(
                    &scores[..half * per_read],
                    half,
                    t_len,
                    alphabet,
                    per_read,
                    order,
                )?;
                first.extend(self.decode_chunked(
                    &scores[half * per_read..],
                    batch - half,
                    t_len,
                    alphabet,
                    per_read,
                    order,
                )?);
                Ok(first)
            }
            other => other,
        }
    }

    /// One batch, one device residency. Every intermediate stays on the device;
    /// only `batch * t_len` bytes of path come back.
    fn decode_one(
        &self,
        scores: &[f32],
        batch: usize,
        t_len: usize,
        alphabet: &[u8],
        order: ScoreOrder,
    ) -> Result<Vec<String>, CrfError> {
        let CrfLayout {
            n_states,
            n_edges,
            n_base,
            n_score,
            ..
        } = self.layout;
        let group = self.layout.group();
        let lattice = batch * (t_len + 1) * n_states;

        let dev = |e: cudarc::driver::DriverError| CrfError::Decode(e.to_string());

        // The encoder's own [dest][edge] order is what every kernel reads, so
        // this is uploaded once and then worked on in place — no transpose pass
        // and no second score buffer. The posterior kernel overwrites it with
        // the log-posteriors that pass 2 consumes.
        let mut work = self.stream.clone_htod(scores).map_err(dev)?;
        let mut alpha = self.stream.alloc_zeros::<f32>(lattice).map_err(dev)?;
        let mut beta = self.stream.alloc_zeros::<f32>(lattice).map_err(dev)?;
        let mut path = self.stream.alloc_zeros::<u8>(batch * t_len).map_err(dev)?;

        let (t_i, ns_i, ne_i, nb_i, g_i) = (
            t_len as i32,
            n_states as i32,
            n_edges as i32,
            n_base as i32,
            group as i32,
        );
        let (sb_i, st_i) = order.strides(batch, t_len);
        // Forward and backward run in the same launch, as blockIdx.y 0 and 1.
        let sweep_cfg = LaunchConfig {
            grid_dim: (batch as u32, 2, 1),
            block_dim: (n_states as u32, 1, 1),
            shared_mem_bytes: (n_states * 4) as u32,
        };
        let step_cfg = LaunchConfig {
            grid_dim: (batch as u32, t_len as u32, 1),
            block_dim: (THREADS as u32, 1, 1),
            // edge scores + one reduction slot per thread (+ an int key slot
            // for the Viterbi argmax, which is the larger of the two).
            shared_mem_bytes: ((n_score + THREADS * 2) * 4) as u32,
        };

        // ---- Pass 1: posteriors under the log semiring ----
        self.sweep(
            &work,
            &mut alpha,
            &mut beta,
            &sweep_cfg,
            t_i,
            ns_i,
            ne_i,
            nb_i,
            g_i,
            sb_i,
            st_i,
            k::SEMIRING_LOG,
        )?;
        {
            let floor = super::lattice::POSTERIOR_FLOOR;
            let f = self.func(k::POSTERIOR_KERNEL_NAME)?;
            let mut b = self.stream.launch_builder(&f);
            b.arg(&mut work)
                .arg(&alpha)
                .arg(&beta)
                .arg(&t_i)
                .arg(&ns_i)
                .arg(&ne_i)
                .arg(&nb_i)
                .arg(&g_i)
                .arg(&sb_i)
                .arg(&st_i)
                .arg(&floor);
            unsafe { b.launch(step_cfg) }.map_err(dev)?;
        }

        // ---- Pass 2: Viterbi over the log-posteriors ----
        self.sweep(
            &work,
            &mut alpha,
            &mut beta,
            &sweep_cfg,
            t_i,
            ns_i,
            ne_i,
            nb_i,
            g_i,
            sb_i,
            st_i,
            k::SEMIRING_MAX,
        )?;
        {
            let f = self.func(k::VITERBI_KERNEL_NAME)?;
            let mut b = self.stream.launch_builder(&f);
            b.arg(&work)
                .arg(&alpha)
                .arg(&beta)
                .arg(&mut path)
                .arg(&t_i)
                .arg(&ns_i)
                .arg(&ne_i)
                .arg(&nb_i)
                .arg(&g_i)
                .arg(&sb_i)
                .arg(&st_i);
            unsafe { b.launch(step_cfg) }.map_err(dev)?;
        }

        self.stream.synchronize().map_err(dev)?;
        let host_path = self.stream.clone_dtoh(&path).map_err(dev)?;

        Ok(host_path
            .chunks(t_len)
            .map(|p| {
                let mut s = String::with_capacity(t_len);
                for &c in p {
                    if c != 0 {
                        s.push(alphabet[c as usize] as char);
                    }
                }
                s
            })
            .collect())
    }

    /// The fused forward+backward launch: one grid, `blockIdx.y` picking the
    /// direction, so both sweeps of a pass go out together.
    #[allow(clippy::too_many_arguments)]
    fn sweep(
        &self,
        scores: &cudarc::driver::CudaSlice<f32>,
        alpha: &mut cudarc::driver::CudaSlice<f32>,
        beta: &mut cudarc::driver::CudaSlice<f32>,
        cfg: &LaunchConfig,
        t: i32,
        ns: i32,
        ne: i32,
        nb: i32,
        g: i32,
        sb: i32,
        st: i32,
        semiring: i32,
    ) -> Result<(), CrfError> {
        let f = self.func(k::SWEEP_KERNEL_NAME)?;
        let mut b = self.stream.launch_builder(&f);
        b.arg(scores)
            .arg(alpha)
            .arg(beta)
            .arg(&t)
            .arg(&ns)
            .arg(&ne)
            .arg(&nb)
            .arg(&g)
            .arg(&sb)
            .arg(&st)
            .arg(&semiring);
        unsafe { b.launch(*cfg) }
            .map(|_| ())
            .map_err(|e| CrfError::Decode(e.to_string()))
    }
}

/// CUDA reports device OOM as a message, like onnxruntime does.
fn is_oom(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("out of memory") || m.contains("outofmemory") || m.contains("oom")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oom_detection_covers_the_driver_shapes() {
        assert!(is_oom("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(is_oom(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)"
        ));
        assert!(!is_oom("invalid device ordinal"));
    }
}

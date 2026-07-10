//! GPU-accelerated banded DTW distance matrix, plus **experimental** GPU
//! primitives for the demux signal chain (SVB16 decode, t-test fingerprint,
//! LLR adapter detect).
//!
//! Enabled via the `gpu` feature. Compiles CUDA kernels at runtime using
//! NVRTC (no `nvcc` / CUDA toolkit required at build time — only the CUDA
//! driver and libnvrtc at runtime).
//!
//! ## Why the prep kernels are experimental (not in the default pipeline)
//!
//! The DTW distance matrix below is a genuine GPU win — dense, regular, f32
//! work. The signal-processing *prep* kernels ([`GpuDtwContext::decode_svb16_batch`],
//! [`GpuDtwContext::fingerprint_batch`], [`GpuDtwContext::detect_adapter_batch`] /
//! [`GpuDtwContext::detect_adapter_batch_block`]) are parity-validated against
//! their CPU references but are **not** wired into `escpod demux`, because
//! measurement shows prep belongs on the CPU:
//!
//! - Prep cost splits ~detect **85%** / fingerprint **12.5%** / decode **2.8%**.
//! - Detect (the dominant stage) is f64-/`ln`-heavy, branchy, with a serial
//!   cumsum dependency — the opposite of what a GPU accelerates. On an A30 the
//!   best block-per-read detect reaches ~2.7–4.5× a *single* CPU thread, but
//!   still loses to a 48-core CPU by ~11–18× (f64-throughput bound, within ~3×
//!   of the card's f64 roofline). Fusing prep onto the GPU made the pipeline
//!   ~15× slower overall.
//! - The real prep speed-up is CPU-side: `--downscale 10` (the WarpDemuX
//!   default) cuts detect ~5.3× with ~98–99.9% barcode agreement.
//!
//! These kernels are kept as validated, reusable primitives — useful on
//! few-core GPU hosts or future fast-f64 cards, and as the record of the
//! measurement. The [`GpuDtwContext`] DTW + SVM classify path remains the
//! production GPU usage.
//!
//! ## Typical use
//!
//! ```no_run
//! # #[cfg(feature = "gpu")] {
//! use escapepod_signal::dtw::GpuDtwContext;
//!
//! let ctx = GpuDtwContext::new()?;
//! let queries = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
//! let refs    = vec![vec![1.0_f32, 2.0, 3.0]];
//! let d = ctx.distance_matrix(&queries, &refs, Some(10))?;
//! assert_eq!(d.shape(), &[2, 1]);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod kernel;
mod llr_detect_block_kernel;
mod llr_detect_kernel;
mod svb16_kernel;
mod svm_kernels;
mod ttest_fp_kernel;

pub use svm_kernels::{
    KERNEL_SRC as SVM_KERNEL_SRC, MODULE_NAME as SVM_MODULE_NAME, OVO_DECISION_KERNEL_NAME,
    RBF_KERNEL_NAME,
};

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileError, compile_ptx};
use ndarray::Array2;
use thiserror::Error;

/// Module name registered with the GPU device for the DTW kernel.
/// Re-exported so downstream crates (escapepod-demux's GPU pipeline)
/// can grab the `CudaFunction` handle via `GpuDtwContext::function`.
pub const DTW_MODULE_NAME: &str = "escapepod_gpu_dtw";
/// Kernel name registered with the GPU device. See `DTW_MODULE_NAME`.
pub const DTW_KERNEL_NAME: &str = "dtw_matrix_kernel";

const MODULE_NAME: &str = DTW_MODULE_NAME;
const KERNEL_NAME: &str = DTW_KERNEL_NAME;

/// Errors from the GPU DTW path.
#[derive(Debug, Error)]
pub enum GpuDtwError {
    /// No CUDA device, missing driver, OOM, kernel launch failure, etc.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    /// NVRTC failed to compile the kernel source.
    #[error("CUDA kernel compilation failed: {0}")]
    Compile(#[from] CompileError),
    /// Kernel not found after loading the module (should not happen in practice).
    #[error("kernel `{0}` not found after module load")]
    KernelMissing(&'static str),
    /// Input too large to address with i32 offsets (the kernel uses `int`).
    #[error("input exceeds GPU kernel limits: {what}")]
    InputTooLarge { what: &'static str },
}

/// A handle to a CUDA device with the DTW kernel pre-loaded.
///
/// Build once and reuse across many `distance_matrix` calls — NVRTC compilation
/// and module load are amortized that way.
pub struct GpuDtwContext {
    /// CUDA context (device's primary context). Kept alive for the lifetime of
    /// the handle and exposed so downstream crates can reach the device.
    ctx: Arc<CudaContext>,
    /// Default stream — every memory transfer and kernel launch runs here, so
    /// operations are serialized without explicit synchronization.
    stream: Arc<CudaStream>,
    // cudarc 0.19 has no device-global function registry keyed by module name;
    // functions are loaded from a module handle, so we hold each module.
    dtw_module: Arc<CudaModule>,
    svm_module: Arc<CudaModule>,
    svb16_module: Arc<CudaModule>,
    ttest_fp_module: Arc<CudaModule>,
    llr_module: Arc<CudaModule>,
    llr_block_module: Arc<CudaModule>,
}

impl GpuDtwContext {
    /// Initialize on CUDA device 0, compile the DTW kernel via NVRTC, and load it.
    pub fn new() -> Result<Self, GpuDtwError> {
        Self::new_on_device(0)
    }

    /// Initialize on a specific CUDA device ordinal. Compiles **both** the
    /// DTW kernel and the post-DTW SVM helper kernels (RBF + OvO decision)
    /// at startup — escapepod-demux's GPU classify path needs them, and
    /// keeping the load here means downstream callers don't have to learn
    /// a second NVRTC compile cycle.
    pub fn new_on_device(ordinal: usize) -> Result<Self, GpuDtwError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();

        let dtw_module = ctx.load_module(compile_ptx(kernel::KERNEL_SRC)?)?;
        let svm_module = ctx.load_module(compile_ptx(SVM_KERNEL_SRC)?)?;
        let svb16_module = ctx.load_module(compile_ptx(svb16_kernel::KERNEL_SRC)?)?;
        let ttest_fp_module = ctx.load_module(compile_ptx(ttest_fp_kernel::KERNEL_SRC)?)?;
        let llr_module = ctx.load_module(compile_ptx(llr_detect_kernel::KERNEL_SRC)?)?;
        let llr_block_module =
            ctx.load_module(compile_ptx(llr_detect_block_kernel::KERNEL_SRC)?)?;

        Ok(Self {
            ctx,
            stream,
            dtw_module,
            svm_module,
            svb16_module,
            ttest_fp_module,
            llr_module,
            llr_block_module,
        })
    }

    /// Borrow the CUDA context. Lets downstream crates query the device or
    /// build their own streams sharing this context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Borrow the default stream. Downstream crates (escapepod-demux's GPU
    /// classify path) issue their memory transfers and kernel launches here so
    /// they share ordering with this context's own work — replaces the old
    /// `device()` accessor now that memory/launch live on the stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Look up a pre-loaded kernel by `(module_name, kernel_name)`.
    /// Used by `escapepod-demux` to grab the SVM helper kernels.
    pub fn function(
        &self,
        module: &'static str,
        kernel_name: &'static str,
    ) -> Result<CudaFunction, GpuDtwError> {
        let module = if module == DTW_MODULE_NAME {
            &self.dtw_module
        } else if module == SVM_MODULE_NAME {
            &self.svm_module
        } else if module == svb16_kernel::MODULE_NAME {
            &self.svb16_module
        } else if module == ttest_fp_kernel::MODULE_NAME {
            &self.ttest_fp_module
        } else if module == llr_detect_kernel::MODULE_NAME {
            &self.llr_module
        } else if module == llr_detect_block_kernel::MODULE_NAME {
            &self.llr_block_module
        } else {
            return Err(GpuDtwError::KernelMissing(kernel_name));
        };
        module.load_function(kernel_name).map_err(GpuDtwError::from)
    }

    /// Compute an (n_queries × n_refs) banded DTW distance matrix on the GPU.
    ///
    /// Inputs and outputs mirror the CPU [`crate::dtw::dtw_distance_matrix`]
    /// exactly, up to f32 summation order; tolerance in parity tests is
    /// `1e-4 * max(1, |cpu|)`.
    pub fn distance_matrix<Q, R>(
        &self,
        queries: &[Q],
        references: &[R],
        window: Option<usize>,
    ) -> Result<Array2<f32>, GpuDtwError>
    where
        Q: AsRef<[f32]>,
        R: AsRef<[f32]>,
    {
        let n_q = queries.len();
        let n_r = references.len();

        if n_q == 0 || n_r == 0 {
            return Ok(Array2::zeros((n_q, n_r)));
        }

        if n_q > i32::MAX as usize || n_r > i32::MAX as usize {
            return Err(GpuDtwError::InputTooLarge {
                what: "too many queries/references for i32 grid dims",
            });
        }

        // Shared memory sizing needs both dimensions: `a_s` + three rolling
        // diagonal buffers are indexed by the query position `i`, while
        // `b_s` caches the reference in shared memory.
        let max_n: usize = queries.iter().map(|q| q.as_ref().len()).max().unwrap_or(0);
        let max_m: usize = references
            .iter()
            .map(|r| r.as_ref().len())
            .max()
            .unwrap_or(0);
        if max_n > i32::MAX as usize {
            return Err(GpuDtwError::InputTooLarge {
                what: "query length exceeds i32::MAX",
            });
        }
        if max_m > i32::MAX as usize {
            return Err(GpuDtwError::InputTooLarge {
                what: "reference length exceeds i32::MAX",
            });
        }

        let (flat_q, q_off) = flatten_with_offsets(queries)?;
        let (flat_r, r_off) = flatten_with_offsets(references)?;

        let stream = &self.stream;
        let queries_dev = stream.clone_htod(&flat_q)?;
        let q_off_dev = stream.clone_htod(&q_off)?;
        let refs_dev = stream.clone_htod(&flat_r)?;
        let r_off_dev = stream.clone_htod(&r_off)?;
        let mut out_dev = stream.alloc_zeros::<f32>(n_q * n_r)?;

        // Sakoe–Chiba band. `-1` = no constraint; otherwise pass an i32.
        let window_i32: i32 = match window {
            None => -1,
            Some(w) if w > i32::MAX as usize => {
                return Err(GpuDtwError::InputTooLarge {
                    what: "window exceeds i32::MAX",
                });
            }
            Some(w) => w as i32,
        };

        // Shared memory: a_s[max_n] + b_s[max_m] + 3 * d_k[max_n + 1] floats.
        let floats: u32 = (max_n as u32)
            .checked_add(max_m as u32)
            .and_then(|v| v.checked_add(3u32.checked_mul((max_n as u32).saturating_add(1))?))
            .ok_or(GpuDtwError::InputTooLarge {
                what: "shared memory size overflowed u32",
            })?;
        let shared_mem_bytes: u32 = floats
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or(GpuDtwError::InputTooLarge {
                what: "shared memory size overflowed u32",
            })?;

        // One warp per block — anti-diagonal DP cooperates within a single
        // warp, so block-internal sync is `__syncwarp()` rather than the
        // heavier `__syncthreads()`. Single-warp blocks also lift the
        // resident-blocks-per-SM ceiling, which is what saturates
        // grid-level parallelism when fingerprints are short (≤ a few
        // hundred samples). The inner stride loop handles bands wider
        // than the warp.
        const THREADS: u32 = 32;

        let cfg = LaunchConfig {
            grid_dim: (n_q as u32, n_r as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes,
        };

        let func = self.function(MODULE_NAME, KERNEL_NAME)?;

        let n_q_i = n_q as i32;
        let n_r_i = n_r as i32;
        let max_n_i = max_n as i32;
        let max_m_i = max_m as i32;
        // No warping penalty for the plain distance matrix (used by training /
        // reference-mode). The SVM batch path launches this kernel directly
        // with the model's penalty.
        let penalty = 0.0f32;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&queries_dev)
            .arg(&q_off_dev)
            .arg(&refs_dev)
            .arg(&r_off_dev)
            .arg(&mut out_dev)
            .arg(&n_q_i)
            .arg(&n_r_i)
            .arg(&max_n_i)
            .arg(&max_m_i)
            .arg(&window_i32)
            .arg(&penalty);
        unsafe {
            builder.launch(cfg)?;
        }

        let host_out = stream.clone_dtoh(&out_dev)?;
        Ok(Array2::from_shape_vec((n_q, n_r), host_out).expect("shape matches"))
    }

    /// Decode a batch of SVB16 byte streams on the GPU, one thread per read.
    ///
    /// `reads` is `(svb16_bytes, sample_count)` per read — exactly the input
    /// the CPU [`escapepod_pod5::compression::svb16::decode`] takes, after the
    /// host has zstd-decompressed each VBZ chunk. Returns the decoded `i16`
    /// signals in input order. Bit-exact with the scalar CPU decoder.
    ///
    /// This host method dtoh's the decoded signal for testing/standalone use;
    /// the fused pipeline keeps the decoded signal on-device and feeds it
    /// straight into the detect/fingerprint kernels.
    pub fn decode_svb16_batch(
        &self,
        reads: &[(&[u8], usize)],
    ) -> Result<Vec<Vec<i16>>, GpuDtwError> {
        if reads.is_empty() {
            return Ok(Vec::new());
        }
        let n = reads.len();
        if n > i32::MAX as usize {
            return Err(GpuDtwError::InputTooLarge {
                what: "too many reads for i32 grid dims",
            });
        }

        let total_bytes: usize = reads.iter().map(|(b, _)| b.len()).sum();
        let total_out: usize = reads.iter().map(|(_, c)| *c).sum();

        let mut data_flat: Vec<u8> = Vec::with_capacity(total_bytes);
        let mut data_off: Vec<i64> = Vec::with_capacity(n + 1);
        let mut counts: Vec<i32> = Vec::with_capacity(n);
        let mut out_off: Vec<i64> = Vec::with_capacity(n + 1);
        data_off.push(0);
        out_off.push(0);
        let mut dacc: i64 = 0;
        let mut oacc: i64 = 0;
        for (bytes, count) in reads {
            data_flat.extend_from_slice(bytes);
            dacc += bytes.len() as i64;
            data_off.push(dacc);
            counts.push(
                i32::try_from(*count).map_err(|_| GpuDtwError::InputTooLarge {
                    what: "sample count exceeds i32::MAX",
                })?,
            );
            oacc += *count as i64;
            out_off.push(oacc);
        }

        let stream = &self.stream;
        let data_dev = stream.clone_htod(&data_flat)?;
        let data_off_dev = stream.clone_htod(&data_off)?;
        let counts_dev = stream.clone_htod(&counts)?;
        let out_off_dev = stream.clone_htod(&out_off)?;
        let mut out_dev = stream.alloc_zeros::<i16>(total_out.max(1))?;

        const THREADS: u32 = 256;
        let blocks = (n as u32).div_ceil(THREADS);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = self.function(svb16_kernel::MODULE_NAME, svb16_kernel::KERNEL_NAME)?;
        let n_i = n as i32;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&data_dev)
            .arg(&data_off_dev)
            .arg(&counts_dev)
            .arg(&out_off_dev)
            .arg(&mut out_dev)
            .arg(&n_i);
        unsafe {
            builder.launch(cfg)?;
        }

        let host_out = stream.clone_dtoh(&out_dev)?;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let s = out_off[i] as usize;
            let e = out_off[i + 1] as usize;
            result.push(host_out[s..e].to_vec());
        }
        Ok(result)
    }

    /// Extract barcode fingerprints on the GPU for the demux path
    /// (`keep_last = Some(n)`, z-score, no dwell).
    ///
    /// `reads` is `(full_signal, adapter_start, adapter_end)` per read — the
    /// kernel slices `[adapter_start - 100 .. adapter_end + 100]` (clamped),
    /// clips ±5·MAD, t-test segments, z-scores the segment means, and keeps the
    /// last `keep_last`. Returns `None` for reads whose adapter region is too
    /// small to segment (matching `extract_fingerprint_from_signal`).
    ///
    /// Reads are length-sorted and split into memory-bounded sub-batches so
    /// rectangular scratch (`max_len` slots/read) tracks each sub-batch rather
    /// than the global maximum.
    pub fn fingerprint_batch(
        &self,
        reads: &[(&[i16], usize, usize)],
        window_width: usize,
        num_segments: usize,
        min_separation: usize,
        keep_last: usize,
    ) -> Result<Vec<Option<Vec<f64>>>, GpuDtwError> {
        const PAD: usize = 100;
        // Scratch budget in "cells" (read × max_len). ~36 B/cell of scratch →
        // ~2.3 GB at 64M cells; comfortable on a 24 GB card.
        const CELL_BUDGET: usize = 64 * 1024 * 1024;

        let n = reads.len();
        let mut result: Vec<Option<Vec<f64>>> = vec![None; n];
        if n == 0 {
            return Ok(result);
        }

        // Per-read slice length L (after pad+clamp); 0 ⇒ skipped (stays None).
        let slice_len = |&(sig, a_s, a_e): &(&[i16], usize, usize)| -> usize {
            let ss = a_s.saturating_sub(PAD);
            let se = (a_e + PAD).min(sig.len());
            se.saturating_sub(ss)
        };

        // Length-sort indices (ascending) so each sub-batch has a tight max_len.
        let mut order: Vec<usize> = (0..n)
            .filter(|&i| slice_len(&reads[i]) >= window_width * 2)
            .collect();
        order.sort_unstable_by_key(|&i| slice_len(&reads[i]));

        let mut start = 0usize;
        while start < order.len() {
            // Grow the sub-batch while cells stay under budget. Since `order`
            // is ascending, max_len is the last element's slice length.
            let mut end = start + 1;
            while end < order.len() {
                let max_len = slice_len(&reads[order[end]]);
                if (end - start + 1) * max_len.max(1) > CELL_BUDGET {
                    break;
                }
                end += 1;
            }
            let batch = &order[start..end];
            self.fingerprint_subbatch(
                reads,
                batch,
                window_width,
                num_segments,
                min_separation,
                keep_last,
                PAD,
                &mut result,
            )?;
            start = end;
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn fingerprint_subbatch(
        &self,
        reads: &[(&[i16], usize, usize)],
        batch: &[usize],
        window_width: usize,
        num_segments: usize,
        min_separation: usize,
        keep_last: usize,
        pad: usize,
        result: &mut [Option<Vec<f64>>],
    ) -> Result<(), GpuDtwError> {
        let bn = batch.len();
        if bn == 0 {
            return Ok(());
        }
        let slice_len = |i: usize| -> usize {
            let (sig, a_s, a_e) = reads[i];
            let ss = a_s.saturating_sub(pad);
            let se = (a_e + pad).min(sig.len());
            se - ss
        };
        let max_len = batch.iter().map(|&i| slice_len(i)).max().unwrap_or(0);
        if max_len == 0 {
            return Ok(());
        }
        let max_cand = max_len.saturating_sub(2 * window_width).max(1);

        // Flatten full signals + per-read offsets / boundaries.
        let total_sig: usize = batch.iter().map(|&i| reads[i].0.len()).sum();
        let mut sig_flat: Vec<i16> = Vec::with_capacity(total_sig);
        let mut sig_off: Vec<i64> = Vec::with_capacity(bn + 1);
        let mut adapter_start: Vec<i32> = Vec::with_capacity(bn);
        let mut adapter_end: Vec<i32> = Vec::with_capacity(bn);
        sig_off.push(0);
        let mut acc: i64 = 0;
        for &i in batch {
            let (sig, a_s, a_e) = reads[i];
            sig_flat.extend_from_slice(sig);
            acc += sig.len() as i64;
            sig_off.push(acc);
            adapter_start.push(a_s as i32);
            adapter_end.push(a_e as i32);
        }

        // Packed scalar params (cudarc 0.12 caps launches at 12 tuple args).
        let mut params = vec![0i32; ttest_fp_kernel::N_PARAMS];
        params[ttest_fp_kernel::P_N_READS] = bn as i32;
        params[ttest_fp_kernel::P_WW] = window_width as i32;
        params[ttest_fp_kernel::P_MIN_SEP] = min_separation as i32;
        params[ttest_fp_kernel::P_NUM_CP] = num_segments.saturating_sub(1) as i32;
        params[ttest_fp_kernel::P_KEEP_LAST] = keep_last as i32;
        params[ttest_fp_kernel::P_MAX_LEN] = max_len as i32;
        params[ttest_fp_kernel::P_MAX_CAND] = max_cand as i32;
        params[ttest_fp_kernel::P_PAD] = pad as i32;

        let stream = &self.stream;
        let sig_dev = stream.clone_htod(&sig_flat)?;
        let sig_off_dev = stream.clone_htod(&sig_off)?;
        let as_dev = stream.clone_htod(&adapter_start)?;
        let ae_dev = stream.clone_htod(&adapter_end)?;
        let params_dev = stream.clone_htod(&params)?;

        // Merged scratch: A|B share one f32 region, cumsum|cumsum_sq one f64.
        let mut scratch_f32 = stream.alloc_zeros::<f32>(bn * 2 * max_len)?;
        let mut scratch_f64 = stream.alloc_zeros::<f64>(bn * 2 * (max_len + 1))?;
        let mut tscores = stream.alloc_zeros::<f64>(bn * max_cand)?;
        let mut peaks = stream.alloc_zeros::<i32>(bn * max_cand)?;
        let mut out = stream.alloc_zeros::<f32>(bn * keep_last)?;
        let mut out_len = stream.alloc_zeros::<i32>(bn)?;

        const THREADS: u32 = 128;
        let blocks = (bn as u32).div_ceil(THREADS);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = self.function(ttest_fp_kernel::MODULE_NAME, ttest_fp_kernel::KERNEL_NAME)?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&sig_dev)
            .arg(&sig_off_dev)
            .arg(&as_dev)
            .arg(&ae_dev)
            .arg(&params_dev)
            .arg(&mut scratch_f32)
            .arg(&mut scratch_f64)
            .arg(&mut tscores)
            .arg(&mut peaks)
            .arg(&mut out)
            .arg(&mut out_len);
        unsafe {
            builder.launch(cfg)?;
        }

        let host_out = stream.clone_dtoh(&out)?;
        let host_len = stream.clone_dtoh(&out_len)?;

        for (b, &i) in batch.iter().enumerate() {
            let k = host_len[b] as usize; // 0 == no fingerprint produced
            if k == 0 {
                continue;
            }
            let base = b * keep_last;
            result[i] = Some(host_out[base..base + k].iter().map(|&v| v as f64).collect());
        }
        Ok(())
    }

    /// Detect adapter boundaries on the GPU (LLR path), mirroring the demux
    /// `normalize_signal` → optional `downscale` → `detect_adapter` pipeline.
    ///
    /// Returns `(adapter_start, adapter_end)` per read (already scaled back from
    /// the downscaled domain). `(0, 0)` means no adapter detected. Reads are
    /// split into memory-bounded sub-batches (scratch is jagged at the full
    /// read length, since LLR scans the whole read).
    pub fn detect_adapter_batch(
        &self,
        signals: &[&[i16]],
        min_adapter: usize,
        border_trim: usize,
        downscale: usize,
    ) -> Result<Vec<(usize, usize)>, GpuDtwError> {
        self.detect_adapter_batch_impl(signals, min_adapter, border_trim, downscale, false)
    }

    /// Block-per-read variant of [`Self::detect_adapter_batch`]: parallelises
    /// `best_split` across a CUDA block and z-score normalizes via a parallel
    /// reduction. Same result as the thread-per-read detector (LLR is affine-
    /// invariant), but built for throughput rather than simplicity.
    pub fn detect_adapter_batch_block(
        &self,
        signals: &[&[i16]],
        min_adapter: usize,
        border_trim: usize,
        downscale: usize,
    ) -> Result<Vec<(usize, usize)>, GpuDtwError> {
        self.detect_adapter_batch_impl(signals, min_adapter, border_trim, downscale, true)
    }

    fn detect_adapter_batch_impl(
        &self,
        signals: &[&[i16]],
        min_adapter: usize,
        border_trim: usize,
        downscale: usize,
        block: bool,
    ) -> Result<Vec<(usize, usize)>, GpuDtwError> {
        // ~40 B/sample of scratch (3×f32 + 2×f64); ~2.5 GB at 64M samples.
        const SAMPLE_BUDGET: usize = 64 * 1024 * 1024;

        let n = signals.len();
        let mut result = vec![(0usize, 0usize); n];
        if n == 0 {
            return Ok(result);
        }

        let mut start = 0usize;
        while start < n {
            let mut end = start;
            let mut acc = 0usize;
            while end < n {
                let len = signals[end].len();
                if end > start && acc + len > SAMPLE_BUDGET {
                    break;
                }
                acc += len;
                end += 1;
            }
            self.detect_subbatch(
                &signals[start..end],
                min_adapter,
                border_trim,
                downscale,
                block,
                &mut result[start..end],
            )?;
            start = end;
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_subbatch(
        &self,
        signals: &[&[i16]],
        min_adapter: usize,
        border_trim: usize,
        downscale: usize,
        block: bool,
        result: &mut [(usize, usize)],
    ) -> Result<(), GpuDtwError> {
        let bn = signals.len();
        if bn == 0 {
            return Ok(());
        }
        let total: usize = signals.iter().map(|s| s.len()).sum();
        let mut sig_flat: Vec<i16> = Vec::with_capacity(total);
        let mut off: Vec<i64> = Vec::with_capacity(bn + 1);
        off.push(0);
        let mut acc: i64 = 0;
        for s in signals {
            sig_flat.extend_from_slice(s);
            acc += s.len() as i64;
            off.push(acc);
        }
        let total = total.max(1);

        let mut params = vec![0i32; llr_detect_kernel::N_PARAMS];
        params[llr_detect_kernel::P_N_READS] = bn as i32;
        params[llr_detect_kernel::P_MIN_ADAPTER] = min_adapter as i32;
        params[llr_detect_kernel::P_BORDER_TRIM] = border_trim as i32;
        params[llr_detect_kernel::P_DOWNSCALE] = downscale.max(1) as i32;

        let stream = &self.stream;
        let sig_dev = stream.clone_htod(&sig_flat)?;
        let off_dev = stream.clone_htod(&off)?;
        let params_dev = stream.clone_htod(&params)?;
        let mut norm = stream.alloc_zeros::<f32>(total)?;
        let mut processed = stream.alloc_zeros::<f32>(total)?;
        let mut medbuf = stream.alloc_zeros::<f32>(total)?;
        let mut cumsum = stream.alloc_zeros::<f64>(total)?;
        let mut cumsumsq = stream.alloc_zeros::<f64>(total)?;
        let mut out_start = stream.alloc_zeros::<i32>(bn)?;
        let mut out_end = stream.alloc_zeros::<i32>(bn)?;

        let (cfg, func) = if block {
            // One block per read, fixed 128 threads (matches BLK in the kernel).
            let cfg = LaunchConfig {
                grid_dim: (bn as u32, 1, 1),
                block_dim: (llr_detect_block_kernel::BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = self.function(
                llr_detect_block_kernel::MODULE_NAME,
                llr_detect_block_kernel::KERNEL_NAME,
            )?;
            (cfg, func)
        } else {
            const THREADS: u32 = 64;
            let blocks = (bn as u32).div_ceil(THREADS);
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = self.function(
                llr_detect_kernel::MODULE_NAME,
                llr_detect_kernel::KERNEL_NAME,
            )?;
            (cfg, func)
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&sig_dev)
            .arg(&off_dev)
            .arg(&params_dev)
            .arg(&mut norm)
            .arg(&mut processed)
            .arg(&mut medbuf)
            .arg(&mut cumsum)
            .arg(&mut cumsumsq)
            .arg(&mut out_start)
            .arg(&mut out_end);
        unsafe {
            builder.launch(cfg)?;
        }

        let hs = stream.clone_dtoh(&out_start)?;
        let he = stream.clone_dtoh(&out_end)?;
        for b in 0..bn {
            result[b] = (hs[b].max(0) as usize, he[b].max(0) as usize);
        }
        Ok(())
    }
}

/// One-shot convenience: build a context, run the matrix, drop the context.
///
/// Prefer `GpuDtwContext::new()` + `distance_matrix()` when doing multiple
/// passes — NVRTC compilation is ~100 ms.
pub fn dtw_distance_matrix_gpu<Q, R>(
    queries: &[Q],
    references: &[R],
    window: Option<usize>,
) -> Result<Array2<f32>, GpuDtwError>
where
    Q: AsRef<[f32]>,
    R: AsRef<[f32]>,
{
    let ctx = GpuDtwContext::new()?;
    ctx.distance_matrix(queries, references, window)
}

fn flatten_with_offsets<T: AsRef<[f32]>>(v: &[T]) -> Result<(Vec<f32>, Vec<i32>), GpuDtwError> {
    let total: usize = v.iter().map(|x| x.as_ref().len()).sum();
    if total > i32::MAX as usize {
        return Err(GpuDtwError::InputTooLarge {
            what: "flattened buffer exceeds i32::MAX",
        });
    }
    let mut flat = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(v.len() + 1);
    offsets.push(0i32);
    let mut acc: i32 = 0;
    for item in v {
        let s = item.as_ref();
        flat.extend_from_slice(s);
        acc += s.len() as i32;
        offsets.push(acc);
    }
    Ok((flat, offsets))
}

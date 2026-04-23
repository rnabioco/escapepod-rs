//! GPU-accelerated banded DTW distance matrix.
//!
//! Enabled via the `gpu` feature. Compiles a CUDA kernel at runtime using
//! NVRTC (no `nvcc` / CUDA toolkit required at build time — only the CUDA
//! driver and libnvrtc at runtime).
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
mod svm_kernels;

pub use svm_kernels::{
    KERNEL_SRC as SVM_KERNEL_SRC, MODULE_NAME as SVM_MODULE_NAME, OVO_DECISION_KERNEL_NAME,
    RBF_KERNEL_NAME,
};

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, DriverError, LaunchAsync, LaunchConfig};
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
    device: Arc<CudaDevice>,
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
        let device = CudaDevice::new(ordinal)?;

        let dtw_ptx = compile_ptx(kernel::KERNEL_SRC)?;
        device.load_ptx(dtw_ptx, MODULE_NAME, &[KERNEL_NAME])?;

        let svm_ptx = compile_ptx(SVM_KERNEL_SRC)?;
        device.load_ptx(
            svm_ptx,
            SVM_MODULE_NAME,
            &[RBF_KERNEL_NAME, OVO_DECISION_KERNEL_NAME],
        )?;

        Ok(Self { device })
    }

    /// Borrow the underlying `Arc<CudaDevice>`. Lets downstream crates
    /// (escapepod-demux's GPU classify path) launch their own kernels —
    /// notably the RBF + OvO decision kernels we pre-load above — without
    /// having to spin up a second context.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Look up a pre-loaded kernel by `(module_name, kernel_name)`.
    /// Used by `escapepod-demux` to grab the SVM helper kernels.
    pub fn function(
        &self,
        module: &'static str,
        kernel_name: &'static str,
    ) -> Result<CudaFunction, GpuDtwError> {
        self.device
            .get_func(module, kernel_name)
            .ok_or(GpuDtwError::KernelMissing(kernel_name))
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

        let dev = &self.device;
        let queries_dev = dev.htod_sync_copy(&flat_q)?;
        let q_off_dev = dev.htod_sync_copy(&q_off)?;
        let refs_dev = dev.htod_sync_copy(&flat_r)?;
        let r_off_dev = dev.htod_sync_copy(&r_off)?;
        let mut out_dev = dev.alloc_zeros::<f32>(n_q * n_r)?;

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

        let func = dev
            .get_func(MODULE_NAME, KERNEL_NAME)
            .ok_or(GpuDtwError::KernelMissing(KERNEL_NAME))?;

        unsafe {
            func.launch(
                cfg,
                (
                    &queries_dev,
                    &q_off_dev,
                    &refs_dev,
                    &r_off_dev,
                    &mut out_dev,
                    n_q as i32,
                    n_r as i32,
                    max_n as i32,
                    max_m as i32,
                    window_i32,
                ),
            )?;
        }

        let host_out = dev.dtoh_sync_copy(&out_dev)?;
        Ok(Array2::from_shape_vec((n_q, n_r), host_out).expect("shape matches"))
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

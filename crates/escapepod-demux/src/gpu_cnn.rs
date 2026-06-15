//! Batched GPU BoundariesCNN inference (cudarc + NVRTC), feature `gpu` +
//! `cnn-detect`. Ports the verified CPU conv stack in `adapter_cnn_compute`
//! to two NVRTC kernels and runs many reads through the network at once —
//! that batching is where GPU beats the per-read `tract-onnx` path.
//!
//! Exactness: reads are zero-padded to a common length, but intermediate
//! activations past a read's true length are bias-driven (nonzero), so each
//! kernel takes a per-read `valid_in` length and treats inputs at/after it as
//! zero. With that mask the batched scores match the per-read CPU reference
//! bit-for-bit (up to f32 summation order), verified in `tests/gpu_cnn.rs`.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rayon::prelude::*;
use thiserror::Error;

use crate::adapter_cnn::{AdapterCnnConfig, mean_pool, median_mad_normalize};
use crate::adapter_cnn_compute::{C, CnnComputeError, CnnWeights, K, conv_out_len, convt_out_len};

const MODULE: &str = "adapter_cnn";
const CONV1D: &str = "conv1d_relu";
const CONVT: &str = "conv_transpose1d";

// Scalar params packed into one `int[7]` device array to stay well under
// cudarc's launch-tuple arity limit: d = [N, Cin, Lin, Cout, Lout, stride, pad].
const KERNEL_SRC: &str = r#"
#define KK 7   // kernel width is fixed for this network

// Direct Conv1d + ReLU (every Conv1d in BoundariesCNN is followed by ReLU).
// One thread per output element [n, co, o]. `valid_in[n]` is the read's true
// input length; inputs at/after it are treated as zero (matches the per-read
// CPU reference's boundary padding).
extern "C" __global__ void conv1d_relu(
    const float* __restrict__ in,    // [N, Cin, Lin]
    const float* __restrict__ w,     // [Cout, Cin, 7]
    const float* __restrict__ b,     // [Cout]
    const int*   __restrict__ valid_in,
    float*       __restrict__ out,   // [N, Cout, Lout]
    const int*   __restrict__ d)
{
    int N=d[0], Cin=d[1], Lin=d[2], Cout=d[3], Lout=d[4], stride=d[5], pad=d[6];
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * Cout * Lout;
    if (idx >= total) return;
    int o  = (int)(idx % Lout);
    int co = (int)((idx / Lout) % Cout);
    int n  = (int)(idx / ((long)Lout * Cout));
    int vin = valid_in[n];
    float acc = b[co];
    int base = o * stride - pad;
    const float* in_n = in + (long)n * Cin * Lin;
    for (int ci = 0; ci < Cin; ++ci) {
        const float* wrow = w + ((long)co * Cin + ci) * KK;
        const float* irow = in_n + (long)ci * Lin;
        for (int k = 0; k < KK; ++k) {
            int ii = base + k;
            if (ii >= 0 && ii < vin) acc += irow[ii] * wrow[k];
        }
    }
    if (acc < 0.f) acc = 0.f;
    out[idx] = acc;
}

// Direct ConvTranspose1d (gather form). Weight layout [Cin, Cout, 7].
extern "C" __global__ void conv_transpose1d(
    const float* __restrict__ in,    // [N, Cin, Lin]
    const float* __restrict__ w,     // [Cin, Cout, 7]
    const float* __restrict__ b,     // [Cout]
    const int*   __restrict__ valid_in,
    float*       __restrict__ out,   // [N, Cout, Lout]
    const int*   __restrict__ d)
{
    int N=d[0], Cin=d[1], Lin=d[2], Cout=d[3], Lout=d[4], stride=d[5], pad=d[6];
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * Cout * Lout;
    if (idx >= total) return;
    int o  = (int)(idx % Lout);
    int co = (int)((idx / Lout) % Cout);
    int n  = (int)(idx / ((long)Lout * Cout));
    int vin = valid_in[n];
    float acc = b[co];
    const float* in_n = in + (long)n * Cin * Lin;
    for (int k = 0; k < KK; ++k) {
        int num = o + pad - k;
        if (num < 0 || (num % stride) != 0) continue;
        int i = num / stride;
        if (i >= vin) continue;
        for (int ci = 0; ci < Cin; ++ci) {
            acc += in_n[(long)ci * Lin + i] * w[((long)ci * Cout + co) * KK + k];
        }
    }
    out[idx] = acc;
}
"#;

#[derive(Debug, Error)]
pub enum GpuCnnError {
    #[error("weights: {0}")]
    Weights(#[from] CnnComputeError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("NVRTC compile: {0}")]
    Compile(#[from] cudarc::nvrtc::CompileError),
    #[error("kernel `{0}` missing from module")]
    KernelMissing(&'static str),
}

/// Batched GPU BoundariesCNN. Build once (NVRTC compile + weight upload
/// amortize), then call [`Self::detect_adapter_end_batch`] per batch of reads.
pub struct GpuCnn {
    device: Arc<CudaDevice>,
    conv1d: CudaFunction,
    convt: CudaFunction,
    w0: cudarc::driver::CudaSlice<f32>,
    b0: cudarc::driver::CudaSlice<f32>,
    w2: cudarc::driver::CudaSlice<f32>,
    b2: cudarc::driver::CudaSlice<f32>,
    w4: cudarc::driver::CudaSlice<f32>,
    b4: cudarc::driver::CudaSlice<f32>,
    w6: cudarc::driver::CudaSlice<f32>,
    b6: cudarc::driver::CudaSlice<f32>,
    config: AdapterCnnConfig,
}

impl GpuCnn {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GpuCnnError> {
        Self::load_with_config(path, AdapterCnnConfig::default(), 0)
    }

    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
        ordinal: usize,
    ) -> Result<Self, GpuCnnError> {
        let weights = CnnWeights::load(path)?;
        let device = CudaDevice::new(ordinal)?;
        let ptx = compile_ptx(KERNEL_SRC)?;
        device.load_ptx(ptx, MODULE, &[CONV1D, CONVT])?;
        let conv1d = device
            .get_func(MODULE, CONV1D)
            .ok_or(GpuCnnError::KernelMissing(CONV1D))?;
        let convt = device
            .get_func(MODULE, CONVT)
            .ok_or(GpuCnnError::KernelMissing(CONVT))?;
        Ok(Self {
            w0: device.htod_sync_copy(&weights.w0)?,
            b0: device.htod_sync_copy(&weights.b0)?,
            w2: device.htod_sync_copy(&weights.w2)?,
            b2: device.htod_sync_copy(&weights.b2)?,
            w4: device.htod_sync_copy(&weights.w4)?,
            b4: device.htod_sync_copy(&weights.b4)?,
            w6: device.htod_sync_copy(&weights.w6)?,
            b6: device.htod_sync_copy(&weights.b6)?,
            device,
            conv1d,
            convt,
            config,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_conv1d(
        &self,
        input: &cudarc::driver::CudaSlice<f32>,
        w: &cudarc::driver::CudaSlice<f32>,
        b: &cudarc::driver::CudaSlice<f32>,
        valid_in: &cudarc::driver::CudaSlice<i32>,
        out: &mut cudarc::driver::CudaSlice<f32>,
        n: usize,
        cin: usize,
        lin: usize,
        cout: usize,
        lout: usize,
        stride: usize,
        pad: usize,
    ) -> Result<(), GpuCnnError> {
        let dims: Vec<i32> = vec![
            n as i32,
            cin as i32,
            lin as i32,
            cout as i32,
            lout as i32,
            stride as i32,
            pad as i32,
        ];
        let dims_dev = self.device.htod_sync_copy(&dims)?;
        let total = (n * cout * lout) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        unsafe {
            self.conv1d
                .clone()
                .launch(cfg, (input, w, b, valid_in, out, &dims_dev))?;
        }
        Ok(())
    }

    /// Detect adapter-end for a batch of calibrated-pA reads. Same per-read
    /// contract as [`crate::adapter_cnn::AdapterCnn::detect_adapter_end`];
    /// reads too short to process yield `0`.
    ///
    /// Reads vary enormously in length (nanopore RNA reads can be >500 k
    /// samples), and a batch is zero-padded to its longest member — so naively
    /// batching everything would size the conv buffers to the longest read and
    /// blow past device memory. Reads are therefore sorted by (pooled) length
    /// and split into memory-bounded sub-batches; padding waste stays small and
    /// a single very long read lands in its own tiny batch.
    pub fn detect_adapter_end_batch(&self, signals: &[&[f32]]) -> Result<Vec<usize>, GpuCnnError> {
        let cfg = self.config;
        let n = signals.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // CPU prep: slice + mean-pool + median/MAD normalize, in parallel across
        // reads (rayon) — otherwise a serial scalar prep starves the GPU. Same
        // math as the CPU/tract path.
        let prepped: Vec<Vec<f32>> = signals
            .par_iter()
            .map(|&sig| {
                if sig.len() <= cfg.min_obs_adapter + cfg.downscale_factor {
                    Vec::new()
                } else {
                    let pooled = mean_pool(&sig[cfg.min_obs_adapter..], cfg.downscale_factor);
                    median_mad_normalize(&pooled)
                }
            })
            .collect();

        // Order by pooled length; greedily pack sub-batches so the dominant conv
        // buffers (3 × n_batch × C × L0 f32) stay near ~5 GB. `n_batch * L0` is
        // the proxy we cap; L0 ≈ Lmax/3.
        const MAX_BATCH_L0_ELEMS: usize = 6_000_000;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| prepped[i].len());

        let mut out = vec![0usize; n];
        let mut start = 0usize;
        while start < n {
            let mut end = start;
            // The batch's Lmax is the last (longest) member, since `order` is
            // ascending — so L0 only grows as we extend the batch.
            while end < n {
                let lmax = prepped[order[end]].len().max(1);
                let l0 = conv_out_len(lmax, K, 3, 3);
                let count = end - start + 1;
                if count > 1 && count * l0 > MAX_BATCH_L0_ELEMS {
                    break;
                }
                end += 1;
            }
            let idxs = &order[start..end];
            let batch: Vec<&[f32]> = idxs.iter().map(|&i| prepped[i].as_slice()).collect();
            let ends = self.run_prepped(&batch)?;
            for (&i, e) in idxs.iter().zip(ends) {
                out[i] = e;
            }
            start = end;
        }
        Ok(out)
    }

    /// Run the conv stack on one already-prepped, memory-safe sub-batch and
    /// return per-read adapter-end positions (in the original signal frame).
    fn run_prepped(&self, prepped: &[&[f32]]) -> Result<Vec<usize>, GpuCnnError> {
        let cfg = self.config;
        let n = prepped.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let lmax = prepped.iter().map(|p| p.len()).max().unwrap_or(0).max(1);

        // Flatten into [N, 1, Lmax], zero-padded; record per-read valid lengths.
        let mut flat = vec![0.0f32; n * lmax];
        let mut valid0 = vec![0i32; n]; // input length
        let mut valid_mid = vec![0i32; n]; // length after conv0 (== conv2/4 in)
        for (i, p) in prepped.iter().enumerate() {
            flat[i * lmax..i * lmax + p.len()].copy_from_slice(p);
            valid0[i] = p.len() as i32;
            valid_mid[i] = if p.is_empty() {
                0
            } else {
                conv_out_len(p.len(), K, 3, 3) as i32
            };
        }

        let l0 = conv_out_len(lmax, K, 3, 3);
        let l6 = convt_out_len(l0, K, 3, 3);

        let dev = &self.device;
        let in_dev = dev.htod_sync_copy(&flat)?;
        let valid0_dev = dev.htod_sync_copy(&valid0)?;
        let valid_mid_dev = dev.htod_sync_copy(&valid_mid)?;
        let mut h0 = dev.alloc_zeros::<f32>(n * C * l0)?;
        let mut h2 = dev.alloc_zeros::<f32>(n * C * l0)?;
        let mut h4 = dev.alloc_zeros::<f32>(n * C * l0)?;
        let mut scores = dev.alloc_zeros::<f32>(n * 2 * l6)?;

        // conv0 (1->C, s3, p3, relu); conv2/conv4 (C->C, s1, p3, relu).
        self.launch_conv1d(
            &in_dev,
            &self.w0,
            &self.b0,
            &valid0_dev,
            &mut h0,
            n,
            1,
            lmax,
            C,
            l0,
            3,
            3,
        )?;
        self.launch_conv1d(
            &h0,
            &self.w2,
            &self.b2,
            &valid_mid_dev,
            &mut h2,
            n,
            C,
            l0,
            C,
            l0,
            1,
            3,
        )?;
        self.launch_conv1d(
            &h2,
            &self.w4,
            &self.b4,
            &valid_mid_dev,
            &mut h4,
            n,
            C,
            l0,
            C,
            l0,
            1,
            3,
        )?;

        // convtranspose6 (C->2, s3, p3).
        let cdims: Vec<i32> = vec![n as i32, C as i32, l0 as i32, 2, l6 as i32, 3, 3];
        let cdims_dev = dev.htod_sync_copy(&cdims)?;
        let total = (n * 2 * l6) as u32;
        unsafe {
            self.convt.clone().launch(
                LaunchConfig::for_num_elems(total),
                (
                    &h4,
                    &self.w6,
                    &self.b6,
                    &valid_mid_dev,
                    &mut scores,
                    &cdims_dev,
                ),
            )?;
        }

        let host = dev.dtoh_sync_copy(&scores)?;

        // Per-read argmax on channel 0 over the read's own valid range.
        let search_cap =
            (cfg.max_obs_adapter.saturating_sub(cfg.min_obs_adapter)) / cfg.downscale_factor;
        let out: Vec<usize> = prepped
            .par_iter()
            .enumerate()
            .map(|(i, p)| {
                if p.is_empty() {
                    return 0;
                }
                let l6_i = convt_out_len(conv_out_len(p.len(), K, 3, 3), K, 3, 3);
                let search = search_cap.min(l6_i);
                let ch0 = &host[i * 2 * l6..i * 2 * l6 + l6]; // channel 0 row
                let mut best_idx = 0usize;
                let mut best = f32::NEG_INFINITY;
                for (k, &v) in ch0.iter().take(search).enumerate() {
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
            })
            .collect();
        Ok(out)
    }
}

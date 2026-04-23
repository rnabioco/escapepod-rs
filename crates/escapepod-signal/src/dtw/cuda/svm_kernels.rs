//! CUDA kernels for the post-DTW SVM pipeline.
//!
//! These two kernels let `escpod demux classify --gpu` keep all per-query
//! work on-device. Without them, the host would dtoh a full
//! `(n_queries × n_refs)` f32 distance matrix per chunk just to apply
//! `(-gamma · d^power).exp()` element-wise and then a 6-pair OvO
//! decision_function per row — for the classify_val workload that's
//! ~16 GB transferred per chunk, of which only ~5 MB of OvO decision
//! values is ever consumed downstream. Moving these two stages onto the
//! GPU shrinks the dtoh per chunk by ~3000×.
//!
//! ## Layout
//!
//! Both kernels are designed for the chunk-batched call shape used by
//! [`crate::dtw::cuda::GpuDtwContext`]: queries fan out as `blockIdx.x`,
//! a second axis (refs / pairs) as `blockIdx.y`. Single-warp blocks
//! (32 threads) keep block-internal sync to `__syncwarp()` and let
//! `__shfl_down_sync` do the within-warp reduction, no shared memory
//! needed.
//!
//! ## `rbf_inplace_kernel`
//!
//! `dist[i] ← exp(-gamma · dist[i]^power)`. Embarrassingly parallel —
//! one thread per cell. In-place so we don't allocate a second
//! `(n_q × n_r)` device buffer (16 GB at the default chunk size).
//!
//! ## `ovo_decision_kernel`
//!
//! For each (query, pair), computes
//!     `decision[q, p] = intercept[p] + Σ_s coef[p, s] · kernel[q, s]`
//! reduced over `n_sv` support vectors.
//!
//! `coef` is pre-flattened on the host into a `(n_pairs × n_sv)`
//! row-major f32 table by [`GpuSvmContext`] (escapepod-demux), with
//! the libsvm OvO sign convention baked in:
//!   * for pair `(i, j)` with `i < j`, an SV of class `i` contributes
//!     `dual_coef[j-1][sv]` and an SV of class `j` contributes
//!     `dual_coef[i][sv]`; SVs of any other class contribute 0.
//!
//! That lets the kernel be a flat dot product without per-SV class
//! branching — ~2× the host memory of the sparse representation but
//! O(n_pairs · n_sv) is small (240 KB at 6 pairs × 10k SVs × f32).

pub const MODULE_NAME: &str = "escapepod_gpu_svm";
pub const RBF_KERNEL_NAME: &str = "rbf_inplace_kernel";
pub const OVO_DECISION_KERNEL_NAME: &str = "ovo_decision_kernel";

pub const KERNEL_SRC: &str = r#"
// In-place RBF transform: dist[i] -> exp(-gamma * dist[i]^power).
//
// Grid-strided so a single 1D launch handles arbitrarily large dist
// buffers (the per-chunk distance matrix is up to 4G cells at the
// default chunk size, well over the per-grid hardware limit on x).
extern "C" __global__
void rbf_inplace_kernel(
    float* __restrict__ dist,
    long long n_cells,
    float gamma,
    float power)
{
    long long idx  = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x  * blockDim.x;
    for (long long i = idx; i < n_cells; i += step) {
        float d = dist[i];
        // Special-case power == 1.0 (common WarpDemuX setting): skip
        // powf, which is a software call. Caller is expected to use
        // power == 1.0 for the default config; the general path below
        // is correct for any power.
        float t = (power == 1.0f) ? d : powf(d, power);
        dist[i] = __expf(-gamma * t);
    }
}

// One-vs-One decision values per (query, pair).
//
// Grid:  (n_q, n_pairs, 1). One block per (query, pair).
// Block: 32 threads (single warp). The strided sum across n_sv ends
// with a shfl_down_sync warp reduction; the warp-leader writes the
// final decision[q, p].
//
// `coef` is the flattened (n_pairs × n_sv) row-major table the host
// builds in `GpuSvmContext::new`.
extern "C" __global__
__launch_bounds__(32, 64)
void ovo_decision_kernel(
    const float* __restrict__ kernel,
    int n_q,
    int n_sv,
    int n_pairs,
    const float* __restrict__ coef,
    const float* __restrict__ intercept,
    float*       __restrict__ decisions)
{
    int qi = blockIdx.x;
    int pi = blockIdx.y;
    if (qi >= n_q || pi >= n_pairs) return;

    int tid = threadIdx.x;
    int nth = blockDim.x;

    // Long-arithmetic offsets — at the default chunk size, n_q * n_sv
    // overflows int (215_000 × 10_000 ≈ 2.15e9).
    const float* k_row = kernel + (long long)qi * (long long)n_sv;
    const float* c_row = coef   + (long long)pi * (long long)n_sv;

    float acc = 0.0f;
    for (int s = tid; s < n_sv; s += nth) {
        acc += c_row[s] * k_row[s];
    }

    // Warp-wide sum reduction. Assumes block_dim.x == 32 (single warp).
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }

    if (tid == 0) {
        decisions[(long long)qi * (long long)n_pairs + (long long)pi] =
            acc + intercept[pi];
    }
}
"#;

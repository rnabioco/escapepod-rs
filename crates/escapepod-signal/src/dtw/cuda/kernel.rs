//! CUDA kernel source for banded DTW.
//!
//! Compiled at runtime by NVRTC (via `cudarc::nvrtc::compile_ptx`). Kept as a
//! string constant so the build has no compile-time dependency on `nvcc` or
//! the CUDA toolkit.
//!
//! ## Kernel layout
//!
//! * Grid = `(n_queries, n_refs, 1)`. One block per (query, ref) pair.
//! * Block = 32 threads (one warp). Anti-diagonal cooperation: cells on a
//!   common `d = i + j` are independent given `d - 1` and `d - 2`, so the
//!   inner loop is fully parallel within the warp. Single-warp blocks let
//!   every block-internal sync be a cheap `__syncwarp()` instead of a full
//!   `__syncthreads()`.
//! * Shared memory layout (all `f32`):
//!   `a_s[max_n]  b_s[max_m]  d2[max_n+1]  d1[max_n+1]  d0[max_n+1]` — the
//!   query and reference fingerprints are pulled in once per block so the
//!   DP loop reads from shared memory instead of hitting global memory
//!   every diagonal.
//! * `__launch_bounds__(32, 64)` tells the compiler to keep register usage
//!   low enough to run 64 resident warps per SM (the hardware max on
//!   Ampere), which is what saturates grid-level parallelism for short
//!   fingerprints.

pub const KERNEL_SRC: &str = r#"
// NVRTC compiles without libc headers, so INFINITY isn't available.
// 0x7f800000 is +inf in IEEE-754 f32.
__device__ __forceinline__ float dtw_inf_f() { return __int_as_float(0x7f800000); }

extern "C" __global__
__launch_bounds__(32, 64)
void dtw_matrix_kernel(
    const float* __restrict__ queries,
    const int*   __restrict__ q_offsets,
    const float* __restrict__ refs,
    const int*   __restrict__ r_offsets,
    float*       __restrict__ out,
    int n_q,
    int n_r,
    int max_n,
    int max_m,
    int window,
    float penalty)
{
    int qi = blockIdx.x;
    int rj = blockIdx.y;
    if (qi >= n_q || rj >= n_r) return;

    int q_start = q_offsets[qi];
    int q_end   = q_offsets[qi + 1];
    int r_start = r_offsets[rj];
    int r_end   = r_offsets[rj + 1];

    int n = q_end - q_start;
    int m = r_end - r_start;

    float* out_cell = out + ((long)qi) * n_r + rj;
    int tid = threadIdx.x;
    int nth = blockDim.x;

    if (n == 0 || m == 0) {
        if (tid == 0) *out_cell = dtw_inf_f();
        return;
    }

    const float* a_glob = queries + q_start;
    const float* b_glob = refs    + r_start;

    // Shared memory layout: a_s | b_s | d2 | d1 | d0.
    extern __shared__ float smem[];
    float* a_s = smem;
    float* b_s = smem + max_n;
    float* d2  = smem + max_n + max_m;
    float* d1  = d2 + (max_n + 1);
    float* d0  = d1 + (max_n + 1);

    // Co-load query and reference, and initialize d2 (diagonal 0) and d1
    // (diagonal 1). Threads stride over each buffer; the warp-wide
    // coalesced loads from global memory are why a_glob/b_glob are kept
    // __restrict__.
    for (int i = tid; i < n; i += nth)    a_s[i] = a_glob[i];
    for (int j = tid; j < m; j += nth)    b_s[j] = b_glob[j];
    for (int i = tid; i <= n; i += nth) { d2[i] = dtw_inf_f(); d1[i] = dtw_inf_f(); }
    if (tid == 0) d2[0] = 0.0f;
    __syncwarp();

    int w = (window < 0) ? (n + m) : window;

    // dtaidistance's `penalty` lives in non-squared space; this DP accumulates
    // squared local costs, so each warping (non-diagonal) step adds penalty^2.
    // Mirrors the CPU `dtw_distance_penalty`. penalty == 0 -> pen == 0 (no-op).
    float pen = penalty * penalty;

    // Anti-diagonal DP. Each thread runtime-checks the band on its strided
    // `i` — the band-bound arithmetic is fiddly enough that a straight
    // runtime predicate is cheaper than precomputing bounds per diagonal.
    for (int d = 2; d <= n + m; ++d) {
        // Clear the new current-diagonal buffer so unreached cells stay INF.
        for (int i = tid; i <= n; i += nth) d0[i] = dtw_inf_f();
        __syncwarp();

        for (int i = 1 + tid; i <= n; i += nth) {
            int j = d - i;
            if (j < 1 || j > m) continue;
            int delta = i - j;
            if (delta < 0) delta = -delta;
            if (delta > w) continue;

            float diff = a_s[i - 1] - b_s[j - 1];
            float cost = diff * diff;
            // d1[i-1] (expansion) and d1[i] (compression) are the off-diagonal
            // predecessors and take the penalty; d2[i-1] (match) is diagonal.
            float m1 = fminf(d1[i - 1] + pen, d1[i] + pen);
            float mp = fminf(m1, d2[i - 1]);
            d0[i] = cost + mp;
        }
        __syncwarp();

        // Rotate: d2 <- d1, d1 <- d0, d0 <- (old d2, will be overwritten).
        // Pointers are per-thread; each thread rotates identically, so no
        // sync is needed after this — the next iteration's clear will
        // __syncwarp before any reads of the new `d1` (= old `d0`).
        float* tmp = d2;
        d2 = d1;
        d1 = d0;
        d0 = tmp;
    }

    if (tid == 0) *out_cell = sqrtf(d1[n]);
}
"#;

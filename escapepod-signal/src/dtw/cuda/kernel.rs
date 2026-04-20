//! CUDA kernel source for banded DTW.
//!
//! Compiled at runtime by NVRTC (via `cudarc::nvrtc::compile_ptx`). Kept as a
//! string constant so the build has no compile-time dependency on `nvcc` or
//! the CUDA toolkit.
//!
//! ## Kernel layout (v1)
//!
//! * Grid = `(n_queries, n_refs, 1)`. One block per (query, ref) pair.
//! * Block = `(1, 1, 1)`. A single thread runs the classic two-row banded DP
//!   for its pair. Grid-level parallelism saturates the device: with
//!   thousands of pairs, one-thread-per-block is wasteful per-SM but the
//!   aggregate throughput still dwarfs CPU rayon for the fingerprint sizes
//!   we see in practice (~100–400 f32).
//! * Dynamic shared memory: `2 * (max_ref_len + 1) * sizeof(float)`, holds
//!   the two rolling DP rows.
//!
//! A later revision can switch to anti-diagonal threading (one thread per
//! band column) to better utilize each SM; the kernel signature and host
//! bindings would not change.

pub const KERNEL_SRC: &str = r#"
// NVRTC compiles without libc headers, so `dtw_inf_f()` is not defined. Bit-pattern
// 0x7f800000 is +inf in IEEE-754 f32.
__device__ __forceinline__ float dtw_inf_f() { return __int_as_float(0x7f800000); }

extern "C" __global__
void dtw_matrix_kernel(
    const float* __restrict__ queries,
    const int*   __restrict__ q_offsets,
    const float* __restrict__ refs,
    const int*   __restrict__ r_offsets,
    float*       __restrict__ out,
    int n_q,
    int n_r,
    int max_m,
    int window)
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

    if (n == 0 || m == 0) {
        if (threadIdx.x == 0) *out_cell = dtw_inf_f();
        return;
    }

    const float* a = queries + q_start;
    const float* b = refs    + r_start;

    extern __shared__ float smem[];
    float* prev = smem;                  // length max_m + 1
    float* curr = smem + (max_m + 1);    // length max_m + 1

    // Only one thread does work in v1; others wait at syncthreads.
    if (threadIdx.x != 0) return;

    // Init prev to INF with prev[0] = 0, curr to INF.
    for (int j = 0; j <= m; ++j) {
        prev[j] = dtw_inf_f();
        curr[j] = dtw_inf_f();
    }
    prev[0] = 0.0f;

    int w = (window < 0) ? m : window;

    for (int i = 1; i <= n; ++i) {
        int j_start = (i > w) ? (i - w) : 1;
        if (j_start < 1) j_start = 1;
        int j_end = (i + w < m) ? (i + w) : m;
        curr[0] = dtw_inf_f();

        float ai = a[i - 1];
        for (int j = j_start; j <= j_end; ++j) {
            float diff = ai - b[j - 1];
            float cost = diff * diff;
            float m1 = fminf(prev[j - 1], prev[j]);
            float mp = fminf(m1, curr[j - 1]);
            curr[j] = cost + mp;
        }

        // Clear the tail beyond j_end so subsequent rows that extend past
        // the previous band don't reuse stale curr[] values via curr[j-1].
        for (int j = j_end + 1; j <= m; ++j) {
            curr[j] = dtw_inf_f();
        }

        // swap prev <-> curr
        float* tmp = prev;
        prev = curr;
        curr = tmp;
    }

    *out_cell = sqrtf(prev[m]);
}
"#;

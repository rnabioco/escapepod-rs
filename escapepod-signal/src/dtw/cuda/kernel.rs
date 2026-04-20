//! CUDA kernel source for banded DTW.
//!
//! Compiled at runtime by NVRTC (via `cudarc::nvrtc::compile_ptx`). Kept as a
//! string constant so the build has no compile-time dependency on `nvcc` or
//! the CUDA toolkit.
//!
//! ## Kernel layout (v2, anti-diagonal)
//!
//! * Grid = `(n_queries, n_refs, 1)`. One block per (query, ref) pair.
//! * Block = `(THREADS, 1, 1)`. Threads cooperate along each anti-diagonal
//!   of the DP table; cells on the same diagonal `d = i + j` are independent
//!   given diagonals `d-1` and `d-2`, so the inner loop over the diagonal's
//!   cells is fully parallel (no cross-thread dependency within a row).
//! * Dynamic shared memory: three rolling diagonal buffers, each indexed by
//!   the query index `i ∈ [0, n]`. Size = `3 * (max_n + 1) * sizeof(float)`.
//!   On diagonal `d`, cell `(i, j=d-i)` reads `d2[i-1]` (the `(i-1,j-1)`
//!   predecessor), `d1[i-1]` (`(i-1,j)`), and `d1[i]` (`(i,j-1)`).
//!
//! The older v1 implementation used a single thread per block and leaned on
//! grid-level parallelism; this one actually uses each warp.

pub const KERNEL_SRC: &str = r#"
// NVRTC compiles without libc headers, so INFINITY isn't available.
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
    int max_n,
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

    // Three rolling diagonal buffers, each indexed by `i ∈ [0, max_n]`.
    extern __shared__ float smem[];
    float* d2 = smem;                              // diagonal d-2
    float* d1 = smem + (max_n + 1);                // diagonal d-1
    float* d0 = smem + 2 * (max_n + 1);            // diagonal d (current)

    int tid = threadIdx.x;
    int nth = blockDim.x;
    float INF = dtw_inf_f();

    // Initialize d2 as diagonal 0: only D[0][0] = 0, the rest INF.
    for (int i = tid; i <= n; i += nth) d2[i] = INF;
    __syncthreads();
    if (tid == 0) d2[0] = 0.0f;

    // Initialize d1 as diagonal 1: D[0][1] and D[1][0] are both INF (base case).
    for (int i = tid; i <= n; i += nth) d1[i] = INF;
    __syncthreads();

    int w = (window < 0) ? (n + m) : window;

    // Walk diagonals d = 2..=n+m. Cells on diagonal d are (i, d-i) with
    // 1 <= i <= n, 1 <= d-i <= m, and |i - (d-i)| = |2i - d| <= w. Each
    // thread strides over candidate i values and runtime-checks the bounds
    // — the band-bound arithmetic lives fine in Rust where it is easy to
    // match the CPU implementation, but here we want the simplest correct
    // mapping and let the compiler predicate the branch.
    for (int d = 2; d <= n + m; ++d) {
        // Clear the current-diagonal buffer so cells outside the band read
        // as INF when they become predecessors on the next diagonal.
        for (int i = tid; i <= n; i += nth) d0[i] = INF;
        __syncthreads();

        for (int i = 1 + tid; i <= n; i += nth) {
            int j = d - i;
            if (j < 1 || j > m) continue;
            int delta = i - j;
            if (delta < 0) delta = -delta;
            if (delta > w) continue;

            float diff = a[i - 1] - b[j - 1];
            float cost = diff * diff;
            float m1 = fminf(d1[i - 1], d1[i]);
            float mp = fminf(m1, d2[i - 1]);
            d0[i] = cost + mp;
        }
        __syncthreads();

        // Rotate: d2 <- d1, d1 <- d0, d0 <- (old d2, will be overwritten).
        float* tmp = d2;
        d2 = d1;
        d1 = d0;
        d0 = tmp;
        __syncthreads();
    }

    // After the last rotation, d1 holds diagonal n+m; D[n][m] lives at i=n.
    if (tid == 0) *out_cell = sqrtf(d1[n]);
}
"#;

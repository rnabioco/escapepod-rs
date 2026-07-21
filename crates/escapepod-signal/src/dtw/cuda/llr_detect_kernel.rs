//! GPU LLR adapter-detect kernel (one thread per read).
//!
//! Port of the demux LLR detect path: `normalize_signal` (MAD) →
//! optional `downscale` → `detect_adapter` (3-split LLR + segment-median
//! branch) → scale boundaries back. Detect is ~85% of CPU prep, so this is
//! the highest-value kernel. First cut is an exact one-thread-per-read
//! transcription to pin parity; `best_split` is an argmax over independent
//! per-position gains and can later move to a block-per-read parallel
//! reduction.
//!
//! Mirrors the CPU reference step for step:
//! - `normalize_signal`: MAD-normalize when len > 10, else raw cast,
//! - `downscale`: mean-pool `ds` samples (when ds > 1),
//! - `LlrTrace`: sequential f64 cumsum / cumsum_sq (inclusive prefix),
//! - `variance(start,end)`: O(1) from the cumsums, with the `start==0` branch,
//! - `best_split`: argmax gain over `[start+min_obs, end-border_trim)`,
//!   strictly-greater so ties prefer the earliest position,
//! - segment medians + the adapter-type branch.
//!
//! **Parity caveat:** like the fingerprint kernel, exact-tie argmax/median
//! cases can rarely diverge from the CPU's float rounding; vanishingly rare on
//! real signal.

/// Module name registered with the GPU device for the LLR detect kernel.
pub const MODULE_NAME: &str = "escapepod_gpu_llr_detect";
/// Kernel name registered with the GPU device. See [`MODULE_NAME`].
pub const KERNEL_NAME: &str = "llr_detect_kernel";

/// Indices into the packed `params` buffer.
pub const P_N_READS: usize = 0;
pub const P_MIN_ADAPTER: usize = 1;
pub const P_BORDER_TRIM: usize = 2;
pub const P_DOWNSCALE: usize = 3;
/// Number of packed scalar params.
pub const N_PARAMS: usize = 4;

/// CUDA-C source compiled at runtime via NVRTC.
pub const KERNEL_SRC: &str = r#"
// kth smallest VALUE (0-indexed) of a[0..n); permutes `a`.
__device__ float qselect(float* a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        float x = a[lo], y = a[mid], z = a[hi];
        float pivot = (x < y) ? ((y < z) ? y : ((x < z) ? z : x))
                              : ((x < z) ? x : ((y < z) ? z : y));
        for (int t = lo; t <= hi; ++t) {
            if (a[t] == pivot) { float tmp = a[t]; a[t] = a[hi]; a[hi] = tmp; break; }
        }
        float pv = a[hi];
        int store = lo;
        for (int t = lo; t < hi; ++t) {
            if (a[t] < pv) { float tmp = a[t]; a[t] = a[store]; a[store] = tmp; ++store; }
        }
        float tmp = a[store]; a[store] = a[hi]; a[hi] = tmp;
        if (k == store) return a[store];
        else if (k < store) hi = store - 1;
        else lo = store + 1;
    }
    return a[lo];
}

// median of src[0..n) using `scratch` as a permutable copy. Matches CPU
// median_slice / median_via_select: empty -> 0; odd -> kth(mid);
// even -> (max of lower half + kth(mid)) / 2.
__device__ float median_copy(const float* src, int n, float* scratch) {
    if (n <= 0) return 0.0f;
    for (int i = 0; i < n; ++i) scratch[i] = src[i];
    int mid = n >> 1;
    float hi = qselect(scratch, n, mid);
    if ((n & 1) == 0 && mid > 0) {
        float lo = -3.402823466e+38f;
        for (int i = 0; i < mid; ++i) if (scratch[i] > lo) lo = scratch[i];
        return (lo + hi) * 0.5f;
    }
    return hi;
}

// variance of processed[start..end) from inclusive-prefix cumsums.
__device__ double seg_variance(const double* cs, const double* css, int start, int end) {
    if (start == end) return 0.0;
    double n = (double)(end - start);
    if (start == 0) {
        double mean = cs[end - 1] / n;
        return css[end - 1] / n - mean * mean;
    }
    double sum_diff = cs[end - 1] - cs[start - 1];
    double sumsq_diff = css[end - 1] - css[start - 1];
    double mean = sum_diff / n;
    return sumsq_diff / n - mean * mean;
}

// best LLR split in [start+min_obs, end-border_trim); returns pos or -1,
// writing the gain to *out_gain. Strictly-greater => earliest on ties.
__device__ int best_split(const double* cs, const double* css, int m,
                          int start, int end, int min_obs, int border_trim,
                          double* out_gain) {
    *out_gain = 0.0;
    if (end > m) return -1;
    double var_full = seg_variance(cs, css, start, end);
    if (var_full <= 0.0) return -1;
    double var_summed = (double)(end - start) * log(var_full);
    int ss = start + min_obs;
    int se = (end >= border_trim) ? (end - border_trim) : 0;
    if (ss >= se) return -1;
    int best_pos = -1;
    double best_gain = 0.0;
    for (int i = ss; i < se; ++i) {
        double vh = seg_variance(cs, css, start, i);
        double vt = seg_variance(cs, css, i, end);
        if (vh <= 0.0 || vt <= 0.0) continue;
        double gain = var_summed
                    - (double)(i - start) * log(vh)
                    - (double)(end - i) * log(vt);
        if (gain > 0.0 && (best_pos < 0 || gain > best_gain)) {
            best_pos = i;
            best_gain = gain;
        }
    }
    *out_gain = best_gain;
    return best_pos;
}

extern "C" __global__
void llr_detect_kernel(
    const short*     __restrict__ signal,    // all reads' i16 signal, concatenated
    const long long* __restrict__ off,       // [n+1] sample offsets (also norm/scratch stride bases)
    const int*       __restrict__ params,     // [N_PARAMS]
    float*           __restrict__ norm,       // n-jagged: normalized signal
    float*           __restrict__ processed,  // n-jagged: downscaled signal (== norm when ds==1)
    float*           __restrict__ medbuf,     // n-jagged: median selection scratch
    double*          __restrict__ cumsum,     // n-jagged: inclusive prefix sum
    double*          __restrict__ cumsumsq,   // n-jagged: inclusive prefix sum of squares
    int*             __restrict__ out_start,  // [n] adapter_start
    int*             __restrict__ out_end)    // [n] adapter_end
{
    int n_reads     = params[0];
    int min_adapter = params[1];
    int border_trim = params[2];
    int ds          = params[3];

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_reads) return;
    out_start[r] = 0;
    out_end[r] = 0;

    long long base = off[r];
    int N = (int)(off[r + 1] - base);
    if (N <= 0) return;

    const short* sig = signal + base;
    float*  nrm = norm      + base;
    float*  prc = processed + base;
    float*  mb  = medbuf    + base;
    double* cs  = cumsum    + base;
    double* css = cumsumsq  + base;

    // 1. normalize_signal: MAD-normalize when N > 10, else raw cast.
    for (int i = 0; i < N; ++i) nrm[i] = (float)sig[i];
    if (N > 10) {
        float med = median_copy(nrm, N, mb);
        for (int i = 0; i < N; ++i) mb[i] = fabsf(nrm[i] - med);
        // median of abs-deviations: reuse mb in place (median_copy copies, but
        // here mb already holds the deviations) — select directly on mb.
        int mid = N >> 1;
        float hi = qselect(mb, N, mid);
        float mad;
        if ((N & 1) == 0 && mid > 0) {
            float lo = -3.402823466e+38f;
            for (int i = 0; i < mid; ++i) if (mb[i] > lo) lo = mb[i];
            mad = (lo + hi) * 0.5f;
        } else {
            mad = hi;
        }
        // CPU mad_normalize panics if mad==0; real signal won't. Guard anyway.
        float inv = (mad != 0.0f) ? (1.0f / mad) : 0.0f;
        for (int i = 0; i < N; ++i) nrm[i] = (nrm[i] - med) * inv;
    }

    // 2. downscale (mean-pool ds samples) when ds > 1; else processed == norm.
    int M, scale;
    if (ds > 1) {
        M = N / ds;          // trunc = M*ds; downscale of normalized[..trunc]
        scale = ds;
        for (int j = 0; j < M; ++j) {
            float acc = 0.0f;
            int b = j * ds;
            for (int t = 0; t < ds; ++t) acc += nrm[b + t];
            prc[j] = acc / (float)ds;
        }
    } else {
        M = N;
        scale = 1;
        for (int i = 0; i < N; ++i) prc[i] = nrm[i];
    }
    if (M <= 0) return;

    // 3. LlrTrace: sequential f64 inclusive-prefix cumsum / cumsum_sq.
    {
        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < M; ++i) {
            double v = (double)prc[i];
            sum += v; sumsq += v * v;
            cs[i] = sum; css[i] = sumsq;
        }
    }

    int min_obs = min_adapter / scale; if (min_obs < 1) min_obs = 1;
    int bt = border_trim / scale; if (bt < 1) bt = 1;

    // 4. three-split detect.
    double g_first;
    int x_first = best_split(cs, css, M, 0, M, min_obs + bt, bt, &g_first);
    if (x_first < 0) return; // (0,0)

    double gain_head, gain_tail;
    int x_head = best_split(cs, css, M, 0, x_first, bt, min_obs, &gain_head);
    if (x_head < 0) { x_head = 1; gain_head = 0.0; }
    int x_tail = best_split(cs, css, M, x_first, M, min_obs, bt, &gain_tail);
    if (x_tail < 0) { x_tail = x_first + 1; gain_tail = 0.0; }

    // segment medians of processed: [0,x_head) [x_head,x_first) [x_first,x_tail) [x_tail,M)
    float m0 = median_copy(prc, x_head, mb);
    float m1 = median_copy(prc + x_head, x_first - x_head, mb);
    float m2 = median_copy(prc + x_first, x_tail - x_first, mb);
    float m3 = median_copy(prc + x_tail, M - x_tail, mb);
    float mean_median = (m0 + m1 + m2 + m3) / 4.0f;
    float diff_1 = m2 - m1;

    int s = 0, e = 0;
    if (diff_1 > 0.0f) {
        if (m0 >= mean_median) { s = x_head; e = x_first; }
        else { s = 0; e = x_first; }
    } else if (gain_tail > gain_head) {
        s = x_first; e = x_tail;
    } else {
        s = 0; e = 0;
    }

    out_start[r] = s * scale;
    out_end[r] = e * scale;
}
"#;

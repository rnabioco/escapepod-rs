//! GPU LLR adapter-detect kernel — **block per read** (parallel best-split).
//!
//! The one-thread-per-read detector ([`super::llr_detect_kernel`]) is bit-exact
//! but ~2.5× slower than a single CPU thread: each thread serially runs three
//! `ln`-heavy `best_split` argmax scans with divergent `qselect` normalize and
//! uncoalesced f64 access. This kernel assigns one CUDA **block** per read and
//! parallelises the dominant work:
//!
//! - **normalize** via a parallel mean/std reduction → z-score. LLR gains are
//!   invariant under any affine transform of the signal (the `ln(var/scale²)`
//!   terms cancel because `n_full = n_head + n_tail`, and the branch only uses
//!   *signs* of segment-median differences), so z-score yields the same
//!   detect result as the CPU's MAD normalize — while avoiding `qselect`.
//! - **best_split** as a parallel per-position gain map + shared-memory argmax
//!   reduction (ties → earliest position, matching the CPU's strict-`>` scan).
//!
//! The sequential f64 cumsum (parity-critical) and the four segment medians
//! stay on thread 0. Block size is fixed at 128 (shared arrays are sized to
//! match); launch with `block_dim = (128, 1, 1)`.

/// Module name registered with the GPU device.
pub const MODULE_NAME: &str = "escapepod_gpu_llr_detect_block";
/// Kernel name registered with the GPU device. See [`MODULE_NAME`].
pub const KERNEL_NAME: &str = "llr_detect_block_kernel";

/// Block size (threads per read). Must match `BLK` in the kernel source.
///
/// The packed `params` buffer layout is shared with the thread-per-read
/// detector (`super::llr_detect_kernel::P_*`); the host builds it once.
pub const BLOCK: u32 = 128;

/// CUDA-C source compiled at runtime via NVRTC.
pub const KERNEL_SRC: &str = r#"
#define BLK 128

// kth smallest VALUE (0-indexed) of a[0..n); permutes `a`. (thread-0 medians)
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

// Parallel best LLR split in [start+min_obs, end-border_trim). All threads
// cooperate; result is written to *out_pos / *out_gain (valid on all threads
// after the final __syncthreads). pos = -1 when no valid split.
__device__ void block_best_split(
    const double* cs, const double* css, int m,
    int start, int end, int min_obs, int border_trim,
    double* sh_gain, int* sh_pos, int* out_pos, double* out_gain)
{
    int tid = threadIdx.x;
    int local_pos = -1;
    double local_gain = 0.0;

    bool ok = (end <= m);
    double var_full = ok ? seg_variance(cs, css, start, end) : -1.0;
    int ss = start + min_obs;
    int se = (end >= border_trim) ? (end - border_trim) : 0;
    if (!ok || var_full <= 0.0 || ss >= se) {
        // no valid split for this read
    } else {
        double var_summed = (double)(end - start) * log(var_full);
        for (int i = ss + tid; i < se; i += BLK) {
            double vh = seg_variance(cs, css, start, i);
            double vt = seg_variance(cs, css, i, end);
            if (vh <= 0.0 || vt <= 0.0) continue;
            double gain = var_summed
                        - (double)(i - start) * log(vh)
                        - (double)(end - i) * log(vt);
            if (gain > 0.0 && (local_pos < 0 || gain > local_gain)) {
                local_gain = gain;
                local_pos = i;
            }
        }
    }

    sh_gain[tid] = local_gain;
    sh_pos[tid] = local_pos;
    __syncthreads();

    // reduce: max gain; ties -> earliest (min) position; ignore pos<0.
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double og = sh_gain[tid + s]; int op = sh_pos[tid + s];
            double mg = sh_gain[tid];     int mp = sh_pos[tid];
            bool take = false;
            if (op >= 0) {
                if (mp < 0) take = true;
                else if (og > mg) take = true;
                else if (og == mg && op < mp) take = true;
            }
            if (take) { sh_gain[tid] = og; sh_pos[tid] = op; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_pos = sh_pos[0];
        *out_gain = (sh_pos[0] >= 0) ? sh_gain[0] : 0.0;
    }
    __syncthreads();
}

extern "C" __global__
void llr_detect_block_kernel(
    const short*     __restrict__ signal,
    const long long* __restrict__ off,
    const int*       __restrict__ params,
    float*           __restrict__ norm,
    float*           __restrict__ processed,
    float*           __restrict__ medbuf,
    double*          __restrict__ cumsum,
    double*          __restrict__ cumsumsq,
    int*             __restrict__ out_start,
    int*             __restrict__ out_end)
{
    int n_reads     = params[0];
    int min_adapter = params[1];
    int border_trim = params[2];
    int ds          = params[3];

    int r = blockIdx.x;
    if (r >= n_reads) return;
    int tid = threadIdx.x;

    __shared__ double sh_a[BLK];
    __shared__ double sh_b[BLK];
    __shared__ int    sh_pos[BLK];
    __shared__ int    sh_xfirst, sh_xhead, sh_xtail, sh_M, sh_scale;
    __shared__ double sh_ghead, sh_gtail, sh_mean, sh_inv;

    long long base = off[r];
    int N = (int)(off[r + 1] - base);
    if (N <= 0) { if (tid == 0) { out_start[r] = 0; out_end[r] = 0; } return; }

    const short* sig = signal + base;
    float*  nrm = norm      + base;
    float*  prc = processed + base;
    float*  mb  = medbuf    + base;
    double* cs  = cumsum    + base;
    double* css = cumsumsq  + base;

    // 1. z-score normalize params via parallel reduction (affine-invariant).
    double ls = 0.0, lq = 0.0;
    for (int i = tid; i < N; i += BLK) { double v = (double)sig[i]; ls += v; lq += v * v; }
    sh_a[tid] = ls; sh_b[tid] = lq; __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (tid < s) { sh_a[tid] += sh_a[tid + s]; sh_b[tid] += sh_b[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) {
        double mean = sh_a[0] / (double)N;
        double var = sh_b[0] / (double)N - mean * mean;
        double std = (var > 0.0) ? sqrt(var) : 0.0;
        sh_mean = mean;
        sh_inv = (std > 0.0) ? (1.0 / std) : 0.0;
    }
    __syncthreads();
    double mean = sh_mean, inv = sh_inv;
    for (int i = tid; i < N; i += BLK) nrm[i] = (float)(((double)sig[i] - mean) * inv);
    __syncthreads();

    // 2. downscale (mean-pool) when ds > 1; else processed aliases norm.
    float* P;
    int M, scale;
    if (ds > 1) {
        M = N / ds; scale = ds;
        for (int j = tid; j < M; j += BLK) {
            float acc = 0.0f;
            int b = j * ds;
            for (int t = 0; t < ds; ++t) acc += nrm[b + t];
            prc[j] = acc / (float)ds;
        }
        P = prc;
        __syncthreads();
    } else {
        M = N; scale = 1; P = nrm;
    }
    if (tid == 0) { sh_M = M; sh_scale = scale; }
    if (M <= 0) { if (tid == 0) { out_start[r] = 0; out_end[r] = 0; } return; }

    // 3. sequential f64 inclusive-prefix cumsum (parity), thread 0.
    if (tid == 0) {
        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < M; ++i) {
            double v = (double)P[i];
            sum += v; sumsq += v * v;
            cs[i] = sum; css[i] = sumsq;
        }
    }
    __syncthreads();

    int min_obs = min_adapter / scale; if (min_obs < 1) min_obs = 1;
    int bt = border_trim / scale; if (bt < 1) bt = 1;

    // 4. three parallel best_splits.
    int xf; double gf;
    block_best_split(cs, css, M, 0, M, min_obs + bt, bt, sh_a, sh_pos, &xf, &gf);
    if (xf < 0) { if (tid == 0) { out_start[r] = 0; out_end[r] = 0; } return; }
    if (tid == 0) sh_xfirst = xf;
    __syncthreads();
    int x_first = sh_xfirst;

    int xh; double gh;
    block_best_split(cs, css, M, 0, x_first, bt, min_obs, sh_a, sh_pos, &xh, &gh);
    if (tid == 0) { sh_xhead = (xh < 0) ? 1 : xh; sh_ghead = (xh < 0) ? 0.0 : gh; }

    int xt; double gt;
    block_best_split(cs, css, M, x_first, M, min_obs, bt, sh_a, sh_pos, &xt, &gt);
    if (tid == 0) { sh_xtail = (xt < 0) ? (x_first + 1) : xt; sh_gtail = (xt < 0) ? 0.0 : gt; }
    __syncthreads();

    // 5. segment medians + branch on thread 0.
    if (tid == 0) {
        int x_head = sh_xhead, x_tail = sh_xtail;
        float m0 = median_copy(P, x_head, mb);
        float m1 = median_copy(P + x_head, x_first - x_head, mb);
        float m2 = median_copy(P + x_first, x_tail - x_first, mb);
        float m3 = median_copy(P + x_tail, M - x_tail, mb);
        float mean_median = (m0 + m1 + m2 + m3) / 4.0f;
        float diff_1 = m2 - m1;

        int s = 0, e = 0;
        if (diff_1 > 0.0f) {
            if (m0 >= mean_median) { s = x_head; e = x_first; }
            else { s = 0; e = x_first; }
        } else if (sh_gtail > sh_ghead) {
            s = x_first; e = x_tail;
        } else {
            s = 0; e = 0;
        }
        out_start[r] = s * scale;
        out_end[r] = e * scale;
    }
}
"#;

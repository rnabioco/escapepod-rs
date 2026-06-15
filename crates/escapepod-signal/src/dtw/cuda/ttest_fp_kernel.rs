//! GPU t-test fingerprint kernel (one thread per read).
//!
//! Port of `escapepod_demux::fingerprint::extract_fingerprint_from_signal`
//! for the demux path: `keep_last = Some(n)`, `NormMethod::ZScore`,
//! `emit_dwell = false`. Mirrors the CPU pipeline step for step:
//!
//! 1. slice `[adapter_start-pad .. adapter_end+pad]` (clamped),
//! 2. `clip_outliers(±5·MAD)` — median/MAD as exact order statistics,
//! 3. windowed t-test (sequential f64 cumsum, identical rounding to CPU),
//! 4. strict local maxima → greedy top-`(num_segments-1)` NMS (min sep),
//! 5. per-segment means (fresh sequential f64 sums),
//! 6. z-score over the full mean population, then keep the last `keep_last`.
//!
//! Inherently serial *within* a read (cumsum, the median selections, the
//! local-maxima scan, and the greedy NMS are all sequential), but
//! embarrassingly parallel *across* reads — one thread handles one read.
//! Scratch is rectangular per sub-batch (`max_len` slots per read); the host
//! length-sorts and memory-bounds sub-batches so `max_len` tracks the batch.
//!
//! **Parity caveat:** peak ordering uses heapsort keyed on the f64 t-score;
//! for *exactly equal* scores the tie-break differs from Rust's `sort_unstable`
//! (pdqsort), which can rarely flip which of two tied peaks survives NMS.
//! Exact f64 ties among peaks are vanishingly rare on real signal.
//!
//! cudarc 0.12 caps kernel launches at 12 tuple args, so scalars are packed
//! into a `params` buffer and the scratch buffers are merged (A|B share one
//! f32 region, cumsum|cumsum_sq share one f64 region). `out_len[r] == 0`
//! flags a read that produced no fingerprint (the invalid sentinel).

/// Module name registered with the GPU device for the fingerprint kernel.
pub const MODULE_NAME: &str = "escapepod_gpu_ttest_fp";
/// Kernel name registered with the GPU device. See [`MODULE_NAME`].
pub const KERNEL_NAME: &str = "ttest_fp_kernel";

/// Indices into the packed `params` buffer.
pub const P_N_READS: usize = 0;
pub const P_WW: usize = 1;
pub const P_MIN_SEP: usize = 2;
pub const P_NUM_CP: usize = 3;
pub const P_KEEP_LAST: usize = 4;
pub const P_MAX_LEN: usize = 5;
pub const P_MAX_CAND: usize = 6;
pub const P_PAD: usize = 7;
/// Number of packed scalar params.
pub const N_PARAMS: usize = 8;

/// CUDA-C source compiled at runtime via NVRTC.
pub const KERNEL_SRC: &str = r#"
// ---- iterative quickselect: kth smallest VALUE (0-indexed) of a[0..n) ----
// Permutes `a`. The kth order statistic is unique, so any correct selection
// reproduces the CPU `select_nth_unstable_by(total_cmp)` *value*.
__device__ float qselect(float* a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        float x = a[lo], y = a[mid], z = a[hi];
        float pivot = (x < y) ? ((y < z) ? y : ((x < z) ? z : x))
                              : ((x < z) ? x : ((y < z) ? z : y));
        // bring a copy of the pivot value to `hi`, then Lomuto-partition.
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

// median matching CPU `median_via_select`: odd -> kth(mid);
// even -> (max of lower half + kth(mid)) / 2, with mid = n/2.
__device__ float median_f32(float* a, int n) {
    int mid = n >> 1;
    float hi = qselect(a, n, mid);
    if ((n & 1) == 0) {
        float lo = -3.402823466e+38f; // -FLT_MAX
        for (int i = 0; i < mid; ++i) if (a[i] > lo) lo = a[i];
        return (lo + hi) * 0.5f;
    }
    return hi;
}

// max-heap sift-down on positions `idx`, keyed by `key[idx[*]]`.
__device__ void sift_down(int* idx, const double* key, int start, int n) {
    int root = start;
    for (;;) {
        int child = 2 * root + 1;
        if (child >= n) break;
        if (child + 1 < n && key[idx[child + 1]] > key[idx[child]]) child++;
        if (key[idx[child]] > key[idx[root]]) {
            int t = idx[root]; idx[root] = idx[child]; idx[child] = t;
            root = child;
        } else break;
    }
}

extern "C" __global__
void ttest_fp_kernel(
    const short*     __restrict__ signal,      // all reads' i16 signal, concatenated
    const long long* __restrict__ sig_off,     // [n+1] sample offsets into `signal`
    const int*       __restrict__ adapter_start,
    const int*       __restrict__ adapter_end,
    const int*       __restrict__ params,       // [N_PARAMS] packed scalars
    float*           __restrict__ scratch_f32,  // n * (2*max_len): A | B
    double*          __restrict__ scratch_f64,  // n * (2*(max_len+1)): cumsum | cumsum_sq
    double*          __restrict__ tscores,      // n * max_cand
    int*             __restrict__ peaks,        // n * max_cand
    float*           __restrict__ out,          // n * keep_last (z-scored tail)
    int*             __restrict__ out_len)      // [n] valid entry count (0 == no fingerprint)
{
    int n_reads   = params[0];
    int ww        = params[1];
    int min_sep   = params[2];
    int num_cp    = params[3];
    int keep_last = params[4];
    int max_len   = params[5];
    int max_cand  = params[6];
    int pad       = params[7];

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_reads) return;
    out_len[r] = 0;

    long long sig_len = sig_off[r + 1] - sig_off[r];
    int a_s = adapter_start[r];
    int a_e = adapter_end[r];

    int ss = a_s - pad; if (ss < 0) ss = 0;          // saturating_sub
    long long se_ll = (long long)a_e + pad; if (se_ll > sig_len) se_ll = sig_len;
    int se = (int)se_ll;
    if (se <= ss || (se - ss) < ww * 2) return;
    int L = se - ss;

    const short* sig = signal + sig_off[r] + ss;
    float*  A   = scratch_f32 + (long long)r * (2 * (long long)max_len);
    float*  B   = A + max_len;
    double* cs  = scratch_f64 + (long long)r * (2 * ((long long)max_len + 1));
    double* css = cs + (max_len + 1);
    double* ts  = tscores  + (long long)r * max_cand;
    int*    pk  = peaks    + (long long)r * max_cand;

    // 1. raw f32
    for (int i = 0; i < L; ++i) A[i] = (float)sig[i];

    // 2. clip_outliers(±5 MAD) — only when L >= 2 and MAD != 0.
    if (L >= 2) {
        for (int i = 0; i < L; ++i) B[i] = A[i];
        float med = median_f32(B, L);
        for (int i = 0; i < L; ++i) B[i] = fabsf(A[i] - med);
        float mad = median_f32(B, L);
        if (mad != 0.0f) {
            float lo = med - 5.0f * mad;
            float hi = med + 5.0f * mad;
            for (int i = 0; i < L; ++i) {
                float x = A[i];
                x = (x < lo) ? lo : x;   // x.max(lo)
                x = (x > hi) ? hi : x;   // .min(hi)
                A[i] = x;
            }
        }
    }

    // 3. sequential f64 cumsum / cumsum_sq over clipped A.
    cs[0] = 0.0; css[0] = 0.0;
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < L; ++i) {
        double v = (double)A[i];
        sum += v; sumsq += v * v;
        cs[i + 1] = sum; css[i + 1] = sumsq;
    }

    // 4. windowed t-test.
    int num_cand = L - 2 * ww;
    if (num_cand < 0) num_cand = 0;
    double w = (double)ww;
    for (int pos = 0; pos < num_cand; ++pos) {
        double sum1 = cs[pos + ww] - cs[pos];
        double sum2 = cs[pos + 2 * ww] - cs[pos + ww];
        double m1 = sum1 / w, m2 = sum2 / w;
        double sq1 = css[pos + ww] - css[pos];
        double sq2 = css[pos + 2 * ww] - css[pos + ww];
        double var1 = sq1 - w * m1 * m1;
        double var2 = sq2 - w * m2 * m2;
        ts[pos] = (var1 + var2 <= 0.0) ? 0.0 : fabs(m1 - m2) / sqrt(var1 + var2);
    }

    // 5. strict local maxima (plateaus -> left-biased midpoint).
    int npk = 0;
    if (num_cand >= 3) {
        int i = 1, imax = num_cand - 1;
        while (i < imax) {
            if (ts[i - 1] < ts[i]) {
                int iah = i + 1;
                while (iah < imax && ts[iah] == ts[i]) iah++;
                if (ts[iah] < ts[i]) {
                    pk[npk++] = (i + iah - 1) / 2;
                    i = iah;
                    continue;
                }
            }
            i++;
        }
    }

    // 6. greedy top-num_cp NMS by descending score (heap over candidate
    //    positions in `pk`, keyed on ts). Kept candidate positions collected
    //    into the cumsum scratch (free after step 4), then sorted ascending.
    for (int s = npk / 2 - 1; s >= 0; --s) sift_down(pk, ts, s, npk);

    int* kept_pos = (int*)cs;   // reuse cumsum scratch (>= num_cp ints)
    int n_kept = 0;
    int heap_n = npk;
    while (heap_n > 0 && n_kept < num_cp) {
        int best = pk[0];
        pk[0] = pk[heap_n - 1];
        heap_n--;
        sift_down(pk, ts, 0, heap_n);
        int ok = 1;
        for (int j = 0; j < n_kept; ++j) {
            int d = best - kept_pos[j]; if (d < 0) d = -d;
            if (d < min_sep) { ok = 0; break; }
        }
        if (ok) kept_pos[n_kept++] = best;
    }
    // insertion sort kept_pos ascending (n_kept <= num_cp, small).
    for (int i = 1; i < n_kept; ++i) {
        int key = kept_pos[i]; int j = i - 1;
        while (j >= 0 && kept_pos[j] > key) { kept_pos[j + 1] = kept_pos[j]; j--; }
        kept_pos[j + 1] = key;
    }

    // 7. per-segment means over clipped A. boundaries = [0, (cp+ww)..., L].
    //    Fresh sequential f64 sums (matches CPU compute_segment_means).
    int n_seg = n_kept + 1;
    double* means = ts;   // ts free after peaks built.
    int prev = 0;
    for (int s = 0; s < n_seg; ++s) {
        int start = prev;
        int end = (s < n_kept) ? (kept_pos[s] + ww) : L;
        prev = end;
        double acc = 0.0;
        for (int t = start; t < end; ++t) acc += (double)A[t];
        means[s] = acc / (double)(end - start);
    }

    // 8. z-score over the full mean population (f32, sequential).
    int k = n_seg;
    if (k <= 0) return;
    float fmean = 0.0f;
    for (int s = 0; s < k; ++s) fmean += (float)means[s];
    fmean /= (float)k;
    float fvar = 0.0f;
    for (int s = 0; s < k; ++s) { float d = (float)means[s] - fmean; fvar += d * d; }
    fvar /= (float)k;
    float fstd = sqrtf(fvar);

    int keep = (k > keep_last) ? keep_last : k;
    int base = k - keep;
    for (int s = 0; s < keep; ++s) {
        float m = (float)means[base + s];
        out[(long long)r * keep_last + s] = (fstd > 0.0f) ? ((m - fmean) / fstd) : m;
    }
    out_len[r] = keep;
}
"#;

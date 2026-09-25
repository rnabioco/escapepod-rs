// SPDX-License-Identifier: MIT

//! CUDA source of the score kernel, compiled by NVRTC at run time (so the
//! build needs no `nvcc` and no CUDA toolkit).
//!
//! # Layout
//!
//! * **One thread per (read, reference) pair.** A block holds one *group* of
//!   references, one per thread (`blockDim.x` = lanes, a multiple of 32), and
//!   the grid is `(persistent blocks, groups)`. Every thread of a block works
//!   on the same read at the same time, so the read's codes are uniform loads
//!   and the per-read loop bounds are uniform; each thread walks its *own*
//!   reference to its own length, so there is no column mask at all — a
//!   thread with a shorter reference simply finishes its column loop first.
//! * **Persistent blocks pull reads from an atomic counter**, in the order the
//!   host gives (longest read first), so a long read starts early instead of
//!   being the tail the whole batch waits behind.
//! * **The panel is resident in shared memory**, loaded once per block: four
//!   bits per reference base (A=1, C=2, G=4, T=8, anything else 0xF), eight
//!   bases per word, transposed so word `w` of lane `t` is `sp[w * lanes + t]`
//!   — consecutive threads, consecutive banks. A read base scores a match iff
//!   its mask and the reference's intersect, which is exactly the wildcard
//!   rule of `Scoring::substitution`.
//! * **The read is walked in stripes of `R` rows held in registers.** For each
//!   reference column the thread updates `R` cells (the scalar recurrence,
//!   verbatim, in `int`), and the only state that leaves registers is the
//!   stripe's bottom row — `H` and `F` of the last row, two `i16`s packed in
//!   one word per column — which the next stripe reads back as its top border.
//!   That boundary lives in a global scratch buffer interleaved by thread
//!   (`scratch[col * total_threads + thread]`, so a warp's accesses coalesce):
//!   one 4-byte load and one store per column per stripe, 0.5 B of traffic per
//!   cell at `R = 16`.
//!
//! The values are exactly the scalar oracle's: the arithmetic is `int` with
//! the same `NEG = i32::MIN / 4` sentinel, and the only narrowing — the packed
//! boundary and the `i16` result — is of values `fits_i16` has bounded.
//!
//! # Semi-global
//!
//! The best cell of the last row is taken column by column in the stripe that
//! holds row `n`; the best cell of the reference's last column is what the
//! stripe's `H` registers hold when the thread's column loop ends, taken per
//! stripe. Local mode takes every valid cell and floors at zero. Rows past the
//! read's end (the last stripe's padding, code 0: matches nothing) are masked
//! out of both, and cannot reach a valid cell because information only flows
//! to larger row indices.

/// Rows per stripe. Changing it changes the query packing (`R / 8` words per
/// stripe) on the host side too.
pub(crate) const STRIPE_ROWS: usize = 16;

pub(crate) const KERNEL_NAME: &str = "align_score_kernel";

pub(crate) const KERNEL_SRC: &str = r#"
#define NEG (-(1 << 29))
#define R 16

template <bool LOCAL, bool TAIL>
__device__ __forceinline__ void stripe(
    const unsigned int* __restrict__ sp, int lanes, int t, int m,
    unsigned int q0, unsigned int q1, int rows, bool first,
    unsigned int* __restrict__ bnd, size_t bstride,
    int ma, int mi, int o, int e, int& best)
{
    int qm[R];
#pragma unroll
    for (int r = 0; r < 8; r++) {
        qm[r] = (q0 >> (4 * r)) & 0xF;
        qm[r + 8] = (q1 >> (4 * r)) & 0xF;
    }
    int H[R];
    int E[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        H[r] = 0;
        E[r] = NEG;
    }
    int diag_top = 0;
    unsigned int rw = 0;
    unsigned int bnext = first ? 0u : bnd[0];
    for (int j = 0; j < m; j++) {
        if ((j & 7) == 0) rw = sp[(j >> 3) * lanes + t];
        const int rm = (rw >> ((j & 7) * 4)) & 0xF;
        int hup;
        int f;
        if (first) {
            hup = 0;
            f = NEG;
        } else {
            const unsigned int b = bnext;
            if (j + 1 < m) bnext = bnd[(size_t)(j + 1) * bstride];
            hup = (int)(short)(b & 0xFFFFu);
            f = (int)(short)(b >> 16);
        }
        int hdiag = diag_top;
        diag_top = hup;
        int last = NEG;
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int hleft = H[r];
            const int ev = max(hleft + o, E[r] + e);
            E[r] = ev;
            f = max(hup + o, f + e);
            const int s = (qm[r] & rm) ? ma : mi;
            int h = max(max(hdiag + s, ev), f);
            if (LOCAL) h = max(h, 0);
            hdiag = hleft;
            H[r] = h;
            hup = h;
            if (LOCAL) {
                if (!TAIL || r < rows) best = max(best, h);
            } else if (TAIL) {
                if (r == rows - 1) last = h;
            }
        }
        if (!TAIL) {
            bnd[(size_t)j * bstride] =
                ((unsigned int)(unsigned short)hup) | (((unsigned int)(unsigned short)f) << 16);
        }
        if (!LOCAL && TAIL) best = max(best, last);
    }
    if (!LOCAL) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (!TAIL || r < rows) best = max(best, H[r]);
        }
    }
}

extern "C" __global__ void align_score_kernel(
    const unsigned int* __restrict__ qwords,
    const unsigned long long* __restrict__ qoff,
    const int* __restrict__ qlen,
    const int* __restrict__ order,
    const unsigned int* __restrict__ panel,
    const int* __restrict__ ref_len,
    const int* __restrict__ ref_idx,
    short* __restrict__ out,
    unsigned int* __restrict__ scratch,
    int* __restrict__ counters,
    int n_work,
    int n_refs,
    int words_per_lane,
    int ma, int mi, int o, int e,
    int local)
{
    extern __shared__ unsigned int sp[];
    __shared__ int next;
    const int lanes = blockDim.x;
    const int t = threadIdx.x;
    const int g = blockIdx.y;
    const unsigned int* gp = panel + (size_t)g * words_per_lane * lanes;
    for (int w = t; w < words_per_lane * lanes; w += lanes) sp[w] = gp[w];
    const int m = ref_len[g * lanes + t];
    const int ri = ref_idx[g * lanes + t];
    const size_t total = (size_t)gridDim.x * gridDim.y * lanes;
    const size_t gtid = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * lanes + t;
    unsigned int* bnd = scratch + gtid;
    for (;;) {
        // First pass: the panel is loaded. Later passes: everyone has read
        // `next` before thread 0 overwrites it.
        __syncthreads();
        if (t == 0) next = atomicAdd(&counters[g], 1);
        __syncthreads();
        const int k = next;
        if (k >= n_work) break;
        const int qi = order[k];
        const int n = qlen[qi];
        const unsigned int* q = qwords + qoff[qi];
        int best = local ? 0 : NEG;
        if (m > 0) {
            for (int i0 = 0; i0 < n; i0 += R) {
                const unsigned int q0 = q[i0 >> 3];
                const unsigned int q1 = q[(i0 >> 3) + 1];
                const int rows = min(R, n - i0);
                const bool first = i0 == 0;
                const bool tail = i0 + R >= n;
                if (local) {
                    if (tail) stripe<true, true>(sp, lanes, t, m, q0, q1, rows, first, bnd, total, ma, mi, o, e, best);
                    else stripe<true, false>(sp, lanes, t, m, q0, q1, rows, first, bnd, total, ma, mi, o, e, best);
                } else {
                    if (tail) stripe<false, true>(sp, lanes, t, m, q0, q1, rows, first, bnd, total, ma, mi, o, e, best);
                    else stripe<false, false>(sp, lanes, t, m, q0, q1, rows, first, bnd, total, ma, mi, o, e, best);
                }
            }
            out[(size_t)qi * n_refs + ri] = (short)best;
        }
    }
}
"#;

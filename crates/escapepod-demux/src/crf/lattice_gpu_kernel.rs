//! CUDA kernels for the CTC-CRF lattice decode, batched across reads.
//!
//! A transcription of [`super::lattice`]'s two passes, with the batch axis as
//! the parallel dimension the CPU path does not have. Read [`super::lattice`]
//! first: the edge indexing is `(destination, dropped_base)` rather than
//! `(source, emitted_base)`, and both passes are load-bearing.
//!
//! # Why this is a good GPU fit when the CPU note says it is not
//!
//! `encoder_gpu`'s note — "sequential in time with a 256-wide inner dimension,
//! a poor fit for the device" — is about decoding **one read**. It is correct
//! there: 256 lanes will not fill an A30. But the fused pipeline holds a whole
//! device batch of reads, and reads are completely independent, so a timestep is
//! `batch * n_states` independent lanes (131 072 at batch 512). Only the `t_len`
//! sweep stays sequential, and it is sequential per read either way.
//!
//! This is also how the algorithm was written upstream: the port's source is
//! bonito's `CTC_CRF`, whose forward/backward scores come from koi's
//! `ctc.{fwd,bwd}_scores_cu_sparse` CUDA kernels.
//!
//! # No transpose
//!
//! The CPU path transposes each timestep from the encoder's `[dest][edge]` into
//! `[edge][dest]` so its SIMD inner loops are unit-stride. The GPU wants the
//! opposite: with one thread per destination state, reading `row[dest*n_edges+e]`
//! gives each thread a contiguous `n_edges`-float run, so a warp covers
//! `32 * n_edges * 4` contiguous bytes — already coalesced. Working in the
//! encoder's native order therefore drops a whole kernel, a second
//! `t_len * n_score` device buffer, and its round trip through memory. It also
//! makes the Viterbi tie-break key the flat index itself, since native order
//! *is* bonito's `dest * n_edges + edge`.
//!
//! # Numerics
//!
//! Same contract as the SIMD backends: **same sequence**, not bit-identical.
//! The block reductions reassociate the softmax denominator exactly as the AVX
//! kernels already do. `expf`/`logf` are used rather than `__expf`/`__logf` —
//! the fast intrinsics are ~2 ulp and this decode feeds an argmax whose ties
//! matter, so the accurate versions keep the GPU *closer* to the scalar
//! reference than AVX is. Ties break toward the lowest `dest * n_edges + edge`,
//! matching every CPU backend, so no two backends can disagree on a tie.
//!
//! # Layout
//!
//! - `scores`: `[batch][t_len][n_states][n_edges]`, the encoder's own order.
//!   Overwritten in place with the pass-1 log-posteriors.
//! - `alpha` / `beta`: `[batch][t_len + 1][n_states]`.
//! - `path`: `[batch][t_len]`, `0` for blank, else `1 + base`.

pub const SWEEP_KERNEL_NAME: &str = "crf_sweep_kernel";
pub const POSTERIOR_KERNEL_NAME: &str = "crf_posterior_kernel";
pub const VITERBI_KERNEL_NAME: &str = "crf_viterbi_kernel";
pub const REFSCAN_KERNEL_NAME: &str = "crf_refscan_kernel";

/// Semiring selector shared with the host side.
pub const SEMIRING_LOG: i32 = 0;
pub const SEMIRING_MAX: i32 = 1;

pub const KERNEL_SRC: &str = r#"
// Largest n_edges we size per-thread accumulators for: n_base + 1 with
// n_base <= 7. The host rejects anything wider before launching.
#define CRF_MAX_EDGES 8

// NVRTC compiles without the host <math.h>, so INFINITY and isfinite are not
// available. Both are spelled out here rather than pulled from a header the
// runtime compiler may or may not have.
#define CRF_INF __int_as_float(0x7f800000)

__device__ __forceinline__ bool crf_finite(float x)
{
    return (x == x) && (x != CRF_INF) && (x != -CRF_INF);
}

// Reduce over acc[0..n_edges], matching lattice.rs's Semiring::reduce:
// max-shifted logsumexp, or a plain max. The !finite early-out reproduces the
// scalar path's behaviour when every incoming edge is -inf.
__device__ __forceinline__ float crf_reduce(
    const float* acc, int n_edges, int semiring)
{
    float m = -CRF_INF;
    for (int e = 0; e < n_edges; ++e) m = fmaxf(m, acc[e]);
    if (semiring == 1) return m;          // Max
    if (!crf_finite(m)) return m;
    float sum = 0.0f;
    // Summed in edge order, like the scalar reference.
    for (int e = 0; e < n_edges; ++e) sum += expf(acc[e] - m);
    return m + logf(sum);
}

// The source state of `edge` into `dest` — lattice.rs's CrfLayout::source_state.
__device__ __forceinline__ int crf_source_state(
    int dest, int edge, int n_base, int group)
{
    return (edge == 0) ? dest : ((edge - 1) * group + dest / n_base);
}

// -------------------------------------------------------------------------
// Forward and backward sweeps, fused into one launch.
//
// Grid (batch, 2): blockIdx.y == 0 runs the forward sweep into `alpha`,
// blockIdx.y == 1 the backward sweep into `beta`. They are independent — both
// only read `scores` — so fusing them doubles the blocks in flight and halves
// the launches. The branch is uniform per block, so no warp diverges.
//
// blockDim.x == n_states, one thread per state. The running row lives in shared
// memory and is mirrored to global because the two consumers below need every
// timestep. Both __syncthreads() per step are required: threads read *other*
// threads' entries, so the shared row cannot be overwritten until every read of
// it has retired.
// -------------------------------------------------------------------------
extern "C" __global__
void crf_sweep_kernel(
    const float* __restrict__ scores,
    float*       __restrict__ alpha,
    float*       __restrict__ beta,
    int t_len, int n_states, int n_edges, int n_base, int group,
    int stride_b, int stride_t,
    int semiring)
{
    extern __shared__ float sh[];
    int b   = blockIdx.x;
    int i   = threadIdx.x;            // dest (forward) or source (backward)
    int n_score = n_states * n_edges;
    long long row_b = (long long)b * stride_b;

    if (blockIdx.y == 0) {
        float* al = alpha + (long long)b * (t_len + 1) * n_states;
        sh[i] = 0.0f;                 // semiring identity
        al[i] = 0.0f;
        __syncthreads();
        for (int t = 0; t < t_len; ++t) {
            const float* row = scores + (row_b + (long long)t * stride_t) * n_score;
            float acc[CRF_MAX_EDGES];
            for (int e = 0; e < n_edges; ++e) {
                acc[e] = sh[crf_source_state(i, e, n_base, group)] + row[i * n_edges + e];
            }
            float v = crf_reduce(acc, n_edges, semiring);
            __syncthreads();                                  // reads retired
            sh[i] = v;
            al[(long long)(t + 1) * n_states + i] = v;
            __syncthreads();                                  // sh visible
        }
    } else {
        // Outgoing edges of state i are the inverse of source_state: the blank
        // edge back to i, plus n_base move edges all carrying edge index
        // 1 + i/group and landing on the block (i % group)*n_base .. + n_base.
        float* be = beta + (long long)b * (t_len + 1) * n_states;
        sh[i] = 0.0f;
        be[(long long)t_len * n_states + i] = 0.0f;
        __syncthreads();
        int edge = 1 + i / group;
        int blk  = (i % group) * n_base;
        for (int t = t_len - 1; t >= 0; --t) {
            const float* row = scores + (row_b + (long long)t * stride_t) * n_score;
            float acc[CRF_MAX_EDGES];
            acc[0] = row[i * n_edges] + sh[i];
            for (int k = 0; k < n_base; ++k) {
                int dest = blk + k;
                acc[1 + k] = row[dest * n_edges + edge] + sh[dest];
            }
            float v = crf_reduce(acc, n_edges, semiring);
            __syncthreads();
            sh[i] = v;
            be[(long long)t * n_states + i] = v;
            __syncthreads();
        }
    }
}

// Per-timestep edge scores alpha[t][source] + score + beta[t+1][dest], in the
// encoder's native [dest][edge] order. Shared by both consumers below.
__device__ __forceinline__ void crf_edge_scores(
    const float* __restrict__ row,
    const float* __restrict__ a,
    const float* __restrict__ bt,
    float* out,
    int n_score, int n_edges, int n_base, int group)
{
    for (int i = threadIdx.x; i < n_score; i += blockDim.x) {
        int dest = i / n_edges;
        int e    = i - dest * n_edges;
        out[i] = a[crf_source_state(dest, e, n_base, group)] + row[i] + bt[dest];
    }
}

// -------------------------------------------------------------------------
// Pass 1 consumer: edge posteriors, written back over `scores` as
// log(softmax + 1e-8) — the input pass 2 runs its Viterbi over.
//
// Grid (batch, t_len), one block per timestep. Overwriting `scores` in place is
// safe for the same reason the CPU scratch reuses one buffer: timestep t's
// posterior depends only on scores[t], alpha[t] and beta[t+1], and both sweeps
// have already completed.
//
// Shared: n_score edge scores, then blockDim.x reduction slots.
// -------------------------------------------------------------------------
extern "C" __global__
void crf_posterior_kernel(
    float*       __restrict__ scores,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    int t_len, int n_states, int n_edges, int n_base, int group,
    int stride_b, int stride_t,
    float floor_val)
{
    extern __shared__ float sh[];
    int b = blockIdx.x;
    int t = blockIdx.y;
    int n_score = n_states * n_edges;

    float* es  = sh;                 // n_score
    float* red = sh + n_score;       // blockDim.x

    float*       row = scores + ((long long)b * stride_b + (long long)t * stride_t) * n_score;
    const float* a   = alpha  + ((long long)b * (t_len + 1) + t) * n_states;
    const float* bt  = beta   + ((long long)b * (t_len + 1) + t + 1) * n_states;

    crf_edge_scores(row, a, bt, es, n_score, n_edges, n_base, group);
    __syncthreads();

    float m = -CRF_INF;
    for (int i = threadIdx.x; i < n_score; i += blockDim.x) m = fmaxf(m, es[i]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float part = 0.0f;
    for (int i = threadIdx.x; i < n_score; i += blockDim.x) {
        float e = expf(es[i] - m);
        es[i] = e;
        part += e;
    }
    red[threadIdx.x] = part;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();

    // The floor is applied to the probability, not the log, so it cannot be
    // folded into x - logsumexp. fmaf matches the scalar path's mul_add.
    for (int i = threadIdx.x; i < n_score; i += blockDim.x) {
        row[i] = logf(fmaf(es[i], inv, floor_val));
    }
}

// -------------------------------------------------------------------------
// Pass 2 consumer: the winning edge per timestep.
//
// In native order the flat index *is* bonito's dest * n_edges + edge, so the
// tie-break key is just `i` and ties resolve to the lowest index — the same
// rule every CPU backend uses.
// -------------------------------------------------------------------------
extern "C" __global__
void crf_viterbi_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    unsigned char* __restrict__ path,
    float*         __restrict__ path_score,
    int t_len, int n_states, int n_edges, int n_base, int group,
    int stride_b, int stride_t)
{
    extern __shared__ float sh[];
    int b = blockIdx.x;
    int t = blockIdx.y;
    int n_score = n_states * n_edges;

    float* es   = sh;                                  // n_score
    float* rval = sh + n_score;                        // blockDim.x
    int*   rkey = (int*)(sh + n_score + blockDim.x);   // blockDim.x

    const float* row = scores + ((long long)b * stride_b + (long long)t * stride_t) * n_score;
    const float* a   = alpha  + ((long long)b * (t_len + 1) + t) * n_states;
    const float* bt  = beta   + ((long long)b * (t_len + 1) + t + 1) * n_states;

    crf_edge_scores(row, a, bt, es, n_score, n_edges, n_base, group);
    __syncthreads();

    float best = -CRF_INF;
    int   bkey = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < n_score; i += blockDim.x) {
        float v = es[i];
        if (v > best || (v == best && i < bkey)) { best = v; bkey = i; }
    }
    rval[threadIdx.x] = best;
    rkey[threadIdx.x] = bkey;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float v = rval[threadIdx.x + s];
            int   k = rkey[threadIdx.x + s];
            if (v > rval[threadIdx.x] || (v == rval[threadIdx.x] && k < rkey[threadIdx.x])) {
                rval[threadIdx.x] = v;
                rkey[threadIdx.x] = k;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int key  = rkey[0];
        int dest = key / n_edges;
        int edge = key - dest * n_edges;
        // 0 = blank; otherwise 1 + emitted base, which the destination alone
        // fixes (see CrfLayout::emitted_base).
        path[(long long)b * t_len + t] =
            (edge == 0) ? (unsigned char)0 : (unsigned char)(1 + (dest % n_base));
        // The winning edge score is the best path's total, identical at every
        // timestep in exact arithmetic. `CrfScratch::path_score` takes the max
        // over t rather than trusting one, so float noise cannot make it depend
        // on which timestep was read; the host does that reduction over this.
        // `path_score` is null when the caller does not want it.
        if (path_score != 0) path_score[(long long)b * t_len + t] = rval[0];
    }
}

// -------------------------------------------------------------------------
// Constrained reference scan: log P(reference | signal) for every reference,
// from the *raw* scores, on the device.
//
// This is `RefChains::forward` with the batch axis added, and it exists so
// `--ref-scores` stops dragging the whole decode back to the host. Without it
// the scoring path copies the entire score tensor out (1.5 MB per read) and
// runs the CPU lattice decode, which is why that flag measured +57% wall and
// +5.5 cores while the device idled (#297).
//
// Placement is forced: the scan needs the raw scores, and `crf_posterior_kernel`
// overwrites them in place with the pass-1 log-posteriors. It runs between the
// two, exactly where the CPU path runs it and for the same reason.
//
// One block per read, a grid-stride loop over cells, and both alpha buffers in
// shared memory. Cells are only ~1 k floats, so double-buffering them costs
// ~8 KB of shared and the whole t-sweep stays on chip; the host falls back to
// the CPU scan when they would not fit. A single __syncthreads() per timestep
// is enough because reads and writes go to *different* buffers within a step —
// unlike `crf_sweep_kernel`, which overwrites the row it reads.
//
// Indices arrive in the encoder's native `dest * n_edges + edge` order. The CPU
// scan's are into its transposed row instead; the host converts on upload,
// because this kernel deliberately never builds a transposed copy (see the
// module note).
//
// `logsumexp` is transcribed term for term from the scalar reference, down to
// seeding the sum at 1.0 and skipping the max's own index: the output is a
// continuous score a user thresholds with `--min-crf-margin`, not an argmax, so
// it is worth staying as close to the reference as the reassociation allows.
// -------------------------------------------------------------------------
extern "C" __global__
void crf_refscan_kernel(
    const float*        __restrict__ scores,
    const float*        __restrict__ alpha,
    const unsigned int* __restrict__ stay,
    const unsigned int* __restrict__ move_off,
    const unsigned int* __restrict__ move_src,
    const unsigned int* __restrict__ move_score,
    const unsigned int* __restrict__ finals,
    float*              __restrict__ out,
    int t_len, int n_score, int n_states, int n_cells, int n_start, int n_refs,
    int stride_b, int stride_t)
{
    extern __shared__ float sh[];
    float* cur  = sh;
    float* next = sh + n_cells;

    int b = blockIdx.x;
    long long row_b = (long long)b * stride_b;

    // Chain position 0 is every legal start; everything else is unreachable
    // until a move gets there.
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x) {
        cur[c] = (c < n_start) ? 0.0f : -CRF_INF;
    }
    __syncthreads();

    for (int t = 0; t < t_len; ++t) {
        const float* row = scores + (row_b + (long long)t * stride_t) * n_score;
        for (int c = threadIdx.x; c < n_cells; c += blockDim.x) {
            float terms[CRF_MAX_EDGES];
            int n = 0;
            terms[n++] = cur[c] + row[stay[c]];
            unsigned int lo = move_off[c];
            unsigned int hi = move_off[c + 1];
            for (unsigned int i = lo; i < hi; ++i) {
                terms[n++] = cur[move_src[i]] + row[move_score[i]];
            }

            // logsumexp, transcribed from the scalar reference: first strict
            // max wins, sum starts at 1.0 for that term, the rest are shifted.
            float m = -CRF_INF;
            int at = 0;
            for (int j = 0; j < n; ++j) {
                if (terms[j] > m) { m = terms[j]; at = j; }
            }
            if (!crf_finite(m)) {
                next[c] = m;
                continue;
            }
            float sum = 1.0f;
            for (int j = 0; j < n; ++j) {
                if (j != at) sum += expf(terms[j] - m);
            }
            next[c] = m + logf(sum);
        }
        __syncthreads();
        float* tmp = cur; cur = next; next = tmp;
    }

    // logZ_full, normalising the chain's raw finals into log P(ref | signal).
    //
    // Computed here rather than on the host so nothing but `n_refs` floats per
    // read leaves the device: `alpha`'s last row alone would be a strided
    // gather over 157 MB at batch 512, which is the transfer this whole path
    // exists to remove. One thread does it because it is 256 values once per
    // read against 300 * n_cells inside the loop, and a sequential reduction is
    // what `Semiring::Log::reduce` does — max, then sum every term including
    // the max's own.
    __shared__ float logz;
    if (threadIdx.x == 0) {
        const float* a = alpha + ((long long)b * (t_len + 1) + t_len) * n_states;
        float m = -CRF_INF;
        for (int i = 0; i < n_states; ++i) {
            if (a[i] > m) m = a[i];
        }
        if (!crf_finite(m)) {
            logz = m;
        } else {
            float sum = 0.0f;
            for (int i = 0; i < n_states; ++i) sum += expf(a[i] - m);
            logz = m + logf(sum);
        }
    }
    __syncthreads();

    for (int r = threadIdx.x; r < n_refs; r += blockDim.x) {
        out[(long long)b * n_refs + r] = cur[finals[r]] - logz;
    }
}
"#;

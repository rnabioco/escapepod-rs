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
    }
}
"#;

//! Sparse CTC-CRF lattice: forward/backward scores, edge posteriors, Viterbi.
//!
//! This is a port of bonito's `CTC_CRF` decode (`bonito/crf/model.py` plus the
//! CUDA kernels behind `koi.ctc.{fwd,bwd}_scores_cu_sparse` and
//! `logZ_cu_sparse`). Bonito's own decode runs entirely on the GPU through
//! `koi`; nothing here needs a GPU, and nothing here is expressible in ONNX,
//! which is why the encoder ships as ONNX and the decode lives in Rust.
//!
//! # The lattice
//!
//! A state is a `state_len`-mer over `n_base` bases, so there are
//! `n_states = n_base ** state_len` of them (256 for the RNA004 barcode model).
//! Each state has `n_edges = n_base + 1` incoming edges: one blank/stay edge and
//! `n_base` move edges. The encoder emits one score per edge per timestep, so
//! `n_score = n_states * n_edges` (1280, *not* 1024 — 1024 is the width of the
//! linear layer before `LinearCRFEncoder` expands blanks to one per state).
//!
//! Edges are indexed `(destination_state, dropped_base)`, not
//! `(source_state, emitted_base)`. bonito builds the map as a tensor
//! (`CTC_CRF.idx`); the closed form it works out to is
//!
//! ```text
//! source(c, 0)     = c                        // blank: stay in place
//! source(c, 1 + j) = j * (n_states / n_base) + c / n_base
//! ```
//!
//! Reading that backwards: destination `c` spells `[c3 c2 c1 c0]` in base
//! `n_base`, the source spells `[j c3 c2 c1]`, so a move drops the oldest base
//! `j` off the front and appends `c0` at the back. The base *emitted* by a move
//! edge is therefore fixed by the destination alone (`c % n_base`) and the edge
//! index only says which base fell out. That inversion is the single easiest
//! thing to get backwards in this port, and it decodes to plausible-looking
//! garbage if you do, so [`CrfLayout::source_state`] is checked against a
//! transcription of bonito's tensor construction in the tests below.
//!
//! # The decode
//!
//! `SeqdistModel.decode_batch` is two passes, and both are needed to match:
//!
//! 1. **Log semiring.** Forward/backward over the lattice, then per-timestep
//!    edge posteriors — `softmax` over all `n_score` edges of
//!    `alpha[t][source] + score + beta[t+1][destination]`. bonito gets these as
//!    the *gradient* of `logZ` (`SequenceDist.posteriors` is literally
//!    `autograd.grad`); the closed form of that gradient is the softmax above.
//! 2. **Max semiring** over `log(posteriors + 1e-8)`. Forward/backward again,
//!    then the argmax edge at each timestep. Under the max semiring the
//!    gradient is a one-hot indicator of the best path, so bonito's
//!    `posteriors(...).argmax(2)` and a plain Viterbi traceback pick out the
//!    same edge.
//!
//! Decoding the raw encoder scores directly — one Viterbi, no posterior pass —
//! is a different and worse decode. The two-pass structure is load-bearing.

use std::fmt;

use super::refchain::RefChains;

/// Natural log of the blank floor bonito adds before taking logs
/// (`posteriors(x) + 1e-8`). Kept as the literal so the port reads like the
/// Python.
pub(super) const POSTERIOR_FLOOR: f32 = 1e-8;

/// Geometry of a bonito-style CTC-CRF lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrfLayout {
    /// Number of real bases (4).
    pub n_base: usize,
    /// Length of the k-mer a state encodes (4).
    pub state_len: usize,
    /// `n_base ** state_len`.
    pub n_states: usize,
    /// `n_base + 1` — one blank edge plus one move edge per dropped base.
    pub n_edges: usize,
    /// `n_states * n_edges` — the encoder's output width.
    pub n_score: usize,
    /// `n_states / n_base` — the stride separating dropped-base groups.
    group: usize,
}

impl CrfLayout {
    /// `n_states / n_base` — the block of destination states that share a
    /// source under a move edge.
    ///
    /// Only the AVX2/AVX-512 kernels need this; the scalar path derives the
    /// source with [`CrfLayout::source_state`] instead.
    #[inline]
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(super) fn group(&self) -> usize {
        self.group
    }
}

impl CrfLayout {
    /// Build the layout for a `state_len`-mer alphabet over `n_base` bases.
    ///
    /// Returns `None` if the geometry would overflow or is degenerate.
    pub fn new(n_base: usize, state_len: usize) -> Option<Self> {
        if n_base < 2 || state_len == 0 || state_len > 8 {
            return None;
        }
        let n_states = n_base.checked_pow(state_len as u32)?;
        let n_edges = n_base + 1;
        let n_score = n_states.checked_mul(n_edges)?;
        Some(Self {
            n_base,
            state_len,
            n_states,
            n_edges,
            n_score,
            group: n_states / n_base,
        })
    }

    /// The source state of edge `edge` into destination state `dest`.
    ///
    /// `edge == 0` is the blank/stay edge; `edge == 1 + j` is the move edge
    /// that drops base `j` off the front of the k-mer.
    #[inline]
    pub fn source_state(&self, dest: usize, edge: usize) -> usize {
        debug_assert!(dest < self.n_states && edge < self.n_edges);
        if edge == 0 {
            dest
        } else {
            (edge - 1) * self.group + dest / self.n_base
        }
    }

    /// The base emitted by edge `edge` into destination `dest`, or `None` for
    /// the blank edge. The value is an index into `alphabet[1..]`.
    #[inline]
    pub fn emitted_base(&self, dest: usize, edge: usize) -> Option<usize> {
        (edge != 0).then(|| dest % self.n_base)
    }

    /// Flat index of edge `edge` into `dest` within a timestep's score row.
    ///
    /// The encoder's output is `dest * n_edges + edge` (bonito reshapes
    /// `[T, N, n_score]` to `[T, N, n_states, n_edges]`), which is *not* the
    /// layout the kernels here use internally — see [`transpose_scores`].
    #[inline]
    pub fn score_index(&self, dest: usize, edge: usize) -> usize {
        dest * self.n_edges + edge
    }
}

/// Errors from the lattice decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrfDecodeError {
    /// The score buffer isn't `t_len * n_score` long.
    ScoreLen { got: usize, expected: usize },
    /// The alphabet doesn't have `n_base + 1` symbols.
    AlphabetLen { got: usize, expected: usize },
}

impl fmt::Display for CrfDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScoreLen { got, expected } => {
                write!(f, "score buffer has {got} floats, expected {expected}")
            }
            Self::AlphabetLen { got, expected } => {
                write!(f, "alphabet has {got} symbols, expected {expected}")
            }
        }
    }
}

impl std::error::Error for CrfDecodeError {}

/// Rewrite one timestep of encoder scores from bonito's `[dest][edge]` order
/// into the `[edge][dest]` order the kernels want.
///
/// Every kernel below walks the destination axis contiguously, so transposing
/// once up front turns each of the five edge classes into its own dense row and
/// keeps all the inner loops unit-stride.
pub fn transpose_scores(layout: &CrfLayout, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), layout.n_score);
    debug_assert_eq!(dst.len(), layout.n_score);
    for edge in 0..layout.n_edges {
        let row = &mut dst[edge * layout.n_states..(edge + 1) * layout.n_states];
        for (dest, out) in row.iter_mut().enumerate() {
            *out = src[dest * layout.n_edges + edge];
        }
    }
}

/// Reduction used to combine the edges arriving at (or leaving) a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Semiring {
    /// `logsumexp` — bonito's `Log`, used for the posterior pass.
    Log,
    /// `max` — bonito's `Max`, used for the Viterbi pass.
    Max,
}

impl Semiring {
    /// Combine `n_edges` values. Matches `torch.logsumexp` / `torch.max`,
    /// including logsumexp's max-shift for stability.
    #[inline]
    fn reduce(self, vals: &[f32]) -> f32 {
        let mut m = f32::NEG_INFINITY;
        for &v in vals {
            if v > m {
                m = v;
            }
        }
        match self {
            Self::Max => m,
            Self::Log => {
                if !m.is_finite() {
                    return m;
                }
                let mut sum = 0.0f32;
                for &v in vals {
                    sum += (v - m).exp();
                }
                m + sum.ln()
            }
        }
    }
}

/// Reusable buffers for decoding one read at a time.
///
/// Allocating these per read shows up immediately at demux scale (one read is
/// ~2 MB of scratch), so the decoder threads one of these through.
#[derive(Debug, Clone, Default)]
pub struct CrfScratch {
    /// `[t][edge][dest]` — encoder scores, transposed; then overwritten in
    /// place by `log(posterior + 1e-8)`, which is pass 2's input.
    ///
    /// One buffer rather than two because the reuse is safe and it is 1 MB per
    /// decoding thread: pass 1's forward and backward sweeps have both finished
    /// before any posterior is written, and timestep `t`'s posterior depends
    /// only on `scores[t]` (plus `alpha[t]` and `beta[t+1]`), so writing it back
    /// over `scores[t]` destroys nothing still needed. It also halves the hot
    /// working set, which matters on a 1 MB L2.
    scores: Vec<f32>,
    /// `[t][state]`, `t` in `0..=t_len`.
    alpha: Vec<f32>,
    /// `[t][state]`, `t` in `0..=t_len`.
    beta: Vec<f32>,
    /// Per-timestep edge scratch, `n_score` long.
    edge: Vec<f32>,
    /// Reshape scratch for the SIMD kernels, `n_base * n_states` long.
    work: Vec<f32>,
    /// Decoded path, `t_len` long: 0 for blank, else `1 + base`.
    path: Vec<u8>,
    /// Winning edge per timestep as a flat `dest * n_edges + edge` index —
    /// bonito's `posteriors(s2, Max).argmax(2)`.
    traceback: Vec<u32>,
    /// `logsumexp(alpha[t_len])` under the log semiring: the partition function
    /// over every path. Recorded because pass 1 has already computed it and it
    /// is what normalises a constrained score into a probability (#241).
    logz: f32,
    /// The best path's total score under the max semiring, in log-posterior
    /// space. Also already computed — `argmax_edge` finds it at every timestep
    /// and used to throw it away.
    path_score: f32,
    /// Double buffer for [`super::refchain::RefChains::forward`], kept here so
    /// a per-read reference scoring allocates nothing.
    chain_cur: Vec<f32>,
    chain_next: Vec<f32>,
}

impl CrfScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Size the buffers for one read.
    ///
    /// Deliberately *not* `clear()` + `resize()`: every buffer here is fully
    /// overwritten before it is read, so re-zeroing costs a ~2 MB memset per
    /// read and buys nothing. Decoding a run of same-length reads — which is
    /// the only thing this is ever used for, since the encoder window is fixed
    /// — then leaves these as no-ops.
    fn reserve(&mut self, layout: &CrfLayout, t_len: usize) {
        let lattice = (t_len + 1) * layout.n_states;
        let edges = t_len * layout.n_score;
        self.scores.resize(edges, 0.0);
        self.alpha.resize(lattice, 0.0);
        self.beta.resize(lattice, 0.0);
        self.edge.resize(layout.n_score, 0.0);
        self.work.resize(layout.n_base * layout.n_states, 0.0);
        self.path.resize(t_len, 0);
        self.traceback.resize(t_len, 0);
    }

    /// The decoded path from the most recent decode: 0 = blank, `1 + base`
    /// otherwise. Same convention as bonito's `CTC_CRF.viterbi`.
    pub fn path(&self) -> &[u8] {
        &self.path
    }

    /// The winning edge per timestep, as the flat `dest * n_edges + edge`
    /// index bonito's `posteriors(scores, Max).argmax(2)` returns.
    ///
    /// Strictly more information than [`path`](Self::path), which keeps only
    /// the emitted base — useful for checking a port against bonito, since it
    /// pins the destination state too.
    pub fn traceback(&self) -> &[u32] {
        &self.traceback
    }

    /// `logZ` over the whole lattice from the most recent decode — the log of
    /// the total probability mass over every path, and the denominator that
    /// turns a constrained score into `log P(sequence | signal)`.
    pub fn logz(&self) -> f32 {
        self.logz
    }

    /// The decoded path's total score from the most recent decode, as a sum of
    /// per-timestep log-posteriors (so `<= 0`, and `path_score / t_len` is the
    /// mean per-timestep log-posterior).
    ///
    /// A measure of how peaked the lattice is around its own answer, which is
    /// *not* the same question as how much it preferred that answer to a
    /// specific alternative — see [`super::refchain`] for the latter.
    pub fn path_score(&self) -> f32 {
        self.path_score
    }
}

/// Forward scores. `alpha[0]` is the semiring identity (0) and
/// `alpha[t + 1][c] = reduce_edge(alpha[t][source(c, edge)] + score[t][edge][c])`.
fn forward(layout: &CrfLayout, scores: &[f32], t_len: usize, s: Semiring, alpha: &mut [f32]) {
    let (n_states, n_edges) = (layout.n_states, layout.n_edges);
    let mut acc = vec![0.0f32; n_edges];
    alpha[..n_states].fill(0.0);
    for t in 0..t_len {
        let (prev, next) = alpha.split_at_mut((t + 1) * n_states);
        let prev = &prev[t * n_states..];
        let row = &scores[t * layout.n_score..(t + 1) * layout.n_score];
        for dest in 0..n_states {
            for (edge, a) in acc.iter_mut().enumerate() {
                *a = prev[layout.source_state(dest, edge)] + row[edge * n_states + dest];
            }
            next[dest] = s.reduce(&acc);
        }
    }
}

/// Backward scores. `beta[t_len]` is the semiring identity and
/// `beta[t][s] = reduce_edge(score[t][edge][dest(s, edge)] + beta[t + 1][dest(s, edge)])`.
///
/// The outgoing edges of state `s` are the inverse of [`CrfLayout::source_state`]:
/// the blank edge back to `s`, plus `n_base` move edges that all carry edge
/// index `1 + s / group` and land on the contiguous block
/// `(s % group) * n_base .. + n_base`.
fn backward(layout: &CrfLayout, scores: &[f32], t_len: usize, s: Semiring, beta: &mut [f32]) {
    let (n_states, n_edges, n_base) = (layout.n_states, layout.n_edges, layout.n_base);
    let mut acc = vec![0.0f32; n_edges];
    beta[t_len * n_states..(t_len + 1) * n_states].fill(0.0);
    for t in (0..t_len).rev() {
        let (cur, next) = beta.split_at_mut((t + 1) * n_states);
        let cur = &mut cur[t * n_states..];
        let row = &scores[t * layout.n_score..(t + 1) * layout.n_score];
        for (src, out) in cur.iter_mut().enumerate() {
            acc[0] = row[src] + next[src];
            let edge = 1 + src / layout.group;
            let block = (src % layout.group) * n_base;
            for k in 0..n_base {
                let dest = block + k;
                acc[1 + k] = row[edge * n_states + dest] + next[dest];
            }
            *out = s.reduce(&acc);
        }
    }
}

/// Per-timestep edge scores `alpha[t][source] + score + beta[t + 1][dest]`.
///
/// This is the quantity bonito differentiates `logZ` to get; under the log
/// semiring its softmax is the edge posterior, and under the max semiring its
/// argmax is the edge on the best path.
fn edge_scores(
    layout: &CrfLayout,
    row: &[f32],
    alpha_t: &[f32],
    beta_next: &[f32],
    out: &mut [f32],
) {
    let n_states = layout.n_states;
    for edge in 0..layout.n_edges {
        let (sl, ol) = (edge * n_states, edge * n_states);
        for dest in 0..n_states {
            out[ol + dest] =
                alpha_t[layout.source_state(dest, edge)] + row[sl + dest] + beta_next[dest];
        }
    }
}

/// Which kernel family the lattice passes run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable reference implementation. Direct index arithmetic, no reshapes.
    Scalar,
    /// AVX2 + FMA, with polynomial `exp`/`ln`. Requires `n_base == 4` and a
    /// state count that is a multiple of 32 — true of every bonito CRF model.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512F, 16 lanes wide. Same kernels as [`Backend::Avx2`]; needs a
    /// state count that is a multiple of 64.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Backend {
    /// The fastest backend this CPU and layout support.
    pub fn best_for(layout: &CrfLayout) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if super::avx512::supported(layout) && super::avx512::available() {
                return Self::Avx512;
            }
            if super::avx2::supported(layout) && super::avx2::available() {
                return Self::Avx2;
            }
        }
        let _ = layout;
        Self::Scalar
    }

    /// Every backend this CPU and layout support, fastest last. Used by tests
    /// and benchmarks to cover each one without hardcoding a CPU model.
    pub fn all_supported(layout: &CrfLayout) -> Vec<Self> {
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut out = vec![Self::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if super::avx2::supported(layout) && super::avx2::available() {
                out.push(Self::Avx2);
            }
            if super::avx512::supported(layout) && super::avx512::available() {
                out.push(Self::Avx512);
            }
        }
        let _ = layout;
        out
    }
}

fn forward_dispatch(
    layout: &CrfLayout,
    scores: &[f32],
    t_len: usize,
    s: Semiring,
    alpha: &mut [f32],
    work: &mut [f32],
    backend: Backend,
) {
    match backend {
        Backend::Scalar => forward(layout, scores, t_len, s, alpha),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `Backend::Avx2` is only produced by `best_for`, which checks
        // both the runtime CPU features and the layout constraints the kernels
        // assume. `work` is sized `n_base * n_states` by `CrfScratch::reserve`.
        Backend::Avx2 => unsafe { super::avx2::forward(layout, scores, t_len, s, alpha, work) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see above; `Avx512` additionally implies AVX512F.
        Backend::Avx512 => unsafe { super::avx512::forward(layout, scores, t_len, s, alpha, work) },
    }
    let _ = work;
}

fn backward_dispatch(
    layout: &CrfLayout,
    scores: &[f32],
    t_len: usize,
    s: Semiring,
    beta: &mut [f32],
    work: &mut [f32],
    backend: Backend,
) {
    match backend {
        Backend::Scalar => backward(layout, scores, t_len, s, beta),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `forward_dispatch`.
        Backend::Avx2 => unsafe { super::avx2::backward(layout, scores, t_len, s, beta, work) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see above; `Avx512` additionally implies AVX512F.
        Backend::Avx512 => unsafe { super::avx512::backward(layout, scores, t_len, s, beta, work) },
    }
    let _ = work;
}

fn edge_scores_dispatch(
    layout: &CrfLayout,
    row: &[f32],
    alpha_t: &[f32],
    beta_next: &[f32],
    out: &mut [f32],
    work: &mut [f32],
    backend: Backend,
) {
    match backend {
        Backend::Scalar => edge_scores(layout, row, alpha_t, beta_next, out),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `forward_dispatch`.
        Backend::Avx2 => unsafe {
            super::avx2::edge_scores(layout, row, alpha_t, beta_next, out, work)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see above.
        Backend::Avx512 => unsafe {
            super::avx512::edge_scores(layout, row, alpha_t, beta_next, out, work)
        },
    }
    let _ = work;
}

fn transpose_dispatch(layout: &CrfLayout, src: &[f32], dst: &mut [f32], backend: Backend) {
    match backend {
        Backend::Scalar => transpose_scores(layout, src, dst),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `forward_dispatch`. The gathers read
        // `dest * n_edges + edge` for `dest < n_states`, whose maximum is
        // `n_score - 1`, so every lane is in bounds.
        Backend::Avx2 => unsafe {
            super::avx2::transpose_scores(src, dst, layout.n_states, layout.n_edges)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see above.
        Backend::Avx512 => unsafe {
            super::avx512::transpose_scores(src, dst, layout.n_states, layout.n_edges)
        },
    }
}

fn log_softmax_dispatch(src: &[f32], dst: &mut [f32], backend: Backend) {
    match backend {
        Backend::Scalar => log_softmax_floored(src, dst),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `forward_dispatch`; `n_score` is a multiple of 8 whenever
        // the AVX2 layout constraints hold.
        Backend::Avx2 => unsafe { super::avx2::log_softmax_floored(src, dst, POSTERIOR_FLOOR) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see above.
        Backend::Avx512 => unsafe { super::avx512::log_softmax_floored(src, dst, POSTERIOR_FLOOR) },
    }
}

/// Decode one read's encoder scores to a sequence over the alphabet, using the
/// fastest kernels this CPU supports.
///
/// `scores` is `t_len * layout.n_score` floats in the encoder's native
/// `[t][dest][edge]` order — one contiguous read sliced out of the time-major
/// `[T, batch, n_score]` output. Returns the sequence; the underlying path is
/// available from [`CrfScratch::path`].
pub fn decode(
    layout: &CrfLayout,
    alphabet: &[u8],
    scores: &[f32],
    t_len: usize,
    scratch: &mut CrfScratch,
) -> Result<String, CrfDecodeError> {
    decode_with(
        layout,
        alphabet,
        scores,
        t_len,
        scratch,
        Backend::best_for(layout),
    )
}

/// [`decode`] pinned to a specific backend.
///
/// Public so the SIMD kernels can be diffed against the scalar reference in
/// tests and benchmarks; production callers want [`decode`].
pub fn decode_with(
    layout: &CrfLayout,
    alphabet: &[u8],
    scores: &[f32],
    t_len: usize,
    scratch: &mut CrfScratch,
    backend: Backend,
) -> Result<String, CrfDecodeError> {
    decode_inner(layout, alphabet, scores, t_len, scratch, backend, None)
}

/// [`decode_with`], additionally scoring every reference in `chains` against
/// the same encoder output.
///
/// `out` receives `log P(reference | signal)` per reference, in the order the
/// chains were built — a genuine probability, because every path through the
/// lattice emits exactly one string, so restricting the forward recursion to
/// the paths that emit a reference and dividing by [`CrfScratch::logz`] is a
/// normalised quantity. Unlike the edit-distance margin to the runner-up it is
/// continuous, which is what makes it usable as a precision/recall gate (#241).
///
/// This has to live inside the decode rather than beside it: the scan reads the
/// **raw** transposed scores, and pass 1 overwrites them in place with
/// log-posteriors. Nothing is copied for the sake of it.
// `decode_with`'s parameters plus the panel and where to put its scores.
// Bundling them into a struct would name the same arguments one indirection
// further away, and this is a leaf called from three places.
#[allow(clippy::too_many_arguments)]
pub fn decode_with_refs(
    layout: &CrfLayout,
    alphabet: &[u8],
    scores: &[f32],
    t_len: usize,
    scratch: &mut CrfScratch,
    backend: Backend,
    chains: &RefChains,
    out: &mut Vec<f32>,
) -> Result<String, CrfDecodeError> {
    let seq = decode_inner(
        layout,
        alphabet,
        scores,
        t_len,
        scratch,
        backend,
        Some((chains, out)),
    )?;
    // `logZ_target - logZ_full`, applied after the scan so the scan itself
    // stays the plain partition function the reference implementation is.
    let logz = scratch.logz;
    for v in out.iter_mut() {
        *v -= logz;
    }
    Ok(seq)
}

fn decode_inner(
    layout: &CrfLayout,
    alphabet: &[u8],
    scores: &[f32],
    t_len: usize,
    scratch: &mut CrfScratch,
    backend: Backend,
    refs: Option<(&RefChains, &mut Vec<f32>)>,
) -> Result<String, CrfDecodeError> {
    if alphabet.len() != layout.n_edges {
        return Err(CrfDecodeError::AlphabetLen {
            got: alphabet.len(),
            expected: layout.n_edges,
        });
    }
    let expected = t_len * layout.n_score;
    if scores.len() != expected {
        return Err(CrfDecodeError::ScoreLen {
            got: scores.len(),
            expected,
        });
    }
    scratch.reserve(layout, t_len);

    for t in 0..t_len {
        let (lo, hi) = (t * layout.n_score, (t + 1) * layout.n_score);
        transpose_dispatch(
            layout,
            &scores[lo..hi],
            &mut scratch.scores[lo..hi],
            backend,
        );
    }

    // Pass 1: posteriors under the log semiring.
    forward_dispatch(
        layout,
        &scratch.scores,
        t_len,
        Semiring::Log,
        &mut scratch.alpha,
        &mut scratch.work,
        backend,
    );
    scratch.logz = Semiring::Log.reduce(&scratch.alpha[t_len * layout.n_states..]);

    // Before the posterior loop below overwrites `scratch.scores`: the
    // constrained scans need the raw scores, and this is the last moment they
    // exist.
    if let Some((chains, out)) = refs {
        chains.forward(
            &scratch.scores,
            t_len,
            layout.n_score,
            &mut scratch.chain_cur,
            &mut scratch.chain_next,
            out,
            backend,
        );
    }

    backward_dispatch(
        layout,
        &scratch.scores,
        t_len,
        Semiring::Log,
        &mut scratch.beta,
        &mut scratch.work,
        backend,
    );
    for t in 0..t_len {
        let (lo, hi) = (t * layout.n_score, (t + 1) * layout.n_score);
        edge_scores_dispatch(
            layout,
            &scratch.scores[lo..hi],
            &scratch.alpha[t * layout.n_states..(t + 1) * layout.n_states],
            &scratch.beta[(t + 1) * layout.n_states..(t + 2) * layout.n_states],
            &mut scratch.edge,
            &mut scratch.work,
            backend,
        );
        log_softmax_dispatch(&scratch.edge, &mut scratch.scores[lo..hi], backend);
    }

    // Pass 2: Viterbi over the log-posteriors.
    forward_dispatch(
        layout,
        &scratch.scores,
        t_len,
        Semiring::Max,
        &mut scratch.alpha,
        &mut scratch.work,
        backend,
    );
    backward_dispatch(
        layout,
        &scratch.scores,
        t_len,
        Semiring::Max,
        &mut scratch.beta,
        &mut scratch.work,
        backend,
    );

    let mut seq = String::with_capacity(t_len);
    scratch.path_score = f32::NEG_INFINITY;
    for t in 0..t_len {
        let (lo, hi) = (t * layout.n_score, (t + 1) * layout.n_score);
        edge_scores_dispatch(
            layout,
            &scratch.scores[lo..hi],
            &scratch.alpha[t * layout.n_states..(t + 1) * layout.n_states],
            &scratch.beta[(t + 1) * layout.n_states..(t + 2) * layout.n_states],
            &mut scratch.edge,
            &mut scratch.work,
            backend,
        );
        let (dest, edge) = argmax_edge(layout, &scratch.edge, backend);
        // Every path takes exactly one edge per timestep, so the winning edge
        // score *is* the best path's total score, the same value at every `t`.
        // Taken as a max rather than off one arbitrary timestep so float noise
        // cannot make it depend on which one.
        let best = scratch.edge[edge * layout.n_states + dest];
        if best > scratch.path_score {
            scratch.path_score = best;
        }
        scratch.traceback[t] = layout.score_index(dest, edge) as u32;
        match layout.emitted_base(dest, edge) {
            Some(base) => {
                scratch.path[t] = 1 + base as u8;
                seq.push(alphabet[1 + base] as char);
            }
            None => scratch.path[t] = 0,
        }
    }
    Ok(seq)
}

/// `log(softmax(x) + 1e-8)`, matching bonito's `posteriors(scores) + 1e-8`
/// followed by `.log()`.
///
/// The floor is applied to the probability, not the log, so it is not a shift
/// and this cannot be folded into `x - logsumexp(x)`.
///
/// `exp(x - max)` is kept and divided by its own sum rather than recomputed as
/// `exp(x - logsumexp)`. That is the same softmax — and the form `torch` itself
/// uses — but it costs one `exp` per element instead of two. This is the
/// hottest kernel in the decode (36% of profiled time before the change), so
/// the saving is the single largest one available.
fn log_softmax_floored(src: &[f32], dst: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in src {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (s - m).exp();
        sum += *d;
    }
    let inv = 1.0 / sum;
    for d in dst.iter_mut() {
        *d = d.mul_add(inv, POSTERIOR_FLOOR).ln();
    }
}

/// The `(dest, edge)` with the highest score, ties broken toward the lowest
/// flat index in bonito's `dest * n_edges + edge` order — which is what
/// `torch.argmax` over that axis returns.
///
/// Both backends break ties by flat index rather than by traversal order, so
/// they cannot disagree on a tie.
fn argmax_edge(layout: &CrfLayout, edges: &[f32], backend: Backend) -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: see `forward_dispatch`.
        let hit = match backend {
            Backend::Avx2 => unsafe {
                super::avx2::argmax_edge(edges, layout.n_states, layout.n_edges)
            },
            Backend::Avx512 => unsafe {
                super::avx512::argmax_edge(edges, layout.n_states, layout.n_edges)
            },
            Backend::Scalar => None,
        };
        if let Some(hit) = hit {
            return hit;
        }
    }
    let _ = backend;
    let n_states = layout.n_states;
    let (mut best, mut best_dest, mut best_edge) = (f32::NEG_INFINITY, 0usize, 0usize);
    for dest in 0..n_states {
        for edge in 0..layout.n_edges {
            let v = edges[edge * n_states + dest];
            if v > best {
                best = v;
                best_dest = dest;
                best_edge = edge;
            }
        }
    }
    (best_dest, best_edge)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Transcription of bonito's `CTC_CRF.idx` tensor construction:
    ///
    /// ```python
    /// torch.cat([
    ///     torch.arange(n_base**state_len)[:, None],
    ///     torch.arange(n_base**state_len)
    ///         .repeat_interleave(n_base)
    ///         .reshape(n_base, -1).T
    /// ], dim=1)
    /// ```
    fn bonito_idx(n_base: usize, state_len: usize) -> Vec<Vec<usize>> {
        let n_states = n_base.pow(state_len as u32);
        // `arange(n_states).repeat_interleave(n_base)` — length n_states*n_base.
        let interleaved: Vec<usize> = (0..n_states)
            .flat_map(|v| std::iter::repeat_n(v, n_base))
            .collect();
        // `.reshape(n_base, -1).T` — element [c][j] is row j, column c.
        (0..n_states)
            .map(|c| {
                let mut row = vec![c];
                row.extend((0..n_base).map(|j| interleaved[j * n_states + c]));
                row
            })
            .collect()
    }

    #[test]
    fn source_state_matches_bonito_idx() {
        for (n_base, state_len) in [(4, 4), (4, 3), (4, 5), (2, 4), (3, 3)] {
            let layout = CrfLayout::new(n_base, state_len).unwrap();
            let idx = bonito_idx(n_base, state_len);
            for (dest, row) in idx.iter().enumerate() {
                for (edge, &want) in row.iter().enumerate() {
                    assert_eq!(
                        layout.source_state(dest, edge),
                        want,
                        "n_base={n_base} state_len={state_len} dest={dest} edge={edge}"
                    );
                }
            }
        }
    }

    #[test]
    fn rna004_geometry() {
        let layout = CrfLayout::new(4, 4).unwrap();
        assert_eq!(layout.n_states, 256);
        assert_eq!(layout.n_edges, 5);
        // 1280, not 1024: `LinearCRFEncoder` expands blanks to one per state.
        assert_eq!(layout.n_score, 1280);
        assert_eq!(layout.source_state(0, 0), 0);
        assert_eq!(layout.source_state(255, 0), 255);
        // Move edges drop base j off the front: source spells [j, c3, c2, c1].
        assert_eq!(layout.source_state(0, 1), 0);
        assert_eq!(layout.source_state(0, 4), 192);
        assert_eq!(layout.source_state(255, 1), 63);
        assert_eq!(layout.source_state(255, 4), 255);
    }

    /// Every state must have exactly `n_edges` outgoing edges, and the
    /// backward pass's closed form must enumerate exactly those.
    #[test]
    fn outgoing_edges_invert_source_state() {
        let layout = CrfLayout::new(4, 4).unwrap();
        for src in 0..layout.n_states {
            let mut expected: Vec<(usize, usize)> = Vec::new();
            for dest in 0..layout.n_states {
                for edge in 0..layout.n_edges {
                    if layout.source_state(dest, edge) == src {
                        expected.push((dest, edge));
                    }
                }
            }
            let mut got = vec![(src, 0)];
            let edge = 1 + src / layout.group;
            let block = (src % layout.group) * layout.n_base;
            got.extend((0..layout.n_base).map(|k| (block + k, edge)));
            expected.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, expected, "state {src}");
        }
    }

    /// The gather-based transposes must be exactly the scalar one — this is a
    /// pure data movement, so unlike the arithmetic kernels there is no
    /// tolerance to allow.
    #[test]
    fn every_backend_transposes_identically() {
        let layout = CrfLayout::new(4, 4).unwrap();
        let src: Vec<f32> = (0..layout.n_score)
            .map(|i| (i as f32) * 0.5 - 3.0)
            .collect();
        let mut want = vec![0.0; layout.n_score];
        transpose_scores(&layout, &src, &mut want);
        for backend in Backend::all_supported(&layout) {
            let mut got = vec![f32::NAN; layout.n_score];
            transpose_dispatch(&layout, &src, &mut got, backend);
            assert_eq!(got, want, "{backend:?}");
        }
    }

    #[test]
    fn transpose_round_trips() {
        let layout = CrfLayout::new(4, 4).unwrap();
        let src: Vec<f32> = (0..layout.n_score).map(|i| i as f32).collect();
        let mut dst = vec![0.0; layout.n_score];
        transpose_scores(&layout, &src, &mut dst);
        for dest in 0..layout.n_states {
            for edge in 0..layout.n_edges {
                assert_eq!(
                    dst[edge * layout.n_states + dest],
                    src[layout.score_index(dest, edge)]
                );
            }
        }
    }

    /// With all edge scores equal, every path is equally likely, so posteriors
    /// are uniform and `alpha[t]` grows by exactly `ln(n_edges)` per step.
    #[test]
    fn uniform_scores_give_uniform_forward() {
        let layout = CrfLayout::new(4, 4).unwrap();
        let t_len = 8;
        let scores = vec![0.0f32; t_len * layout.n_score];
        let mut alpha = vec![0.0f32; (t_len + 1) * layout.n_states];
        forward(&layout, &scores, t_len, Semiring::Log, &mut alpha);
        let expected = (layout.n_edges as f32).ln();
        for t in 1..=t_len {
            for s in 0..layout.n_states {
                let want = expected * t as f32;
                assert!(
                    (alpha[t * layout.n_states + s] - want).abs() < 1e-3,
                    "t={t} s={s} got {} want {want}",
                    alpha[t * layout.n_states + s]
                );
            }
        }
    }

    #[test]
    fn max_semiring_forward_is_zero_for_uniform_scores() {
        let layout = CrfLayout::new(4, 4).unwrap();
        let t_len = 8;
        let scores = vec![0.0f32; t_len * layout.n_score];
        let mut alpha = vec![0.0f32; (t_len + 1) * layout.n_states];
        forward(&layout, &scores, t_len, Semiring::Max, &mut alpha);
        assert!(alpha.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn log_softmax_floored_sums_to_one() {
        let src: Vec<f32> = (0..1280).map(|i| ((i % 37) as f32 - 18.0) / 4.0).collect();
        let mut dst = vec![0.0f32; src.len()];
        log_softmax_floored(&src, &mut dst);
        let total: f64 = dst.iter().map(|&v| (v as f64).exp()).sum();
        assert!((total - 1.0).abs() < 1e-4, "posteriors sum to {total}");
    }

    /// Deterministic stand-in for encoder output, reproducible from a formula
    /// so tests need no fixture file.
    pub(crate) fn synthetic_scores(layout: &CrfLayout, t_len: usize, seed: f64) -> Vec<f32> {
        (0..t_len * layout.n_score)
            .map(|i| {
                let t = (i / layout.n_score) as f64;
                let c = (i % layout.n_score) as f64;
                ((0.1 * t + 0.01 * c + seed).sin() * 2.0 + (0.013 * i as f64).cos()) as f32
            })
            .collect()
    }

    /// The SIMD kernels use polynomial `exp`/`ln` and reassociate one sum, so
    /// they are not bit-identical to the scalar reference. What must hold is
    /// that every backend decodes the same sequence and stays within a tight
    /// tolerance.
    #[test]
    fn simd_backends_agree_with_scalar() {
        let layout = CrfLayout::new(4, 4).unwrap();
        let backends = Backend::all_supported(&layout);
        if backends.len() == 1 {
            eprintln!("skipping: no SIMD backend on this host");
            return;
        }
        let t_len = 60;
        for seed in [0.0, 0.7, 2.3] {
            let scores = synthetic_scores(&layout, t_len, seed);
            let mut sc = CrfScratch::new();
            let want =
                decode_with(&layout, b"NACGT", &scores, t_len, &mut sc, Backend::Scalar).unwrap();
            let want_alpha = sc.alpha.clone();
            let want_post = sc.scores.clone();

            for backend in backends.iter().copied().filter(|&b| b != Backend::Scalar) {
                let mut sa = CrfScratch::new();
                let got = decode_with(&layout, b"NACGT", &scores, t_len, &mut sa, backend).unwrap();

                assert_eq!(got, want, "{backend:?}: sequence differs (seed {seed})");
                assert_eq!(
                    sa.path(),
                    sc.path(),
                    "{backend:?}: path differs (seed {seed})"
                );
                assert_eq!(
                    sa.traceback(),
                    sc.traceback(),
                    "{backend:?}: traceback differs (seed {seed})"
                );

                let worst_alpha = max_abs_diff(&sa.alpha, &want_alpha);
                let worst_post = max_abs_diff(&sa.scores, &want_post);
                assert!(
                    worst_alpha < 2e-3,
                    "{backend:?}: alpha diverges by {worst_alpha:e}"
                );
                assert!(
                    worst_post < 2e-4,
                    "{backend:?}: posteriors diverge by {worst_post:e}"
                );
            }
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// A decode must only ever emit symbols from `alphabet[1..]`; blanks
    /// contribute nothing. Guards against an off-by-one in the edge→base map.
    #[test]
    fn decoded_symbols_come_from_the_alphabet() {
        let layout = CrfLayout::new(4, 4).unwrap();
        let t_len = 40;
        let scores = synthetic_scores(&layout, t_len, 1.1);
        let mut scratch = CrfScratch::new();
        let seq = decode(&layout, b"NACGT", &scores, t_len, &mut scratch).unwrap();
        assert!(seq.bytes().all(|b| b"ACGT".contains(&b)), "got {seq:?}");
        assert_eq!(
            seq.len(),
            scratch.path().iter().filter(|&&p| p != 0).count()
        );
        assert!(scratch.path().iter().all(|&p| p <= 4));
    }

    #[test]
    fn rejects_bad_lengths() {
        let layout = CrfLayout::new(4, 4).unwrap();
        let mut scratch = CrfScratch::new();
        let err = decode(&layout, b"NACGT", &[0.0; 10], 4, &mut scratch).unwrap_err();
        assert!(matches!(err, CrfDecodeError::ScoreLen { .. }));
        let err = decode(&layout, b"NACG", &[0.0; 1280], 1, &mut scratch).unwrap_err();
        assert!(matches!(err, CrfDecodeError::AlphabetLen { .. }));
    }
}

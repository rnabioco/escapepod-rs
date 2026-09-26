// SPDX-License-Identifier: MIT

//! Score-only inter-sequence kernels.
//!
//! **Inter-sequence** layout: one SIMD lane per *reference*, so a vector holds
//! the same DP cell `(i, j)` of 16 (AVX2) or 32 (AVX-512BW) different
//! read-vs-reference problems, all driven by one read. Nothing is shuffled
//! between lanes, no lane waits on another, and the recurrence is the scalar
//! one verbatim — which is why the result can be required to *equal* the
//! scalar oracle rather than approximate it.
//!
//! # Layout
//!
//! References are sorted by length (longest first) and cut into groups of one
//! lane width, so a group's padding is only the spread of lengths inside it.
//! Each group carries a *profile*: for every reference column `j` and every
//! read code `c`, the lane-wise substitution score `s(c, ref_lane[j])`. The DP
//! walks the reference column by column (`j` outer) and the read row by row
//! (`i` inner), so a column's five profile vectors stay in L1 while the read
//! streams through, and the per-row state (`H` and `E` of the previous column)
//! is two vectors per read base — ~16 KB for a median tRNA read at 32 lanes.
//!
//! A lane whose reference is shorter than the group's longest keeps computing
//! past its end; those cells cannot feed back into valid ones (information
//! only flows to larger `j`), and a per-column lane mask keeps them out of the
//! running best. Local mode takes each valid column's maximum; semi-global
//! mode takes the last row of each valid column and the whole of the lane's
//! own last column.
//!
//! # Two loop orders: row-major and reference-major
//!
//! The walk above keeps two vectors of state *per read base*, so its working
//! set grows with the read: fine for the ~130 nt median tRNA read, and 50 MB
//! per lane group for the 395 kb read one real sample carries — streamed once
//! per reference column, which makes the kernel memory-bound on exactly the
//! reads that already cost the most (2e9 cells/s against ~4.5e9 in cache).
//! [`Kernel::Transposed`] runs the same recurrence with the loops swapped:
//! read row outer, reference column inner, so the state is the previous
//! *row* — `H` and `F` per reference column, ~25 KB for a 200 nt group at 32
//! lanes, whatever the read's length — and `E` travels along the row in a
//! register. It reads a second profile laid out code-major
//! (`tprof[(c * len + j) * lanes + lane]`), so one read row streams one
//! contiguous slice of it.
//!
//! Padding columns are handled differently too. The row-major kernel masks
//! each column; here a padding column scores `i16::MIN` in the transposed
//! profile, so no padding cell can ever take the diagonal. In local mode that
//! makes every padding cell a valid cell plus gap penalties, or the zero
//! floor, so it can never exceed the lane's best and the running maximum
//! needs no mask — for any scoring, `mismatch >= 0` included. Semi-global
//! mode takes the lane's own last column inside the row loop (only at the at
//! most `lanes` columns that end a lane, `Group::ends`) and the last row in
//! one masked pass after it.
//!
//! [`Aligner`](crate::Aligner) picks the transposed kernel for reads of at
//! least [`TRANSPOSED_MIN_READ_LEN`] bases; below that the row-major kernel
//! is the measured, tuned default and stays so.
//!
//! # i16 is exact, not approximate
//!
//! Lanes are `i16` with saturating arithmetic, and `i16::MIN` stands for −∞.
//! Every real value of a cell is bounded by `min(n, L) · max(|match|,
//! |mismatch|)` in magnitude (a path is at most that many substitutions from
//! a zero border), plus one gap open and extend for the gap states; when that
//! bound is below 32 000 no real value can saturate, and saturation of the
//! −∞ sentinel is exactly what −∞ should do. [`fits_i16`] checks it per read,
//! and a read that does not fit — only possible for references well beyond the
//! intended ≤ ~1 kb — is scored by the scalar path instead.
//!
//! # Dispatch
//!
//! In the style of `escapepod_signal::lstm`: runtime `is_x86_feature_detected!`
//! (never a baseline bump), kernels `#[inline(never)]`, and
//! `ESCAPEPOD_ALIGN_BACKEND=scalar|avx2|avx512` capping the choice for A/Bs.
//! Tests sweep [`Backend::available`], not the one the dispatch prefers.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::alphabet::N_CODES;
use crate::panel::Panel;
use crate::scalar::{self, Alignment};
use crate::scoring::{Mode, Scoring};

/// Which kernel scores the panel. Ordered slowest to fastest, so a cap is a
/// comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Backend {
    /// The scalar oracle, one reference at a time.
    Scalar,
    /// 16 × i16 lanes.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 32 × i16 lanes (AVX-512BW).
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// The environment variable that caps the dispatch.
pub const BACKEND_ENV: &str = "ESCAPEPOD_ALIGN_BACKEND";

/// Reads at least this long are scored with the reference-major
/// ([`Kernel::Transposed`]) kernel by default; shorter ones with the
/// row-major one. Chosen from `examples/long_read_probe.rs` on Cascade Lake
/// (`benchmarks/README.md`).
pub const TRANSPOSED_MIN_READ_LEN: usize = 1024;

/// Loop order of a SIMD score kernel (see the module docs). Both compute the
/// same scores; they differ only in what stays in cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Reference column outer, read row inner: state per read base.
    RowMajor,
    /// Read row outer, reference column inner: state per reference column.
    Transposed,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Kernel::RowMajor => "row-major",
            Kernel::Transposed => "transposed",
        }
    }
}

impl Backend {
    #[cfg(target_arch = "x86_64")]
    const ALL: [Backend; 3] = [Backend::Scalar, Backend::Avx2, Backend::Avx512];
    #[cfg(not(target_arch = "x86_64"))]
    const ALL: [Backend; 1] = [Backend::Scalar];

    /// Lanes per vector (references scored at once).
    pub fn lanes(self) -> usize {
        match self {
            Backend::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => 16,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => 32,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => "avx512",
        }
    }

    /// Parse a kernel name as `ESCAPEPOD_ALIGN_BACKEND` spells it.
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.name() == s)
    }

    /// Whether this machine can run the kernel. Every `unsafe` call into a
    /// `#[target_feature]` kernel rests on this having said yes.
    pub fn supported(self) -> bool {
        match self {
            Backend::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => {
                is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            }
        }
    }

    /// Every kernel this machine can run, slowest to fastest.
    pub fn available() -> Vec<Backend> {
        Self::ALL.into_iter().filter(|b| b.supported()).collect()
    }

    /// The cap `ESCAPEPOD_ALIGN_BACKEND` sets: `Ok(None)` when unset,
    /// `Err(value)` when it names no kernel (the caller decides whether to
    /// warn; this crate does not log).
    pub fn cap_from_env() -> Result<Option<Backend>, String> {
        // `*_BACKEND` caps are a name, not a flag/knob shape, and keep their
        // own parse (root CLAUDE.md's env-switch table).
        #[allow(clippy::disallowed_methods)]
        match std::env::var(BACKEND_ENV) {
            Err(_) => Ok(None),
            Ok(v) => Self::from_name(v.trim()).map(Some).ok_or(v),
        }
    }

    /// The fastest kernel this machine can run, under the environment cap
    /// (an unrecognised cap is ignored). A cap never raises the choice.
    pub fn best() -> Backend {
        let cap = Self::cap_from_env().ok().flatten();
        Self::ALL
            .into_iter()
            .rev()
            .find(|&b| cap.is_none_or(|c| b <= c) && b.supported())
            .unwrap_or(Backend::Scalar)
    }
}

/// Whether every real DP value of a read of `n` bases against references of
/// at most `max_ref_len` bases fits the kernels' i16 lanes (see module docs).
pub fn fits_i16(n: usize, max_ref_len: usize, scoring: &Scoring) -> bool {
    let k = n.min(max_ref_len) as i64;
    let m = scoring.match_score.abs().max(scoring.mismatch.abs()) as i64;
    let bound = k * m + scoring.gap_open.abs() as i64 + scoring.gap_extend.abs() as i64;
    bound < 32_000 && max_ref_len < 32_000
}

/// One lane-width group of references.
#[derive(Debug, Clone)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Group {
    /// Longest reference in the group: the number of columns walked.
    pub(crate) len: usize,
    /// Per-lane reference length (0 for an empty lane).
    pub(crate) lens: Vec<i16>,
    /// Per-lane panel index (`usize::MAX` for an empty lane).
    pub(crate) refs: Vec<usize>,
    /// `prof[(j * N_CODES + c) * lanes + lane]` = `s(c, ref_lane[j])`.
    pub(crate) prof: Vec<i16>,
    /// The transposed kernel's profile, built from `prof` the first time that
    /// kernel runs on this group (see [`Group::tprof`]) — at the defaults of
    /// `escpod align` it never does, and the copy would double the profile.
    tprof: OnceLock<Vec<i16>>,
    /// `ends[j]`: bit `lane` set when column `j` is that lane's last.
    pub(crate) ends: Vec<u32>,
}

impl Group {
    /// The transposed kernel's profile: `tprof[(c * len + j) * lanes + lane]`
    /// = `s(c, ref_lane[j])`, and `i16::MIN` past the lane's end (no padding
    /// cell can take the diagonal). The same substitution scores as `prof`,
    /// transposed; built once, on first use, by whichever thread gets there.
    pub(crate) fn tprof(&self) -> &[i16] {
        self.tprof.get_or_init(|| {
            let (len, lanes) = (self.len, self.lens.len());
            let mut t = vec![i16::MIN; N_CODES * len * lanes];
            for (lane, &n) in self.lens.iter().enumerate() {
                for j in 0..n as usize {
                    for c in 0..N_CODES {
                        t[(c * len + j) * lanes + lane] =
                            self.prof[(j * N_CODES + c) * lanes + lane];
                    }
                }
            }
            t
        })
    }

    /// Whether [`Group::tprof`] has been built.
    #[cfg(test)]
    pub(crate) fn tprof_built(&self) -> bool {
        self.tprof.get().is_some()
    }
}

/// A panel laid out for one lane width under one scoring.
#[derive(Debug, Clone)]
pub(crate) struct Profile {
    pub(crate) lanes: usize,
    pub(crate) groups: Vec<Group>,
}

impl Profile {
    pub(crate) fn build(panel: &Panel, scoring: &Scoring, lanes: usize) -> Self {
        let codes: Vec<&[u8]> = panel.references().iter().map(|r| &r.codes[..]).collect();
        Self::from_codes(&codes, scoring, lanes)
    }

    /// Lay out `refs` (codes); a lane's `refs` entry is its index in `refs`.
    pub(crate) fn from_codes(refs: &[&[u8]], scoring: &Scoring, lanes: usize) -> Self {
        let mut order: Vec<usize> = (0..refs.len()).collect();
        // Longest first, stable: groups of similar length waste little padding.
        order.sort_by_key(|&i| std::cmp::Reverse(refs[i].len()));
        let groups = order
            .chunks(lanes)
            .map(|chunk| {
                let len = chunk.iter().map(|&i| refs[i].len()).max().unwrap_or(0);
                let mut lens = vec![0i16; lanes];
                let mut idx = vec![usize::MAX; lanes];
                // Padding scores as a mismatch; it is masked out, never read.
                let mut prof = vec![scoring.mismatch as i16; len * N_CODES * lanes];
                let mut ends = vec![0u32; len];
                for (lane, &ri) in chunk.iter().enumerate() {
                    let codes = refs[ri];
                    lens[lane] = codes.len() as i16;
                    idx[lane] = ri;
                    if let Some(last) = codes.len().checked_sub(1) {
                        ends[last] |= 1 << lane;
                    }
                    for (j, &r) in codes.iter().enumerate() {
                        for c in 0..N_CODES {
                            let s = scoring.substitution(c as u8, r) as i16;
                            prof[(j * N_CODES + c) * lanes + lane] = s;
                        }
                    }
                }
                Group {
                    len,
                    lens,
                    refs: idx,
                    prof,
                    tprof: OnceLock::new(),
                    ends,
                }
            })
            .collect();
        Profile { lanes, groups }
    }
}

/// Gap scores as the kernels take them.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Gaps {
    pub(crate) open: i16,
    pub(crate) extend: i16,
}

thread_local! {
    /// Per-thread `H` and `E` rows, reused across reads and groups.
    static SCRATCH: RefCell<(Vec<i16>, Vec<i16>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    /// Per-thread traceback bytes for [`align_pairs`].
    static TRACE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Largest traceback buffer [`align_pairs`] allocates per thread (bytes); a
/// group of pairs that would need more is traced by the scalar path instead.
const MAX_TRACE_BYTES: usize = 16 << 20;

/// Per-lane result of a traceback kernel: best score and its 1-based end cell
/// (`i == 0` when nothing scored, as in [`scalar::walk`]).
pub(crate) struct LaneEnds {
    pub(crate) best: [i16; 32],
    pub(crate) end_i: [i16; 32],
    pub(crate) end_j: [i16; 32],
}

/// Up to one lane width of (read, reference) pairs, transposed for a pairs
/// kernel: `qt[i * lanes + lane]` is lane `lane`'s read code at row `i`,
/// `rt[j * lanes + lane]` its reference code at column `j`. Rows and columns
/// past a lane's own lengths are padding the kernel masks out.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct PairGroup {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) qlens: Vec<i16>,
    pub(crate) rlens: Vec<i16>,
    pub(crate) qt: Vec<i16>,
    pub(crate) rt: Vec<i16>,
}

/// All four scores as the pairs kernels take them.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Scores {
    pub(crate) match_score: i16,
    pub(crate) mismatch: i16,
    pub(crate) open: i16,
    pub(crate) extend: i16,
}

/// Full alignments (with tracebacks) of each `(query, reference)` pair (codes),
/// in order — identical, field for field, to [`scalar::align`] on each.
///
/// This is how winners are traced. Every lane carries its own read *and* its
/// own reference, so the unique winners of many reads share one kernel call as
/// readily as the members of one read's tie set do. Both were once scalar, and
/// both were expensive in different ways: a read that is nothing but adapter
/// ties with every reference of an adapter-flanked panel (all 164 of the
/// sacCer3 dual-adapter panel, for ~5% of one sample's reads), and the unique
/// winner of every other read, at ~45 instructions per cell, cost a third of
/// the run between them. The kernels run the scalar recurrence lane by lane,
/// write the scalar path's traceback byte, track each lane's end cell in the
/// scalar path's scan order, and hand every lane to the same [`scalar::walk`]
/// — so the result is the oracle's by construction, and pinned by equality in
/// the tests.
///
/// Pairs are sorted by read length before they are cut into lane groups, so a
/// group's padding is only the spread of lengths inside it. A pair too long
/// for i16 positions or outside the i16 value bound, a group whose traceback
/// would exceed [`MAX_TRACE_BYTES`], and everything under the scalar backend
/// go through [`scalar::align`] instead.
pub(crate) fn align_pairs(
    backend: Backend,
    pairs: &[(&[u8], &[u8])],
    scoring: &Scoring,
    mode: Mode,
) -> Vec<Alignment> {
    let mut out: Vec<Option<Alignment>> = vec![None; pairs.len()];
    let simd_ok = |(q, r): &(&[u8], &[u8])| {
        !q.is_empty()
            && !r.is_empty()
            && q.len() < 32_000
            && r.len() < 32_000
            && fits_i16(q.len(), r.len(), scoring)
    };
    let mut order: Vec<usize> = Vec::with_capacity(pairs.len());
    for (k, p) in pairs.iter().enumerate() {
        if backend != Backend::Scalar && simd_ok(p) {
            order.push(k);
        } else {
            out[k] = Some(scalar::align(p.0, p.1, scoring, mode));
        }
    }
    if order.is_empty() {
        return out.into_iter().map(Option::unwrap).collect();
    }
    debug_assert!(backend.supported());
    order.sort_by_key(|&k| (pairs[k].0.len(), pairs[k].1.len()));
    let lanes = backend.lanes();
    let scores = Scores {
        match_score: scoring.match_score as i16,
        mismatch: scoring.mismatch as i16,
        open: scoring.gap_open as i16,
        extend: scoring.gap_extend as i16,
    };
    SCRATCH.with(|s| {
        TRACE.with(|t| {
            let (hbuf, ebuf) = &mut *s.borrow_mut();
            let tb = &mut *t.borrow_mut();
            for chunk in order.chunks(lanes) {
                let rows = chunk.iter().map(|&k| pairs[k].0.len()).max().unwrap_or(0);
                let cols = chunk.iter().map(|&k| pairs[k].1.len()).max().unwrap_or(0);
                let need = (rows + 1) * (cols + 1) * lanes;
                if need > MAX_TRACE_BYTES {
                    for &k in chunk {
                        out[k] = Some(scalar::align(pairs[k].0, pairs[k].1, scoring, mode));
                    }
                    continue;
                }
                let mut g = PairGroup {
                    rows,
                    cols,
                    qlens: vec![0; lanes],
                    rlens: vec![0; lanes],
                    qt: vec![0; rows * lanes],
                    rt: vec![0; cols * lanes],
                };
                for (lane, &k) in chunk.iter().enumerate() {
                    let (q, r) = pairs[k];
                    g.qlens[lane] = q.len() as i16;
                    g.rlens[lane] = r.len() as i16;
                    for (i, &c) in q.iter().enumerate() {
                        g.qt[i * lanes + lane] = c as i16;
                    }
                    for (j, &c) in r.iter().enumerate() {
                        g.rt[j * lanes + lane] = c as i16;
                    }
                }
                if hbuf.len() < rows * lanes {
                    hbuf.resize(rows * lanes, 0);
                    ebuf.resize(rows * lanes, 0);
                }
                if tb.len() < need {
                    tb.resize(need, 0);
                }
                let mut ends = LaneEnds {
                    best: [0; 32],
                    end_i: [0; 32],
                    end_j: [0; 32],
                };
                run_trace_pairs(backend, &g, scores, mode, hbuf, ebuf, tb, &mut ends);
                let stride = rows + 1;
                for (lane, &k) in chunk.iter().enumerate() {
                    let (bi, bj) = (ends.end_i[lane] as usize, ends.end_j[lane] as usize);
                    let tb = &*tb;
                    out[k] = Some(scalar::walk(ends.best[lane] as i32, bi, bj, |i, j| {
                        tb[(j * stride + i) * lanes + lane]
                    }));
                }
            }
        })
    });
    out.into_iter()
        .map(|a| a.expect("every pair is traced"))
        .collect()
}

/// Score `query` (codes) against every reference in `profile` with a SIMD
/// kernel, writing each reference's best score to `out[panel_index]`.
///
/// The caller has checked [`fits_i16`] and that `backend` is supported and
/// matches `profile.lanes`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_profile(
    backend: Backend,
    profile: &Profile,
    query: &[u8],
    scoring: &Scoring,
    mode: Mode,
    kernel: Kernel,
    out: &mut [i32],
) {
    for g in 0..profile.groups.len() {
        score_profile_group(
            backend,
            profile,
            g,
            query,
            scoring,
            mode,
            kernel,
            |ri, s| out[ri] = s,
        );
    }
}

/// [`score_profile`] for one lane group: `emit(panel_index, score)` once per
/// reference of group `group`. The unit a caller can spread over threads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_profile_group(
    backend: Backend,
    profile: &Profile,
    group: usize,
    query: &[u8],
    scoring: &Scoring,
    mode: Mode,
    kernel: Kernel,
    mut emit: impl FnMut(usize, i32),
) {
    debug_assert!(backend.supported());
    debug_assert_eq!(backend.lanes(), profile.lanes);
    let gaps = Gaps {
        open: scoring.gap_open as i16,
        extend: scoring.gap_extend as i16,
    };
    let g = &profile.groups[group];
    let lanes = profile.lanes;
    let mut best = [0i16; 32];
    SCRATCH.with(|s| {
        let (hbuf, ebuf) = &mut *s.borrow_mut();
        let need = match kernel {
            Kernel::RowMajor => query.len() * lanes,
            Kernel::Transposed => g.len * lanes,
        };
        if hbuf.len() < need {
            hbuf.resize(need, 0);
            ebuf.resize(need, 0);
        }
        run_score_group(backend, kernel, query, g, gaps, mode, hbuf, ebuf, &mut best);
    });
    for (lane, &ri) in g.refs.iter().enumerate() {
        if ri != usize::MAX {
            emit(ri, best[lane] as i32);
        }
    }
}

/// One score kernel call, dispatched on `backend`.
///
/// Split out, with a stub below for other architectures, so the code around
/// it compiles identically everywhere: on aarch64 there are no SIMD backends,
/// a profile is never built, and this is never reached.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn run_score_group(
    backend: Backend,
    kernel: Kernel,
    query: &[u8],
    g: &Group,
    gaps: Gaps,
    mode: Mode,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    best: &mut [i16; 32],
) {
    // SAFETY: a SIMD `Backend` reaches here only through `Aligner`, which
    // refuses one the CPU does not support; the caller sized the buffers for
    // `query.len() * lanes` (row-major) or `g.len * lanes` (transposed) and
    // the profile for `lanes`.
    match (backend, kernel) {
        (Backend::Scalar, _) => unreachable!("scalar has no profile"),
        (Backend::Avx2, Kernel::RowMajor) => unsafe {
            avx2::score_group(query, g, gaps, mode, hbuf, ebuf, best)
        },
        (Backend::Avx2, Kernel::Transposed) => unsafe {
            avx2::score_group_transposed(query, g, gaps, mode, hbuf, ebuf, best)
        },
        (Backend::Avx512, Kernel::RowMajor) => unsafe {
            avx512::score_group(query, g, gaps, mode, hbuf, ebuf, best)
        },
        (Backend::Avx512, Kernel::Transposed) => unsafe {
            avx512::score_group_transposed(query, g, gaps, mode, hbuf, ebuf, best)
        },
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn run_score_group(
    _backend: Backend,
    _kernel: Kernel,
    _query: &[u8],
    _g: &Group,
    _gaps: Gaps,
    _mode: Mode,
    _hbuf: &mut [i16],
    _ebuf: &mut [i16],
    _best: &mut [i16; 32],
) {
    unreachable!("no SIMD score kernels on this architecture")
}

/// One pairs traceback kernel call, dispatched on `backend` (see
/// [`run_score_group`] for the stub).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn run_trace_pairs(
    backend: Backend,
    g: &PairGroup,
    scores: Scores,
    mode: Mode,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    tb: &mut [u8],
    ends: &mut LaneEnds,
) {
    match backend {
        Backend::Scalar => unreachable!("scalar pairs are traced by scalar::align"),
        // SAFETY: as in `run_score_group`; the caller sized `hbuf`/`ebuf` for
        // `g.rows * lanes` and `tb` for `(g.rows + 1) * (g.cols + 1) * lanes`.
        Backend::Avx2 => unsafe { avx2::trace_pairs(g, scores, mode, hbuf, ebuf, tb, ends) },
        Backend::Avx512 => unsafe { avx512::trace_pairs(g, scores, mode, hbuf, ebuf, tb, ends) },
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn run_trace_pairs(
    _backend: Backend,
    _g: &PairGroup,
    _scores: Scores,
    _mode: Mode,
    _hbuf: &mut [i16],
    _ebuf: &mut [i16],
    _tb: &mut [u8],
    _ends: &mut LaneEnds,
) {
    unreachable!("no SIMD traceback kernels on this architecture")
}

/// Scalar panel scoring: the oracle, one reference at a time.
pub(crate) fn score_scalar(
    panel: &Panel,
    query: &[u8],
    scoring: &Scoring,
    mode: Mode,
    out: &mut [i32],
) {
    for (slot, r) in out.iter_mut().zip(panel.references()) {
        *slot = scalar::score(query, &r.codes, scoring, mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Aligner;
    use crate::alphabet::encode;

    /// xorshift64*, so the random tests need no dependency and are
    /// reproducible from their seed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn chance(&mut self, p: f64) -> bool {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
        }
    }

    fn random_seq(rng: &mut Rng, len: usize, n_rate: f64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                if rng.chance(n_rate) {
                    b'N'
                } else {
                    b"ACGT"[rng.below(4)]
                }
            })
            .collect()
    }

    /// Nanopore-ish: substitutions, insertions and deletions at a few % each.
    fn mutate(rng: &mut Rng, s: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(s.len() + 8);
        for &b in s {
            if rng.chance(0.04) {
                continue;
            }
            if rng.chance(0.05) {
                out.push(b"ACGT"[rng.below(4)]);
            } else {
                out.push(b);
            }
            if rng.chance(0.03) {
                out.push(b"ACGT"[rng.below(4)]);
            }
        }
        out
    }

    fn random_panel(rng: &mut Rng, n_refs: usize) -> Panel {
        Panel::new((0..n_refs).map(|k| {
            let len = 20 + rng.below(280);
            (format!("r{k}"), random_seq(rng, len, 0.01))
        }))
        .unwrap()
    }

    fn schemes() -> [Scoring; 3] {
        [
            Scoring::default(),
            Scoring::new(1, -1, -2, -1).unwrap(),
            Scoring::new(3, -2, -5, -2).unwrap(),
        ]
    }

    /// Every available SIMD backend against the scalar oracle, by equality,
    /// on reads drawn from the panel (so real maxima exist) and unrelated
    /// random reads (so the zero floor and negative semi-global cells do).
    #[cfg(target_arch = "x86_64")]
    fn simd_matches_scalar_random(backend: Backend, seed: u64) {
        let mut rng = Rng(seed);
        // 37 references: more than one group at both widths, and a ragged last group.
        let panel = random_panel(&mut rng, 37);
        for scoring in schemes() {
            for mode in [Mode::Local, Mode::SemiGlobal] {
                let simd = Aligner::with_backend(panel.clone(), scoring, mode, backend).unwrap();
                let oracle =
                    Aligner::with_backend(panel.clone(), scoring, mode, Backend::Scalar).unwrap();
                for t in 0..40 {
                    let read = if t % 4 == 3 {
                        let len = 1 + rng.below(300);
                        random_seq(&mut rng, len, 0.02)
                    } else {
                        let r = &panel.get(rng.below(panel.len())).seq;
                        let a = rng.below(r.len() / 3 + 1);
                        let b = r.len() - rng.below(r.len() / 3 + 1);
                        let (lead, tail) = (rng.below(30), rng.below(30));
                        let mut read = random_seq(&mut rng, lead, 0.0);
                        read.extend(mutate(&mut rng, &r[a..b.max(a)]));
                        read.extend(random_seq(&mut rng, tail, 0.0));
                        read
                    };
                    let q = encode(&read);
                    let (mut got, mut want) = (Vec::new(), Vec::new());
                    simd.score_all(&q, &mut got);
                    oracle.score_all(&q, &mut want);
                    assert_eq!(
                        got,
                        want,
                        "{} {} {scoring} read {}",
                        backend.name(),
                        mode.name(),
                        String::from_utf8_lossy(&read)
                    );
                }
            }
        }
    }

    #[test]
    fn avx2_matches_scalar_random() {
        #[cfg(target_arch = "x86_64")]
        if Backend::Avx2.supported() {
            simd_matches_scalar_random(Backend::Avx2, 0x5eed_0001);
            return;
        }
        eprintln!("avx2_matches_scalar_random: AVX2 not available on this machine; skipped");
    }

    #[test]
    fn avx512_matches_scalar_random() {
        #[cfg(target_arch = "x86_64")]
        if Backend::Avx512.supported() {
            simd_matches_scalar_random(Backend::Avx512, 0x5eed_0002);
            return;
        }
        eprintln!("avx512_matches_scalar_random: AVX-512BW not available on this machine; skipped");
    }

    /// Every backend the machine has, on the real tRNA reads and the real
    /// 47-reference panel from escapepod-classify's fixtures (reads taken from
    /// this crate's golden, which carries them as plain strings).
    #[test]
    fn simd_matches_scalar_on_fixture_reads() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let golden: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("tests/fixtures/align_golden.json")).unwrap(),
        )
        .unwrap();
        let reads: Vec<Vec<u8>> = golden["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["source"].as_str() != Some("random"))
            .map(|p| p["query"].as_str().unwrap().as_bytes().to_vec())
            .collect();
        assert!(reads.len() >= 5, "golden carries too few fixture reads");
        for fasta in ["trna_reference.fa", "trna_reference_ambiguous.fa"] {
            let file = std::fs::File::open(
                root.join("../escapepod-classify/tests/fixtures")
                    .join(fasta),
            )
            .unwrap();
            let refs = crate::fasta::read_fasta(std::io::BufReader::new(file)).unwrap();
            let panel = Panel::new(refs).unwrap();
            assert_eq!(panel.len(), 47);
            for scoring in schemes() {
                for mode in [Mode::Local, Mode::SemiGlobal] {
                    let oracle =
                        Aligner::with_backend(panel.clone(), scoring, mode, Backend::Scalar)
                            .unwrap();
                    for backend in Backend::available() {
                        let a =
                            Aligner::with_backend(panel.clone(), scoring, mode, backend).unwrap();
                        for read in &reads {
                            let q = encode(read);
                            let (mut got, mut want) = (Vec::new(), Vec::new());
                            a.score_all(&q, &mut got);
                            oracle.score_all(&q, &mut want);
                            assert_eq!(
                                got,
                                want,
                                "{} {} {scoring} {fasta}",
                                backend.name(),
                                mode.name()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The pairs traceback kernels against [`scalar::align`], field for
    /// field — same score, end cell, start and CIGAR — on every backend, over
    /// ragged groups (reads and references of mixed lengths in one call, some
    /// related and some not, so every lane mask is exercised).
    #[test]
    fn traceback_kernels_match_scalar() {
        let mut rng = Rng(0x5eed_0003);
        let panel = random_panel(&mut rng, 37);
        for backend in Backend::available() {
            for scoring in schemes() {
                for mode in [Mode::Local, Mode::SemiGlobal] {
                    let mut reads: Vec<Vec<u8>> = Vec::new();
                    let mut refs: Vec<usize> = Vec::new();
                    for t in 0..80 {
                        let ri = rng.below(panel.len());
                        let read = if t % 4 == 3 {
                            let len = 1 + rng.below(250);
                            random_seq(&mut rng, len, 0.02)
                        } else {
                            let r = &panel.get(ri).seq;
                            let (lead, tail) = (rng.below(20), rng.below(20));
                            let mut read = random_seq(&mut rng, lead, 0.0);
                            read.extend(mutate(&mut rng, r));
                            read.extend(random_seq(&mut rng, tail, 0.0));
                            read
                        };
                        reads.push(encode(&read));
                        refs.push(ri);
                    }
                    let pairs: Vec<(&[u8], &[u8])> = reads
                        .iter()
                        .zip(&refs)
                        .map(|(q, &ri)| (&q[..], &panel.get(ri).codes[..]))
                        .collect();
                    let got = align_pairs(backend, &pairs, &scoring, mode);
                    for (k, (a, (q, r))) in got.iter().zip(&pairs).enumerate() {
                        let want = scalar::align(q, r, &scoring, mode);
                        assert_eq!(
                            a,
                            &want,
                            "{} {} {scoring} pair {k}",
                            backend.name(),
                            mode.name()
                        );
                    }
                }
            }
        }
    }

    /// The three schemes above plus the edges `Scoring::validate` accepts and
    /// the transposed kernel's padding argument has to survive: a mismatch
    /// of zero, and a positive one.
    fn transposed_schemes() -> [Scoring; 5] {
        let [a, b, c] = schemes();
        [
            a,
            b,
            c,
            Scoring::new(2, 0, -10, -1).unwrap(),
            Scoring::new(3, 1, -5, -2).unwrap(),
        ]
    }

    /// A read drawn from the panel: pieces of mutated references between
    /// stretches of random sequence, `len` bases in all.
    fn chimeric_read(rng: &mut Rng, panel: &Panel, len: usize) -> Vec<u8> {
        let mut read = Vec::with_capacity(len + 300);
        while read.len() < len {
            let gap = rng.below(400);
            read.extend(random_seq(rng, gap, 0.01));
            let r = &panel.get(rng.below(panel.len())).seq;
            read.extend(mutate(rng, r));
        }
        read.truncate(len);
        read
    }

    /// The transposed kernel, forced, against the scalar oracle by equality:
    /// every SIMD backend, both modes, five schemes (two with a mismatch of
    /// zero or above), ragged lane groups, and reads short, either side of
    /// [`TRANSPOSED_MIN_READ_LEN`], and 20–60 kb long.
    #[test]
    fn transposed_matches_scalar_random() {
        let mut rng = Rng(0x5eed_0004);
        let panel = random_panel(&mut rng, 37);
        let backends: Vec<Backend> = Backend::available()
            .into_iter()
            .filter(|&b| b != Backend::Scalar)
            .collect();
        if backends.is_empty() {
            eprintln!("transposed_matches_scalar_random: no SIMD backend on this machine; skipped");
            return;
        }
        let t = TRANSPOSED_MIN_READ_LEN;
        let mut combo = 0usize;
        for scoring in transposed_schemes() {
            for mode in [Mode::Local, Mode::SemiGlobal] {
                let oracle =
                    Aligner::with_backend(panel.clone(), scoring, mode, Backend::Scalar).unwrap();
                let simd: Vec<Aligner> = backends
                    .iter()
                    .map(|&b| Aligner::with_backend(panel.clone(), scoring, mode, b).unwrap())
                    .collect();
                let mut reads: Vec<Vec<u8>> = Vec::new();
                for k in 0..12 {
                    let len = 1 + rng.below(320);
                    reads.push(if k % 3 == 2 {
                        random_seq(&mut rng, len, 0.02)
                    } else {
                        chimeric_read(&mut rng, &panel, len)
                    });
                }
                for len in [t - 1, t + 1] {
                    reads.push(chimeric_read(&mut rng, &panel, len));
                }
                // One long read per scheme and mode, 20–56 kb across them.
                reads.push(chimeric_read(&mut rng, &panel, 20_000 + 4_000 * combo));
                combo += 1;
                for read in &reads {
                    let q = encode(read);
                    let mut want = Vec::new();
                    oracle.score_all(&q, &mut want);
                    for a in &simd {
                        let mut got = Vec::new();
                        a.score_all_with(&q, Kernel::Transposed, &mut got);
                        assert_eq!(
                            got,
                            want,
                            "{} {} {scoring} read of {} nt",
                            a.backend().name(),
                            mode.name(),
                            q.len()
                        );
                    }
                }
            }
        }
    }

    /// Both loop orders on the real tRNA reads against both 47-reference
    /// fixture panels: equal, reference for reference.
    #[test]
    fn transposed_matches_row_major_on_fixture_reads() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let golden: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("tests/fixtures/align_golden.json")).unwrap(),
        )
        .unwrap();
        let reads: Vec<Vec<u8>> = golden["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["source"].as_str() != Some("random"))
            .map(|p| p["query"].as_str().unwrap().as_bytes().to_vec())
            .collect();
        assert!(reads.len() >= 5, "golden carries too few fixture reads");
        for fasta in ["trna_reference.fa", "trna_reference_ambiguous.fa"] {
            let file = std::fs::File::open(
                root.join("../escapepod-classify/tests/fixtures")
                    .join(fasta),
            )
            .unwrap();
            let refs = crate::fasta::read_fasta(std::io::BufReader::new(file)).unwrap();
            let panel = Panel::new(refs).unwrap();
            for scoring in transposed_schemes() {
                for mode in [Mode::Local, Mode::SemiGlobal] {
                    for backend in Backend::available() {
                        if backend == Backend::Scalar {
                            continue;
                        }
                        let a =
                            Aligner::with_backend(panel.clone(), scoring, mode, backend).unwrap();
                        for read in &reads {
                            let q = encode(read);
                            let (mut row, mut tr) = (Vec::new(), Vec::new());
                            a.score_all_with(&q, Kernel::RowMajor, &mut row);
                            a.score_all_with(&q, Kernel::Transposed, &mut tr);
                            assert_eq!(
                                tr,
                                row,
                                "{} {} {scoring} {fasta}",
                                backend.name(),
                                mode.name()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The dispatch takes the transposed kernel from the named threshold on,
    /// the row-major one below it, neither under the scalar backend, and
    /// never once the threshold is switched off.
    #[test]
    fn dispatch_picks_transposed_above_threshold() {
        let mut rng = Rng(0x5eed_0005);
        let panel = random_panel(&mut rng, 37);
        let t = TRANSPOSED_MIN_READ_LEN;
        for backend in Backend::available() {
            let a = Aligner::with_backend(panel.clone(), Scoring::default(), Mode::Local, backend)
                .unwrap();
            assert_eq!(a.transposed_min_len(), Some(t));
            if backend == Backend::Scalar {
                assert_eq!(a.kernel_for(t), None);
                continue;
            }
            assert_eq!(a.kernel_for(1), Some(Kernel::RowMajor));
            assert_eq!(a.kernel_for(t - 1), Some(Kernel::RowMajor));
            assert_eq!(a.kernel_for(t), Some(Kernel::Transposed));
            assert_eq!(a.kernel_for(400_000), Some(Kernel::Transposed));
            let off = a.clone().with_transposed_min_len(None);
            assert_eq!(off.kernel_for(400_000), Some(Kernel::RowMajor));
            let low = a.with_transposed_min_len(Some(10));
            assert_eq!(low.kernel_for(9), Some(Kernel::RowMajor));
            assert_eq!(low.kernel_for(10), Some(Kernel::Transposed));
        }
    }

    /// The transposed profile is built only when the transposed kernel runs:
    /// never for reads below the threshold, nor with the switch off, and on
    /// first use for a long read — which then scores as before.
    #[test]
    fn row_major_scoring_does_not_build_tprof() {
        let mut rng = Rng(0x5eed_0006);
        let panel = random_panel(&mut rng, 37);
        let short = encode(&chimeric_read(&mut rng, &panel, 300));
        let long = encode(&chimeric_read(
            &mut rng,
            &panel,
            TRANSPOSED_MIN_READ_LEN + 50,
        ));
        for backend in Backend::available() {
            if backend == Backend::Scalar {
                continue;
            }
            let built = |a: &Aligner| a.profile_for_tests().groups.iter().any(Group::tprof_built);
            let mut out = Vec::new();
            let a = Aligner::with_backend(panel.clone(), Scoring::default(), Mode::Local, backend)
                .unwrap();
            let off = a.clone().with_transposed_min_len(None);
            a.score_all(&short, &mut out);
            assert!(!built(&a), "{}: short read built tprof", backend.name());
            off.score_all(&long, &mut out);
            assert!(
                !built(&off),
                "{}: switched-off aligner built tprof",
                backend.name()
            );
            let row = out.clone();
            a.score_all(&long, &mut out);
            assert!(
                a.profile_for_tests().groups.iter().all(Group::tprof_built),
                "{}: long read did not build tprof",
                backend.name()
            );
            assert_eq!(out, row, "{}", backend.name());
        }
    }

    #[test]
    fn cap_names_round_trip() {
        for b in Backend::ALL {
            assert_eq!(Backend::from_name(b.name()), Some(b));
        }
        assert_eq!(Backend::from_name("sse9"), None);
    }

    #[test]
    fn i16_bound() {
        let s = Scoring::default();
        assert!(fits_i16(5000, 200, &s));
        assert!(!fits_i16(20_000, 20_000, &s));
    }
}

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
pub(crate) struct Group {
    /// Longest reference in the group: the number of columns walked.
    pub(crate) len: usize,
    /// Per-lane reference length (0 for an empty lane).
    pub(crate) lens: Vec<i16>,
    /// Per-lane panel index (`usize::MAX` for an empty lane).
    pub(crate) refs: Vec<usize>,
    /// `prof[(j * N_CODES + c) * lanes + lane]` = `s(c, ref_lane[j])`.
    pub(crate) prof: Vec<i16>,
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
                for (lane, &ri) in chunk.iter().enumerate() {
                    let codes = refs[ri];
                    lens[lane] = codes.len() as i16;
                    idx[lane] = ri;
                    for (j, &r) in codes.iter().enumerate() {
                        for c in 0..N_CODES {
                            prof[(j * N_CODES + c) * lanes + lane] =
                                scoring.substitution(c as u8, r) as i16;
                        }
                    }
                }
                Group {
                    len,
                    lens,
                    refs: idx,
                    prof,
                }
            })
            .collect();
        Profile { lanes, groups }
    }
}

/// Gap scores as the kernels take them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Gaps {
    pub(crate) open: i16,
    pub(crate) extend: i16,
}

thread_local! {
    /// Per-thread `H` and `E` rows, reused across reads and groups.
    static SCRATCH: RefCell<(Vec<i16>, Vec<i16>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    /// Per-thread traceback bytes for [`align_many`].
    static TRACE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Largest traceback buffer [`align_many`] allocates per thread (bytes); a
/// read whose group would need more is traced by the scalar path instead.
const MAX_TRACE_BYTES: usize = 16 << 20;

/// Per-lane result of a traceback kernel: best score and its 1-based end cell
/// (`i == 0` when nothing scored, as in [`scalar::walk`]).
pub(crate) struct LaneEnds {
    pub(crate) best: [i16; 32],
    pub(crate) end_i: [i16; 32],
    pub(crate) end_j: [i16; 32],
}

/// Full alignments (with tracebacks) of `query` against each of `refs`, in
/// `refs` order — identical, field for field, to [`scalar::align`] on each.
///
/// This is what makes a large tie set affordable. A read that is nothing but
/// adapter ties with every reference of an adapter-flanked panel (164 on the
/// sacCer3 dual-adapter panel, ~5% of one sample's reads), and tracing those
/// one reference at a time cost twice what scoring the whole panel did. The
/// kernels here run the same DP inter-sequence, write the scalar path's
/// traceback byte per lane, track each lane's end cell in the scalar path's
/// scan order, and hand every lane to the same [`scalar::walk`] — so the
/// result is the oracle's by construction and pinned by equality in tests.
///
/// Falls back to [`scalar::align`] per reference for the scalar backend, a
/// single reference, a read too long for i16 positions or the i16 value bound,
/// or a traceback buffer over [`MAX_TRACE_BYTES`].
pub(crate) fn align_many(
    backend: Backend,
    query: &[u8],
    refs: &[&[u8]],
    scoring: &Scoring,
    mode: Mode,
) -> Vec<Alignment> {
    let n = query.len();
    let max_len = refs.iter().map(|r| r.len()).max().unwrap_or(0);
    let lanes = backend.lanes();
    let scalar_all = || {
        refs.iter()
            .map(|r| scalar::align(query, r, scoring, mode))
            .collect()
    };
    if backend == Backend::Scalar
        || refs.len() < 2
        || n == 0
        || n >= 32_000
        || !fits_i16(n, max_len, scoring)
        || (n + 1) * (max_len + 1) * lanes > MAX_TRACE_BYTES
    {
        return scalar_all();
    }
    debug_assert!(backend.supported());
    let profile = Profile::from_codes(refs, scoring, lanes);
    let gaps = Gaps {
        open: scoring.gap_open as i16,
        extend: scoring.gap_extend as i16,
    };
    let rows = n + 1;
    let mut out: Vec<Option<Alignment>> = vec![None; refs.len()];
    SCRATCH.with(|s| {
        TRACE.with(|t| {
            let (hbuf, ebuf) = &mut *s.borrow_mut();
            let tb = &mut *t.borrow_mut();
            if hbuf.len() < n * lanes {
                hbuf.resize(n * lanes, 0);
                ebuf.resize(n * lanes, 0);
            }
            for g in &profile.groups {
                let need = rows * (g.len + 1) * lanes;
                if tb.len() < need {
                    tb.resize(need, 0);
                }
                let mut ends = LaneEnds {
                    best: [0; 32],
                    end_i: [0; 32],
                    end_j: [0; 32],
                };
                match backend {
                    Backend::Scalar => unreachable!("scalar returned above"),
                    // SAFETY: `backend.supported()` holds (it is only ever
                    // constructed through a supported check); the buffers were
                    // sized above for this group and read length.
                    #[cfg(target_arch = "x86_64")]
                    Backend::Avx2 => unsafe {
                        avx2::trace_group(query, g, gaps, mode, hbuf, ebuf, tb, &mut ends)
                    },
                    #[cfg(target_arch = "x86_64")]
                    Backend::Avx512 => unsafe {
                        avx512::trace_group(query, g, gaps, mode, hbuf, ebuf, tb, &mut ends)
                    },
                }
                for (lane, &ri) in g.refs.iter().enumerate() {
                    if ri == usize::MAX {
                        continue;
                    }
                    let (bi, bj) = (ends.end_i[lane] as usize, ends.end_j[lane] as usize);
                    let tb = &*tb;
                    out[ri] = Some(scalar::walk(ends.best[lane] as i32, bi, bj, |i, j| {
                        tb[(j * rows + i) * lanes + lane]
                    }));
                }
            }
        })
    });
    out.into_iter()
        .map(|a| a.expect("every reference is in one lane"))
        .collect()
}

/// Score `query` (codes) against every reference in `profile` with a SIMD
/// kernel, writing each reference's best score to `out[panel_index]`.
///
/// The caller has checked [`fits_i16`] and that `backend` is supported and
/// matches `profile.lanes`.
pub(crate) fn score_profile(
    backend: Backend,
    profile: &Profile,
    query: &[u8],
    scoring: &Scoring,
    mode: Mode,
    out: &mut [i32],
) {
    debug_assert!(backend.supported());
    debug_assert_eq!(backend.lanes(), profile.lanes);
    let gaps = Gaps {
        open: scoring.gap_open as i16,
        extend: scoring.gap_extend as i16,
    };
    let lanes = profile.lanes;
    SCRATCH.with(|s| {
        let (hbuf, ebuf) = &mut *s.borrow_mut();
        let need = query.len() * lanes;
        if hbuf.len() < need {
            hbuf.resize(need, 0);
            ebuf.resize(need, 0);
        }
        let mut best = [0i16; 32];
        for g in &profile.groups {
            match backend {
                Backend::Scalar => unreachable!("scalar has no profile"),
                // SAFETY: `backend.supported()` was checked by the caller
                // (debug-asserted above); the buffers hold `query.len() *
                // lanes` elements and the profile was built for `lanes`.
                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 => unsafe {
                    avx2::score_group(query, g, gaps, mode, hbuf, ebuf, &mut best)
                },
                #[cfg(target_arch = "x86_64")]
                Backend::Avx512 => unsafe {
                    avx512::score_group(query, g, gaps, mode, hbuf, ebuf, &mut best)
                },
            }
            for (lane, &ri) in g.refs.iter().enumerate() {
                if ri != usize::MAX {
                    out[ri] = best[lane] as i32;
                }
            }
        }
    });
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
            let text = std::fs::read_to_string(
                root.join("../escapepod-classify/tests/fixtures")
                    .join(fasta),
            )
            .unwrap();
            let mut refs: Vec<(String, Vec<u8>)> = Vec::new();
            for line in text.lines() {
                if let Some(h) = line.strip_prefix('>') {
                    refs.push((h.split_whitespace().next().unwrap().to_string(), Vec::new()));
                } else {
                    refs.last_mut().unwrap().1.extend(line.trim().bytes());
                }
            }
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

    /// The traceback kernels against [`scalar::align`], field for field: same
    /// score, same end cell, same start, same CIGAR — on every backend.
    #[test]
    fn traceback_kernels_match_scalar() {
        let mut rng = Rng(0x5eed_0003);
        let panel = random_panel(&mut rng, 37);
        let refs: Vec<&[u8]> = panel.references().iter().map(|r| &r.codes[..]).collect();
        for backend in Backend::available() {
            for scoring in schemes() {
                for mode in [Mode::Local, Mode::SemiGlobal] {
                    for t in 0..12 {
                        let read = if t % 4 == 3 {
                            let len = 1 + rng.below(250);
                            random_seq(&mut rng, len, 0.02)
                        } else {
                            let r = &panel.get(rng.below(panel.len())).seq;
                            let (lead, tail) = (rng.below(20), rng.below(20));
                            let mut read = random_seq(&mut rng, lead, 0.0);
                            read.extend(mutate(&mut rng, r));
                            read.extend(random_seq(&mut rng, tail, 0.0));
                            read
                        };
                        let q = encode(&read);
                        let got = align_many(backend, &q, &refs, &scoring, mode);
                        for (k, (a, r)) in got.iter().zip(&refs).enumerate() {
                            let want = scalar::align(&q, r, &scoring, mode);
                            assert_eq!(
                                a,
                                &want,
                                "{} {} {scoring} ref {k} read {}",
                                backend.name(),
                                mode.name(),
                                String::from_utf8_lossy(&read)
                            );
                        }
                    }
                }
            }
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

// SPDX-License-Identifier: MIT

//! Scalar Gotoh: the reference implementation.
//!
//! Integer scoring throughout, so every other path in this crate — the SIMD
//! kernels, the winner selection — is tested against this one by *equality*,
//! with no tolerances anywhere.
//!
//! # Recurrence
//!
//! With the read (query) down the rows `i = 1..=n` and the reference across
//! the columns `j = 1..=m`:
//!
//! ```text
//! E[i][j] = max(H[i][j-1] + gap_open, E[i][j-1] + gap_extend)   deletion  (ref consumed)
//! F[i][j] = max(H[i-1][j] + gap_open, F[i-1][j] + gap_extend)   insertion (read consumed)
//! H[i][j] = max(H[i-1][j-1] + s(q_i, r_j), E[i][j], F[i][j] [, 0 in local mode])
//! ```
//!
//! Both modes have a zero border (`H[i][0] = H[0][j] = 0`, `E`/`F` = −∞ on it):
//! in local mode because an alignment may start anywhere, in semi-global mode
//! because skipping a prefix of either sequence is free. Local mode floors
//! every cell at zero and ends at the best cell anywhere; semi-global mode
//! floors nothing and ends at the best cell of the last row or last column.
//!
//! # Which optimum, when there are several
//!
//! Optimal alignments are rarely unique, so the choices are fixed and pinned
//! against parasail (`tests/parasail_golden.rs`), whose `sw_trace` /
//! `sg_trace` make the same ones:
//!
//! * **End cell:** the smallest reference end, then the smallest read end —
//!   the first maximum of a column-major scan.
//! * **Traceback from a cell:** diagonal, then insertion, then deletion; inside
//!   a gap, extending is preferred over opening; in local mode the path stops
//!   at the first cell whose `H` is zero, even if a zero-scoring extension
//!   exists before it.

use crate::scoring::{Mode, Scoring};

/// Stand-in for −∞ that cannot overflow when a penalty is added to it.
const NEG: i32 = i32::MIN / 4;

/// One CIGAR operation of the aligned region (soft clips are added when
/// writing SAM, see [`crate::sam::cigar_string`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarOp {
    /// Read base against reference base, match or mismatch (`M`).
    Match,
    /// Read base against nothing (`I`).
    Ins,
    /// Reference base against nothing (`D`).
    Del,
}

impl CigarOp {
    pub fn as_char(self) -> char {
        match self {
            Self::Match => 'M',
            Self::Ins => 'I',
            Self::Del => 'D',
        }
    }
}

/// A scored alignment with its traceback. Coordinates are 0-based and
/// half-open: the aligned read is `query[query_start..query_end]`, the aligned
/// reference `reference[ref_start..ref_end]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alignment {
    pub score: i32,
    pub query_start: usize,
    pub query_end: usize,
    pub ref_start: usize,
    pub ref_end: usize,
    /// Run-length encoded operations over the aligned region, in order.
    pub ops: Vec<(CigarOp, u32)>,
}

impl Alignment {
    /// Whether any read base is aligned to any reference base. A local
    /// alignment with score zero has none, and is not an alignment.
    pub fn is_empty(&self) -> bool {
        !self.ops.iter().any(|(op, _)| *op == CigarOp::Match)
    }

    /// Score the traceback under `scoring`, independently of the DP that
    /// produced it. Equal to [`Alignment::score`] for a correct traceback — an
    /// optimal CIGAR is not unique, so this is how one is checked.
    pub fn rescore(&self, query: &[u8], reference: &[u8], scoring: &Scoring) -> i32 {
        let (mut i, mut j) = (self.query_start, self.ref_start);
        let mut total = 0;
        for &(op, len) in &self.ops {
            match op {
                CigarOp::Match => {
                    for _ in 0..len {
                        total += scoring.substitution(query[i], reference[j]);
                        i += 1;
                        j += 1;
                    }
                }
                CigarOp::Ins => {
                    total += scoring.gap(len);
                    i += len as usize;
                }
                CigarOp::Del => {
                    total += scoring.gap(len);
                    j += len as usize;
                }
            }
        }
        total
    }
}

/// Best score of `query` against `reference`, without a traceback.
///
/// O(n) memory. This is what the SIMD kernels must equal, lane for lane.
/// Both inputs are codes from [`crate::alphabet`].
pub fn score(query: &[u8], reference: &[u8], scoring: &Scoring, mode: Mode) -> i32 {
    let n = query.len();
    let m = reference.len();
    if n == 0 || m == 0 {
        return 0;
    }
    let (o, e) = (scoring.gap_open, scoring.gap_extend);
    let local = mode == Mode::Local;
    // Column j-1's H and E, indexed by row i = 1..=n (slot 0 unused).
    let mut hcol = vec![0i32; n + 1];
    let mut ecol = vec![NEG; n + 1];
    let mut best = if local { 0 } else { NEG };
    for (j, &rj) in reference.iter().enumerate() {
        let mut hdiag = 0; // H[0][j-1]
        let mut hup = 0; // H[0][j]
        let mut f = NEG;
        let mut colmax = NEG;
        for i in 1..=n {
            let hleft = hcol[i];
            let ev = (hleft + o).max(ecol[i] + e);
            ecol[i] = ev;
            f = (hup + o).max(f + e);
            let mut h = (hdiag + scoring.substitution(query[i - 1], rj))
                .max(ev)
                .max(f);
            if local {
                h = h.max(0);
            }
            hdiag = hleft;
            hcol[i] = h;
            hup = h;
            colmax = colmax.max(h);
        }
        if local {
            best = best.max(colmax);
        } else {
            best = best.max(hup);
            if j + 1 == m {
                best = best.max(colmax);
            }
        }
    }
    best
}

// Traceback byte, one per cell.
const SRC_MASK: u8 = 0b11;
const SRC_STOP: u8 = 0;
const SRC_DIAG: u8 = 1;
const SRC_INS: u8 = 2;
const SRC_DEL: u8 = 3;
/// `E[i][j]` extended `E[i][j-1]` (rather than opening from `H[i][j-1]`).
const E_EXTENDED: u8 = 0b100;
/// `F[i][j]` extended `F[i-1][j]`.
const F_EXTENDED: u8 = 0b1000;

/// Best alignment of `query` against `reference`, with its traceback.
///
/// Holds one byte per DP cell (`(n+1)·(m+1)`) for the traceback — about
/// 0.9 MB for a 4 kb read against a 200 nt reference — so it is run for the
/// winners only. Both inputs are codes from [`crate::alphabet`].
pub fn align(query: &[u8], reference: &[u8], scoring: &Scoring, mode: Mode) -> Alignment {
    let n = query.len();
    let m = reference.len();
    let empty = Alignment {
        score: 0,
        query_start: 0,
        query_end: 0,
        ref_start: 0,
        ref_end: 0,
        ops: Vec::new(),
    };
    if n == 0 || m == 0 {
        return empty;
    }
    let (o, e) = (scoring.gap_open, scoring.gap_extend);
    let local = mode == Mode::Local;
    let rows = n + 1;
    // Column-major: trace[j * rows + i].
    let mut trace = vec![0u8; rows * (m + 1)];
    let mut hcol = vec![0i32; rows];
    let mut ecol = vec![NEG; rows];
    let mut best = if local { 0 } else { NEG };
    let (mut best_i, mut best_j) = (0usize, 0usize);

    for j in 1..=m {
        let rj = reference[j - 1];
        let col = &mut trace[j * rows..(j + 1) * rows];
        let mut hdiag = 0;
        let mut hup = 0;
        let mut f = NEG;
        for i in 1..=n {
            let hleft = hcol[i];
            let e_open = hleft + o;
            let e_ext = ecol[i] + e;
            let ev = e_open.max(e_ext);
            let f_open = hup + o;
            let f_ext = f + e;
            f = f_open.max(f_ext);
            let hd = hdiag + scoring.substitution(query[i - 1], rj);
            let mut h = hd.max(ev).max(f);
            if local {
                h = h.max(0);
            }
            let src = if local && h == 0 {
                SRC_STOP
            } else if h == hd {
                SRC_DIAG
            } else if h == f {
                SRC_INS
            } else {
                SRC_DEL
            };
            let mut t = src;
            if ev == e_ext {
                t |= E_EXTENDED;
            }
            if f == f_ext {
                t |= F_EXTENDED;
            }
            col[i] = t;
            ecol[i] = ev;
            hdiag = hleft;
            hcol[i] = h;
            hup = h;
            // First maximum of a column-major scan: smallest j, then smallest i.
            let candidate = local || i == n || j == m;
            if candidate && h > best {
                best = h;
                best_i = i;
                best_j = j;
            }
        }
    }

    if best_i == 0 {
        // Local mode with no positive cell: nothing aligns.
        return empty;
    }

    #[derive(PartialEq)]
    enum State {
        H,
        E,
        F,
    }
    let (mut i, mut j) = (best_i, best_j);
    let mut state = State::H;
    let mut rev_ops: Vec<CigarOp> = Vec::new();
    while i > 0 && j > 0 {
        let t = trace[j * rows + i];
        match state {
            State::H => match t & SRC_MASK {
                SRC_STOP => break,
                SRC_DIAG => {
                    rev_ops.push(CigarOp::Match);
                    i -= 1;
                    j -= 1;
                }
                SRC_INS => state = State::F,
                _ => state = State::E,
            },
            State::F => {
                rev_ops.push(CigarOp::Ins);
                if t & F_EXTENDED == 0 {
                    state = State::H;
                }
                i -= 1;
            }
            State::E => {
                rev_ops.push(CigarOp::Del);
                if t & E_EXTENDED == 0 {
                    state = State::H;
                }
                j -= 1;
            }
        }
    }

    let mut ops: Vec<(CigarOp, u32)> = Vec::new();
    for op in rev_ops.into_iter().rev() {
        match ops.last_mut() {
            Some((last, len)) if *last == op => *len += 1,
            _ => ops.push((op, 1)),
        }
    }
    Alignment {
        score: best,
        query_start: i,
        query_end: best_i,
        ref_start: j,
        ref_end: best_j,
        ops,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::encode;

    fn cigar(a: &Alignment) -> String {
        a.ops
            .iter()
            .map(|(op, n)| format!("{n}{}", op.as_char()))
            .collect()
    }

    #[test]
    fn exact_match_local() {
        let q = encode(b"ACGTACGT");
        let r = encode(b"TTACGTACGTTT");
        let s = Scoring::default();
        let a = align(&q, &r, &s, Mode::Local);
        assert_eq!(a.score, 16);
        assert_eq!((a.query_start, a.query_end), (0, 8));
        assert_eq!((a.ref_start, a.ref_end), (2, 10));
        assert_eq!(cigar(&a), "8M");
        assert_eq!(score(&q, &r, &s, Mode::Local), 16);
    }

    #[test]
    fn wildcard_scores_as_match() {
        let s = Scoring::default();
        let q = encode(b"ACGTA");
        let r = encode(b"ACNTA");
        assert_eq!(score(&q, &r, &s, Mode::Local), 10);
    }

    #[test]
    fn gap_is_opened_once() {
        // 1-base deletion in the read, flanked by enough matches to pay for it.
        let s = Scoring::new(2, -4, -3, -1).unwrap();
        let r = encode(b"AAAACCCCGGGGTTTT");
        let q = encode(b"AAAACCCGGGGTTTT");
        let a = align(&q, &r, &s, Mode::Local);
        assert_eq!(a.score, 15 * 2 - 3);
        assert_eq!(a.rescore(&q, &r, &s), a.score);
        assert!(cigar(&a).contains("1D"), "{}", cigar(&a));
    }

    #[test]
    fn semiglobal_skips_overhangs_free() {
        let s = Scoring::default();
        // Read overhangs the reference's end; reference overhangs the read's start.
        let r = encode(b"GGGGGACGTACGT");
        let q = encode(b"ACGTACGTCCCCC");
        let a = align(&q, &r, &s, Mode::SemiGlobal);
        assert_eq!(a.score, 16);
        assert_eq!((a.ref_start, a.ref_end), (5, 13));
        assert_eq!((a.query_start, a.query_end), (0, 8));
        assert_eq!(score(&q, &r, &s, Mode::SemiGlobal), 16);
    }

    #[test]
    fn local_zero_is_empty() {
        let s = Scoring::default();
        let a = align(&encode(b"AAAA"), &encode(b"CCCC"), &s, Mode::Local);
        assert!(a.is_empty());
        assert_eq!(a.score, 0);
    }
}

// SPDX-License-Identifier: MIT

//! 16 × i16 lanes. See the parent module for the layout and the i16 bound.

use std::arch::x86_64::*;

use super::{Gaps, Group, LaneEnds, PairGroup, Scores};
use crate::alphabet::{N_CODES, WILDCARD};
use crate::scalar::{E_EXTENDED, F_EXTENDED, SRC_DEL, SRC_DIAG, SRC_INS};
use crate::scoring::Mode;

const W: usize = 16;

/// Score `query` against the 16 references of `g`, best score per lane into
/// `best[..16]`.
///
/// # Safety
///
/// The CPU must support AVX2, and `hbuf`/`ebuf` must hold at least
/// `query.len() * 16` elements.
#[inline(never)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn score_group(
    query: &[u8],
    g: &Group,
    gaps: Gaps,
    mode: Mode,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    best: &mut [i16; 32],
) {
    match mode {
        // SAFETY: forwarded from this function's contract.
        Mode::Local => unsafe { kernel::<true>(query, g, gaps, hbuf, ebuf, best) },
        Mode::SemiGlobal => unsafe { kernel::<false>(query, g, gaps, hbuf, ebuf, best) },
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn kernel<const LOCAL: bool>(
    query: &[u8],
    g: &Group,
    gaps: Gaps,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    best: &mut [i16; 32],
) {
    let n = query.len();
    assert!(hbuf.len() >= n * W && ebuf.len() >= n * W);
    assert_eq!(g.lens.len(), W);
    assert!(g.prof.len() >= g.len * N_CODES * W);
    let zero = _mm256_setzero_si256();
    let neg = _mm256_set1_epi16(i16::MIN);
    let vo = _mm256_set1_epi16(gaps.open);
    let ve = _mm256_set1_epi16(gaps.extend);
    let hp = hbuf.as_mut_ptr() as *mut __m256i;
    let ep = ebuf.as_mut_ptr() as *mut __m256i;
    let pp = g.prof.as_ptr();
    // SAFETY: every pointer offset below is `< n` vectors into buffers of at
    // least `n * W` i16 (asserted above), or `< g.len * N_CODES` vectors into
    // the profile; `query` codes are `< N_CODES` by construction.
    unsafe {
        for i in 0..n {
            _mm256_storeu_si256(hp.add(i), zero);
            _mm256_storeu_si256(ep.add(i), neg);
        }
        let lens = _mm256_loadu_si256(g.lens.as_ptr() as *const __m256i);
        let mut acc = if LOCAL { zero } else { neg };
        for j in 0..g.len {
            let pj = pp.add(j * N_CODES * W);
            let mut hdiag = zero;
            let mut hup = zero;
            let mut f = neg;
            let mut colmax = neg;
            for (i, &c) in query.iter().enumerate() {
                let s = _mm256_loadu_si256(pj.add(c as usize * W) as *const __m256i);
                let hleft = _mm256_loadu_si256(hp.add(i));
                let e = _mm256_max_epi16(
                    _mm256_adds_epi16(hleft, vo),
                    _mm256_adds_epi16(_mm256_loadu_si256(ep.add(i)), ve),
                );
                _mm256_storeu_si256(ep.add(i), e);
                let mut h = _mm256_max_epi16(_mm256_adds_epi16(hdiag, s), e);
                hdiag = hleft;
                f = _mm256_max_epi16(_mm256_adds_epi16(hup, vo), _mm256_adds_epi16(f, ve));
                h = _mm256_max_epi16(h, f);
                if LOCAL {
                    h = _mm256_max_epi16(h, zero);
                }
                _mm256_storeu_si256(hp.add(i), h);
                hup = h;
                colmax = _mm256_max_epi16(colmax, h);
            }
            let valid = _mm256_cmpgt_epi16(lens, _mm256_set1_epi16(j as i16));
            if LOCAL {
                acc = _mm256_blendv_epi8(acc, _mm256_max_epi16(acc, colmax), valid);
            } else {
                acc = _mm256_blendv_epi8(acc, _mm256_max_epi16(acc, hup), valid);
                let last = _mm256_cmpeq_epi16(lens, _mm256_set1_epi16(j as i16 + 1));
                acc = _mm256_blendv_epi8(acc, _mm256_max_epi16(acc, colmax), last);
            }
        }
        _mm256_storeu_si256(best.as_mut_ptr() as *mut __m256i, acc);
    }
}

/// Traceback DP over up to 16 independent (read, reference) pairs (see the
/// AVX-512 `trace_pairs`; this is the same kernel at half the width, with
/// blend vectors for masks).
///
/// # Safety
///
/// The CPU must support AVX2; `hbuf`/`ebuf` must hold `g.rows * 16` elements
/// and `tb` `(g.rows + 1) * (g.cols + 1) * 16` bytes; `g.rows` and `g.cols`
/// must be below `i16::MAX`.
#[inline(never)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn trace_pairs(
    g: &PairGroup,
    sc: Scores,
    mode: Mode,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    tb: &mut [u8],
    ends: &mut LaneEnds,
) {
    match mode {
        // SAFETY: forwarded from this function's contract.
        Mode::Local => unsafe { pairs_kernel::<true>(g, sc, hbuf, ebuf, tb, ends) },
        Mode::SemiGlobal => unsafe { pairs_kernel::<false>(g, sc, hbuf, ebuf, tb, ends) },
    }
}

/// `mask ? b : a`, lane-wise, for an all-ones/all-zeros i16 `mask`.
#[inline]
#[target_feature(enable = "avx2")]
fn select(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
    _mm256_blendv_epi8(a, b, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pairs_kernel<const LOCAL: bool>(
    g: &PairGroup,
    sc: Scores,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    tb: &mut [u8],
    ends: &mut LaneEnds,
) {
    let (n, m) = (g.rows, g.cols);
    let stride = n + 1;
    assert!(hbuf.len() >= n * W && ebuf.len() >= n * W);
    assert!(tb.len() >= stride * (m + 1) * W);
    assert!(n < i16::MAX as usize && m < i16::MAX as usize);
    assert!(g.qlens.len() == W && g.rlens.len() == W);
    assert!(g.qt.len() >= n * W && g.rt.len() >= m * W);
    let zero = _mm256_setzero_si256();
    let neg = _mm256_set1_epi16(i16::MIN);
    let wild = _mm256_set1_epi16(WILDCARD as i16);
    let vm = _mm256_set1_epi16(sc.match_score);
    let vx = _mm256_set1_epi16(sc.mismatch);
    let vo = _mm256_set1_epi16(sc.open);
    let ve = _mm256_set1_epi16(sc.extend);
    let (t_diag, t_ins, t_del) = (
        _mm256_set1_epi16(SRC_DIAG as i16),
        _mm256_set1_epi16(SRC_INS as i16),
        _mm256_set1_epi16(SRC_DEL as i16),
    );
    let (t_eext, t_fext) = (
        _mm256_set1_epi16(E_EXTENDED as i16),
        _mm256_set1_epi16(F_EXTENDED as i16),
    );
    let hp = hbuf.as_mut_ptr() as *mut __m256i;
    let ep = ebuf.as_mut_ptr() as *mut __m256i;
    let qp = g.qt.as_ptr() as *const __m256i;
    let rp = g.rt.as_ptr() as *const __m256i;
    let tp = tb.as_mut_ptr();
    // SAFETY: every offset stays inside the buffers whose sizes are asserted
    // above; a traceback row is 16 bytes at `((j+1) * stride + i+1) * 16`.
    unsafe {
        for i in 0..n {
            _mm256_storeu_si256(hp.add(i), zero);
            _mm256_storeu_si256(ep.add(i), neg);
        }
        let qlens = _mm256_loadu_si256(g.qlens.as_ptr() as *const __m256i);
        let rlens = _mm256_loadu_si256(g.rlens.as_ptr() as *const __m256i);
        let mut best = if LOCAL { zero } else { neg };
        let mut bi = zero;
        let mut bj = zero;
        for j in 0..m {
            let rv = _mm256_loadu_si256(rp.add(j));
            let rwild = _mm256_cmpeq_epi16(rv, wild);
            let jv = _mm256_set1_epi16(j as i16 + 1);
            let jvalid = _mm256_cmpgt_epi16(rlens, _mm256_set1_epi16(j as i16));
            let jlast = _mm256_cmpeq_epi16(rlens, jv);
            let tcol = tp.add((j + 1) * stride * W);
            let mut hdiag = zero;
            let mut hup = zero;
            let mut f = neg;
            for i in 0..n {
                let qv = _mm256_loadu_si256(qp.add(i));
                let i0 = _mm256_set1_epi16(i as i16);
                let iv = _mm256_set1_epi16(i as i16 + 1);
                let same = _mm256_or_si256(
                    _mm256_or_si256(_mm256_cmpeq_epi16(qv, rv), rwild),
                    _mm256_cmpeq_epi16(qv, wild),
                );
                let s = select(vx, vm, same);
                let hleft = _mm256_loadu_si256(hp.add(i));
                let e_ext = _mm256_adds_epi16(_mm256_loadu_si256(ep.add(i)), ve);
                let e = _mm256_max_epi16(_mm256_adds_epi16(hleft, vo), e_ext);
                _mm256_storeu_si256(ep.add(i), e);
                let f_ext = _mm256_adds_epi16(f, ve);
                f = _mm256_max_epi16(_mm256_adds_epi16(hup, vo), f_ext);
                let hd = _mm256_adds_epi16(hdiag, s);
                hdiag = hleft;
                let mut h = _mm256_max_epi16(_mm256_max_epi16(hd, e), f);
                if LOCAL {
                    h = _mm256_max_epi16(h, zero);
                }
                let mut t = t_del;
                t = select(t, t_ins, _mm256_cmpeq_epi16(h, f));
                t = select(t, t_diag, _mm256_cmpeq_epi16(h, hd));
                if LOCAL {
                    t = select(t, zero, _mm256_cmpeq_epi16(h, zero));
                }
                t = _mm256_or_si256(t, _mm256_and_si256(_mm256_cmpeq_epi16(e, e_ext), t_eext));
                t = _mm256_or_si256(t, _mm256_and_si256(_mm256_cmpeq_epi16(f, f_ext), t_fext));
                // 16 × i16 → 16 bytes: pack within 128-bit halves, then gather
                // the two halves' low quadwords.
                let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(t, t), 0b1000);
                _mm_storeu_si128(
                    tcol.add((i + 1) * W) as *mut __m128i,
                    _mm256_castsi256_si128(packed),
                );
                _mm256_storeu_si256(hp.add(i), h);
                hup = h;
                let ivalid = _mm256_cmpgt_epi16(qlens, i0);
                let cand = if LOCAL {
                    _mm256_and_si256(ivalid, jvalid)
                } else {
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_cmpeq_epi16(qlens, iv), jvalid),
                        _mm256_and_si256(jlast, ivalid),
                    )
                };
                let up = _mm256_and_si256(_mm256_cmpgt_epi16(h, best), cand);
                best = select(best, h, up);
                bi = select(bi, iv, up);
                bj = select(bj, jv, up);
            }
        }
        _mm256_storeu_si256(ends.best.as_mut_ptr() as *mut __m256i, best);
        _mm256_storeu_si256(ends.end_i.as_mut_ptr() as *mut __m256i, bi);
        _mm256_storeu_si256(ends.end_j.as_mut_ptr() as *mut __m256i, bj);
    }
}

/// Score `query` against the 16 references of `g` with the reference-major
/// (transposed) kernel, best score per lane into `best[..16]` (see the AVX-512
/// `score_group_transposed`; this is the same kernel at half the width, with
/// blend vectors for masks).
///
/// # Safety
///
/// The CPU must support AVX2, and `hbuf`/`fbuf` must hold at least
/// `g.len * 16` elements.
#[inline(never)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn score_group_transposed(
    query: &[u8],
    g: &Group,
    gaps: Gaps,
    mode: Mode,
    hbuf: &mut [i16],
    fbuf: &mut [i16],
    best: &mut [i16; 32],
) {
    match mode {
        // SAFETY: forwarded from this function's contract.
        Mode::Local => unsafe { transposed::<true>(query, g, gaps, hbuf, fbuf, best) },
        Mode::SemiGlobal => unsafe { transposed::<false>(query, g, gaps, hbuf, fbuf, best) },
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transposed<const LOCAL: bool>(
    query: &[u8],
    g: &Group,
    gaps: Gaps,
    hbuf: &mut [i16],
    fbuf: &mut [i16],
    best: &mut [i16; 32],
) {
    let l = g.len;
    assert!(hbuf.len() >= l * W && fbuf.len() >= l * W);
    assert_eq!(g.lens.len(), W);
    assert!(g.tprof.len() >= N_CODES * l * W && g.ends.len() >= l);
    let zero = _mm256_setzero_si256();
    let neg = _mm256_set1_epi16(i16::MIN);
    let vo = _mm256_set1_epi16(gaps.open);
    let ve = _mm256_set1_epi16(gaps.extend);
    let hp = hbuf.as_mut_ptr() as *mut __m256i;
    let fp = fbuf.as_mut_ptr() as *mut __m256i;
    let tp = g.tprof.as_ptr();
    let ends = g.ends.as_ptr();
    // SAFETY: as in the AVX-512 kernel — every offset stays inside buffers
    // whose sizes are asserted above, and query codes are `< N_CODES`.
    unsafe {
        for j in 0..l {
            _mm256_storeu_si256(hp.add(j), zero);
            _mm256_storeu_si256(fp.add(j), neg);
        }
        let lens = _mm256_loadu_si256(g.lens.as_ptr() as *const __m256i);
        let mut acc = if LOCAL { zero } else { neg };
        for &c in query {
            let row = tp.add(c as usize * l * W);
            let mut hdiag = zero;
            let mut hleft = zero;
            let mut e = neg;
            for j in 0..l {
                let s = _mm256_loadu_si256(row.add(j * W) as *const __m256i);
                let hup = _mm256_loadu_si256(hp.add(j));
                let f = _mm256_max_epi16(
                    _mm256_adds_epi16(hup, vo),
                    _mm256_adds_epi16(_mm256_loadu_si256(fp.add(j)), ve),
                );
                _mm256_storeu_si256(fp.add(j), f);
                e = _mm256_max_epi16(_mm256_adds_epi16(hleft, vo), _mm256_adds_epi16(e, ve));
                let mut h = _mm256_max_epi16(_mm256_adds_epi16(hdiag, s), e);
                hdiag = hup;
                h = _mm256_max_epi16(h, f);
                if LOCAL {
                    h = _mm256_max_epi16(h, zero);
                }
                _mm256_storeu_si256(hp.add(j), h);
                hleft = h;
                if LOCAL {
                    acc = _mm256_max_epi16(acc, h);
                } else if *ends.add(j) != 0 {
                    let last = _mm256_cmpeq_epi16(lens, _mm256_set1_epi16(j as i16 + 1));
                    acc = _mm256_blendv_epi8(acc, _mm256_max_epi16(acc, h), last);
                }
            }
        }
        if !LOCAL && !query.is_empty() {
            for j in 0..l {
                let valid = _mm256_cmpgt_epi16(lens, _mm256_set1_epi16(j as i16));
                let h = _mm256_loadu_si256(hp.add(j));
                acc = _mm256_blendv_epi8(acc, _mm256_max_epi16(acc, h), valid);
            }
        }
        _mm256_storeu_si256(best.as_mut_ptr() as *mut __m256i, acc);
    }
}

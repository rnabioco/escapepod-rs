// SPDX-License-Identifier: MIT

//! 16 × i16 lanes. See the parent module for the layout and the i16 bound.

use std::arch::x86_64::*;

use super::{Gaps, Group, LaneEnds};
use crate::alphabet::N_CODES;
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

/// The score kernel plus a traceback byte per cell and lane, and each lane's
/// end cell in the scalar path's scan order (see `super::align_many`).
///
/// # Safety
///
/// As [`score_group`], and `tb` must hold `(query.len() + 1) * (g.len + 1) *
/// 16` bytes; `query.len()` must be below `i16::MAX`.
#[inline(never)]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn trace_group(
    query: &[u8],
    g: &Group,
    gaps: Gaps,
    mode: Mode,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    tb: &mut [u8],
    ends: &mut LaneEnds,
) {
    match mode {
        // SAFETY: forwarded from this function's contract.
        Mode::Local => unsafe { trace_kernel::<true>(query, g, gaps, hbuf, ebuf, tb, ends) },
        Mode::SemiGlobal => unsafe { trace_kernel::<false>(query, g, gaps, hbuf, ebuf, tb, ends) },
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
unsafe fn trace_kernel<const LOCAL: bool>(
    query: &[u8],
    g: &Group,
    gaps: Gaps,
    hbuf: &mut [i16],
    ebuf: &mut [i16],
    tb: &mut [u8],
    ends: &mut LaneEnds,
) {
    let n = query.len();
    let rows = n + 1;
    assert!(hbuf.len() >= n * W && ebuf.len() >= n * W);
    assert!(tb.len() >= rows * (g.len + 1) * W);
    assert!(n < i16::MAX as usize && g.len < i16::MAX as usize);
    assert_eq!(g.lens.len(), W);
    assert!(g.prof.len() >= g.len * N_CODES * W);
    let zero = _mm256_setzero_si256();
    let neg = _mm256_set1_epi16(i16::MIN);
    let vo = _mm256_set1_epi16(gaps.open);
    let ve = _mm256_set1_epi16(gaps.extend);
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
    let pp = g.prof.as_ptr();
    let tp = tb.as_mut_ptr();
    // SAFETY: offsets stay inside the buffers whose sizes are asserted above;
    // a traceback row is 16 bytes at `((j+1) * rows + i+1) * 16`, `j < g.len`,
    // `i < n`.
    unsafe {
        for i in 0..n {
            _mm256_storeu_si256(hp.add(i), zero);
            _mm256_storeu_si256(ep.add(i), neg);
        }
        let lens = _mm256_loadu_si256(g.lens.as_ptr() as *const __m256i);
        let mut best = if LOCAL { zero } else { neg };
        let mut bi = zero;
        let mut bj = zero;
        for j in 0..g.len {
            let pj = pp.add(j * N_CODES * W);
            let jv = _mm256_set1_epi16(j as i16 + 1);
            let valid = _mm256_cmpgt_epi16(lens, _mm256_set1_epi16(j as i16));
            let tcol = tp.add((j + 1) * rows * W);
            let mut hdiag = zero;
            let mut hup = zero;
            let mut f = neg;
            let mut colmax = neg;
            let mut colarg = zero;
            for (i, &c) in query.iter().enumerate() {
                let iv = _mm256_set1_epi16(i as i16 + 1);
                let s = _mm256_loadu_si256(pj.add(c as usize * W) as *const __m256i);
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
                if LOCAL {
                    let up = _mm256_and_si256(_mm256_cmpgt_epi16(h, best), valid);
                    best = select(best, h, up);
                    bi = select(bi, iv, up);
                    bj = select(bj, jv, up);
                } else {
                    let up = _mm256_cmpgt_epi16(h, colmax);
                    colmax = select(colmax, h, up);
                    colarg = select(colarg, iv, up);
                }
            }
            if !LOCAL {
                let last = _mm256_cmpeq_epi16(lens, jv);
                let up = _mm256_and_si256(_mm256_cmpgt_epi16(colmax, best), last);
                best = select(best, colmax, up);
                bi = select(bi, colarg, up);
                bj = select(bj, jv, up);
                let up = _mm256_andnot_si256(
                    last,
                    _mm256_and_si256(_mm256_cmpgt_epi16(hup, best), valid),
                );
                best = select(best, hup, up);
                bi = select(bi, _mm256_set1_epi16(n as i16), up);
                bj = select(bj, jv, up);
            }
        }
        _mm256_storeu_si256(ends.best.as_mut_ptr() as *mut __m256i, best);
        _mm256_storeu_si256(ends.end_i.as_mut_ptr() as *mut __m256i, bi);
        _mm256_storeu_si256(ends.end_j.as_mut_ptr() as *mut __m256i, bj);
    }
}

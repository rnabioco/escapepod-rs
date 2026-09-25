// SPDX-License-Identifier: MIT

//! 32 × i16 lanes (AVX-512BW). See the parent module for the layout and the
//! i16 bound. Same recurrence as the AVX2 kernel; the per-column lane masks
//! are mask registers rather than blend vectors.

use std::arch::x86_64::*;

use super::{Gaps, Group, LaneEnds};
use crate::alphabet::N_CODES;
use crate::scalar::{E_EXTENDED, F_EXTENDED, SRC_DEL, SRC_DIAG, SRC_INS};
use crate::scoring::Mode;

const W: usize = 32;

/// Score `query` against the 32 references of `g`, best score per lane into
/// `best`.
///
/// # Safety
///
/// The CPU must support AVX-512F and AVX-512BW, and `hbuf`/`ebuf` must hold
/// at least `query.len() * 32` elements.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw")]
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
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_si512();
    let neg = _mm512_set1_epi16(i16::MIN);
    let vo = _mm512_set1_epi16(gaps.open);
    let ve = _mm512_set1_epi16(gaps.extend);
    let hp = hbuf.as_mut_ptr() as *mut __m512i;
    let ep = ebuf.as_mut_ptr() as *mut __m512i;
    let pp = g.prof.as_ptr();
    // SAFETY: as in the AVX2 kernel — every offset stays inside buffers whose
    // sizes are asserted above, and query codes are `< N_CODES`.
    unsafe {
        for i in 0..n {
            _mm512_storeu_si512(hp.add(i), zero);
            _mm512_storeu_si512(ep.add(i), neg);
        }
        let lens = _mm512_loadu_si512(g.lens.as_ptr() as *const __m512i);
        let mut acc = if LOCAL { zero } else { neg };
        for j in 0..g.len {
            let pj = pp.add(j * N_CODES * W);
            let mut hdiag = zero;
            let mut hup = zero;
            let mut f = neg;
            let mut colmax = neg;
            for (i, &c) in query.iter().enumerate() {
                let s = _mm512_loadu_si512(pj.add(c as usize * W) as *const __m512i);
                let hleft = _mm512_loadu_si512(hp.add(i));
                let e = _mm512_max_epi16(
                    _mm512_adds_epi16(hleft, vo),
                    _mm512_adds_epi16(_mm512_loadu_si512(ep.add(i)), ve),
                );
                _mm512_storeu_si512(ep.add(i), e);
                let mut h = _mm512_max_epi16(_mm512_adds_epi16(hdiag, s), e);
                hdiag = hleft;
                f = _mm512_max_epi16(_mm512_adds_epi16(hup, vo), _mm512_adds_epi16(f, ve));
                h = _mm512_max_epi16(h, f);
                if LOCAL {
                    h = _mm512_max_epi16(h, zero);
                }
                _mm512_storeu_si512(hp.add(i), h);
                hup = h;
                colmax = _mm512_max_epi16(colmax, h);
            }
            let valid = _mm512_cmpgt_epi16_mask(lens, _mm512_set1_epi16(j as i16));
            if LOCAL {
                acc = _mm512_mask_max_epi16(acc, valid, acc, colmax);
            } else {
                acc = _mm512_mask_max_epi16(acc, valid, acc, hup);
                let last = _mm512_cmpeq_epi16_mask(lens, _mm512_set1_epi16(j as i16 + 1));
                acc = _mm512_mask_max_epi16(acc, last, acc, colmax);
            }
        }
        _mm512_storeu_si512(best.as_mut_ptr() as *mut __m512i, acc);
    }
}

/// The score kernel plus a traceback byte per cell and lane, and each lane's
/// end cell in the scalar path's scan order (see `super::align_many`).
///
/// # Safety
///
/// As [`score_group`], and `tb` must hold `(query.len() + 1) * (g.len + 1) *
/// 32` bytes; `query.len()` must be below `i16::MAX`.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw")]
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

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_si512();
    let neg = _mm512_set1_epi16(i16::MIN);
    let vo = _mm512_set1_epi16(gaps.open);
    let ve = _mm512_set1_epi16(gaps.extend);
    let (t_diag, t_ins, t_del) = (
        _mm512_set1_epi16(SRC_DIAG as i16),
        _mm512_set1_epi16(SRC_INS as i16),
        _mm512_set1_epi16(SRC_DEL as i16),
    );
    let (t_eext, t_fext) = (
        _mm512_set1_epi16(E_EXTENDED as i16),
        _mm512_set1_epi16(F_EXTENDED as i16),
    );
    let hp = hbuf.as_mut_ptr() as *mut __m512i;
    let ep = ebuf.as_mut_ptr() as *mut __m512i;
    let pp = g.prof.as_ptr();
    let tp = tb.as_mut_ptr();
    // SAFETY: offsets stay inside the buffers whose sizes are asserted above;
    // a traceback row is 32 bytes at `((j+1) * rows + i+1) * 32`, `j < g.len`,
    // `i < n`.
    unsafe {
        for i in 0..n {
            _mm512_storeu_si512(hp.add(i), zero);
            _mm512_storeu_si512(ep.add(i), neg);
        }
        let lens = _mm512_loadu_si512(g.lens.as_ptr() as *const __m512i);
        let mut best = if LOCAL { zero } else { neg };
        let mut bi = zero;
        let mut bj = zero;
        for j in 0..g.len {
            let pj = pp.add(j * N_CODES * W);
            let jv = _mm512_set1_epi16(j as i16 + 1);
            let valid = _mm512_cmpgt_epi16_mask(lens, _mm512_set1_epi16(j as i16));
            let tcol = tp.add((j + 1) * rows * W);
            let mut hdiag = zero;
            let mut hup = zero;
            let mut f = neg;
            // Semi-global: the first maximum of this column, for lanes whose
            // last column it is.
            let mut colmax = neg;
            let mut colarg = zero;
            for (i, &c) in query.iter().enumerate() {
                let iv = _mm512_set1_epi16(i as i16 + 1);
                let s = _mm512_loadu_si512(pj.add(c as usize * W) as *const __m512i);
                let hleft = _mm512_loadu_si512(hp.add(i));
                let e_ext = _mm512_adds_epi16(_mm512_loadu_si512(ep.add(i)), ve);
                let e = _mm512_max_epi16(_mm512_adds_epi16(hleft, vo), e_ext);
                _mm512_storeu_si512(ep.add(i), e);
                let f_ext = _mm512_adds_epi16(f, ve);
                f = _mm512_max_epi16(_mm512_adds_epi16(hup, vo), f_ext);
                let hd = _mm512_adds_epi16(hdiag, s);
                hdiag = hleft;
                let mut h = _mm512_max_epi16(_mm512_max_epi16(hd, e), f);
                if LOCAL {
                    h = _mm512_max_epi16(h, zero);
                }
                // Source, in the scalar path's order of preference.
                let mut t = t_del;
                t = _mm512_mask_mov_epi16(t, _mm512_cmpeq_epi16_mask(h, f), t_ins);
                t = _mm512_mask_mov_epi16(t, _mm512_cmpeq_epi16_mask(h, hd), t_diag);
                if LOCAL {
                    t = _mm512_mask_mov_epi16(t, _mm512_cmpeq_epi16_mask(h, zero), zero);
                }
                t = _mm512_mask_add_epi16(t, _mm512_cmpeq_epi16_mask(e, e_ext), t, t_eext);
                t = _mm512_mask_add_epi16(t, _mm512_cmpeq_epi16_mask(f, f_ext), t, t_fext);
                _mm256_storeu_si256(
                    tcol.add((i + 1) * W) as *mut __m256i,
                    _mm512_cvtepi16_epi8(t),
                );
                _mm512_storeu_si512(hp.add(i), h);
                hup = h;
                if LOCAL {
                    let up = _mm512_cmpgt_epi16_mask(h, best) & valid;
                    best = _mm512_mask_mov_epi16(best, up, h);
                    bi = _mm512_mask_mov_epi16(bi, up, iv);
                    bj = _mm512_mask_mov_epi16(bj, up, jv);
                } else {
                    let up = _mm512_cmpgt_epi16_mask(h, colmax);
                    colmax = _mm512_mask_mov_epi16(colmax, up, h);
                    colarg = _mm512_mask_mov_epi16(colarg, up, iv);
                }
            }
            if !LOCAL {
                let last = _mm512_cmpeq_epi16_mask(lens, jv);
                // A lane's last column: its first maximum over the column.
                let up = _mm512_cmpgt_epi16_mask(colmax, best) & last;
                best = _mm512_mask_mov_epi16(best, up, colmax);
                bi = _mm512_mask_mov_epi16(bi, up, colarg);
                bj = _mm512_mask_mov_epi16(bj, up, jv);
                // Any other valid column: its last row.
                let up = _mm512_cmpgt_epi16_mask(hup, best) & valid & !last;
                best = _mm512_mask_mov_epi16(best, up, hup);
                bi = _mm512_mask_mov_epi16(bi, up, _mm512_set1_epi16(n as i16));
                bj = _mm512_mask_mov_epi16(bj, up, jv);
            }
        }
        _mm512_storeu_si512(ends.best.as_mut_ptr() as *mut __m512i, best);
        _mm512_storeu_si512(ends.end_i.as_mut_ptr() as *mut __m512i, bi);
        _mm512_storeu_si512(ends.end_j.as_mut_ptr() as *mut __m512i, bj);
    }
}

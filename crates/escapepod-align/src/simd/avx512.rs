// SPDX-License-Identifier: MIT

//! 32 × i16 lanes (AVX-512BW). See the parent module for the layout and the
//! i16 bound. Same recurrence as the AVX2 kernel; the per-column lane masks
//! are mask registers rather than blend vectors.

use std::arch::x86_64::*;

use super::{Gaps, Group};
use crate::alphabet::N_CODES;
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

// SPDX-License-Identifier: MIT

//! 16 × i16 lanes. See the parent module for the layout and the i16 bound.

use std::arch::x86_64::*;

use super::{Gaps, Group};
use crate::alphabet::N_CODES;
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

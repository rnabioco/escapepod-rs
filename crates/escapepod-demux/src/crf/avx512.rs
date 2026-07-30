//! AVX-512 kernels for the CTC-CRF lattice.
//!
//! A 16-wide mirror of [`super::avx2`], structured identically so the two are
//! diffable: same reshapes ([`expand`], [`deinterleave4`]), same five-row
//! reduction, same Cephes `exp`/`ln`. Only the vector width and the mask
//! handling differ — AVX-512 comparisons produce `__mmask16` registers rather
//! than lane masks, which makes the `ln` fold and the argmax scan cheaper.
//!
//! Worth having because after the cheaper wins (one fewer `exp` in the
//! softmax, a `movemask` argmax, no per-read memset) the decode is still ~78%
//! three vector kernels — `log_softmax_floored`, `forward`, `backward` — all
//! transcendental-bound and all trivially wider.
//!
//! Gated on `avx512f` alone. `_mm512_and_ps`/`_mm512_or_ps` would pull in
//! AVX512DQ, so the bit-twiddling in [`ln16`] goes through `_mm512_and_si512`
//! with casts instead; that keeps this usable on every AVX-512 part rather than
//! only the ones with DQ.
//!
//! Per the repository's build policy this is runtime-dispatched, not a baseline
//! bump: the pinned `target-cpu=x86-64-v3` stays portable across the Broadwell
//! login node, Cascade Lake `rna`, and Ice Lake `gpu`, and this kernel is only
//! selected where the CPU actually reports the feature.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::lattice::{CrfLayout, Semiring};

/// Same layout constraints as the AVX2 path, but on 64-element blocks: the
/// backward fold consumes `4 * 16` destinations per pass.
pub(super) fn supported(layout: &CrfLayout) -> bool {
    layout.n_base == 4 && layout.n_states.is_multiple_of(64) && layout.n_states >= 64
}

pub(super) fn available() -> bool {
    is_x86_feature_detected!("avx512f")
}

// ---------------------------------------------------------------------------
// Transcendentals
// ---------------------------------------------------------------------------

const EXP_HI: f32 = 88.376_26;
const EXP_LO: f32 = -88.376_26;
const LOG2EF: f32 = std::f32::consts::LOG2_E;
const EXP_C1: f32 = 0.693_359_4;
const EXP_C2: f32 = -2.121_944_4e-4;
const SQRTHF: f32 = 0.707_106_77;

/// `exp(x)` for sixteen lanes. Cephes `expf`; see [`super::avx2::exp8`] — the
/// polynomial and range reduction are identical.
#[inline]
#[target_feature(enable = "avx512f")]
fn exp16(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let x = _mm512_min_ps(_mm512_set1_ps(EXP_HI), x);
    let x = _mm512_max_ps(_mm512_set1_ps(EXP_LO), x);

    // round-to-nearest, suppress exceptions
    let fx = _mm512_roundscale_ps::<0x08>(_mm512_mul_ps(x, _mm512_set1_ps(LOG2EF)));
    let r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(EXP_C1), x);
    let r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(EXP_C2), r);

    let z = _mm512_mul_ps(r, r);
    let mut y = _mm512_set1_ps(1.987_569_1e-4);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.398_199_9e-3));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.333_452e-3));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.166_579_6e-2));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.666_666_5e-1));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.000_000_6e-1));
    y = _mm512_fmadd_ps(y, z, _mm512_add_ps(r, one));

    let n = _mm512_cvttps_epi32(fx);
    let n = _mm512_add_epi32(n, _mm512_set1_epi32(0x7f));
    let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(n));
    _mm512_mul_ps(y, pow2n)
}

/// `ln(x)` for sixteen lanes, for strictly positive `x`. Cephes `logf`; see
/// [`super::avx2::ln8`].
#[inline]
#[target_feature(enable = "avx512f")]
fn ln16(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let x = _mm512_max_ps(x, _mm512_set1_ps(f32::MIN_POSITIVE));

    let e = _mm512_srli_epi32::<23>(_mm512_castps_si512(x));
    let e = _mm512_sub_epi32(e, _mm512_set1_epi32(0x7f));
    let mut e = _mm512_cvtepi32_ps(e);
    // Mantissa into [0.5, 1): clear the exponent field, set it to 2^-1.
    let mant = _mm512_and_si512(
        _mm512_castps_si512(x),
        _mm512_set1_epi32(!0x7f80_0000u32 as i32),
    );
    let m = _mm512_castsi512_ps(_mm512_or_si512(mant, _mm512_set1_epi32(0x3f00_0000)));
    e = _mm512_add_ps(e, one);

    // Fold [0.5, 1) around 1 into [sqrt(1/2), sqrt(2)): below sqrt(1/2) the
    // mantissa doubles (2m - 1) and the exponent drops by one; above it, m - 1.
    let lt = _mm512_cmp_ps_mask::<_CMP_LT_OS>(m, _mm512_set1_ps(SQRTHF));
    let shifted = _mm512_sub_ps(m, one);
    let m = _mm512_mask_add_ps(shifted, lt, shifted, m);
    e = _mm512_mask_sub_ps(e, lt, e, one);

    let z = _mm512_mul_ps(m, m);
    let mut y = _mm512_set1_ps(7.037_683_6e-2);
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(-1.151_461e-1));
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(1.167_699_9e-1));
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(-1.242_014_1e-1));
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(1.424_932_3e-1));
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(-1.666_805_8e-1));
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(2.000_071_5e-1));
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(-2.499_999_4e-1));
    y = _mm512_fmadd_ps(y, m, _mm512_set1_ps(3.333_333_3e-1));
    y = _mm512_mul_ps(_mm512_mul_ps(y, m), z);

    y = _mm512_fmadd_ps(e, _mm512_set1_ps(EXP_C2), y);
    y = _mm512_fnmadd_ps(_mm512_set1_ps(0.5), z, y);
    let out = _mm512_add_ps(m, y);
    _mm512_fmadd_ps(e, _mm512_set1_ps(EXP_C1), out)
}

// ---------------------------------------------------------------------------
// Semiring reductions
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx512f")]
fn max5(v: [__m512; 5]) -> __m512 {
    let a = _mm512_max_ps(v[0], v[1]);
    let b = _mm512_max_ps(v[2], v[3]);
    _mm512_max_ps(_mm512_max_ps(a, b), v[4])
}

#[inline]
#[target_feature(enable = "avx512f")]
fn lse5(v: [__m512; 5]) -> __m512 {
    let m = max5(v);
    let mut s = exp16(_mm512_sub_ps(v[0], m));
    s = _mm512_add_ps(s, exp16(_mm512_sub_ps(v[1], m)));
    s = _mm512_add_ps(s, exp16(_mm512_sub_ps(v[2], m)));
    s = _mm512_add_ps(s, exp16(_mm512_sub_ps(v[3], m)));
    s = _mm512_add_ps(s, exp16(_mm512_sub_ps(v[4], m)));
    _mm512_add_ps(m, ln16(s))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn reduce5(v: [__m512; 5], s: Semiring) -> __m512 {
    match s {
        Semiring::Max => max5(v),
        Semiring::Log => lse5(v),
    }
}

// ---------------------------------------------------------------------------
// Reshapes
// ---------------------------------------------------------------------------

/// `dst[j * n_states + c] = src[j * group + c / 4]`. See [`super::avx2::expand`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn expand(src: &[f32], n_states: usize, group: usize, dst: &mut [f32]) {
    unsafe {
        debug_assert_eq!(dst.len(), 4 * n_states);
        for j in 0..4 {
            let row = &src[j * group..j * group + group];
            let out = &mut dst[j * n_states..(j + 1) * n_states];
            // 16 inputs become 64 outputs: each source value four times.
            for (chunk, o) in row.chunks_exact(16).zip(out.chunks_exact_mut(64)) {
                let v = _mm512_loadu_ps(chunk.as_ptr());
                for i in 0..4 {
                    let base = 4 * i as i32;
                    let idx = _mm512_setr_epi32(
                        base,
                        base,
                        base,
                        base,
                        base + 1,
                        base + 1,
                        base + 1,
                        base + 1,
                        base + 2,
                        base + 2,
                        base + 2,
                        base + 2,
                        base + 3,
                        base + 3,
                        base + 3,
                        base + 3,
                    );
                    _mm512_storeu_ps(o.as_mut_ptr().add(i * 16), _mm512_permutexvar_ps(idx, v));
                }
            }
        }
    }
}

/// `dst[k * (n / 4) + u] = src[4 * u + k]`. See [`super::avx2::deinterleave4`].
///
/// Each 64-element block feeds two `permutex2var` pairs per output row: the
/// first eight results come from the block's low 32 lanes, the second eight
/// from its high 32, and `shuffle_f32x4` splices the two halves.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn deinterleave4(src: &[f32], dst: &mut [f32]) {
    unsafe {
        let n = src.len();
        let quarter = n / 4;
        debug_assert!(n.is_multiple_of(64));
        debug_assert_eq!(dst.len(), n);

        for (blk, u0) in (0..n).step_by(64).zip((0..quarter).step_by(16)) {
            let a = _mm512_loadu_ps(src.as_ptr().add(blk));
            let b = _mm512_loadu_ps(src.as_ptr().add(blk + 16));
            let c = _mm512_loadu_ps(src.as_ptr().add(blk + 32));
            let d = _mm512_loadu_ps(src.as_ptr().add(blk + 48));

            for k in 0..4 {
                let j = k as i32;
                // Lanes 0..7 pick 4l+k from the (a,b) 32-lane space; lanes 8..15
                // repeat them so the same index vector serves the (c,d) pair.
                let idx = _mm512_setr_epi32(
                    j,
                    j + 4,
                    j + 8,
                    j + 12,
                    j + 16,
                    j + 20,
                    j + 24,
                    j + 28,
                    j,
                    j + 4,
                    j + 8,
                    j + 12,
                    j + 16,
                    j + 20,
                    j + 24,
                    j + 28,
                );
                let lo = _mm512_permutex2var_ps(a, idx, b);
                let hi = _mm512_permutex2var_ps(c, idx, d);
                // Take the low 256 bits of each: [lo0, lo1, hi0, hi1].
                let row = _mm512_shuffle_f32x4::<0x44>(lo, hi);
                _mm512_storeu_ps(dst.as_mut_ptr().add(k * quarter + u0), row);
            }
        }
    }
}

/// `dst[edge * n_states + dest] = src[dest * n_edges + edge]` — the
/// `[dest][edge]` → `[edge][dest]` transpose of one timestep.
///
/// `n_edges` is 5, so this is a five-way de-interleave that no shuffle network
/// handles cleanly (5 does not divide the register width). A strided gather per
/// edge row does, and the scalar version of this was ~19% of the AVX-512
/// decode — the largest remaining single item once the lattice passes were
/// widened.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn transpose_scores(
    src: &[f32],
    dst: &mut [f32],
    n_states: usize,
    n_edges: usize,
) {
    unsafe {
        let lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let step = _mm512_mullo_epi32(lane, _mm512_set1_epi32(n_edges as i32));
        let bump = _mm512_set1_epi32((16 * n_edges) as i32);
        for edge in 0..n_edges {
            let base = src.as_ptr().add(edge);
            let out = dst.as_mut_ptr().add(edge * n_states);
            let mut idx = step;
            for d in (0..n_states).step_by(16) {
                _mm512_storeu_ps(out.add(d), _mm512_i32gather_ps::<4>(idx, base));
                idx = _mm512_add_epi32(idx, bump);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lattice passes
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn edge_term(work: &[f32], row: &[f32], j: usize, c: usize, n_states: usize) -> __m512 {
    unsafe {
        _mm512_add_ps(
            _mm512_loadu_ps(work.as_ptr().add(j * n_states + c)),
            _mm512_loadu_ps(row.as_ptr().add((j + 1) * n_states + c)),
        )
    }
}

/// Forward scores; see [`super::lattice`] for the recurrence.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn forward(
    layout: &CrfLayout,
    scores: &[f32],
    t_len: usize,
    s: Semiring,
    alpha: &mut [f32],
    work: &mut [f32],
) {
    unsafe {
        let (n_states, group) = (layout.n_states, layout.group());
        alpha[..n_states].fill(0.0);
        for t in 0..t_len {
            let (prev, next) = alpha.split_at_mut((t + 1) * n_states);
            let prev = &prev[t * n_states..];
            let row = &scores[t * layout.n_score..(t + 1) * layout.n_score];
            expand(prev, n_states, group, work);
            for c in (0..n_states).step_by(16) {
                let v = [
                    _mm512_add_ps(
                        _mm512_loadu_ps(prev.as_ptr().add(c)),
                        _mm512_loadu_ps(row.as_ptr().add(c)),
                    ),
                    edge_term(work, row, 0, c, n_states),
                    edge_term(work, row, 1, c, n_states),
                    edge_term(work, row, 2, c, n_states),
                    edge_term(work, row, 3, c, n_states),
                ];
                _mm512_storeu_ps(next.as_mut_ptr().add(c), reduce5(v, s));
            }
        }
    }
}

/// Backward scores; see [`super::lattice`] for the recurrence.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn backward(
    layout: &CrfLayout,
    scores: &[f32],
    t_len: usize,
    s: Semiring,
    beta: &mut [f32],
    work: &mut [f32],
) {
    unsafe {
        let (n_states, group) = (layout.n_states, layout.group());
        let (summed, rest) = work.split_at_mut(n_states);
        let folded = &mut rest[..n_states];
        beta[t_len * n_states..(t_len + 1) * n_states].fill(0.0);
        for t in (0..t_len).rev() {
            let (cur, next) = beta.split_at_mut((t + 1) * n_states);
            let cur = &mut cur[t * n_states..];
            let row = &scores[t * layout.n_score..(t + 1) * layout.n_score];
            for j in 0..4 {
                let move_row = &row[(j + 1) * n_states..(j + 2) * n_states];
                for c in (0..n_states).step_by(16) {
                    _mm512_storeu_ps(
                        summed.as_mut_ptr().add(c),
                        _mm512_add_ps(
                            _mm512_loadu_ps(move_row.as_ptr().add(c)),
                            _mm512_loadu_ps(next.as_ptr().add(c)),
                        ),
                    );
                }
                deinterleave4(summed, folded);
                let out = &mut cur[j * group..(j + 1) * group];
                for u in (0..group).step_by(16) {
                    let stay = _mm512_add_ps(
                        _mm512_loadu_ps(row.as_ptr().add(j * group + u)),
                        _mm512_loadu_ps(next.as_ptr().add(j * group + u)),
                    );
                    let v = [
                        stay,
                        _mm512_loadu_ps(folded.as_ptr().add(u)),
                        _mm512_loadu_ps(folded.as_ptr().add(group + u)),
                        _mm512_loadu_ps(folded.as_ptr().add(2 * group + u)),
                        _mm512_loadu_ps(folded.as_ptr().add(3 * group + u)),
                    ];
                    _mm512_storeu_ps(out.as_mut_ptr().add(u), reduce5(v, s));
                }
            }
        }
    }
}

/// Per-timestep edge scores `alpha[t][source] + score + beta[t + 1][dest]`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn edge_scores(
    layout: &CrfLayout,
    row: &[f32],
    alpha_t: &[f32],
    beta_next: &[f32],
    out: &mut [f32],
    work: &mut [f32],
) {
    unsafe {
        let (n_states, group) = (layout.n_states, layout.group());
        expand(alpha_t, n_states, group, work);
        for c in (0..n_states).step_by(16) {
            let b = _mm512_loadu_ps(beta_next.as_ptr().add(c));
            let stay = _mm512_add_ps(
                _mm512_add_ps(
                    _mm512_loadu_ps(alpha_t.as_ptr().add(c)),
                    _mm512_loadu_ps(row.as_ptr().add(c)),
                ),
                b,
            );
            _mm512_storeu_ps(out.as_mut_ptr().add(c), stay);
            for j in 0..4 {
                let v = _mm512_add_ps(edge_term(work, row, j, c, n_states), b);
                _mm512_storeu_ps(out.as_mut_ptr().add((j + 1) * n_states + c), v);
            }
        }
    }
}

/// `log(softmax(src) + floor)` over a whole timestep's edge scores.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn log_softmax_floored(src: &[f32], dst: &mut [f32], floor: f32) {
    unsafe {
        let n = src.len();
        debug_assert!(n.is_multiple_of(16));

        let mut mv = _mm512_set1_ps(f32::NEG_INFINITY);
        for c in (0..n).step_by(16) {
            mv = _mm512_max_ps(mv, _mm512_loadu_ps(src.as_ptr().add(c)));
        }
        let m = _mm512_set1_ps(_mm512_reduce_max_ps(mv));

        let mut sv = _mm512_setzero_ps();
        for c in (0..n).step_by(16) {
            let e = exp16(_mm512_sub_ps(_mm512_loadu_ps(src.as_ptr().add(c)), m));
            _mm512_storeu_ps(dst.as_mut_ptr().add(c), e);
            sv = _mm512_add_ps(sv, e);
        }
        let inv = _mm512_set1_ps(1.0 / _mm512_reduce_add_ps(sv));

        let fl = _mm512_set1_ps(floor);
        for c in (0..n).step_by(16) {
            let e = _mm512_loadu_ps(dst.as_ptr().add(c));
            _mm512_storeu_ps(dst.as_mut_ptr().add(c), ln16(_mm512_fmadd_ps(e, inv, fl)));
        }
    }
}

/// The `(dest, edge)` with the highest score, ties broken toward the lowest
/// `dest * n_edges + edge`. See [`super::avx2::argmax_edge`].
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn argmax_edge(
    edges: &[f32],
    n_states: usize,
    n_edges: usize,
) -> Option<(usize, usize)> {
    unsafe {
        let mut mv = _mm512_set1_ps(f32::NEG_INFINITY);
        for c in (0..edges.len()).step_by(16) {
            mv = _mm512_max_ps(mv, _mm512_loadu_ps(edges.as_ptr().add(c)));
        }
        let peak = _mm512_set1_ps(_mm512_reduce_max_ps(mv));

        let mut best: Option<(usize, usize)> = None;
        let mut best_flat = usize::MAX;
        for edge in 0..n_edges {
            let row = edges.as_ptr().add(edge * n_states);
            for c in (0..n_states).step_by(16) {
                let hits = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(_mm512_loadu_ps(row.add(c)), peak);
                if hits != 0 {
                    let dest = c + hits.trailing_zeros() as usize;
                    let flat = dest * n_edges + edge;
                    if flat < best_flat {
                        best_flat = flat;
                        best = Some((dest, edge));
                    }
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval16(f: unsafe fn(__m512) -> __m512, xs: [f32; 16]) -> [f32; 16] {
        unsafe {
            let mut out = [0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), f(_mm512_loadu_ps(xs.as_ptr())));
            out
        }
    }

    #[test]
    fn exp16_matches_std_over_the_normal_range() {
        if !available() {
            return;
        }
        let mut worst = 0f32;
        for base in -870..880 {
            let xs: [f32; 16] = std::array::from_fn(|i| (base as f32 + i as f32 / 16.0) / 10.0);
            for (x, g) in xs.iter().zip(eval16(exp16, xs)) {
                let want = x.exp();
                if want < f32::MIN_POSITIVE {
                    continue;
                }
                worst = worst.max(((g - want) / want).abs());
            }
        }
        assert!(worst < 2e-6, "worst relative error {worst:e}");
    }

    #[test]
    fn ln16_matches_std() {
        if !available() {
            return;
        }
        let mut worst = 0f32;
        for base in 1..2000 {
            let xs: [f32; 16] =
                std::array::from_fn(|i| (base as f32 + i as f32 / 16.0) * 1e-4 + 1e-9);
            for (x, g) in xs.iter().zip(eval16(ln16, xs)) {
                worst = worst.max((g - x.ln()).abs() / x.ln().abs().max(1.0));
            }
        }
        assert!(worst < 2e-6, "worst error {worst:e}");
    }

    #[test]
    fn expand_replicates_each_source_four_times() {
        if !available() {
            return;
        }
        let (n_states, group) = (256usize, 64usize);
        let src: Vec<f32> = (0..n_states).map(|i| i as f32).collect();
        let mut dst = vec![0f32; 4 * n_states];
        unsafe { expand(&src, n_states, group, &mut dst) };
        for j in 0..4 {
            for c in 0..n_states {
                assert_eq!(dst[j * n_states + c], src[j * group + c / 4], "j={j} c={c}");
            }
        }
    }

    #[test]
    fn deinterleave4_is_a_stride_4_transpose() {
        if !available() {
            return;
        }
        let n = 256usize;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut dst = vec![0f32; n];
        unsafe { deinterleave4(&src, &mut dst) };
        for k in 0..4 {
            for u in 0..n / 4 {
                assert_eq!(dst[k * (n / 4) + u], src[4 * u + k], "k={k} u={u}");
            }
        }
    }
}

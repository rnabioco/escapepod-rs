//! AVX2 kernels for the CTC-CRF lattice.
//!
//! The decode is transcendental-bound, not memory-bound: one 200-timestep read
//! costs roughly 770k `exp` and 260k `ln` calls across the two forward/backward
//! passes and the posterior softmax. Scalar `f32::exp` puts the decode at the
//! same order as the ONNX encoder on CPU, and makes it *dominate* once the
//! encoder moves to the GPU — so vectorising it is what lets the GPU path pay
//! off at all.
//!
//! Everything here is validated against [`super::lattice`]'s scalar
//! implementation rather than trusted: the polynomial `exp`/`ln` below are
//! approximations, so the contract is "same decoded sequence, floats within a
//! tight tolerance", not bit-identity.
//!
//! # Data movement
//!
//! The forward and backward recurrences are duals and neither is naturally
//! unit-stride on both sides, so each gets one cheap reshape:
//!
//! * **Forward** needs `alpha[source(c, edge)]` laid out along `c`. For move
//!   edges `source` is `j * group + c / n_base`, i.e. each of `group` values
//!   repeated `n_base` times — [`expand`] materialises that.
//! * **Backward** reduces `n_base` consecutive destinations onto one source, so
//!   it needs the transpose: [`deinterleave4`] turns `v[4u + k]` into four
//!   unit-stride rows indexed by `u`.
//!
//! After those, every inner loop is five unit-stride rows reduced elementwise,
//! which is the same shape for both passes and both semirings.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::lattice::{CrfLayout, Semiring};

/// The AVX2 path hardcodes `n_base == 4` (so `n_edges == 5`) and works on
/// 32-element blocks of the state axis. Both hold for every bonito CRF model;
/// anything else falls back to scalar.
pub(super) fn supported(layout: &CrfLayout) -> bool {
    layout.n_base == 4 && layout.n_states.is_multiple_of(32) && layout.n_states >= 32
}

/// Runtime feature check. `.cargo/config.toml` pins `target-cpu=x86-64-v3`
/// locally, which implies both, but the musl release artifacts are built
/// without it — so this is a real check, not a formality.
pub(super) fn available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

// ---------------------------------------------------------------------------
// Transcendentals (Cephes single-precision kernels, vectorised)
// ---------------------------------------------------------------------------

const EXP_HI: f32 = 88.376_26;
const EXP_LO: f32 = -88.376_26;
/// `log2(e)`, for the `exp` range reduction.
const LOG2EF: f32 = std::f32::consts::LOG2_E;
// ln(2) split high/low so the range reduction keeps full precision.
const EXP_C1: f32 = 0.693_359_4;
const EXP_C2: f32 = -2.121_944_4e-4;

/// `exp(x)` for eight lanes. Cephes `expf`: reduce `x` to `r + n*ln2` with
/// `|r| <= ln2/2`, evaluate a degree-5 polynomial on `r`, then scale by `2^n`
/// by writing the exponent field directly.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp8(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x = _mm256_min_ps(_mm256_set1_ps(EXP_HI), x);
    let x = _mm256_max_ps(_mm256_set1_ps(EXP_LO), x);

    // n = round(x / ln2)
    let fx = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(LOG2EF),
    ));
    // r = x - n*ln2, in two pieces.
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXP_C1), x);
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXP_C2), r);

    let z = _mm256_mul_ps(r, r);
    let mut y = _mm256_set1_ps(1.987_569_1e-4);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_199_9e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_452e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_6e-2));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_5e-1));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.000_000_6e-1));
    y = _mm256_fmadd_ps(y, z, _mm256_add_ps(r, one));

    // 2^n by constructing the exponent field.
    let n = _mm256_cvttps_epi32(fx);
    let n = _mm256_add_epi32(n, _mm256_set1_epi32(0x7f));
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(n));
    _mm256_mul_ps(y, pow2n)
}

const SQRTHF: f32 = 0.707_106_77;

/// `ln(x)` for eight lanes, for strictly positive `x`. Cephes `logf`: split off
/// the binary exponent, fold the mantissa into `[sqrt(1/2), sqrt(2))`, then a
/// degree-8 polynomial in `m - 1`.
///
/// Callers here only ever pass a `logsumexp` denominator (`>= 1`) or a
/// floored probability (`>= 1e-8`), so the `x <= 0` branch of Cephes is
/// dropped; the minimum-normal clamp is kept as a cheap guard.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn ln8(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x = _mm256_max_ps(x, _mm256_set1_ps(f32::MIN_POSITIVE));

    // e = exponent - 127, m = mantissa in [0.5, 1)
    let e = _mm256_srli_epi32::<23>(_mm256_castps_si256(x));
    let e = _mm256_sub_epi32(e, _mm256_set1_epi32(0x7f));
    let mut e = _mm256_cvtepi32_ps(e);
    let m = _mm256_and_ps(
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(!0x7f80_0000u32 as i32)),
    );
    let m = _mm256_or_ps(m, _mm256_set1_ps(0.5));
    e = _mm256_add_ps(e, one);

    // Fold [0.5, 1) into [sqrt(1/2), sqrt(2)) around 1.
    let mask = _mm256_cmp_ps::<_CMP_LT_OS>(m, _mm256_set1_ps(SQRTHF));
    let extra = _mm256_and_ps(m, mask);
    let m = _mm256_sub_ps(m, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    let m = _mm256_add_ps(m, extra);

    let z = _mm256_mul_ps(m, m);
    let mut y = _mm256_set1_ps(7.037_683_6e-2);
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-1.151_461e-1));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(1.167_699_9e-1));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-1.242_014_1e-1));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(1.424_932_3e-1));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-1.666_805_8e-1));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(2.000_071_5e-1));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-2.499_999_4e-1));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(3.333_333_3e-1));
    y = _mm256_mul_ps(_mm256_mul_ps(y, m), z);

    y = _mm256_fmadd_ps(e, _mm256_set1_ps(EXP_C2), y);
    y = _mm256_fnmadd_ps(_mm256_set1_ps(0.5), z, y);
    let out = _mm256_add_ps(m, y);
    _mm256_fmadd_ps(e, _mm256_set1_ps(EXP_C1), out)
}

// ---------------------------------------------------------------------------
// Semiring reductions over five lane-aligned rows
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn max5(v: [__m256; 5]) -> __m256 {
    let a = _mm256_max_ps(v[0], v[1]);
    let b = _mm256_max_ps(v[2], v[3]);
    _mm256_max_ps(_mm256_max_ps(a, b), v[4])
}

/// `logsumexp` over five lanes-aligned vectors, with the same max-shift the
/// scalar path (and `torch.logsumexp`) uses.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn lse5(v: [__m256; 5]) -> __m256 {
    unsafe {
        let m = max5(v);
        let mut s = exp8(_mm256_sub_ps(v[0], m));
        s = _mm256_add_ps(s, exp8(_mm256_sub_ps(v[1], m)));
        s = _mm256_add_ps(s, exp8(_mm256_sub_ps(v[2], m)));
        s = _mm256_add_ps(s, exp8(_mm256_sub_ps(v[3], m)));
        s = _mm256_add_ps(s, exp8(_mm256_sub_ps(v[4], m)));
        _mm256_add_ps(m, ln8(s))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn reduce5(v: [__m256; 5], s: Semiring) -> __m256 {
    unsafe {
        match s {
            Semiring::Max => max5(v),
            Semiring::Log => lse5(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Reshapes
// ---------------------------------------------------------------------------

/// `dst[j * n_states + c] = src[j * group + c / 4]` — replicate each of the
/// `group` source values four times, once per dropped-base group.
///
/// This is the forward pass's `alpha[source(c, 1 + j)]` gather, precomputed so
/// the recurrence itself stays unit-stride.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn expand(src: &[f32], n_states: usize, group: usize, dst: &mut [f32]) {
    unsafe {
        debug_assert_eq!(dst.len(), 4 * n_states);
        for j in 0..4 {
            let row = &src[j * group..j * group + group];
            let out = &mut dst[j * n_states..(j + 1) * n_states];
            // Each 8 inputs become 32 outputs: lanes [0,0,0,0,1,1,1,1] and so on.
            let (src_chunks, _) = row.as_chunks::<8>();
            let (dst_chunks, _) = out.as_chunks_mut::<32>();
            for (chunk, o) in src_chunks.iter().zip(dst_chunks) {
                let v = _mm256_loadu_ps(chunk.as_ptr());
                let q0 = _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));
                let q1 = _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3));
                let q2 = _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5));
                let q3 = _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7));
                _mm256_storeu_ps(o.as_mut_ptr(), q0);
                _mm256_storeu_ps(o.as_mut_ptr().add(8), q1);
                _mm256_storeu_ps(o.as_mut_ptr().add(16), q2);
                _mm256_storeu_ps(o.as_mut_ptr().add(24), q3);
            }
        }
    }
}

/// `dst[k * (n / 4) + u] = src[4 * u + k]` — a stride-4 de-interleave.
///
/// The backward pass folds four consecutive destination states onto one source
/// state, so this turns that 4:1 fold into four unit-stride rows that the same
/// [`reduce5`] kernel can consume.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn deinterleave4(src: &[f32], dst: &mut [f32]) {
    unsafe {
        let n = src.len();
        let quarter = n / 4;
        debug_assert_eq!(n % 32, 0);
        debug_assert_eq!(dst.len(), n);
        // Lane picks that turn [a0,a1,b0,b1 | a4,a5,b4,b5] into [a0,a4,b0,b4 | …].
        let even = _mm256_setr_epi32(0, 4, 2, 6, 0, 4, 2, 6);
        let odd = _mm256_setr_epi32(1, 5, 3, 7, 1, 5, 3, 7);
        for (blk, u0) in (0..n).step_by(32).zip((0..quarter).step_by(8)) {
            let a = _mm256_loadu_ps(src.as_ptr().add(blk));
            let b = _mm256_loadu_ps(src.as_ptr().add(blk + 8));
            let c = _mm256_loadu_ps(src.as_ptr().add(blk + 16));
            let d = _mm256_loadu_ps(src.as_ptr().add(blk + 24));

            // Pair up so each 128-bit lane holds two elements from each source.
            let ab0 = _mm256_shuffle_ps::<0b01_00_01_00>(a, b);
            let ab1 = _mm256_shuffle_ps::<0b11_10_11_10>(a, b);
            let cd0 = _mm256_shuffle_ps::<0b01_00_01_00>(c, d);
            let cd1 = _mm256_shuffle_ps::<0b11_10_11_10>(c, d);

            let k0 = _mm256_permute2f128_ps::<0x20>(
                _mm256_permutevar8x32_ps(ab0, even),
                _mm256_permutevar8x32_ps(cd0, even),
            );
            let k1 = _mm256_permute2f128_ps::<0x20>(
                _mm256_permutevar8x32_ps(ab0, odd),
                _mm256_permutevar8x32_ps(cd0, odd),
            );
            let k2 = _mm256_permute2f128_ps::<0x20>(
                _mm256_permutevar8x32_ps(ab1, even),
                _mm256_permutevar8x32_ps(cd1, even),
            );
            let k3 = _mm256_permute2f128_ps::<0x20>(
                _mm256_permutevar8x32_ps(ab1, odd),
                _mm256_permutevar8x32_ps(cd1, odd),
            );
            _mm256_storeu_ps(dst.as_mut_ptr().add(u0), k0);
            _mm256_storeu_ps(dst.as_mut_ptr().add(quarter + u0), k1);
            _mm256_storeu_ps(dst.as_mut_ptr().add(2 * quarter + u0), k2);
            _mm256_storeu_ps(dst.as_mut_ptr().add(3 * quarter + u0), k3);
        }
    }
}

/// `dst[edge * n_states + dest] = src[dest * n_edges + edge]` — the
/// `[dest][edge]` → `[edge][dest]` transpose of one timestep.
///
/// See [`super::avx512::transpose_scores`]: `n_edges` is 5, which no shuffle
/// network de-interleaves cleanly, so this gathers one edge row at a time.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn transpose_scores(
    src: &[f32],
    dst: &mut [f32],
    n_states: usize,
    n_edges: usize,
) {
    unsafe {
        let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let step = _mm256_mullo_epi32(lane, _mm256_set1_epi32(n_edges as i32));
        let bump = _mm256_set1_epi32((8 * n_edges) as i32);
        for edge in 0..n_edges {
            let base = src.as_ptr().add(edge);
            let out = dst.as_mut_ptr().add(edge * n_states);
            let mut idx = step;
            for d in (0..n_states).step_by(8) {
                _mm256_storeu_ps(out.add(d), _mm256_i32gather_ps::<4>(base, idx));
                idx = _mm256_add_epi32(idx, bump);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lattice passes
// ---------------------------------------------------------------------------

/// Forward scores; see [`super::lattice`] for the recurrence.
///
/// `work` is `4 * n_states` of scratch for the expanded alpha.
#[target_feature(enable = "avx2,fma")]
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
            for c in (0..n_states).step_by(8) {
                let v = [
                    _mm256_add_ps(
                        _mm256_loadu_ps(prev.as_ptr().add(c)),
                        _mm256_loadu_ps(row.as_ptr().add(c)),
                    ),
                    edge_term(work, row, 0, c, n_states),
                    edge_term(work, row, 1, c, n_states),
                    edge_term(work, row, 2, c, n_states),
                    edge_term(work, row, 3, c, n_states),
                ];
                _mm256_storeu_ps(next.as_mut_ptr().add(c), reduce5(v, s));
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn edge_term(work: &[f32], row: &[f32], j: usize, c: usize, n_states: usize) -> __m256 {
    unsafe {
        _mm256_add_ps(
            _mm256_loadu_ps(work.as_ptr().add(j * n_states + c)),
            _mm256_loadu_ps(row.as_ptr().add((j + 1) * n_states + c)),
        )
    }
}

/// Backward scores; see [`super::lattice`] for the recurrence.
///
/// `work` is `4 * n_states` of scratch: `n_states` for the summed move row and
/// `n_states` for its de-interleave, per dropped-base group.
#[target_feature(enable = "avx2,fma")]
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
                // v[c] = move_score[c] + beta[t+1][c], contiguous over destinations…
                let move_row = &row[(j + 1) * n_states..(j + 2) * n_states];
                for c in (0..n_states).step_by(8) {
                    _mm256_storeu_ps(
                        summed.as_mut_ptr().add(c),
                        _mm256_add_ps(
                            _mm256_loadu_ps(move_row.as_ptr().add(c)),
                            _mm256_loadu_ps(next.as_ptr().add(c)),
                        ),
                    );
                }
                // …then fold every four destinations onto their shared source.
                deinterleave4(summed, folded);
                let out = &mut cur[j * group..(j + 1) * group];
                for u in (0..group).step_by(8) {
                    let stay = _mm256_add_ps(
                        _mm256_loadu_ps(row.as_ptr().add(j * group + u)),
                        _mm256_loadu_ps(next.as_ptr().add(j * group + u)),
                    );
                    let v = [
                        stay,
                        _mm256_loadu_ps(folded.as_ptr().add(u)),
                        _mm256_loadu_ps(folded.as_ptr().add(group + u)),
                        _mm256_loadu_ps(folded.as_ptr().add(2 * group + u)),
                        _mm256_loadu_ps(folded.as_ptr().add(3 * group + u)),
                    ];
                    _mm256_storeu_ps(out.as_mut_ptr().add(u), reduce5(v, s));
                }
            }
        }
    }
}

/// Per-timestep edge scores `alpha[t][source] + score + beta[t + 1][dest]`,
/// written in `[edge][dest]` order.
#[target_feature(enable = "avx2,fma")]
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
        for c in (0..n_states).step_by(8) {
            let b = _mm256_loadu_ps(beta_next.as_ptr().add(c));
            let stay = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_loadu_ps(alpha_t.as_ptr().add(c)),
                    _mm256_loadu_ps(row.as_ptr().add(c)),
                ),
                b,
            );
            _mm256_storeu_ps(out.as_mut_ptr().add(c), stay);
            for j in 0..4 {
                let v = _mm256_add_ps(edge_term(work, row, j, c, n_states), b);
                _mm256_storeu_ps(out.as_mut_ptr().add((j + 1) * n_states + c), v);
            }
        }
    }
}

/// `log(softmax(src) + 1e-8)` over a whole timestep's edge scores.
///
/// Pure unit-stride, and the single largest block of transcendentals in the
/// decode: one `exp` and one `ln` per edge per timestep.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn log_softmax_floored(src: &[f32], dst: &mut [f32], floor: f32) {
    unsafe {
        let n = src.len();
        debug_assert_eq!(n % 8, 0);

        let mut mv = _mm256_set1_ps(f32::NEG_INFINITY);
        for c in (0..n).step_by(8) {
            mv = _mm256_max_ps(mv, _mm256_loadu_ps(src.as_ptr().add(c)));
        }
        let m = _mm256_set1_ps(hmax(mv));

        // Keep exp(x - max) in `dst` and divide by its own sum, rather than
        // taking a second exp against the logsumexp. Same softmax — and the
        // form torch itself uses — for one exp per element instead of two, in
        // the decode's hottest kernel.
        let mut sv = _mm256_setzero_ps();
        for c in (0..n).step_by(8) {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(src.as_ptr().add(c)), m));
            _mm256_storeu_ps(dst.as_mut_ptr().add(c), e);
            sv = _mm256_add_ps(sv, e);
        }
        // The vector loop accumulates in eight independent lanes, so this sum is
        // reassociated relative to the scalar path's sequential one — a ~1 ulp
        // difference, not a bug, and the reason the SIMD contract is "same decode"
        // rather than "same bits".
        let inv = _mm256_set1_ps(1.0 / hsum(sv));

        let fl = _mm256_set1_ps(floor);
        for c in (0..n).step_by(8) {
            let e = _mm256_loadu_ps(dst.as_ptr().add(c));
            _mm256_storeu_ps(dst.as_mut_ptr().add(c), ln8(_mm256_fmadd_ps(e, inv, fl)));
        }
    }
}

/// One timestep of the constrained-chain scan's fan-in-1 cells
/// ([`super::refchain`]), eight at a time. Returns how many were done, leaving
/// the remainder to the caller's scalar loop.
///
/// `cur[base + i]` and `next[base + i]` are unit-stride because
/// `RefChains::partition` made that class of cell contiguous — which is what
/// makes this vectorisable on AVX2 at all, since there is no scatter to write
/// results back to scattered cells with. The three gathers that remain are the
/// stay score, the move's source cell, and the move score.
///
/// `log(exp(a) + exp(b))` as `max + ln(1 + exp(-|a - b|))`: one `exp` and one
/// `ln` per cell, against the general `logsumexp`'s two.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn chain_tail(
    cur: &[f32],
    next: &mut [f32],
    row: &[f32],
    stay: &[u32],
    src: &[u32],
    mv: &[u32],
    base: usize,
) -> usize {
    unsafe {
        let n = stay.len();
        let (zero, one) = (_mm256_setzero_ps(), _mm256_set1_ps(1.0));
        let mut i = 0;
        while i + 8 <= n {
            let idx = |p: *const u32| _mm256_loadu_si256(p.add(i) as *const __m256i);
            let a = _mm256_add_ps(
                _mm256_loadu_ps(cur.as_ptr().add(base + i)),
                _mm256_i32gather_ps::<4>(row.as_ptr(), idx(stay.as_ptr())),
            );
            let b = _mm256_add_ps(
                _mm256_i32gather_ps::<4>(cur.as_ptr(), idx(src.as_ptr())),
                _mm256_i32gather_ps::<4>(row.as_ptr(), idx(mv.as_ptr())),
            );

            let m = _mm256_max_ps(a, b);
            // A cell no path has reached yet has `-inf` on both terms, so their
            // difference is `NaN`. `min(_, 0)` turns that into 0 — `_mm256_min_ps`
            // returns its second operand for an unordered compare — and the
            // result is `-inf + ln(2)`, still `-inf`. For a reachable cell the
            // difference is already `<= 0`, so this is a no-op.
            //
            // Deleting it does not change any output today, and no test catches
            // it: `ln8`'s `_mm256_max_ps(x, MIN_POSITIVE)` clamp scrubs the same
            // `NaN` by the same unordered rule. That is an argument-order
            // accident two functions away, and it would reverse silently — every
            // scored read `NaN` — if `ln8` ever clamped the other way round. One
            // `vminps` per eight lanes to not depend on it.
            let d = _mm256_min_ps(_mm256_sub_ps(_mm256_min_ps(a, b), m), zero);
            let r = _mm256_add_ps(m, ln8(_mm256_add_ps(one, exp8(d))));
            _mm256_storeu_ps(next.as_mut_ptr().add(base + i), r);
            i += 8;
        }
        i
    }
}

/// One timestep of the constrained-chain scan's fan-in-4 cells — the
/// unresolved-prefix head ([`super::refchain`]) — eight at a time. Returns how
/// many were done.
///
/// Five terms per cell against [`chain_tail`]'s two, which is why the head is a
/// third of the scan's transcendental work despite being a tenth of its cells.
/// `src` and `score` are `[edge][cell]`, so each edge's indices are a
/// unit-stride load and only their values need gathering; the reduction is the
/// same [`lse5`] the decode's own forward pass uses.
// Three index arrays, three buffers, and where in each the head starts. A
// struct would name the same things one indirection away from the intrinsics.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn chain_head(
    cur: &[f32],
    next: &mut [f32],
    row: &[f32],
    stay: &[u32],
    src: &[u32],
    score: &[u32],
    base: usize,
    n: usize,
) -> usize {
    unsafe {
        debug_assert_eq!(src.len(), 4 * n);
        let mut i = 0;
        while i + 8 <= n {
            let idx = |p: *const u32, at: usize| _mm256_loadu_si256(p.add(at) as *const __m256i);
            let mut v = [_mm256_setzero_ps(); 5];
            v[0] = _mm256_add_ps(
                _mm256_loadu_ps(cur.as_ptr().add(base + i)),
                _mm256_i32gather_ps::<4>(row.as_ptr(), idx(stay.as_ptr(), i)),
            );
            for d in 0..4 {
                v[1 + d] = _mm256_add_ps(
                    _mm256_i32gather_ps::<4>(cur.as_ptr(), idx(src.as_ptr(), d * n + i)),
                    _mm256_i32gather_ps::<4>(row.as_ptr(), idx(score.as_ptr(), d * n + i)),
                );
            }
            _mm256_storeu_ps(next.as_mut_ptr().add(base + i), lse5(v));
            i += 8;
        }
        i
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hmax(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let m = _mm_max_ps(lo, hi);
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    let m = _mm_max_ss(m, _mm_shuffle_ps::<0b01>(m, m));
    _mm_cvtss_f32(m)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
    _mm_cvtss_f32(s)
}

/// The `(dest, edge)` with the highest score, ties broken toward the lowest
/// `dest * n_edges + edge` — what `torch.argmax` over that axis returns.
///
/// Two passes, both unit-stride: a vector max over everything, then a
/// compare-and-`movemask` scan that stops at the first hit in each edge row.
/// The obvious scalar form instead walks `dest` outermost, which strides the
/// `[edge][dest]` layout by `n_states` and mispredicts on every new maximum;
/// it measured as the bulk of an 18% profile entry.
///
/// Tie-breaking is exact rather than lane-order dependent: for a fixed edge the
/// flat index increases with `dest`, so the first match in a row is that row's
/// best candidate, and the minimum over rows is the global first.
///
/// Returns `None` if nothing compares equal to the maximum — unreachable for
/// finite input, but a NaN row would otherwise fall through, so the caller
/// keeps the scalar path as a fallback.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn argmax_edge(
    edges: &[f32],
    n_states: usize,
    n_edges: usize,
) -> Option<(usize, usize)> {
    unsafe {
        let mut mv = _mm256_set1_ps(f32::NEG_INFINITY);
        for c in (0..edges.len()).step_by(8) {
            mv = _mm256_max_ps(mv, _mm256_loadu_ps(edges.as_ptr().add(c)));
        }
        let peak = _mm256_set1_ps(hmax(mv));

        let mut best: Option<(usize, usize)> = None;
        let mut best_flat = usize::MAX;
        for edge in 0..n_edges {
            let row = edges.as_ptr().add(edge * n_states);
            for c in (0..n_states).step_by(8) {
                let hits = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(
                    _mm256_loadu_ps(row.add(c)),
                    peak,
                ));
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

    fn eval8(f: unsafe fn(__m256) -> __m256, xs: [f32; 8]) -> [f32; 8] {
        unsafe {
            let mut out = [0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), f(_mm256_loadu_ps(xs.as_ptr())));
            out
        }
    }

    /// Accuracy is only claimed over the normal range. Below about -87 the true
    /// result is subnormal, where Cephes' `2^n` exponent-field construction
    /// stops being exact — irrelevant here because every `exp` in the decode
    /// feeds a sum whose largest term is `exp(0) = 1`.
    #[test]
    fn exp8_matches_std_over_the_normal_range() {
        if !available() {
            return;
        }
        let mut worst = 0f32;
        for base in -870..880 {
            let xs: [f32; 8] = std::array::from_fn(|i| (base as f32 + i as f32 / 8.0) / 10.0);
            let got = eval8(exp8, xs);
            for (x, g) in xs.iter().zip(got) {
                let want = x.exp();
                if want < f32::MIN_POSITIVE {
                    continue;
                }
                worst = worst.max(((g - want) / want).abs());
            }
        }
        assert!(worst < 2e-6, "worst relative error {worst:e}");
    }

    /// Inputs far below the clamp must still collapse to zero rather than
    /// wrapping around through the exponent-field trick.
    /// A chain cell no path has reached yet has `-inf` on both terms, so their
    /// difference is `NaN`. The kernel must still produce `-inf`.
    ///
    /// Pins the behaviour, not the line that implements it: two mechanisms
    /// currently deliver it (see the `min(_, 0)` in `chain_tail`), so removing
    /// either one alone leaves this passing. It is here because the scan cannot
    /// show the difference at all — an escaped `NaN` is added to `-inf` and
    /// disappears — so without it nothing in the suite touches the case.
    #[test]
    fn chain_tail_absorbs_unreached() {
        if !available() {
            return;
        }
        let cur = vec![f32::NEG_INFINITY; 8];
        let mut next = vec![0.0f32; 8];
        let row = vec![0.0f32; 8];
        let (stay, src, mv) = ([0u32; 8], [0u32; 8], [0u32; 8]);
        // SAFETY: `available()` checked; every index is 0 and in range.
        let done = unsafe { chain_tail(&cur, &mut next, &row, &stay, &src, &mv, 0) };
        assert_eq!(done, 8);
        for (i, v) in next.iter().enumerate() {
            assert!(v.is_infinite() && v.is_sign_negative(), "lane {i} is {v}");
        }
    }

    /// The ordinary case, against the scalar `logaddexp` the scan falls back to.
    #[test]
    fn chain_tail_matches_scalar_logaddexp() {
        if !available() {
            return;
        }
        let cur: Vec<f32> = (0..16).map(|i| -0.5 * i as f32).collect();
        let row: Vec<f32> = (0..16).map(|i| 0.25 * i as f32 - 2.0).collect();
        let stay: [u32; 8] = [0, 3, 5, 7, 9, 11, 13, 15];
        let src: [u32; 8] = [1, 2, 4, 6, 8, 10, 12, 14];
        let mv: [u32; 8] = [15, 12, 10, 8, 6, 4, 2, 0];
        let mut next = vec![0.0f32; 16];
        // SAFETY: `available()` checked; indices are all below 16.
        unsafe { chain_tail(&cur, &mut next, &row, &stay, &src, &mv, 8) };
        for i in 0..8 {
            let a = cur[8 + i] + row[stay[i] as usize];
            let b = cur[src[i] as usize] + row[mv[i] as usize];
            let m = a.max(b);
            let want = m + (-(a - b).abs()).exp().ln_1p();
            assert!(
                (next[8 + i] - want).abs() < 1e-5,
                "lane {i}: got {} want {want}",
                next[8 + i]
            );
        }
    }

    #[test]
    fn exp8_underflows_to_zero() {
        if !available() {
            return;
        }
        let xs = [-100.0, -1e3, -1e6, -1e30, -f32::MAX, -200.0, -90.0, -88.5];
        for g in eval8(exp8, xs) {
            assert!((0.0..1e-37).contains(&g), "expected ~0, got {g:e}");
        }
    }

    #[test]
    fn ln8_matches_std() {
        if !available() {
            return;
        }
        let mut worst = 0f32;
        for base in 1..2000 {
            let xs: [f32; 8] =
                std::array::from_fn(|i| (base as f32 + i as f32 / 8.0) * 1e-4 + 1e-9);
            let got = eval8(ln8, xs);
            for (x, g) in xs.iter().zip(got) {
                let want = x.ln();
                worst = worst.max((g - want).abs() / want.abs().max(1.0));
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

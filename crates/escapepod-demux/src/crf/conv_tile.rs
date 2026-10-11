// SPDX-License-Identifier: MIT

//! Register-tiled accumulation for the native CRF encoder's convolutions
//! (`encoder_native::conv1d_silu`).
//!
//! The reference loop is one axpy per `(oc, ic, kk)`: load the output row,
//! load a window of the (polyphase) input, `mul`, `add`, store the row back.
//! Stage 3 (`16 -> 96`, `k = 19`, stride 10, 300 outputs) is 29 k of those
//! axpys over 1.2 KB rows, so it is bound by loads and stores, not by the
//! arithmetic. Here a tile of `OCB` output channels by `OTB` vectors of output
//! positions lives in registers for the whole `(ic, kk)` reduction: each
//! window load serves `OCB` channels and each weight broadcast serves `OTB`
//! vectors, and nothing is stored until the tile is done.
//!
//! **Bit-identical to the reference by construction.** Every output element
//! is still `bias`, then `+= w * x` for `ic` ascending and, within it, `kk`
//! ascending — tiling only changes *which elements* share a register, never
//! the order in which one element's terms are added. Each product is an
//! explicit `mul` followed by an `add` (never an FMA), exactly how the scalar
//! loop rounds. Padding lanes (positions past `out_len`) compute throw-away
//! values from zero-filled input and are never read back.
//!
//! Runtime-dispatched on the caller's [`escapepod_signal::lstm::LstmBackend`]
//! (`supported()` guards the `unsafe`), never a baseline bump; the kernels
//! are `#[inline(never)]` like every other kernel in this stack.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// The shape of one convolution's polyphase accumulation.
#[derive(Clone, Copy, Debug)]
pub(super) struct Geom {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub stride: usize,
    /// Distance between two phase rows in the `phases` buffer.
    pub phase_row: usize,
    /// Output positions per channel, rounded up to the vector width. The
    /// accumulator rows are this long; `phase_row` must cover
    /// `(k - 1) / stride + out_pad` so no tile reads past its row.
    pub out_pad: usize,
}

/// Fill `acc` (`[out_c][out_pad]`) with `bias + sum w * x`, AVX2 (8 lanes).
///
pub(super) fn conv_avx2(w: &[f32], bias: &[f32], phases: &[f32], g: Geom, acc: &mut [f32]) {
    check(w, bias, phases, g, acc, 8);
    assert!(
        is_x86_feature_detected!("avx2"),
        "AVX2 kernel on a CPU without AVX2"
    );
    // Safety: bounds and CPU feature checked above.
    unsafe { drive_avx2(w, bias, phases, g, acc) }
}

/// As [`conv_avx2`], AVX-512 (16 lanes).
pub(super) fn conv_avx512(w: &[f32], bias: &[f32], phases: &[f32], g: Geom, acc: &mut [f32]) {
    check(w, bias, phases, g, acc, 16);
    assert!(
        is_x86_feature_detected!("avx512f"),
        "AVX-512 kernel on a CPU without AVX-512F"
    );
    // Safety: bounds and CPU feature checked above.
    unsafe { drive_avx512(w, bias, phases, g, acc) }
}

/// The preconditions the pointer arithmetic below leans on.
fn check(w: &[f32], bias: &[f32], phases: &[f32], g: Geom, acc: &[f32], lanes: usize) {
    assert!(g.stride > 0 && g.k > 0 && g.out_pad > 0);
    assert_eq!(g.out_pad % lanes, 0, "out_pad is a multiple of the width");
    assert_eq!(w.len(), g.out_c * g.in_c * g.k);
    assert_eq!(bias.len(), g.out_c);
    assert_eq!(acc.len(), g.out_c * g.out_pad);
    assert!(g.phase_row >= (g.k - 1) / g.stride + g.out_pad);
    assert!(phases.len() >= g.in_c * g.stride * g.phase_row);
}

macro_rules! drive {
    ($name:ident, $tile:ident, $feat:literal, $lanes:expr, $ocb:expr, $otb:expr) => {
        #[target_feature(enable = $feat)]
        unsafe fn $name(w: &[f32], bias: &[f32], phases: &[f32], g: Geom, acc: &mut [f32]) {
            const OCB: usize = $ocb;
            const OTB: usize = $otb;
            let nv = g.out_pad / $lanes;
            let mut oc = 0;
            while oc < g.out_c {
                let ocn = if g.out_c - oc >= OCB { OCB } else { 1 };
                let mut j = 0;
                while j < nv {
                    let rem = nv - j;
                    let t0 = j * $lanes;
                    // Safety: preconditions from `check`; `oc + ocn <= out_c`
                    // and `t0 + otb * lanes <= out_pad` by construction.
                    let full = rem >= OTB;
                    unsafe {
                        if ocn == OCB && full {
                            $tile::<OCB, OTB>(w, bias, phases, g, oc, t0, acc);
                        } else if ocn == OCB {
                            $tile::<OCB, 1>(w, bias, phases, g, oc, t0, acc);
                        } else if full {
                            $tile::<1, OTB>(w, bias, phases, g, oc, t0, acc);
                        } else {
                            $tile::<1, 1>(w, bias, phases, g, oc, t0, acc);
                        }
                    }
                    j += if full { OTB } else { 1 };
                }
                oc += ocn;
            }
        }
    };
}

// Tile shapes: AVX-512 holds 4 x 3 accumulators + 3 window loads + 1
// broadcast in 32 ZMM; AVX2 holds 4 x 2 + 2 + 1 in 16 YMM.
drive!(drive_avx512, tile_avx512, "avx512f", 16, 4, 3);
drive!(drive_avx2, tile_avx2, "avx2", 8, 4, 2);

macro_rules! tile {
    ($name:ident, $feat:literal, $lanes:expr, $reg:ty,
     $zero:ident, $set1:ident, $loadu:ident, $storeu:ident, $mul:ident, $add:ident) => {
        #[target_feature(enable = $feat)]
        #[inline(never)]
        unsafe fn $name<const OCB: usize, const OTB: usize>(
            w: &[f32],
            bias: &[f32],
            phases: &[f32],
            g: Geom,
            oc0: usize,
            t0: usize,
            acc: &mut [f32],
        ) {
            let per_oc = g.in_c * g.k;
            let mut a: [[$reg; OTB]; OCB] = [[$zero(); OTB]; OCB];
            unsafe {
                for (o, row) in a.iter_mut().enumerate() {
                    let b = $set1(*bias.get_unchecked(oc0 + o));
                    *row = [b; OTB];
                }
                let wp = w.as_ptr();
                let qn = g.k.div_ceil(g.stride);
                for ic in 0..g.in_c {
                    for q in 0..qn {
                        for r in 0..g.stride {
                            let kk = q * g.stride + r;
                            if kk >= g.k {
                                break;
                            }
                            let xp = phases
                                .as_ptr()
                                .add((ic * g.stride + r) * g.phase_row + q + t0);
                            let mut x: [$reg; OTB] = [$zero(); OTB];
                            for (j, xj) in x.iter_mut().enumerate() {
                                *xj = $loadu(xp.add(j * $lanes));
                            }
                            for (o, row) in a.iter_mut().enumerate() {
                                let wv = $set1(*wp.add((oc0 + o) * per_oc + ic * g.k + kk));
                                for (aj, &xj) in row.iter_mut().zip(&x) {
                                    *aj = $add(*aj, $mul(wv, xj));
                                }
                            }
                        }
                    }
                }
                for (o, row) in a.iter().enumerate() {
                    let dst = acc.as_mut_ptr().add((oc0 + o) * g.out_pad + t0);
                    for (j, &aj) in row.iter().enumerate() {
                        $storeu(dst.add(j * $lanes), aj);
                    }
                }
            }
        }
    };
}

tile!(
    tile_avx512,
    "avx512f",
    16,
    __m512,
    _mm512_setzero_ps,
    _mm512_set1_ps,
    _mm512_loadu_ps,
    _mm512_storeu_ps,
    _mm512_mul_ps,
    _mm512_add_ps
);
tile!(
    tile_avx2,
    "avx2",
    8,
    __m256,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_mul_ps,
    _mm256_add_ps
);

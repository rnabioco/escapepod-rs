// SPDX-License-Identifier: MIT

//! Register-blocked SIMD kernel for the `dwell_idx`-outer sweep inside
//! [`dp_step_with_dwell_penalty`](super::fill::dp_step_with_dwell_penalty).
//!
//! That function's doc comment derives `lo(dwell_idx)`/`hi(dwell_idx)` — the
//! contiguous `band_pos` range a fixed `dwell_idx` updates — and explains why
//! visiting `dwell_idx` in the outer loop makes the inner `band_pos` loop a
//! plain unit-stride scan. The flat scalar form of that loop
//! ([`run_scalar_sweep`] below) is written so LLVM can autovectorize it, but
//! only 4 `f64` lanes wide (the width-limiting operation), and it
//! round-trips `cand`/`cand_tb` through memory on *every* `dwell_idx` pass:
//! one reload plus two masked stores per 4-element chunk, because from the
//! compiler's point of view a fresh `band_pos` vector loop starts from
//! scratch each time `dwell_idx` advances.
//!
//! **An 8-wide (AVX2) register-blocked kernel was tried here and measured
//! *slower* than LLVM's own autovectorization of the flat loop** — 1.10–1.12×
//! slower end to end on the `resquiggle_dwell_dp` criterion bench (rna,
//! 2026-09-09), for both benchmarked shapes. It has been removed; see
//! `benchmarks/README.md`'s dead-end entry for the numbers. The design idea
//! (hold `cand`/`cand_tb` in registers across every `dwell_idx` that fully
//! covers a fixed block of `band_pos`, via [`full_coverage_range`], with
//! [`scalar_dwell_idx_over_block`] handling the ragged boundary `dwell_idx`
//! values at each block's edge) survives only in the 16-wide AVX-512 kernel
//! below, which is the one width that *does* beat the compiler's own 4-wide
//! code — 1.08–1.30× faster, same bench. AVX-512 is kept for that reason
//! alone; everything else (aarch64, and every AVX2-only x86_64 machine —
//! Alpine's Zen3 `amilan` nodes, the release artifact's Haswell baseline)
//! takes the flat scalar path and gets LLVM's autovectorization instead of a
//! hand-written kernel.
//!
//! **Bit-identity.** The vector body computes the exact same expression in
//! the exact same association as the scalar one:
//! `(previous_scores[..] + running_pos_score) + pen`, where
//! `running_pos_score = (cum[band_pos + 1] - cum[band_pos - dwell_idx]) as
//! f32` — an `f64` subtraction (`_mm512_sub_pd`), then a round-to-nearest-even
//! narrowing cast (`_mm512_cvtpd_ps`, which rounds the same way as Rust's
//! `as f32`). There is no multiply anywhere in the body, so there is no
//! FMA-contraction risk to guard against, and the kernel below does not
//! enable `fma`. The comparison uses `_CMP_LT_OQ` (ordered, quiet), matching
//! `<`'s behavior on `NaN`. `dwell_idx` is visited in ascending order within
//! a block (the vector loop counts up; the leading ragged phase runs
//! *before* the vector phase and the trailing ragged phase *after*), so the
//! strict-`<` first-writer-wins tie-break is unchanged from the scalar
//! sweep.
//!
//! **Safety shape.** The kernel is a small, `#[inline(never)]` leaf that
//! only ever runs on a block whose full extent was already proven in-bounds
//! by [`full_coverage_range`]'s caller; see its `# Safety` section.
//! `#[inline(never)]` is deliberate, not stylistic: `crates/escapepod-signal/
//! src/lstm.rs` (`run_avx2`'s doc comment) records an LLVM miscompile where an
//! AVX2 kernel inlined beside a sibling returned wrong values for part of its
//! input, cleared only by preventing the inlining. The kernel here gets the
//! same treatment on the same evidence.

use super::fill::DWELL_TABLE_SIZE;

/// Which kernel runs the `dwell_idx`-outer sweep. Ordered slowest to
/// fastest, so a cap from `ESCAPEPOD_DP_BACKEND` is a comparison — mirrors
/// `escapepod_signal::lstm::LstmBackend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum DpBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// `ESCAPEPOD_DP_BACKEND=scalar|avx512` caps the kernel the dispatch may
/// pick. Never raises: a cap the machine cannot run is simply not reached.
/// `avx2` is accepted but ignored (with a warning): there is no AVX2 kernel
/// any more — see the module doc comment's dead-end note.
pub(super) fn backend_cap() -> Option<DpBackend> {
    static CAP: std::sync::OnceLock<Option<DpBackend>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let v = std::env::var("ESCAPEPOD_DP_BACKEND").ok()?;
        match v.as_str() {
            "scalar" => Some(DpBackend::Scalar),
            #[cfg(target_arch = "x86_64")]
            "avx512" => Some(DpBackend::Avx512),
            "avx2" => {
                tracing::warn!(
                    "ESCAPEPOD_DP_BACKEND=avx2: no AVX2 kernel exists any more (measured \
                     1.10-1.12x slower than LLVM's own autovectorization of the flat scalar \
                     loop, see benchmarks/README.md); ignoring cap"
                );
                None
            }
            other => {
                tracing::warn!("ESCAPEPOD_DP_BACKEND={other}: not a kernel name, ignored");
                None
            }
        }
    })
}

impl DpBackend {
    /// Every kernel, slowest to fastest.
    #[cfg(target_arch = "x86_64")]
    const ALL: [DpBackend; 2] = [DpBackend::Scalar, DpBackend::Avx512];
    #[cfg(not(target_arch = "x86_64"))]
    const ALL: [DpBackend; 1] = [DpBackend::Scalar];

    /// The block width this backend tiles `band_pos` into (`1` for the flat
    /// scalar sweep, which has no blocking).
    pub(super) fn lanes(self) -> usize {
        match self {
            DpBackend::Scalar => 1,
            #[cfg(target_arch = "x86_64")]
            DpBackend::Avx512 => 16,
        }
    }

    /// Whether this machine can run the kernel. The one place the `unsafe`
    /// call into the `#[target_feature]` kernel rests on: every constructor
    /// that picks a backend consults it. The vector kernel uses no `fma`
    /// (there is no multiply in the body) and is kept to `avx512f` alone —
    /// no `avx512dq` — matching `escapepod_demux::crf::avx512`'s policy of
    /// staying usable on every AVX-512 part, not only the ones with DQ.
    pub(super) fn supported(self) -> bool {
        match self {
            DpBackend::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            DpBackend::Avx512 => is_x86_feature_detected!("avx512f"),
        }
    }

    /// Every kernel this machine can run, slowest to fastest — what an
    /// equivalence test sweeps, whichever one the dispatch prefers. Test-only:
    /// production always goes through [`DpBackend::best_for`].
    #[cfg(test)]
    pub(super) fn available() -> Vec<DpBackend> {
        Self::ALL.into_iter().filter(|b| b.supported()).collect()
    }

    /// The fastest kernel this machine can run, capped by
    /// `ESCAPEPOD_DP_BACKEND` if set. AVX-512 is runtime-detected, never a
    /// baseline bump: the release artifact is built for `x86-64-v3`
    /// (Haswell-equivalent), so a Broadwell login node, Alpine's Zen3
    /// `amilan` nodes, and the release artifact itself all fall through to
    /// the flat scalar sweep (LLVM's own autovectorization of it), and only
    /// a machine with `avx512f` ever reaches the AVX-512 kernel.
    pub(super) fn best_for() -> DpBackend {
        let cap = backend_cap();
        Self::ALL
            .into_iter()
            .rev()
            .find(|&b| cap.is_none_or(|c| b <= c) && b.supported())
            .unwrap_or(DpBackend::Scalar)
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            DpBackend::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            DpBackend::Avx512 => "avx512",
        }
    }
}

/// Run the whole `dwell_idx in 0..max_check` sweep, updating `cand`/`cand_tb`
/// exactly as the scalar `dwell_idx`-outer loop would, dispatched to
/// `backend`.
///
/// `cum` is `len + 1` elements (the prefix sum, `cum[0] = 0`); `cand`/
/// `cand_tb` are `len` elements, already filled with the "no valid
/// transition" sentinel by the caller.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_dwell_sweep(
    cum: &[f64],
    previous_scores: &[f32],
    penalty_table: &[f32],
    cand: &mut [f32],
    cand_tb: &mut [i32],
    len: usize,
    max_check: usize,
    prev_band_offset: usize,
    backend: DpBackend,
) {
    debug_assert!(max_check <= DWELL_TABLE_SIZE);
    let zero_offset_shift = if prev_band_offset == 0 { 1 } else { 0 };
    // Called on every arch regardless of which arm below is taken, so
    // neither `lanes()` nor `name()` is ever unreachable — see the module
    // doc comment's aarch64 note and rnabioco/escapepod-rs#329/#333.
    let lanes = backend.lanes();
    tracing::trace!("dp dwell sweep backend: {} ({lanes} lanes)", backend.name());
    match backend {
        DpBackend::Scalar => {
            debug_assert_eq!(lanes, 1);
            run_scalar_sweep(
                cum,
                previous_scores,
                penalty_table,
                cand,
                cand_tb,
                len,
                max_check,
                prev_band_offset,
                zero_offset_shift,
            );
        }
        #[cfg(target_arch = "x86_64")]
        DpBackend::Avx512 => {
            debug_assert_eq!(lanes, 16);
            // Safety: `backend` is only ever `Avx512` after `DpBackend::
            // best_for`/`available` confirmed `supported()` on this machine.
            unsafe {
                sweep_avx512(
                    cum,
                    previous_scores,
                    penalty_table,
                    cand,
                    cand_tb,
                    len,
                    max_check,
                    prev_band_offset,
                    zero_offset_shift,
                );
            }
        }
    }
}

/// Scalar `dwell_idx`-outer sweep: the same loop
/// [`super::fill::dp_step_with_dwell_penalty`] used before this module
/// existed, restored to its exact flat shape after a first cut through this
/// function (delegating to [`scalar_dwell_idx_over_block`] with the whole
/// array as "the block") measured a ~4.7-5.1x regression against it —
/// see the module doc comment's dead-end note and `benchmarks/README.md`.
///
/// The regression's mechanism: [`scalar_dwell_idx_over_block`] takes
/// `block_start`/`block_end` as ordinary `usize` parameters and computes
/// `hi = (.. ).min(block_end)`. With `block_end` opaque to this call site,
/// LLVM could no longer prove `band_pos + 1 <= cum.len()` (needed for the
/// `cum[band_pos + 1]` read below), so a bounds check — and its panic
/// branch — stayed inside the loop and blocked autovectorization entirely.
/// This function instead computes `hi` against `len` directly and
/// pre-narrows `cand`/`cand_tb`/`cum` to their true, locally-known lengths
/// (`len` and `len + 1`), which is what lets LLVM prove `band_pos < len` and
/// `band_pos + 1 <= len + 1` from the loop bounds alone and drop the check.
/// Confirmed by disassembly of the bench binary: the loop below compiles to
/// `vsubpd` → `vcvtpd2ps` → `vaddps`/`vaddps` → `vcmpltps` → `vmaskmovps`,
/// 4 `f64` lanes wide, no `panic_bounds_check` inside the loop.
///
/// Still the exact-equality baseline the property tests in `fill.rs`
/// compare against (same operations, same association, same visitation
/// order as [`scalar_dwell_idx_over_block`] — just without the opaque block
/// parameters that defeated the compiler), and also the production `Scalar`
/// backend.
#[allow(clippy::too_many_arguments)]
fn run_scalar_sweep(
    cum: &[f64],
    previous_scores: &[f32],
    penalty_table: &[f32],
    cand: &mut [f32],
    cand_tb: &mut [i32],
    len: usize,
    max_check: usize,
    prev_band_offset: usize,
    zero_offset_shift: usize,
) {
    let prev_len = previous_scores.len();
    // Pre-narrow to the exact, locally-known length so LLVM can prove every
    // `cand`/`cand_tb`/`cum` access below in-bounds from `hi <= len` alone.
    let cum = &cum[..=len];
    let cand = &mut cand[..len];
    let cand_tb = &mut cand_tb[..len];

    for dwell_idx in 0..max_check {
        let lo = dwell_idx + zero_offset_shift;
        // `lo` is strictly increasing in `dwell_idx`, so once it reaches
        // `len` no later `dwell_idx` can produce a non-empty range either.
        if lo >= len {
            break;
        }
        let hi = ((prev_len - prev_band_offset) + dwell_idx + 1).min(len);
        if hi <= lo {
            continue;
        }
        let pen = penalty_table[dwell_idx];
        for band_pos in lo..hi {
            let dwell_offset = (band_pos - dwell_idx) + prev_band_offset - 1;
            let running_pos_score = (cum[band_pos + 1] - cum[band_pos - dwell_idx]) as f32;
            let pos_score = previous_scores[dwell_offset] + running_pos_score + pen;
            if pos_score < cand[band_pos] {
                cand[band_pos] = pos_score;
                cand_tb[band_pos] = dwell_idx as i32;
            }
        }
    }
}

/// Apply one `dwell_idx`'s update to every `band_pos` in `[block_start,
/// block_end)` that the untransposed scalar sweep would have touched — i.e.
/// the real (global) `lo(dwell_idx)..hi(dwell_idx)` range intersected with
/// the block. A no-op when the intersection is empty, which is exactly what
/// happens when this `dwell_idx` doesn't reach this block at all — safe to
/// call for any `dwell_idx` in `0..max_check` against any block.
///
/// This is the scalar body [`super::fill::dp_step_with_dwell_penalty`]'s
/// `dwell_idx`-outer loop runs per `band_pos`, unchanged: same
/// `(previous_scores[..] + running_pos_score) + pen`, same strict `<`
/// update.
///
/// Only ever called from the `x86_64` AVX-512 tiling driver ([`sweep_avx512`])
/// as its ragged-boundary fallback — `#[cfg]`-gated to match, rather than
/// left as dead code on any other target (the flat [`run_scalar_sweep`]
/// above is what the `Scalar` backend uses, and it doesn't call this).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn scalar_dwell_idx_over_block(
    cum: &[f64],
    previous_scores: &[f32],
    penalty_table: &[f32],
    cand: &mut [f32],
    cand_tb: &mut [i32],
    dwell_idx: usize,
    prev_band_offset: usize,
    zero_offset_shift: usize,
    block_start: usize,
    block_end: usize,
) {
    let lo = dwell_idx + zero_offset_shift;
    if lo >= block_end {
        return;
    }
    let prev_len = previous_scores.len();
    let hi = ((prev_len - prev_band_offset) + dwell_idx + 1).min(block_end);
    let sub_lo = lo.max(block_start);
    if hi <= sub_lo {
        return;
    }
    let pen = penalty_table[dwell_idx];
    for band_pos in sub_lo..hi {
        let dwell_offset = (band_pos - dwell_idx) + prev_band_offset - 1;
        let running_pos_score = (cum[band_pos + 1] - cum[band_pos - dwell_idx]) as f32;
        let pos_score = previous_scores[dwell_offset] + running_pos_score + pen;
        if pos_score < cand[band_pos] {
            cand[band_pos] = pos_score;
            cand_tb[band_pos] = dwell_idx as i32;
        }
    }
}

/// For a `width`-wide block `[block_start, block_start + width)`, the
/// inclusive `dwell_idx` range `[full_lo, full_hi]` such that *every*
/// `dwell_idx` in it fully covers the block, i.e. `lo(dwell_idx) <=
/// block_start` and `hi(dwell_idx) >= block_start + width` hold
/// simultaneously. `None` when no such `dwell_idx` exists (e.g. `max_check <
/// width`, or the block straddles the previous band's edge sharply enough
/// that no single `dwell_idx` reaches every lane) — the caller falls back to
/// [`scalar_dwell_idx_over_block`] for the whole block in that case.
///
/// Only ever called from the `x86_64` tiling driver below (there is no
/// blocking to do without a vector width to block by) — `#[cfg]`-gated to
/// match, rather than left as dead code on any other target.
///
/// Derived from the same `lo(dwell_idx) = dwell_idx + zero_offset_shift`
/// (strictly increasing) and `hi(dwell_idx) = min(len, prev_len -
/// prev_band_offset + dwell_idx + 1)` (non-decreasing) that
/// `dp_step_with_dwell_penalty`'s doc comment derives. `lo(dwell_idx) <=
/// block_start` rearranges directly to `dwell_idx <= block_start -
/// zero_offset_shift`. For `hi`, only the *unsaturated* linear term is used
/// below — deliberately, not as an oversight: every caller only tiles blocks
/// with `block_start + width <= len`, and for such a block the saturated
/// case (`len` binding instead of the linear term) satisfies `hi(dwell_idx)
/// == len >= block_start + width` automatically, for *every* `dwell_idx`. So
/// using the linear form alone never produces a false positive on coverage
/// (it can only be *more* conservative than the true saturated `hi`, pushing
/// a `dwell_idx` that would have counted as "fully covering" into the
/// caller's ragged scalar phase instead — still correct, just occasionally
/// short of maximally vectorized).
///
/// The two boundary zones this leaves for the caller (`dwell_idx` below
/// `full_lo`, and above `full_hi`) are each at most `width - 1` values wide:
/// algebraically, `full_lo`'s raw (unclamped) form is exactly `width - 1`
/// more than the smallest `dwell_idx` with *any* overlap on the block's
/// right edge, and `full_hi`'s raw form is exactly `width` less than the
/// smallest `dwell_idx` with *no* overlap on the left edge — both because
/// `lo`/`hi` advance by exactly one per unit of `block_start`. A caller that
/// scans a `width`-wide margin on each side of `[full_lo, full_hi]` (via
/// [`scalar_dwell_idx_over_block`], itself a safe no-op outside the true
/// overlap) therefore never misses a boundary `dwell_idx`, for *every* block
/// this function is called on — not just the boundary ones; interior blocks
/// simply have an empty margin to scan, which is the "no ragged work at all"
/// case the module doc comment describes.
#[cfg(target_arch = "x86_64")]
fn full_coverage_range(
    block_start: usize,
    width: usize,
    prev_len: usize,
    prev_band_offset: usize,
    zero_offset_shift: usize,
    max_check: usize,
) -> Option<(usize, usize)> {
    // Invariant restated from `dp_step_with_dwell_penalty`: a real forward
    // pass always has `prev_band_offset <= prev_len`.
    debug_assert!(prev_band_offset <= prev_len);
    let base = prev_len as i64 - prev_band_offset as i64;
    let full_lo_raw = (block_start + width) as i64 - 1 - base;
    let full_hi_raw = block_start as i64 - zero_offset_shift as i64;
    let full_lo = full_lo_raw.clamp(0, max_check as i64);
    let full_hi = full_hi_raw.min(max_check as i64 - 1);
    (full_lo <= full_hi).then_some((full_lo as usize, full_hi as usize))
}

#[cfg(target_arch = "x86_64")]
/// Tile `[0, len)` into 16-wide blocks and sweep each with
/// [`dwell_block_kernel_avx512`], falling back to
/// [`scalar_dwell_idx_over_block`] for the boundary `dwell_idx` values each
/// block's [`full_coverage_range`] doesn't cover, for blocks with no
/// covering range at all, and for the trailing remainder shorter than 16.
///
/// # Safety
/// The caller must have verified `DpBackend::Avx512.supported()` (`avx512f`
/// present).
#[allow(clippy::too_many_arguments)]
unsafe fn sweep_avx512(
    cum: &[f64],
    previous_scores: &[f32],
    penalty_table: &[f32],
    cand: &mut [f32],
    cand_tb: &mut [i32],
    len: usize,
    max_check: usize,
    prev_band_offset: usize,
    zero_offset_shift: usize,
) {
    const WIDTH: usize = 16;
    let prev_len = previous_scores.len();
    let mut b = 0usize;
    while b + WIDTH <= len {
        match full_coverage_range(
            b,
            WIDTH,
            prev_len,
            prev_band_offset,
            zero_offset_shift,
            max_check,
        ) {
            Some((full_lo, full_hi)) => {
                for dwell_idx in full_lo.saturating_sub(WIDTH)..full_lo {
                    scalar_dwell_idx_over_block(
                        cum,
                        previous_scores,
                        penalty_table,
                        cand,
                        cand_tb,
                        dwell_idx,
                        prev_band_offset,
                        zero_offset_shift,
                        b,
                        b + WIDTH,
                    );
                }
                // Safety: `full_coverage_range` guarantees every `dwell_idx`
                // in `full_lo..=full_hi` satisfies `lo(dwell_idx) <= b` and
                // `hi(dwell_idx) >= b + WIDTH`, so every lane's
                // `previous_scores`/`cum` access the kernel makes is in
                // bounds (the same invariant the untransposed scalar sweep
                // relies on for the same range, just checked once here for
                // the whole block instead of once per `band_pos`).
                unsafe {
                    dwell_block_kernel_avx512(
                        cum,
                        previous_scores,
                        penalty_table,
                        cand,
                        cand_tb,
                        b,
                        full_lo,
                        full_hi,
                        prev_band_offset,
                    );
                }
                for dwell_idx in (full_hi + 1)..(full_hi + 1 + WIDTH).min(max_check) {
                    scalar_dwell_idx_over_block(
                        cum,
                        previous_scores,
                        penalty_table,
                        cand,
                        cand_tb,
                        dwell_idx,
                        prev_band_offset,
                        zero_offset_shift,
                        b,
                        b + WIDTH,
                    );
                }
            }
            None => {
                for dwell_idx in 0..max_check {
                    scalar_dwell_idx_over_block(
                        cum,
                        previous_scores,
                        penalty_table,
                        cand,
                        cand_tb,
                        dwell_idx,
                        prev_band_offset,
                        zero_offset_shift,
                        b,
                        b + WIDTH,
                    );
                }
            }
        }
        b += WIDTH;
    }
    if b < len {
        for dwell_idx in 0..max_check {
            scalar_dwell_idx_over_block(
                cum,
                previous_scores,
                penalty_table,
                cand,
                cand_tb,
                dwell_idx,
                prev_band_offset,
                zero_offset_shift,
                b,
                len,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
/// Runs the fully-covering `dwell_idx` range `[dwell_lo, dwell_hi]` for one
/// 16-wide block `[block_start, block_start + 16)`, holding `cand`/
/// `cand_tb` in registers for the whole range and writing back once — the
/// change that deletes the per-`dwell_idx` reload and masked stores the
/// autovectorized scalar loop pays. See the module doc comment for the
/// bit-identity argument. The `f64 -> f32` narrowing cast is done as two
/// 8-wide `_mm512_cvtpd_ps` calls (each producing a `__m256`) bridged
/// through a 16-element stack buffer into one `_mm512_loadu_ps`, rather than
/// reaching for a cross-256-bit-lane insert intrinsic — those need
/// `avx512dq` on some parts, and this kernel is `avx512f`-only by policy
/// (see [`DpBackend::supported`]).
///
/// # Safety
/// - AVX-512 (`avx512f`) must be available (checked by the dispatch before
///   this backend is ever selected; never called directly on unchecked
///   hardware).
/// - `dwell_lo..=dwell_hi` must be within [`full_coverage_range`]'s
///   guarantee for `(block_start, 16)`: every `dwell_idx` in it must
///   satisfy `lo(dwell_idx) <= block_start` and `hi(dwell_idx) >=
///   block_start + 16`.
/// - `block_start + 16 <= cand.len()` (`== cand_tb.len()`) and
///   `block_start + 17 <= cum.len()`.
#[inline(never)]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn dwell_block_kernel_avx512(
    cum: &[f64],
    previous_scores: &[f32],
    penalty_table: &[f32],
    cand: &mut [f32],
    cand_tb: &mut [i32],
    block_start: usize,
    dwell_lo: usize,
    dwell_hi: usize,
    prev_band_offset: usize,
) {
    use std::arch::x86_64::*;
    // Safety: see this function's `# Safety` section.
    unsafe {
        let cand_ptr = cand.as_mut_ptr().add(block_start);
        let tb_ptr = cand_tb.as_mut_ptr().add(block_start);
        let mut cand_v = _mm512_loadu_ps(cand_ptr);
        let mut tb_v = _mm512_loadu_epi32(tb_ptr as *const i32);

        let cum_hi_ptr = cum.as_ptr().add(block_start + 1);
        let cum_hi_lo = _mm512_loadu_pd(cum_hi_ptr);
        let cum_hi_hi = _mm512_loadu_pd(cum_hi_ptr.add(8));

        // Bridge buffer for the two 8-wide `f64 -> f32` conversions below —
        // see this function's doc comment.
        let mut rps_buf = [0.0f32; 16];

        // `dwell_idx` drives several derived offsets below (`cum_off`, the
        // `previous_scores` base, the broadcast traceback value) besides
        // indexing `penalty_table`, so the `enumerate()` clippy suggests
        // would not simplify this loop.
        #[allow(clippy::needless_range_loop)]
        for dwell_idx in dwell_lo..=dwell_hi {
            let pen_v = _mm512_set1_ps(penalty_table[dwell_idx]);

            let cum_off = block_start - dwell_idx;
            let cum_lo_ptr = cum.as_ptr().add(cum_off);
            let cum_lo_lo = _mm512_loadu_pd(cum_lo_ptr);
            let cum_lo_hi = _mm512_loadu_pd(cum_lo_ptr.add(8));

            let diff_lo = _mm512_sub_pd(cum_hi_lo, cum_lo_lo);
            let diff_hi = _mm512_sub_pd(cum_hi_hi, cum_lo_hi);
            _mm256_storeu_ps(rps_buf.as_mut_ptr(), _mm512_cvtpd_ps(diff_lo));
            _mm256_storeu_ps(rps_buf.as_mut_ptr().add(8), _mm512_cvtpd_ps(diff_hi));
            let rps = _mm512_loadu_ps(rps_buf.as_ptr());

            let base = cum_off + prev_band_offset - 1;
            let prev_v = _mm512_loadu_ps(previous_scores.as_ptr().add(base));

            let sum = _mm512_add_ps(_mm512_add_ps(prev_v, rps), pen_v);

            let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(sum, cand_v);
            cand_v = _mm512_mask_blend_ps(mask, cand_v, sum);
            let dwell_bcast = _mm512_set1_epi32(dwell_idx as i32);
            tb_v = _mm512_mask_blend_epi32(mask, tb_v, dwell_bcast);
        }

        _mm512_storeu_ps(cand_ptr, cand_v);
        _mm512_storeu_epi32(tb_ptr, tb_v);
    }
}

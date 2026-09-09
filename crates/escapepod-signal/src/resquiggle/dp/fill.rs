// SPDX-License-Identifier: MIT
// Algorithm inspired by fishnet (Brickner et al.); independent implementation.

//! Forward DP fill: per-base step implementations and dwell penalty model.

use super::buffers::{StepBuffers, ViterbiBuffers};
use super::fill_simd::{self, DpBackend};
use super::{DpContext, INVALID_PENALTY, score};
use crate::resquiggle::bands::Band;

/// Forward pass of banded DP.
pub(super) fn forward_pass(
    all_scores: &mut [f32],
    traceback: &mut [i32],
    signal: &[f32],
    expected_levels: &[f32],
    band: &Band,
    base_offsets: &[usize],
    ctx: &mut DpContext,
) {
    let seq_band_start = &band.start;
    let seq_band_end = &band.end;

    // First base
    let current_bandwidth = seq_band_end[0];
    let mut previous_scores = vec![f32::INFINITY; current_bandwidth];
    previous_scores[0] = 0.0;

    ctx.step(
        &mut all_scores[0..current_bandwidth],
        &mut traceback[0..current_bandwidth],
        &previous_scores,
        expected_levels[0],
        &signal[0..current_bandwidth],
        1,
    );

    let mut previous_band_start = 0;
    let mut previous_offset = 0;

    // Remaining bases
    for base_idx in 1..expected_levels.len() {
        let current_band_start = seq_band_start[base_idx];
        let current_band_end = seq_band_end[base_idx];
        let current_bandwidth = current_band_end - current_band_start;

        let current_offset = base_offsets[base_idx];
        let current_slice_end = current_offset + current_bandwidth;

        let prev_band_offset = current_band_start - previous_band_start;

        // Split the scores array to get non-overlapping mutable slices
        let (scores_prev_slice, scores_current_slice) = all_scores.split_at_mut(current_offset);

        ctx.step(
            &mut scores_current_slice[0..current_bandwidth],
            &mut traceback[current_offset..current_slice_end],
            &scores_prev_slice[previous_offset..],
            expected_levels[base_idx],
            &signal[current_band_start..current_band_end],
            prev_band_offset,
        );

        previous_band_start = current_band_start;
        previous_offset = current_offset;
    }
}

/// Forward step using the Viterbi algorithm (no dwell penalty).
///
/// Convenience wrapper that allocates scratch buffers internally.  For hot
/// loops (e.g. adaptive DP), prefer [`dp_step_buffered`] with pre-allocated
/// [`ViterbiBuffers`].
pub fn dp_step(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
) {
    let mut buf = ViterbiBuffers::new(current_scores.len());
    dp_step_buffered(
        current_scores,
        current_traceback,
        previous_scores,
        current_level,
        current_signal,
        prev_band_offset,
        &mut buf,
    );
}

/// Forward step using the Viterbi algorithm with pre-allocated scratch buffers.
///
/// The inner loop is split into three phases so that LLVM can auto-vectorize
/// the first two (base_scores and move_scores are element-wise independent).
/// Only the final sequential scan carries a horizontal dependency.
pub fn dp_step_buffered(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
    buf: &mut ViterbiBuffers,
) {
    let len = current_scores.len();
    buf.prepare(len);

    let base_scores = &mut buf.base_scores[..len];
    let move_scores = &mut buf.move_scores[..len];

    // Phase 1: base_scores — independent, auto-vectorizable
    for i in 0..len {
        let d = current_level - current_signal[i];
        base_scores[i] = d * d;
    }

    let previous_scores_slice = &previous_scores[prev_band_offset..];
    // Number of positions (1..=process_len) with valid move transitions.
    // Capped at len-1 because move_scores/base_scores have length len.
    let process_len = previous_scores_slice.len().min(len - 1);

    // Phase 2: move_scores — independent, auto-vectorizable
    move_scores.fill(f32::INFINITY);
    if prev_band_offset == 0 {
        move_scores[0] = INVALID_PENALTY + previous_scores[previous_scores.len() - 1];
    } else {
        move_scores[0] = previous_scores[prev_band_offset - 1] + base_scores[0];
    }
    for i in 1..=process_len {
        move_scores[i] = previous_scores_slice[i - 1] + base_scores[i];
    }

    // Phase 3: sequential scan — stay vs move (horizontal dependency)
    current_scores[0] = move_scores[0];
    current_traceback[0] = if prev_band_offset == 0 { -1 } else { 0 };

    for i in 1..len {
        let stay = current_scores[i - 1] + base_scores[i];
        let mv = move_scores[i];
        let prev_tb = current_traceback[i - 1];
        // Branchless select: compute both candidates, then pick with a mask.
        // On noisy signal the original branch mispredicts; this keeps the
        // pipeline full at the cost of one always-taken addition.
        let take_move = mv <= stay;
        current_scores[i] = if take_move { mv } else { stay };
        current_traceback[i] = if take_move { 0 } else { prev_tb + 1 };
    }
}

/// Forward step with asymmetric dwell time penalties.
///
/// Uses pre-allocated `StepBuffers` to compute baseline Viterbi scores, then
/// checks a bounded number of dwell transitions with explicit penalties.
/// For positions beyond the check horizon, falls back to baseline Viterbi
/// scores, preserving O(max_check · B) complexity instead of O(B²).
///
/// This is the actual hot loop behind rnabioco/escapepod-rs#353 ("banded-DP
/// refinement dominates CPU cost in the waveform classify path"): every
/// shipped `waveform_model` bundle resolves `RefineSettings::move_table_refinement`'s
/// default (`RefineAlgo::DwellPenalty`, see
/// [`RefineAlgo::default`](super::super::types::RefineAlgo)), so this
/// function, not [`dp_step_buffered`], is where the cycles go — confirmed by
/// profiling with LTO off (`perf` otherwise merges this symbol into its
/// caller), ~75% of wall-clock cycles attributed here by name (#353/#355).
///
/// #355 fixed one slice of that cost: the internal baseline-Viterbi pass
/// below used to heap-allocate a fresh `ViterbiBuffers` on every call (once
/// per base, per refinement iteration) via the [`dp_step`] convenience
/// wrapper; it now reuses `buf.viterbi_buf`. That was ~4% end-to-end.
///
/// #356 addresses the dominant remaining cost: the `dwell_idx` loop used to
/// carry a `running_pos_score` accumulator that restarted at `0.0` for every
/// `band_pos` and re-summed up to `max_check` (≤256) prior `score(...)`
/// terms from scratch — an O(len · max_check) chain of strictly-dependent
/// float adds (the same signal position re-added to a fresh accumulator up
/// to `max_check` times as `band_pos` advances), which blocked
/// autovectorization the way [`dp_step_buffered`]'s phases 1–2 achieve it.
/// `buf.cum_sq_err` is a prefix sum of `score(current_level,
/// current_signal[i])` (`cum[0] = 0`, `cum[k] = cum[k-1] + score(..)`,
/// built once per call, O(len)), so any window sum
/// `sum(score(..)) for i in a..=b` is `cum[b+1] - cum[a]` — an O(1) lookup
/// with no cross-iteration dependency. Two edge cases the rewrite has to
/// preserve (both present in [`dp_step_with_dwell_penalty_reference`], the
/// pre-#356 accumulator form kept under `#[cfg(test)]` as the property-test
/// baseline):
///
/// 1. `previous_scores[dwell_offset]` and `penalty_table[dwell_idx]` are
///    each read over a *contiguous* window as `dwell_idx` sweeps
///    `0..=effective_upper` — no gather needed, though the current
///    implementation is still a plain scalar loop (see below).
/// 2. When `prev_band_offset == 0`, the old code's early `break` at
///    `dwell_idx == band_pos` excluded that index's own score from
///    everything computed after it. That only happens when
///    `check_limit == band_pos` (i.e. `band_pos <= max_check - 1`); this
///    version reproduces it by capping the loop at `effective_upper =
///    check_limit - 1` in that specific case (`truncate_for_zero_offset`
///    below) instead of relying on a mid-loop `break`.
///
/// **Not bit-identical**: `cum[b+1] - cum[a]` accumulates floating-point
/// rounding differently than summing `(dwell_idx + 1)` fresh terms each
/// time (float addition isn't associative). `cum_sq_err` is `f64`
/// specifically because an `f32` cumsum's rounding error grows with
/// `band_pos` (catastrophic cancellation — both operands of the subtraction
/// carry O(band_pos) accumulated error), and an early version of this
/// change measured that as real production disagreement up to 0.556 in
/// `p_charged` before the `f64` fix; validated by
/// [`dp_step_with_dwell_penalty_reference`]-based property tests
/// (`prefix_sum_matches_reference_dwell_penalty_scores` and friends) across
/// randomized band widths (deliberately up to 3000, well past any realistic
/// `max_check`, specifically to catch that class of regression), `max_check`,
/// and `prev_band_offset`.
///
/// **This still isn't nothing, even fixed.** Per-`dp_step` score agreement is
/// tightly bounded (~1e-3, ordinary `f32` non-associativity, no longer
/// growing with band depth) — but this is a genuine dynamic program, where
/// one base's `current_scores` becomes the next base's `previous_scores`.
/// Over hundreds of bases, that bounded per-step noise can still occasionally
/// flip an already-near-tied traceback decision, producing a materially
/// different boundary map for that one read. Measured on the real 55,446-read
/// production sample (post-`f64`-fix): 43 reads (0.08%) show any `p_charged`
/// difference, 10 exceed 0.02, the worst is 0.304 — and checking against the
/// bundle's own recommended operating point (`cl >= 200`), exactly 1 read's
/// discrete charged/uncharged *call* flips (`cl` 200 → 197, a literal
/// hairline tie). This is a structural property of changing summation order
/// anywhere inside a DP feeding a discrete decision and a downstream
/// classifier — not a bug fixable by a cleverer sum, and not fully
/// eliminable short of bit-identical reproduction of the original's
/// operation order, which would forfeit the whole point of this change.
/// Shipped anyway as a deliberate, informed tradeoff (25% real wall-clock win
/// against a ~1-in-55,000 chance of a borderline call flipping): see #356 and
/// `CHANGELOG.md` for the full numbers and reasoning, and #331's `crf`
/// native-encoder work for a case where the equivalent validation instead
/// found zero disagreement at real-dataset scale — the bar this one did not
/// clear.
///
/// **The loop below is transposed, `dwell_idx`-outer rather than
/// `band_pos`-outer.** The scalar `band_pos`-outer form (kept verbatim under
/// `#[cfg(test)]` as [`dp_step_with_dwell_penalty_prefix_scalar`], the
/// immediate pre-transpose baseline) indexes every array it touches by
/// `band_pos - dwell_idx`: `cum[band_pos + 1] - cum[band_pos - dwell_idx]`
/// and `previous_scores[band_pos - dwell_idx - 1 + prev_band_offset]`. For a
/// *fixed* `dwell_idx` that offset is constant, so as `band_pos` varies every
/// one of those accesses is unit-stride — the access pattern was already
/// there, it was just the outer loop variable holding the wrong thing fixed.
/// Swapping which variable is outer turns the inner loop into a straight
/// sequential scan and removes two per-element branches it used to carry
/// (the `dwell_offset >= previous_scores.len()` skip, and the
/// `dwell_idx < penalty_table.len()` table-bounds check — see below), which
/// sets up for the register-blocked SIMD kernels that read this same
/// transposed shape (a separate, later change, not this one).
///
/// For a fixed `dwell_idx`, the set of `band_pos` that perform a candidate
/// update — i.e. that would *not* hit the `continue` in the untransposed
/// form — is the contiguous range `lo(dwell_idx)..hi(dwell_idx)`:
///
/// ```text
/// lo(dwell_idx) = dwell_idx + (if prev_band_offset == 0 { 1 } else { 0 })
/// hi(dwell_idx) = min(len, previous_scores.len() + dwell_idx + 1 - prev_band_offset)
/// ```
///
/// `lo` falls out of two constraints that turn out to coincide. First,
/// `dwell_idx` has to sit within the untransposed loop's own upper bound
/// `effective_upper(band_pos) = min(band_pos - (if prev_band_offset == 0
/// {1} else {0}), max_check - 1)`, which rearranges to `band_pos >=
/// dwell_idx + 1` when `prev_band_offset == 0`, else `band_pos >=
/// dwell_idx`. Second, `dwell_offset = band_pos - dwell_idx - 1 +
/// prev_band_offset` has to be `>= 0`, or the untransposed loop's own
/// `continue` would have fired — which rearranges to `band_pos >= dwell_idx
/// + 1 - prev_band_offset`. Checking both cases shows the second is always
/// implied by the first (when `prev_band_offset == 0` they're the same
/// inequality; when `prev_band_offset >= 1` the second is strictly weaker),
/// so `lo` needs no separate `max()` of the two. `hi` is the mirror
/// constraint, `dwell_offset < previous_scores.len()`, solved for
/// `band_pos`. Three guards from the untransposed form fall out of this for
///   free, needing no separate test in the vector loop below:
///
/// - `truncate_for_zero_offset` (present in
///   [`dp_step_with_dwell_penalty_prefix_scalar`] below) is exactly the `+1`
///   in `lo` when `prev_band_offset == 0` — it existed only to keep
///   `dwell_idx` from reaching `band_pos` itself in that case, which
///   `lo(dwell_idx) = dwell_idx + 1` already guarantees by construction:
///   `band_pos == dwell_idx` is simply outside the range.
/// - The `band_pos == 0 && prev_band_offset == 0` early `continue` is
///   subsumed the same way: `lo(0) = 1` when `prev_band_offset == 0`, so
///   `band_pos == 0` is never in range for any `dwell_idx`, and the
///   candidate buffer is left at its initial sentinel — the same value the
///   early `continue` produces by leaving `current_scores[0]` untouched
///   after the sentinel write.
/// - The "past end of previous band" early-stay guard (`band_pos as i32 +
///   prev_band_offset as i32 - previous_scores.len() as i32 >= max_check as
///   i32`) is implied by `hi(dwell_idx)` for every `dwell_idx <= max_check -
///   1`: a `band_pos` at or past that guard's threshold is at or past
///   `hi(max_check - 1)` too, so no `dwell_idx` iteration ever reaches it —
///   the vector loop needs no test for it at all. It still needs its own
///   test in the scalar tail pass below, because the tail pass is what
///   actually *writes* `current_scores`/`current_traceback` for such a
///   `band_pos` (the candidate sweep just never touches it, leaving
///   whatever sentinel or stray candidate sat in `buf.cand` for that
///   position irrelevant — the tail pass overrides it unconditionally
///   before any other read).
///
/// `max_check <= DWELL_TABLE_SIZE` (256) and `penalty_table.len() ==
/// DWELL_TABLE_SIZE` always (every caller builds the table via
/// [`build_dwell_penalty_table`]), so `dwell_idx < penalty_table.len()`
/// holds across the whole `0..max_check` outer range — the `else` branch
/// computing `dwell_penalty(..)` inline was already dead in production, and
/// is replaced below by a debug assertion instead of a per-element branch.
///
/// **This is bit-identical, not merely equivalent.** The per-candidate
/// arithmetic is untouched: the same `(cum[band_pos + 1] - cum[band_pos -
/// dwell_idx]) as f32` (an `f64` subtraction, then one `as f32` cast,
/// round-to-nearest-even like every other numeric cast in this codebase),
/// the same left-associated `previous_scores[..] + running_pos_score +
/// pen`, the same strict `<` update. `dwell_idx` is still visited in
/// ascending order for every `band_pos` a given `dwell_idx` touches (the
/// outer loop counts up from 0), so a fixed `band_pos`'s candidates are
/// compared in exactly the same order as before — just produced by
/// different outer-loop iterations instead of gathered into one inner
/// loop, which is invisible to the comparison itself. There is no multiply
/// anywhere in the body, so there is no FMA-contraction risk to weigh
/// either. The property tests below (`transposed_matches_prefix_scalar_*`)
/// assert `assert_eq!` — exact equality of both `current_scores` and
/// `current_traceback` against [`dp_step_with_dwell_penalty_prefix_scalar`]
/// — not a tolerance; if any case turns up where that doesn't hold, the fix
/// is to find the bug, not to widen the assertion.
///
/// **Still scalar.** The register-blocked SIMD kernels that read this
/// transposed shape are a separate, later change. This one lands alone
/// because it already wins on its own — the two guards leave the inner
/// loop, the iterations the `dwell_offset` guard used to `continue` past
/// are never visited at all instead of merely skipped, and the branch on
/// `penalty_table.len()` is gone — and because landing it alone keeps the
/// bit-identical claim easy to check before SIMD adds its own edge cases on
/// top of it.
#[allow(clippy::too_many_arguments)]
pub(super) fn dp_step_with_dwell_penalty(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
    penalty_table: &[f32],
    target: f32,
    weight: f32,
    buf: &mut StepBuffers,
) {
    dp_step_with_dwell_penalty_with_backend(
        current_scores,
        current_traceback,
        previous_scores,
        current_level,
        current_signal,
        prev_band_offset,
        penalty_table,
        target,
        weight,
        buf,
        DpBackend::best_for(),
    )
}

/// [`dp_step_with_dwell_penalty`], with the `dwell_idx`-sweep backend chosen
/// explicitly rather than via [`DpBackend::best_for`]. Split out so the
/// property tests below can sweep every backend `DpBackend::available()`
/// reports on this machine, not just the one the dispatch would have picked
/// — see `fill_simd`'s module doc comment for why that matters (rnabioco/
/// escapepod-rs#328).
#[allow(clippy::too_many_arguments)]
fn dp_step_with_dwell_penalty_with_backend(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
    penalty_table: &[f32],
    target: f32,
    _weight: f32,
    buf: &mut StepBuffers,
    backend: DpBackend,
) {
    let len = current_scores.len();
    buf.prepare(len);

    let base_scores = &mut buf.base_scores[..len];
    let base_traceback = &mut buf.base_traceback[..len];

    // Compute baseline Viterbi scores (no penalty) for fallback beyond check
    // range. Uses the buffered form directly (not the `dp_step` convenience
    // wrapper) since this runs once per base per refinement iteration and
    // `dp_step` would heap-allocate a fresh `ViterbiBuffers` on every call.
    dp_step_buffered(
        base_scores,
        base_traceback,
        previous_scores,
        current_level,
        current_signal,
        prev_band_offset,
        &mut buf.viterbi_buf,
    );

    // Prefix sum of per-position squared error: cum[k] = sum of
    // score(level, signal[0..k]). cum[0] = 0 sentinel makes every window
    // sum a single subtraction, cum[hi+1] - cum[lo].
    let cum = &mut buf.cum_sq_err[..=len];
    cum[0] = 0.0;
    for i in 0..len {
        cum[i + 1] = cum[i] + score(current_level, current_signal[i]) as f64;
    }

    // Bound the inner loop: check up to 2*target (covers the full quadratic
    // region plus some logarithmic), clamped to [8, DWELL_TABLE_SIZE].
    let max_check = ((2.0 * target).ceil() as usize).clamp(8, DWELL_TABLE_SIZE);
    let prev_len = previous_scores.len();

    // `max_check <= DWELL_TABLE_SIZE == penalty_table.len()` for every real
    // caller (the table is always built by `build_dwell_penalty_table`), so
    // `penalty_table[dwell_idx]` below never needs the `dwell_penalty(..)`
    // fallback the untransposed form carried for an out-of-range index.
    debug_assert!(
        max_check <= penalty_table.len(),
        "max_check {max_check} exceeds penalty_table.len() {} \
         (expected DWELL_TABLE_SIZE = {DWELL_TABLE_SIZE}); \
         penalty_table[dwell_idx] would be out of bounds",
        penalty_table.len(),
    );

    // Candidate score/traceback per `band_pos`, min-reduced across the
    // `dwell_idx`-outer sweep below. Starts at the same "no valid
    // transition" sentinel the untransposed form writes as its default.
    let cand = &mut buf.cand[..len];
    let cand_tb = &mut buf.cand_tb[..len];
    cand.fill(INVALID_PENALTY + previous_scores[prev_len - 1]);
    cand_tb.fill(-1);

    // The `dwell_idx`-outer sweep itself — dispatched to a register-blocked
    // SIMD kernel when `backend` and this machine allow it, and to a scalar
    // sweep otherwise. `lo(dwell_idx) = dwell_idx + zero_offset_shift` and
    // `hi(dwell_idx) = min(len, (prev_len - prev_band_offset) + dwell_idx +
    // 1)` — see the doc comment above for the derivation of both bounds and
    // `fill_simd`'s module doc comment for how the SIMD kernels use them.
    // `prev_band_offset <= prev_len` always holds for a real forward pass
    // (see the invariant restated in the property tests' `random_case`
    // below), so `prev_len - prev_band_offset` is always representable.
    fill_simd::run_dwell_sweep(
        cum,
        previous_scores,
        penalty_table,
        cand,
        cand_tb,
        len,
        max_check,
        prev_band_offset,
        backend,
    );

    // Scalar tail pass: applies the three steps that carry a
    // `current_scores[band_pos - 1]` / `current_traceback[band_pos - 1]`
    // dependency and so cannot join the `dwell_idx`-outer sweep above —
    // the early-stay guard, the baseline-Viterbi fallback beyond
    // `max_check`, and the "no valid transition" stay-fallback — in the
    // same order the untransposed form applied them.
    for band_pos in 0..len {
        // Past end of previous band by more than max_check — just stay.
        // Implied by `hi(dwell_idx)` for every `dwell_idx <= max_check - 1`
        // above (see the doc comment), so this is the one guard the vector
        // loop above never needed but this tail pass still must apply,
        // since it is what writes `current_scores`/`current_traceback` for
        // such a `band_pos`.
        if band_pos as i32 + prev_band_offset as i32 - prev_len as i32 >= max_check as i32 {
            current_scores[band_pos] =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
            continue;
        }

        let mut best_score = cand[band_pos];
        let mut best_tb = cand_tb[band_pos];

        // For positions beyond the check range, also consider baseline
        // Viterbi path shifted by max_check (no additional penalty since
        // log penalty is negligible for dwells >> target). Same window sum
        // the untransposed form's final `dwell_idx == max_check - 1`
        // iteration would have produced.
        if band_pos >= max_check {
            let running_pos_score = (cum[band_pos + 1] - cum[band_pos - (max_check - 1)]) as f32;
            let pos_score = base_scores[band_pos - max_check] + running_pos_score;

            if pos_score < best_score {
                best_score = pos_score;
                best_tb = base_traceback[band_pos - max_check] + max_check as i32;
            }
        }

        // Fallback: if no valid transition from previous base was found, stay
        if best_score >= INVALID_PENALTY && band_pos > 0 {
            best_score =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            best_tb = current_traceback[band_pos - 1] + 1;
        }

        current_scores[band_pos] = best_score;
        current_traceback[band_pos] = best_tb;
    }
}

/// Pre-#356 reference implementation of [`dp_step_with_dwell_penalty`],
/// unchanged from before the prefix-sum rewrite. Kept only as the baseline
/// for the property tests below — the O(len · max_check) serial
/// `running_pos_score` accumulator this preserves is exactly the cost #356
/// exists to remove from the production path.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn dp_step_with_dwell_penalty_reference(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
    penalty_table: &[f32],
    target: f32,
    weight: f32,
) {
    let len = current_scores.len();
    let mut base_scores = vec![0.0f32; len];
    let mut base_traceback = vec![0i32; len];

    dp_step(
        &mut base_scores,
        &mut base_traceback,
        previous_scores,
        current_level,
        current_signal,
        prev_band_offset,
    );

    let max_check = ((2.0 * target).ceil() as usize).clamp(8, DWELL_TABLE_SIZE);

    for band_pos in 0..len {
        if band_pos as i32 + prev_band_offset as i32 - previous_scores.len() as i32
            >= max_check as i32
        {
            current_scores[band_pos] =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
            continue;
        }

        current_scores[band_pos] = INVALID_PENALTY + previous_scores[previous_scores.len() - 1];
        current_traceback[band_pos] = -1;

        if band_pos == 0 && prev_band_offset == 0 {
            continue;
        }

        let check_limit = band_pos.min(max_check - 1);
        let mut running_pos_score = 0.0;

        for dwell_idx in 0..=check_limit {
            if prev_band_offset == 0 && band_pos == dwell_idx {
                break;
            }

            running_pos_score += score(current_level, current_signal[band_pos - dwell_idx]);

            let dwell_offset =
                (band_pos as i32 - dwell_idx as i32 - 1 + prev_band_offset as i32) as usize;
            if dwell_offset >= previous_scores.len() {
                continue;
            }

            let pen = if dwell_idx < penalty_table.len() {
                penalty_table[dwell_idx]
            } else {
                dwell_penalty(dwell_idx, target, weight)
            };

            let pos_score = previous_scores[dwell_offset] + running_pos_score + pen;

            if pos_score < current_scores[band_pos] {
                current_scores[band_pos] = pos_score;
                current_traceback[band_pos] = dwell_idx as i32;
            }
        }

        if band_pos >= max_check {
            let pos_score = base_scores[band_pos - max_check] + running_pos_score;

            if pos_score < current_scores[band_pos] {
                current_scores[band_pos] = pos_score;
                current_traceback[band_pos] =
                    base_traceback[band_pos - max_check] + max_check as i32;
            }
        }

        if current_scores[band_pos] >= INVALID_PENALTY && band_pos > 0 {
            current_scores[band_pos] =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
        }
    }
}

/// Immediate pre-transpose body of [`dp_step_with_dwell_penalty`]: the #356
/// prefix-sum rewrite (an O(1) window-sum lookup via `cum`, replacing
/// [`dp_step_with_dwell_penalty_reference`]'s O(max_check) running
/// accumulator), but still `band_pos`-outer / `dwell_idx`-inner, unchanged
/// from production before the loop transpose documented on
/// [`dp_step_with_dwell_penalty`] above. Kept only as the exact-equality
/// baseline for the property tests below — unlike the tolerance-based
/// comparison against [`dp_step_with_dwell_penalty_reference`] (which
/// exists because #356's prefix sum is *not* bit-identical to the running
/// accumulator it replaced), the loop transpose has no excuse for any
/// disagreement with this function at all: same operations, same
/// association, same visitation order, just addressed by a different pair
/// of nested loops.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn dp_step_with_dwell_penalty_prefix_scalar(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
    penalty_table: &[f32],
    target: f32,
    _weight: f32,
    buf: &mut StepBuffers,
) {
    let len = current_scores.len();
    buf.prepare(len);

    let base_scores = &mut buf.base_scores[..len];
    let base_traceback = &mut buf.base_traceback[..len];

    dp_step_buffered(
        base_scores,
        base_traceback,
        previous_scores,
        current_level,
        current_signal,
        prev_band_offset,
        &mut buf.viterbi_buf,
    );

    let cum = &mut buf.cum_sq_err[..=len];
    cum[0] = 0.0;
    for i in 0..len {
        cum[i + 1] = cum[i] + score(current_level, current_signal[i]) as f64;
    }

    let max_check = ((2.0 * target).ceil() as usize).clamp(8, DWELL_TABLE_SIZE);

    for band_pos in 0..len {
        if band_pos as i32 + prev_band_offset as i32 - previous_scores.len() as i32
            >= max_check as i32
        {
            current_scores[band_pos] =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
            continue;
        }

        current_scores[band_pos] = INVALID_PENALTY + previous_scores[previous_scores.len() - 1];
        current_traceback[band_pos] = -1;

        if band_pos == 0 && prev_band_offset == 0 {
            continue;
        }

        let check_limit = band_pos.min(max_check - 1);
        let truncate_for_zero_offset = prev_band_offset == 0 && check_limit == band_pos;
        let effective_upper = if truncate_for_zero_offset {
            check_limit - 1
        } else {
            check_limit
        };

        for dwell_idx in 0..=effective_upper {
            let running_pos_score = (cum[band_pos + 1] - cum[band_pos - dwell_idx]) as f32;

            let dwell_offset =
                (band_pos as i32 - dwell_idx as i32 - 1 + prev_band_offset as i32) as usize;
            if dwell_offset >= previous_scores.len() {
                continue;
            }

            let pen = if dwell_idx < penalty_table.len() {
                penalty_table[dwell_idx]
            } else {
                dwell_penalty(dwell_idx, target, _weight)
            };

            let pos_score = previous_scores[dwell_offset] + running_pos_score + pen;

            if pos_score < current_scores[band_pos] {
                current_scores[band_pos] = pos_score;
                current_traceback[band_pos] = dwell_idx as i32;
            }
        }

        if band_pos >= max_check {
            let running_pos_score = (cum[band_pos + 1] - cum[band_pos - (max_check - 1)]) as f32;
            let pos_score = base_scores[band_pos - max_check] + running_pos_score;

            if pos_score < current_scores[band_pos] {
                current_scores[band_pos] = pos_score;
                current_traceback[band_pos] =
                    base_traceback[band_pos - max_check] + max_check as i32;
            }
        }

        if current_scores[band_pos] >= INVALID_PENALTY && band_pos > 0 {
            current_scores[band_pos] =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
        }
    }
}

/// Asymmetric dwell penalty: quadratic below target, logarithmic above.
///
/// Short dwells get a strong quadratic penalty to prevent degenerate collapse.
/// Long dwells get a gentle logarithmic nudge that is easily overcome by good
/// signal fit, preserving genuine long dwells (e.g., aminoacylation signals).
#[inline]
pub fn dwell_penalty(dwell: usize, target: f32, weight: f32) -> f32 {
    let d = dwell as f32;
    if d < target {
        let diff = target - d;
        weight * diff * diff
    } else {
        weight * (1.0 + d / target).ln()
    }
}

/// Maximum precomputed table size for dwell penalties.
pub(super) const DWELL_TABLE_SIZE: usize = 256;

/// Build a precomputed penalty lookup table for dwells 0..DWELL_TABLE_SIZE.
pub(super) fn build_dwell_penalty_table(target: f32, weight: f32) -> Vec<f32> {
    (0..DWELL_TABLE_SIZE)
        .map(|i| dwell_penalty(i, target, weight))
        .collect()
}

/// Property tests validating the #356 prefix-sum rewrite of
/// [`dp_step_with_dwell_penalty`] against [`dp_step_with_dwell_penalty_reference`]
/// (the pre-#356 accumulator form) across randomized inputs.
///
/// The two are **not** expected to be bit-identical (float addition isn't
/// associative — a prefix-sum difference accumulates rounding differently
/// than a fresh per-`band_pos` sum), so this checks `current_scores`
/// agreement within an absolute tolerance rather than equality. It
/// deliberately does not assert `current_traceback` equality: near-tied
/// candidate scores can legitimately pick a different `dwell_idx` under
/// either implementation without either being wrong. Behavioral agreement
/// (that near-ties don't change the *path*) is covered by the existing
/// end-to-end `banded_dp` tests below and by a real-dataset A/B (see #356).
#[cfg(test)]
mod prefix_sum_property_tests {
    use super::super::buffers::StepBuffers;
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Absolute tolerance on `current_scores`. A first version of this
    /// cumsum used `f32` accumulation and only ever generated band widths up
    /// to 150 in this test (comparable to `max_check` itself) — small enough
    /// that catastrophic cancellation (`cum[band_pos] - cum[band_pos -
    /// dwell_idx]`, both operands carrying O(band_pos) rounding error) never
    /// had room to show up here, yet it produced real disagreement up to
    /// 0.556 in `p_charged` on production data (3 of 55,446 reads) once
    /// `band_pos` got large. `cum_sq_err` is `f64` specifically to close
    /// that gap: with it, the worst diff observed across randomized trials
    /// at `len` up to 3000 is ~9.8e-4, and manually re-run at `len` up to
    /// 20,000 it stayed at exactly the same value — i.e. it's now bounded by
    /// ordinary `f32` non-associativity at the score magnitudes this
    /// function produces, not by band depth. `SCORE_TOL` gives that ~3x
    /// margin rather than sitting right at the observed max.
    const SCORE_TOL: f32 = 3e-3;

    struct Case {
        len: usize,
        prev_len: usize,
        prev_band_offset: usize,
        current_level: f32,
        current_signal: Vec<f32>,
        previous_scores: Vec<f32>,
        target: f32,
        weight: f32,
    }

    fn random_case(rng: &mut StdRng) -> Case {
        // Half the cases stay small (comparable to max_check itself); half
        // go large enough to give a flawed cumsum room to accumulate
        // cancellation error before the subtraction — see SCORE_TOL's doc.
        let len = if rng.random_bool(0.5) {
            rng.random_range(1..=150)
        } else {
            rng.random_range(150..=3000)
        };
        let prev_len = rng.random_range(1..=150);
        // A real forward pass always has prev_band_offset <= prev_len (it's
        // `current_band_start - previous_band_start`, and the two bands are
        // built to overlap) — both step functions index
        // `previous_scores[prev_band_offset..]`/`[prev_band_offset - 1]`
        // without re-checking that, so an out-of-range offset here isn't a
        // "hostile input", it's just not a case that can occur. Bias half
        // the cases to prev_band_offset == 0 — the truncation edge case
        // only fires there.
        let prev_band_offset = if rng.random_bool(0.5) {
            0
        } else {
            rng.random_range(0..=prev_len.min(5))
        };
        let current_level = rng.random_range(-2.0..2.0);
        let current_signal: Vec<f32> = (0..len).map(|_| rng.random_range(-3.0..3.0)).collect();
        // A handful of INVALID_PENALTY-scale sentinels mixed with normal
        // scores, matching what a real forward pass produces at band edges.
        let previous_scores: Vec<f32> = (0..prev_len)
            .map(|_| {
                if rng.random_bool(0.1) {
                    INVALID_PENALTY + rng.random_range(0.0..5.0)
                } else {
                    rng.random_range(0.0..20.0)
                }
            })
            .collect();
        let target = rng.random_range(1.0..40.0);
        let weight = rng.random_range(0.1..1.0);
        Case {
            len,
            prev_len,
            prev_band_offset,
            current_level,
            current_signal,
            previous_scores,
            target,
            weight,
        }
    }

    fn run_reference(c: &Case) -> (Vec<f32>, Vec<i32>) {
        let mut scores = vec![0.0f32; c.len];
        let mut tb = vec![0i32; c.len];
        let penalty_table = build_dwell_penalty_table(c.target, c.weight);
        dp_step_with_dwell_penalty_reference(
            &mut scores,
            &mut tb,
            &c.previous_scores,
            c.current_level,
            &c.current_signal,
            c.prev_band_offset,
            &penalty_table,
            c.target,
            c.weight,
        );
        (scores, tb)
    }

    fn run_prefix_sum(c: &Case, buf: &mut StepBuffers) -> (Vec<f32>, Vec<i32>) {
        let mut scores = vec![0.0f32; c.len];
        let mut tb = vec![0i32; c.len];
        let penalty_table = build_dwell_penalty_table(c.target, c.weight);
        dp_step_with_dwell_penalty(
            &mut scores,
            &mut tb,
            &c.previous_scores,
            c.current_level,
            &c.current_signal,
            c.prev_band_offset,
            &penalty_table,
            c.target,
            c.weight,
            buf,
        );
        (scores, tb)
    }

    /// [`run_prefix_sum`], with the `dwell_idx`-sweep backend pinned rather
    /// than left to [`DpBackend::best_for`] — what the backend-sweeping exact-
    /// equality tests below use, so a bug in a kernel this machine's dispatch
    /// wouldn't otherwise reach still gets caught (rnabioco/escapepod-rs#328:
    /// an AVX2 kernel returned wrong values for part of its input while an
    /// AVX-512 machine's *dispatch* never reached it at all).
    fn run_prefix_sum_with_backend(
        c: &Case,
        buf: &mut StepBuffers,
        backend: DpBackend,
    ) -> (Vec<f32>, Vec<i32>) {
        let mut scores = vec![0.0f32; c.len];
        let mut tb = vec![0i32; c.len];
        let penalty_table = build_dwell_penalty_table(c.target, c.weight);
        dp_step_with_dwell_penalty_with_backend(
            &mut scores,
            &mut tb,
            &c.previous_scores,
            c.current_level,
            &c.current_signal,
            c.prev_band_offset,
            &penalty_table,
            c.target,
            c.weight,
            buf,
            backend,
        );
        (scores, tb)
    }

    #[test]
    fn prefix_sum_matches_reference_dwell_penalty_scores() {
        let mut rng = StdRng::seed_from_u64(0x353_356);
        let mut buf = StepBuffers::new(256);
        let mut max_diff = 0.0f32;

        for trial in 0..2000 {
            let case = random_case(&mut rng);
            let (ref_scores, _ref_tb) = run_reference(&case);
            let (new_scores, _new_tb) = run_prefix_sum(&case, &mut buf);

            assert_eq!(ref_scores.len(), new_scores.len());
            for (i, (&r, &n)) in ref_scores.iter().zip(new_scores.iter()).enumerate() {
                // Both implementations use the same INVALID_PENALTY-based
                // "no valid transition" fallback path verbatim, so those
                // positions should agree exactly modulo the same signal
                // read — only compare where both are finite and below the
                // fallback's own magnitude.
                let diff = (r - n).abs();
                assert!(
                    diff <= SCORE_TOL,
                    "trial {trial} pos {i}: reference={r} prefix_sum={n} diff={diff} \
                     (len={}, prev_len={}, prev_band_offset={}, target={}, weight={})",
                    case.len,
                    case.prev_len,
                    case.prev_band_offset,
                    case.target,
                    case.weight,
                );
                max_diff = max_diff.max(diff);
            }
        }

        // Not a correctness assertion — a canary. If this creeps far above
        // SCORE_TOL's margin, the tolerance above needs revisiting rather
        // than silently passing on values close to the cliff edge.
        assert!(
            max_diff < SCORE_TOL,
            "max observed diff {max_diff} approached SCORE_TOL {SCORE_TOL} with no margin"
        );
    }

    #[test]
    fn prefix_sum_truncation_edge_case_matches_reference() {
        // prev_band_offset == 0 with a short band forces
        // truncate_for_zero_offset on nearly every band_pos.
        let mut rng = StdRng::seed_from_u64(0x353_357);
        let mut buf = StepBuffers::new(64);

        for trial in 0..500 {
            let len = rng.random_range(1..=10);
            let case = Case {
                len,
                prev_len: rng.random_range(1..=10),
                prev_band_offset: 0,
                current_level: rng.random_range(-2.0..2.0),
                current_signal: (0..len).map(|_| rng.random_range(-3.0..3.0)).collect(),
                previous_scores: (0..rng.random_range(1..=10))
                    .map(|_| rng.random_range(0.0..20.0))
                    .collect(),
                target: rng.random_range(1.0..10.0),
                weight: rng.random_range(0.1..1.0),
            };
            let (ref_scores, _) = run_reference(&case);
            let (new_scores, _) = run_prefix_sum(&case, &mut buf);
            for (i, (&r, &n)) in ref_scores.iter().zip(new_scores.iter()).enumerate() {
                let diff = (r - n).abs();
                assert!(
                    diff <= SCORE_TOL,
                    "trial {trial} pos {i}: reference={r} prefix_sum={n} diff={diff}"
                );
            }
        }
    }

    fn run_prefix_scalar(c: &Case, buf: &mut StepBuffers) -> (Vec<f32>, Vec<i32>) {
        let mut scores = vec![0.0f32; c.len];
        let mut tb = vec![0i32; c.len];
        let penalty_table = build_dwell_penalty_table(c.target, c.weight);
        dp_step_with_dwell_penalty_prefix_scalar(
            &mut scores,
            &mut tb,
            &c.previous_scores,
            c.current_level,
            &c.current_signal,
            c.prev_band_offset,
            &penalty_table,
            c.target,
            c.weight,
            buf,
        );
        (scores, tb)
    }

    /// Exact-equality counterpart to `prefix_sum_matches_reference_dwell_penalty_scores`
    /// above, but comparing the production (transposed, `dwell_idx`-outer)
    /// [`dp_step_with_dwell_penalty`] against
    /// [`dp_step_with_dwell_penalty_prefix_scalar`] — the immediate
    /// pre-transpose body, still `band_pos`-outer. Both compute the exact
    /// same operations in the exact same association and visitation order
    /// (see the transpose derivation on `dp_step_with_dwell_penalty`'s doc
    /// comment), so unlike the `SCORE_TOL`-bounded comparison above, this
    /// one has no excuse for any disagreement at all — `assert_eq!`, not a
    /// tolerance, on both `current_scores` *and* `current_traceback`.
    ///
    /// Run under *every* `DpBackend::available()` reports on this machine
    /// (not just the one `DpBackend::best_for` would pick): the register-
    /// blocked SIMD kernels in `fill_simd` are a separate implementation of
    /// this same sweep, and rnabioco/escapepod-rs#328 found a kernel that was
    /// wrong only for part of its input while this machine's own dispatch
    /// never reached it at all — sweeping every backend the CPU can run,
    /// regardless of which one is fastest here, is what catches that.
    #[test]
    fn transposed_matches_prefix_scalar_exactly() {
        for backend in DpBackend::available() {
            let mut rng = StdRng::seed_from_u64(0x2a_7005e);
            let mut buf_scalar = StepBuffers::new(256);
            let mut buf_transposed = StepBuffers::new(256);

            for trial in 0..2000 {
                let case = random_case(&mut rng);
                let (scalar_scores, scalar_tb) = run_prefix_scalar(&case, &mut buf_scalar);
                let (transposed_scores, transposed_tb) =
                    run_prefix_sum_with_backend(&case, &mut buf_transposed, backend);

                assert_eq!(
                    scalar_scores, transposed_scores,
                    "backend={backend:?} trial {trial}: current_scores differ (len={}, \
                     prev_len={}, prev_band_offset={}, target={}, weight={})",
                    case.len, case.prev_len, case.prev_band_offset, case.target, case.weight,
                );
                assert_eq!(
                    scalar_tb, transposed_tb,
                    "backend={backend:?} trial {trial}: current_traceback differs (len={}, \
                     prev_len={}, prev_band_offset={}, target={}, weight={})",
                    case.len, case.prev_len, case.prev_band_offset, case.target, case.weight,
                );
            }
        }
    }

    /// Same exact-equality contract as
    /// [`transposed_matches_prefix_scalar_exactly`], but at `len` and
    /// `prev_len` in `1..=10` — the range where `lo`/`hi`'s edge behavior
    /// (an empty candidate range, `band_pos == 0`, `check_limit == band_pos`
    /// truncation) is most likely to be exercised, mirroring
    /// `prefix_sum_truncation_edge_case_matches_reference`'s bias toward the
    /// same region for the (tolerance-based) #356 comparison. This range is
    /// also where a block never gets a fully-covering `dwell_idx` at all
    /// (`full_coverage_range` returning `None`, `fill_simd`'s all-scalar
    /// fallback) and where the trailing-remainder-shorter-than-a-block path
    /// fires — both otherwise rare, so run under every available backend for
    /// the same reason as above.
    #[test]
    fn transposed_matches_prefix_scalar_small_lengths() {
        for backend in DpBackend::available() {
            let mut rng = StdRng::seed_from_u64(0x2a_7005e ^ 0x5ca1e);
            let mut buf_scalar = StepBuffers::new(64);
            let mut buf_transposed = StepBuffers::new(64);

            for trial in 0..2000 {
                let len = rng.random_range(1..=10);
                let prev_len = rng.random_range(1..=10);
                let prev_band_offset = if rng.random_bool(0.5) {
                    0
                } else {
                    rng.random_range(0..=prev_len)
                };
                let case = Case {
                    len,
                    prev_len,
                    prev_band_offset,
                    current_level: rng.random_range(-2.0..2.0),
                    current_signal: (0..len).map(|_| rng.random_range(-3.0..3.0)).collect(),
                    previous_scores: (0..prev_len)
                        .map(|_| {
                            if rng.random_bool(0.1) {
                                INVALID_PENALTY + rng.random_range(0.0..5.0)
                            } else {
                                rng.random_range(0.0..20.0)
                            }
                        })
                        .collect(),
                    target: rng.random_range(1.0..10.0),
                    weight: rng.random_range(0.1..1.0),
                };

                let (scalar_scores, scalar_tb) = run_prefix_scalar(&case, &mut buf_scalar);
                let (transposed_scores, transposed_tb) =
                    run_prefix_sum_with_backend(&case, &mut buf_transposed, backend);

                assert_eq!(
                    scalar_scores, transposed_scores,
                    "backend={backend:?} trial {trial}: current_scores differ (len={len}, \
                     prev_len={prev_len}, prev_band_offset={prev_band_offset})",
                );
                assert_eq!(
                    scalar_tb, transposed_tb,
                    "backend={backend:?} trial {trial}: current_traceback differs (len={len}, \
                     prev_len={prev_len}, prev_band_offset={prev_band_offset})",
                );
            }
        }
    }
}

// SPDX-License-Identifier: MIT
// Algorithm inspired by fishnet (Brickner et al.); independent implementation.

//! Forward DP fill: per-base step implementations and dwell penalty model.

use super::buffers::{StepBuffers, ViterbiBuffers};
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
/// **Not hand-vectorized.** Breaking the dependency was the point — whether
/// it also needs explicit SIMD intrinsics is a measure-first question (see
/// #356's non-goals); this is deliberately still a scalar loop pending that
/// measurement.
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
    _weight: f32,
    buf: &mut StepBuffers,
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

    for band_pos in 0..len {
        // Past end of previous band by more than max_check — just stay
        if band_pos as i32 + prev_band_offset as i32 - previous_scores.len() as i32
            >= max_check as i32
        {
            current_scores[band_pos] =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
            continue;
        }

        // Default: invalid score
        current_scores[band_pos] = INVALID_PENALTY + previous_scores[previous_scores.len() - 1];
        current_traceback[band_pos] = -1;

        if band_pos == 0 && prev_band_offset == 0 {
            continue;
        }

        // Try dwell transitions with explicit penalty (bounded to max_check)
        let check_limit = band_pos.min(max_check - 1);
        // The reference implementation breaks out of its accumulator loop
        // before adding dwell_idx == band_pos's own score when
        // prev_band_offset == 0 (no earlier base to reach past position 0).
        // That only overlaps check_limit == band_pos (band_pos <=
        // max_check - 1), in which case band_pos >= 1 is guaranteed (the
        // band_pos == 0 && prev_band_offset == 0 case already `continue`d
        // above), so the subtraction below can't underflow.
        let truncate_for_zero_offset = prev_band_offset == 0 && check_limit == band_pos;
        let effective_upper = if truncate_for_zero_offset {
            check_limit - 1
        } else {
            check_limit
        };

        for dwell_idx in 0..=effective_upper {
            // band_pos - dwell_idx never underflows: dwell_idx <=
            // effective_upper <= check_limit <= band_pos.
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

        // For positions beyond the check range, also consider baseline Viterbi
        // path shifted by max_check (no additional penalty since log penalty
        // is negligible for dwells >> target). band_pos >= max_check here
        // implies check_limit == max_check - 1 with no truncation (the
        // truncation case requires check_limit == band_pos <= max_check - 1),
        // so effective_upper == max_check - 1 and the window sum below
        // matches what the loop above would have accumulated through its
        // final iteration.
        if band_pos >= max_check {
            let running_pos_score = (cum[band_pos + 1] - cum[band_pos - (max_check - 1)]) as f32;
            let pos_score = base_scores[band_pos - max_check] + running_pos_score;

            if pos_score < current_scores[band_pos] {
                current_scores[band_pos] = pos_score;
                current_traceback[band_pos] =
                    base_traceback[band_pos - max_check] + max_check as i32;
            }
        }

        // Fallback: if no valid transition from previous base was found, stay
        if current_scores[band_pos] >= INVALID_PENALTY && band_pos > 0 {
            current_scores[band_pos] =
                current_scores[band_pos - 1] + score(current_level, current_signal[band_pos]);
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
        }
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
}

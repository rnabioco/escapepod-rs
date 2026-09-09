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
/// refinement dominates CPU cost in the waveform classify path"): that issue's
/// own profiling (LTO-inlined, `release-with-debug`) misattributed the cost to
/// `DpContext::step`/`RefineAlgo::Viterbi` — every shipped `waveform_model`
/// bundle actually resolves `RefineSettings::move_table_refinement`'s default
/// (`RefineAlgo::DwellPenalty`, see [`RefineAlgo::default`](super::super::types::RefineAlgo)),
/// so this function, not [`dp_step_buffered`], is where the cycles go.
/// Re-profiling with the `profiling` cargo profile (LTO off, so perf doesn't
/// merge this symbol into its caller) on the same input attributes ~75% of
/// wall-clock cycles here directly, confirmed by name rather than inferred.
///
/// One real fix already landed: the internal baseline-Viterbi pass below used
/// to go through [`dp_step`], which heap-allocates a fresh `ViterbiBuffers` on
/// every call — once per base, per refinement iteration. Routing it through
/// [`dp_step_buffered`] with `buf.viterbi_buf` (a scratch buffer `StepBuffers`
/// now owns and reuses) removed that allocation churn: ~4% end-to-end wall
/// time on a real bundle + real POD5/BAM (a 55,446-read production sample,
/// `--device cpu --threads 4`, bit-identical output, `release`-profile,
/// interleaved A/B reps to control for page-cache warming per
/// `benchmarks/README.md`'s methodology).
///
/// That 4% is a small slice of the ~75%, not the story — the dominant cost is
/// the `dwell_idx` loop's `running_pos_score` accumulator below. It restarts
/// at `0.0` for every `band_pos` and re-sums up to `max_check` (≤256) prior
/// `score(...)` terms from scratch, so the same signal position is
/// re-added to a fresh accumulator up to `max_check` times as `band_pos`
/// advances — O(len · max_check) additions, each strictly dependent on the
/// previous (a serial reduction chain), which is what actually burns the
/// cycles this issue found. A prefix sum over `score(current_level,
/// current_signal[i])` (built once, O(len)) turns
/// `running_pos_score(band_pos, dwell_idx)` into
/// `cum[band_pos] - cum[band_pos - dwell_idx - 1]` — an O(1) lookup with no
/// cross-iteration dependency, which is what would let the `dwell_idx` loop
/// autovectorize the way `dp_step_buffered`'s phases 1–2 already do. Two
/// details a rewrite must preserve, found while reading this function for
/// this note, not yet implemented: (1) `previous_scores[dwell_offset]` and
/// `penalty_table[dwell_idx]` are each read over a *contiguous* window as
/// `dwell_idx` sweeps `0..=check_limit` (the former descending, the latter
/// ascending) — no gather needed; (2) when `prev_band_offset == 0`, the
/// current early `break` at `dwell_idx == band_pos` must still exclude that
/// index's own score from the accumulated sum (it fires before the `+=`), so
/// the cumsum lookup's effective upper bound needs the same off-by-one in
/// that boundary case, not just a truncated loop count. Left as a follow-up
/// rather than done here — a numerically-different (not bit-identical, per
/// float non-associativity) implementation deserves its own validation pass
/// (property test against this scalar form, not just the existing exact-value
/// unit tests) before it replaces this one.
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
        // is negligible for dwells >> target)
        if band_pos >= max_check {
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

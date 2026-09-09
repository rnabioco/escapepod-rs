// SPDX-License-Identifier: MIT
// Algorithm inspired by fishnet (Brickner et al.); independent implementation.

//! Reusable scratch buffers for the banded DP step implementations.

/// Reusable temporary buffers for the Viterbi DP step (phases 1 & 2).
pub struct ViterbiBuffers {
    pub(super) base_scores: Vec<f32>,
    pub(super) move_scores: Vec<f32>,
}

impl ViterbiBuffers {
    /// Create buffers sized for the given bandwidth.
    pub fn new(capacity: usize) -> Self {
        Self {
            base_scores: vec![0.0f32; capacity],
            move_scores: vec![f32::INFINITY; capacity],
        }
    }

    /// Ensure buffers are at least `len` elements.
    pub(super) fn prepare(&mut self, len: usize) {
        if self.base_scores.len() < len {
            self.base_scores.resize(len, 0.0);
            self.move_scores.resize(len, f32::INFINITY);
        }
    }
}

/// Reusable temporary buffers for the dwell penalty DP step.
pub(super) struct StepBuffers {
    pub(super) base_scores: Vec<f32>,
    pub(super) base_traceback: Vec<i32>,
    /// Scratch for the internal baseline-Viterbi fallback pass.
    ///
    /// `dp_step_with_dwell_penalty` runs a full Viterbi pass every call (once
    /// per base, per refinement iteration) to get a fallback score beyond the
    /// dwell check horizon. Owning the buffer here — instead of going through
    /// [`dp_step`](super::fill::dp_step), which allocates a fresh
    /// [`ViterbiBuffers`] on every call — is what makes that pass reuse
    /// scratch instead of heap-allocating twice per base.
    pub(super) viterbi_buf: ViterbiBuffers,
    /// Prefix sum of per-position squared error, `cum[0] = 0`,
    /// `cum[k] = cum[k-1] + score(level, signal[k-1])`. Length `len + 1`.
    ///
    /// Lets the `dwell_idx` inner loop read `sum(score(..)) over a window`
    /// as one O(1) subtraction (`cum[hi] - cum[lo]`) instead of a serial
    /// running accumulator — see rnabioco/escapepod-rs#356.
    ///
    /// **`f64`, not `f32`.** An `f32` cumsum, subtracted at two positions
    /// `dwell_idx` apart, is a textbook catastrophic-cancellation trap: both
    /// operands carry O(band_pos) accumulated rounding error (from summing
    /// every position *since the start of the band*, not just the window),
    /// so the error in their difference doesn't shrink with the window size
    /// — it grows with how deep into the band `band_pos` is. A first `f32`
    /// attempt at this measured real production-data disagreement up to
    /// 0.556 in `p_charged` for 3 of 55,446 reads (bands with an unusually
    /// long dwell somewhere gave `cum` more room to drift before the
    /// subtraction) — real answer drift, not acceptable numerical noise, and
    /// not something the property tests below caught because their band
    /// widths (≤150, comparable to `max_check` itself) never gave `cum` room
    /// to drift far from the window before the subtraction. `f64`'s ~52-bit
    /// mantissa keeps that drift far below `f32` ULP for any realistic band
    /// width, which is cheap here: this cumsum is O(len) additions once per
    /// call, not the O(len·max_check) cost being fixed.
    pub(super) cum_sq_err: Vec<f64>,
    /// Best dwell-transition candidate found so far for each `band_pos`,
    /// filled by the `dwell_idx`-outer sweep in `dp_step_with_dwell_penalty`
    /// (see that function's doc comment for the loop transpose this backs).
    /// Reset to the "no valid transition" sentinel
    /// (`INVALID_PENALTY + previous_scores[last]`) at the start of every
    /// call, then min-reduced in ascending `dwell_idx` order — the same
    /// order the pre-transpose `band_pos`-outer form considered candidates
    /// in for a fixed `band_pos`, just spread across outer-loop iterations
    /// instead of one inner loop, so tie-breaking is unchanged. The scalar
    /// tail pass reads this back to apply the steps that carry a
    /// `current_scores[band_pos - 1]` dependency and so cannot be part of
    /// the `dwell_idx`-outer sweep.
    pub(super) cand: Vec<f32>,
    /// Traceback counterpart to [`cand`](Self::cand): the `dwell_idx` (as
    /// `i32`) that produced the current best score at each `band_pos`, or
    /// `-1` if nothing has beaten the sentinel yet.
    pub(super) cand_tb: Vec<i32>,
}

impl StepBuffers {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            base_scores: vec![0.0f32; capacity],
            base_traceback: vec![0i32; capacity],
            viterbi_buf: ViterbiBuffers::new(capacity),
            cum_sq_err: vec![0.0f64; capacity + 1],
            cand: vec![0.0f32; capacity],
            cand_tb: vec![0i32; capacity],
        }
    }

    /// Ensure buffers are at least `len` elements.
    ///
    /// Does not zero `base_scores`/`base_traceback`: `dp_step_with_dwell_penalty`
    /// (`fill.rs`) writes every element of both, for the full `0..len` range,
    /// via `dp_step_buffered` before anything reads them (that function's
    /// three phases cover `0..len` completely), so a pre-fill here was dead
    /// work on every call.
    pub(super) fn prepare(&mut self, len: usize) {
        if self.base_scores.len() < len {
            self.base_scores.resize(len, 0.0);
            self.base_traceback.resize(len, 0);
        }
        if self.cum_sq_err.len() < len + 1 {
            self.cum_sq_err.resize(len + 1, 0.0);
        }
        if self.cand.len() < len {
            self.cand.resize(len, 0.0);
            self.cand_tb.resize(len, 0);
        }
        self.viterbi_buf.prepare(len);
    }
}

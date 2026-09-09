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
}

impl StepBuffers {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            base_scores: vec![0.0f32; capacity],
            base_traceback: vec![0i32; capacity],
            viterbi_buf: ViterbiBuffers::new(capacity),
        }
    }

    /// Ensure buffers are at least `len` elements and zero them.
    pub(super) fn prepare(&mut self, len: usize) {
        if self.base_scores.len() < len {
            self.base_scores.resize(len, 0.0);
            self.base_traceback.resize(len, 0);
        }
        self.base_scores[..len].fill(0.0);
        self.base_traceback[..len].fill(0);
        self.viterbi_buf.prepare(len);
    }
}

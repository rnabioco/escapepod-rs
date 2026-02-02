// SPDX-License-Identifier: GPL-3.0-or-later
// Inspired by fishnet, licensed under the GNU General Public License v3.0.

//! Banded dynamic programming for sequence-to-signal alignment.

use super::bands::Band;
use super::types::RefineAlgo;

/// Penalty score for invalid or out-of-band transitions.
const INVALID_PENALTY: f32 = 100.0;

/// Squared error score between expected and measured signal levels.
#[inline]
pub fn score(expected: f32, measured: f32) -> f32 {
    let d = measured - expected;
    d * d
}

/// Perform banded DP to find optimal sequence-to-signal alignment.
///
/// Returns a vector of signal indices (length = levels.len() + 1) where
/// path[i] is the signal position where base i starts, and path[last]
/// is the end of the last base.
pub fn banded_dp(signal: &[f32], levels: &[f32], band: &Band, method: &RefineAlgo) -> Vec<usize> {
    // Build base offsets for flattened score array
    let mut base_offsets = Vec::with_capacity(band.len() + 1);
    base_offsets.push(0);
    let mut offset_cumsum = 0;
    let mut max_bandwidth = 0usize;
    for i in 0..band.len() {
        let bw = band.end[i] - band.start[i];
        max_bandwidth = max_bandwidth.max(bw);
        offset_cumsum += bw;
        base_offsets.push(offset_cumsum);
    }

    let band_len = offset_cumsum;
    let mut all_scores = vec![f32::INFINITY; band_len];
    let mut traceback = vec![0i32; band_len];

    // Pre-allocate dwell penalty buffers (reused across all bases)
    let mut dp_buf = match method {
        RefineAlgo::DwellPenalty { .. } => Some(StepBuffers::new(max_bandwidth)),
        RefineAlgo::Viterbi => None,
    };

    forward_pass(
        &mut all_scores,
        &mut traceback,
        signal,
        levels,
        band,
        &base_offsets,
        method,
        &mut dp_buf,
    );

    let mut path = vec![0usize; levels.len() + 1];
    banded_traceback(&mut path, band, &base_offsets, &traceback);

    path
}

/// Reusable temporary buffers for the dwell penalty DP step.
struct StepBuffers {
    base_scores: Vec<f32>,
    base_traceback: Vec<i32>,
}

impl StepBuffers {
    fn new(capacity: usize) -> Self {
        Self {
            base_scores: vec![0.0f32; capacity],
            base_traceback: vec![0i32; capacity],
        }
    }

    /// Ensure buffers are at least `len` elements and zero them.
    fn prepare(&mut self, len: usize) {
        if self.base_scores.len() < len {
            self.base_scores.resize(len, 0.0);
            self.base_traceback.resize(len, 0);
        }
        self.base_scores[..len].fill(0.0);
        self.base_traceback[..len].fill(0);
    }
}

/// Forward pass of banded DP.
#[allow(clippy::too_many_arguments)]
fn forward_pass(
    all_scores: &mut [f32],
    traceback: &mut [i32],
    signal: &[f32],
    expected_levels: &[f32],
    band: &Band,
    base_offsets: &[usize],
    method: &RefineAlgo,
    dp_buf: &mut Option<StepBuffers>,
) {
    let mut short_dwell_penalty_vec = Vec::new();
    let use_dwell_penalty = match method {
        RefineAlgo::DwellPenalty {
            target,
            limit,
            weight,
        } => {
            short_dwell_penalty_vec = build_dwell_penalties(*target, *limit, *weight);
            true
        }
        RefineAlgo::Viterbi => false,
    };

    let seq_band_start = &band.start;
    let seq_band_end = &band.end;

    // First base
    let current_bandwidth = seq_band_end[0];
    let mut previous_scores = vec![f32::INFINITY; current_bandwidth];
    previous_scores[0] = 0.0;

    if use_dwell_penalty {
        dp_step_with_dwell_penalty(
            &mut all_scores[0..current_bandwidth],
            &mut traceback[0..current_bandwidth],
            &previous_scores,
            expected_levels[0],
            &signal[0..current_bandwidth],
            1,
            &short_dwell_penalty_vec,
            dp_buf.as_mut().unwrap(),
        );
    } else {
        dp_step(
            &mut all_scores[0..current_bandwidth],
            &mut traceback[0..current_bandwidth],
            &previous_scores,
            expected_levels[0],
            &signal[0..current_bandwidth],
            1,
        );
    }

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

        if use_dwell_penalty {
            dp_step_with_dwell_penalty(
                &mut scores_current_slice[0..current_bandwidth],
                &mut traceback[current_offset..current_slice_end],
                &scores_prev_slice[previous_offset..],
                expected_levels[base_idx],
                &signal[current_band_start..current_band_end],
                prev_band_offset,
                &short_dwell_penalty_vec,
                dp_buf.as_mut().unwrap(),
            );
        } else {
            dp_step(
                &mut scores_current_slice[0..current_bandwidth],
                &mut traceback[current_offset..current_slice_end],
                &scores_prev_slice[previous_offset..],
                expected_levels[base_idx],
                &signal[current_band_start..current_band_end],
                prev_band_offset,
            );
        }

        previous_band_start = current_band_start;
        previous_offset = current_offset;
    }
}

/// Forward step using the Viterbi algorithm (no dwell penalty).
pub fn dp_step(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
) {
    // Handle start position
    if prev_band_offset == 0 {
        current_scores[0] = INVALID_PENALTY + previous_scores[previous_scores.len() - 1];
        current_traceback[0] = -1;
    } else {
        let base_score = score(current_level, current_signal[0]);
        current_scores[0] = previous_scores[prev_band_offset - 1] + base_score;
        current_traceback[0] = 0;
    }

    let previous_scores_slice = &previous_scores[prev_band_offset..];

    let process_len = if previous_scores_slice.len() == current_scores.len() {
        previous_scores_slice.len() - 1
    } else {
        previous_scores_slice.len()
    };

    // Overlapping region: both move and stay transitions possible
    for band_pos in 1..=process_len {
        let base_score = score(current_level, current_signal[band_pos]);
        let move_score = previous_scores_slice[band_pos - 1] + base_score;
        let stay_score = current_scores[band_pos - 1] + base_score;

        if move_score < stay_score {
            current_scores[band_pos] = move_score;
            current_traceback[band_pos] = 0;
        } else {
            current_scores[band_pos] = stay_score;
            current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
        }
    }

    // Remaining: only stay transitions
    for band_pos in (process_len + 1)..current_scores.len() {
        let base_score = score(current_level, current_signal[band_pos]);
        let stay_score = current_scores[band_pos - 1] + base_score;
        current_scores[band_pos] = stay_score;
        current_traceback[band_pos] = current_traceback[band_pos - 1] + 1;
    }
}

/// Forward step with dwell time penalties.
///
/// Uses pre-allocated `StepBuffers` to avoid per-call heap allocations.
#[allow(clippy::too_many_arguments)]
fn dp_step_with_dwell_penalty(
    current_scores: &mut [f32],
    current_traceback: &mut [i32],
    previous_scores: &[f32],
    current_level: f32,
    current_signal: &[f32],
    prev_band_offset: usize,
    dwell_penalty: &[f32],
    buf: &mut StepBuffers,
) {
    let len = current_scores.len();
    buf.prepare(len);

    let base_scores = &mut buf.base_scores[..len];
    let base_traceback = &mut buf.base_traceback[..len];

    dp_step(
        base_scores,
        base_traceback,
        previous_scores,
        current_level,
        current_signal,
        prev_band_offset,
    );

    let max_penalized_len = dwell_penalty.len();

    for band_pos in 0..current_scores.len() {
        // Past end of previous band — stay until the end
        if band_pos as i32 + prev_band_offset as i32 - previous_scores.len() as i32
            >= max_penalized_len as i32
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

        let mut running_pos_score = 0.0;
        for dwell_idx in 0..dwell_penalty.len() {
            if dwell_idx > band_pos || (prev_band_offset == 0 && band_pos == dwell_idx) {
                break;
            }

            running_pos_score += score(current_level, current_signal[band_pos - dwell_idx]);

            let dwell_offset =
                (band_pos as i32 - dwell_idx as i32 - 1 + prev_band_offset as i32) as usize;
            if dwell_offset >= previous_scores.len() {
                continue;
            }

            let pos_score =
                previous_scores[dwell_offset] + running_pos_score + dwell_penalty[dwell_idx];

            if pos_score < current_scores[band_pos] {
                current_scores[band_pos] = pos_score;
                current_traceback[band_pos] = dwell_idx as i32;
            }
        }

        if band_pos >= max_penalized_len {
            let pos_score = base_scores[band_pos - max_penalized_len] + running_pos_score;

            if pos_score < current_scores[band_pos] {
                current_scores[band_pos] = pos_score;
                current_traceback[band_pos] =
                    base_traceback[band_pos - max_penalized_len] + max_penalized_len as i32;
            }
        }
    }
}

/// Calculate the dwell penalty vector.
fn build_dwell_penalties(target: f32, limit: f32, weight: f32) -> Vec<f32> {
    let actual_limit = if limit > target { target } else { limit };
    let size = actual_limit as usize;
    (0..size)
        .map(|i| weight * (i as f32 - target).powi(2))
        .collect()
}

/// Traceback to reconstruct the optimal path.
pub fn banded_traceback(
    path: &mut [usize],
    band: &Band,
    base_offsets: &[usize],
    traceback: &[i32],
) {
    let seq_band_start = &band.start;
    let seq_band_end = &band.end;

    path[0] = 0;
    let last = path.len() - 1;
    path[last] = seq_band_end[seq_band_end.len() - 1];

    for base_idx in (1..last).rev() {
        let sig_lookup_pos = path[base_idx + 1] - 1;
        let base_offset = base_offsets[base_idx];
        let band_start = seq_band_start[base_idx];
        let traceback_idx = base_offset + (sig_lookup_pos - band_start);
        let next_sig_offset = traceback[traceback_idx];
        path[base_idx] = if next_sig_offset >= 0 {
            sig_lookup_pos - (next_sig_offset as usize)
        } else {
            // Traceback hit an unreachable cell (band too narrow or bad inputs).
            // Fall back to band start to avoid underflow.
            band_start
        };
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision, clippy::useless_vec)]
mod tests {
    use super::*;

    fn round_to(value: f32, decimal_places: u32) -> f32 {
        let multiplier = 10f32.powi(decimal_places as i32);
        (value * multiplier).round() / multiplier
    }

    #[test]
    fn test_build_dwell_penalties() {
        let vec = build_dwell_penalties(4.0, 3.0, 0.5);
        assert_eq!(vec, vec![8.0, 4.5, 2.0]);
    }

    #[test]
    fn test_viterbi_first_iter_scores() {
        let band_width = 46;
        let mut scores = vec![0.0; 50];
        let mut tb = vec![-1i32; 50];
        let mut prev_scores = vec![1000000000.0f32; band_width];
        prev_scores[0] = 0.0;
        let level = 0.0;
        let signal: Vec<f32> = vec![
            0.53498218,
            0.55017991,
            0.65656397,
            0.545114,
            0.59577308,
            0.55017991,
            0.53498218,
            0.49445492,
            0.54004809,
            0.47419129,
            0.64136625,
            0.24115953,
            0.40326858,
            0.42353221,
            0.7376185,
            0.4843231,
            0.48938901,
            0.55524581,
            0.57550944,
            0.56537763,
            0.4843231,
            0.66669579,
            0.56031172,
            0.545114,
            0.46912538,
            0.6109708,
            0.65656397,
            0.51471855,
            0.60083898,
            0.58057535,
            0.58564126,
            0.44379584,
            0.45899356,
            0.545114,
            0.545114,
            0.545114,
            0.60590489,
            0.4843231,
            0.63630034,
            0.58057535,
            0.35767541,
            0.50458673,
            0.4843231,
            0.24622543,
            0.2259618,
            -0.08305858,
        ];

        dp_step(
            &mut scores[..band_width],
            &mut tb[..band_width],
            &prev_scores,
            level,
            &signal,
            1,
        );

        let expected_scores: Vec<f32> = vec![
            0.28620595, 0.5889039, 1.0199802, 1.3171295, 1.672075, 1.9747729, 2.260979, 2.5054646,
            2.7971165, 3.0219738, 3.4333246, 3.4914825, 3.654108, 3.8334875, 4.3775687, 4.612138,
            4.8516393, 5.1599374, 5.4911485, 5.8108006, 6.0453696, 6.489853, 6.803802, 7.100951,
            7.3210297, 7.694315, 8.125391, 8.3903265, 8.751334, 9.088402, 9.431377, 9.628332,
            9.839007, 10.136157, 10.433307, 10.730456, 11.097577, 11.332146, 11.737023, 12.074091,
            12.202023, 12.456631, 12.691199, 12.751826, 12.802885, 12.809784, 0.0, 0.0, 0.0, 0.0,
        ];

        assert_eq!(
            scores
                .iter()
                .map(|&el| round_to(el, 5))
                .collect::<Vec<f32>>(),
            expected_scores
                .iter()
                .map(|&el| round_to(el, 5))
                .collect::<Vec<f32>>()
        );
    }

    #[test]
    fn test_dwell_penalty_first_iter_scores() {
        let band_width = 46;
        let mut scores = vec![0.0; 50];
        let mut tb = vec![-1i32; 50];
        let mut prev_scores = vec![1000000000.0f32; band_width];
        prev_scores[0] = 0.0;
        let level = 0.0;
        let signal: Vec<f32> = vec![
            0.53498218,
            0.55017991,
            0.65656397,
            0.545114,
            0.59577308,
            0.55017991,
            0.53498218,
            0.49445492,
            0.54004809,
            0.47419129,
            0.64136625,
            0.24115953,
            0.40326858,
            0.42353221,
            0.7376185,
            0.4843231,
            0.48938901,
            0.55524581,
            0.57550944,
            0.56537763,
            0.4843231,
            0.66669579,
            0.56031172,
            0.545114,
            0.46912538,
            0.6109708,
            0.65656397,
            0.51471855,
            0.60083898,
            0.58057535,
            0.58564126,
            0.44379584,
            0.45899356,
            0.545114,
            0.545114,
            0.545114,
            0.60590489,
            0.4843231,
            0.63630034,
            0.58057535,
            0.35767541,
            0.50458673,
            0.4843231,
            0.24622543,
            0.2259618,
            -0.08305858,
        ];
        let dwell_penalty = vec![8., 4.5, 2.];
        let mut buf = StepBuffers::new(band_width);

        dp_step_with_dwell_penalty(
            &mut scores[..band_width],
            &mut tb[..band_width],
            &prev_scores,
            level,
            &signal,
            1,
            &dwell_penalty,
            &mut buf,
        );

        let expected_scores: Vec<f32> = vec![
            8.28620593,
            5.08890386,
            3.01998011,
            1.31712938,
            1.67207494,
            1.97477287,
            2.2609788,
            2.50546447,
            2.79711641,
            3.02197378,
            3.43332445,
            3.49148236,
            3.65410791,
            3.83348744,
            4.37756849,
            4.61213735,
            4.85163896,
            5.15993687,
            5.49114799,
            5.81079985,
            6.04536872,
            6.48985199,
            6.80380122,
            7.10095049,
            7.32102911,
            7.69431443,
            8.12539067,
            8.39032586,
            8.75133334,
            9.08840108,
            9.43137677,
            9.62833152,
            9.83900661,
            10.13615588,
            10.43330515,
            10.73045442,
            11.09757516,
            11.33214403,
            11.73702215,
            12.07408989,
            12.20202158,
            12.45662936,
            12.69119822,
            12.75182519,
            12.80288392,
            12.80978265,
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        assert_eq!(
            scores
                .iter()
                .map(|&el| round_to(el, 4))
                .collect::<Vec<f32>>(),
            expected_scores
                .iter()
                .map(|&el| round_to(el, 4))
                .collect::<Vec<f32>>()
        );
    }

    #[test]
    fn test_traceback_simple() {
        let n_bases = 3;
        let mut path = vec![0; n_bases + 1];

        let band = Band::new(vec![0, 3, 5], vec![3, 5, 10], true);
        let base_offsets = vec![0, 3, 5];
        let traceback = vec![0, 1, 2, 0, 1, 0, 1, 2, 3, 4];

        banded_traceback(&mut path, &band, &base_offsets, &traceback);
        assert_eq!(path, vec![0, 3, 5, 10]);
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later
// Inspired by fishnet, licensed under the GNU General Public License v3.0.

//! Banded dynamic programming for sequence-to-signal alignment.
//!
//! The DP is split into cohesive submodules:
//! - [`buffers`]: reusable scratch buffers ([`ViterbiBuffers`], `StepBuffers`).
//! - [`fill`]: forward pass and per-base step implementations plus the dwell
//!   penalty model.
//! - [`traceback`]: optimal path reconstruction ([`banded_traceback`]).

mod buffers;
mod fill;
mod traceback;

use super::bands::Band;
use super::types::RefineAlgo;

use buffers::StepBuffers;
use fill::{build_dwell_penalty_table, dp_step_with_dwell_penalty, forward_pass};

// Public surface — re-exported so existing paths (`resquiggle::dp::*`) resolve
// unchanged after the split.
pub use buffers::ViterbiBuffers;
pub use fill::{dp_step, dp_step_buffered, dwell_penalty};
pub use traceback::banded_traceback;

/// Penalty score for invalid or out-of-band transitions.
const INVALID_PENALTY: f32 = 100.0;

/// Encapsulates DP method dispatch and its associated buffers/tables.
///
/// Created once per DP run, this struct owns the penalty table (if needed),
/// dwell parameters, and pre-allocated scratch buffers.  The [`step`] method
/// dispatches to the correct DP step implementation.
pub(super) struct DpContext {
    penalty_table: Option<Vec<f32>>,
    dwell_target: f32,
    dwell_weight: f32,
    viterbi_buf: ViterbiBuffers,
    step_buf: Option<StepBuffers>,
}

impl DpContext {
    /// Build a context from the refinement algorithm and maximum expected bandwidth.
    pub(super) fn new(method: &RefineAlgo, max_bandwidth: usize) -> Self {
        match method {
            RefineAlgo::DwellPenalty { target, weight } => Self {
                penalty_table: Some(build_dwell_penalty_table(*target, *weight)),
                dwell_target: *target,
                dwell_weight: *weight,
                viterbi_buf: ViterbiBuffers::new(max_bandwidth),
                step_buf: Some(StepBuffers::new(max_bandwidth)),
            },
            RefineAlgo::Viterbi => Self {
                penalty_table: None,
                dwell_target: 0.0,
                dwell_weight: 0.0,
                viterbi_buf: ViterbiBuffers::new(max_bandwidth),
                step_buf: None,
            },
        }
    }

    /// Execute one DP step, dispatching to the correct implementation.
    pub(super) fn step(
        &mut self,
        current_scores: &mut [f32],
        current_traceback: &mut [i32],
        previous_scores: &[f32],
        current_level: f32,
        current_signal: &[f32],
        prev_band_offset: usize,
    ) {
        if let Some(ref table) = self.penalty_table {
            dp_step_with_dwell_penalty(
                current_scores,
                current_traceback,
                previous_scores,
                current_level,
                current_signal,
                prev_band_offset,
                table,
                self.dwell_target,
                self.dwell_weight,
                self.step_buf.as_mut().unwrap(),
            );
        } else {
            dp_step_buffered(
                current_scores,
                current_traceback,
                previous_scores,
                current_level,
                current_signal,
                prev_band_offset,
                &mut self.viterbi_buf,
            );
        }
    }
}

/// Squared error score between expected and measured signal levels.
#[inline]
pub fn score(expected: f32, measured: f32) -> f32 {
    let d = measured - expected;
    d * d
}

/// Perform banded DP to find optimal sequence-to-signal alignment.
///
/// Returns a vector of signal indices (length = levels.len() + 1) where
/// path\[i\] is the signal position where base i starts, and path\[last\]
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

    let mut ctx = DpContext::new(method, max_bandwidth);

    forward_pass(
        &mut all_scores,
        &mut traceback,
        signal,
        levels,
        band,
        &base_offsets,
        &mut ctx,
    );

    let mut path = vec![0usize; levels.len() + 1];
    banded_traceback(&mut path, band, &base_offsets, &traceback);

    path
}

#[cfg(test)]
#[allow(
    clippy::excessive_precision,
    clippy::useless_vec,
    clippy::needless_range_loop
)]
mod tests {
    use super::fill::DWELL_TABLE_SIZE;
    use super::*;

    fn round_to(value: f32, decimal_places: u32) -> f32 {
        let multiplier = 10f32.powi(decimal_places as i32);
        (value * multiplier).round() / multiplier
    }

    #[test]
    fn test_dwell_penalty_asymmetric() {
        let target = 36.0;
        let weight = 0.5;

        // Below target: quadratic penalty
        let p0 = dwell_penalty(0, target, weight);
        assert!(
            (p0 - weight * target * target).abs() < 1e-6,
            "dwell=0 should be weight*target^2"
        );

        let p10 = dwell_penalty(10, target, weight);
        let p20 = dwell_penalty(20, target, weight);
        assert!(
            p0 > p10 && p10 > p20,
            "penalty should decrease as dwell approaches target"
        );

        // At target: logarithmic penalty = weight * ln(2)
        let p_target = dwell_penalty(36, target, weight);
        assert!((p_target - weight * 2.0_f32.ln()).abs() < 1e-6);

        // Above target: logarithmic, modest penalty
        let p_2x = dwell_penalty(72, target, weight);
        assert!((p_2x - weight * 3.0_f32.ln()).abs() < 1e-6);

        // 50x target: penalty should still be modest (< 4 * weight)
        let p_50x = dwell_penalty(1800, target, weight);
        assert!(p_50x < 4.0 * weight, "50x dwell penalty should be modest");
        assert!((p_50x - weight * 51.0_f32.ln()).abs() < 1e-4);

        // Short-side penalty at target-5 should be much stronger than 50x long dwell
        let p_short = dwell_penalty(31, target, weight);
        assert!(
            p_short > p_50x,
            "short dwell penalty should exceed 50x long dwell penalty"
        );
    }

    #[test]
    fn test_dwell_penalty_table_matches_inline() {
        let target = 36.0;
        let weight = 0.5;
        let table = build_dwell_penalty_table(target, weight);
        assert_eq!(table.len(), DWELL_TABLE_SIZE);
        for (i, &val) in table.iter().enumerate() {
            let expected = dwell_penalty(i, target, weight);
            assert!((val - expected).abs() < 1e-6, "table[{}] mismatch", i);
        }
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
        let target = 4.0;
        let weight = 0.5;
        let penalty_table = build_dwell_penalty_table(target, weight);
        let mut buf = StepBuffers::new(band_width);

        dp_step_with_dwell_penalty(
            &mut scores[..band_width],
            &mut tb[..band_width],
            &prev_scores,
            level,
            &signal,
            1,
            &penalty_table,
            target,
            weight,
            &mut buf,
        );

        // Structural checks (exact values changed with asymmetric formula)
        // First position (dwell_idx=0, dwell=1): should include short-dwell penalty
        let short_dwell_pen = dwell_penalty(0, target, weight);
        let signal_score_0 = score(level, signal[0]);
        assert!(
            (scores[0] - (signal_score_0 + short_dwell_pen)).abs() < 1e-4,
            "scores[0] should be signal_score + short_dwell_penalty"
        );

        // All scores within band should be finite and positive
        for i in 0..band_width {
            assert!(scores[i].is_finite(), "scores[{}] should be finite", i);
            assert!(scores[i] >= 0.0, "scores[{}] should be non-negative", i);
        }

        // Scores should generally increase monotonically (more signal = more error)
        // but the last position should be the highest
        assert!(scores[band_width - 1] >= scores[0]);

        // Unused positions beyond band should remain 0
        for i in band_width..scores.len() {
            assert_eq!(scores[i], 0.0);
        }
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

    #[test]
    fn test_banded_dp_viterbi_end_to_end() {
        // 5 bases, each with exactly 10 signal samples at the expected level
        let n_bases = 5;
        let samples_per_base = 10;
        let signal_len = n_bases * samples_per_base;

        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0];
        let mut signal = vec![0.0f32; signal_len];
        for (i, &level) in levels.iter().enumerate() {
            for j in 0..samples_per_base {
                signal[i * samples_per_base + j] = level;
            }
        }

        // Build a simple sequence band with half_bandwidth=3
        let half_bw = 3;
        let start: Vec<usize> = (0..n_bases)
            .map(|i| (i * samples_per_base).saturating_sub(half_bw))
            .collect();
        let end: Vec<usize> = (0..n_bases)
            .map(|i| ((i + 1) * samples_per_base + half_bw).min(signal_len))
            .collect();
        let band = Band::new(start, end, true);

        let path = banded_dp(&signal, &levels, &band, &RefineAlgo::Viterbi);

        assert_eq!(path.len(), n_bases + 1);
        assert_eq!(path[0], 0);
        assert_eq!(path[n_bases], signal_len);

        // Path must be strictly increasing
        for w in path.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing: {:?}", path);
        }

        // With perfect signal, boundaries should be near the true boundaries
        for i in 0..=n_bases {
            let expected = i * samples_per_base;
            let diff = (path[i] as i64 - expected as i64).unsigned_abs() as usize;
            assert!(
                diff <= half_bw + 1,
                "path[{}]={} far from expected {}",
                i,
                path[i],
                expected
            );
        }
    }

    #[test]
    fn test_banded_dp_dwell_penalty_end_to_end() {
        // Same setup as Viterbi but with dwell penalty
        let n_bases = 5;
        let samples_per_base = 10;
        let signal_len = n_bases * samples_per_base;

        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0];
        let mut signal = vec![0.0f32; signal_len];
        for (i, &level) in levels.iter().enumerate() {
            for j in 0..samples_per_base {
                signal[i * samples_per_base + j] = level;
            }
        }

        let half_bw = 3;
        let start: Vec<usize> = (0..n_bases)
            .map(|i| (i * samples_per_base).saturating_sub(half_bw))
            .collect();
        let end: Vec<usize> = (0..n_bases)
            .map(|i| ((i + 1) * samples_per_base + half_bw).min(signal_len))
            .collect();
        let band = Band::new(start, end, true);

        let method = RefineAlgo::DwellPenalty {
            target: samples_per_base as f32,
            weight: 0.5,
        };
        let path = banded_dp(&signal, &levels, &band, &method);

        assert_eq!(path.len(), n_bases + 1);
        assert_eq!(path[0], 0);
        assert_eq!(path[n_bases], signal_len);

        for w in path.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing: {:?}", path);
        }

        // Dwell penalty should also find near-optimal boundaries
        for i in 0..=n_bases {
            let expected = i * samples_per_base;
            let diff = (path[i] as i64 - expected as i64).unsigned_abs() as usize;
            assert!(
                diff <= half_bw + 1,
                "path[{}]={} far from expected {}",
                i,
                path[i],
                expected
            );
        }
    }

    #[test]
    fn test_banded_dp_dwell_penalty_discourages_short_dwells() {
        // Construct a scenario where Viterbi would produce a short dwell
        // but dwell penalty should avoid it.
        //
        // 3 bases, 30 samples total. Signal: base0 at level 0, base1 at level 1,
        // base2 at level 0. The band is wide enough that Viterbi might assign
        // just 1 sample to base1 if the level transition is sharp.
        let signal_len = 30;
        let mut signal = vec![0.0f32; signal_len];
        // Base0: 0..10, Base1: 10..20, Base2: 20..30
        for i in 10..20 {
            signal[i] = 1.0;
        }
        let levels = vec![0.0, 1.0, 0.0];

        // Wide band to allow various alignments
        let band = Band::new(vec![0, 5, 15], vec![15, 25, 30], true);

        let viterbi_path = banded_dp(&signal, &levels, &band, &RefineAlgo::Viterbi);
        let dwell_path = banded_dp(
            &signal,
            &levels,
            &band,
            &RefineAlgo::DwellPenalty {
                target: 10.0,
                weight: 0.5,
            },
        );

        // Both should be valid paths
        assert_eq!(viterbi_path[0], 0);
        assert_eq!(dwell_path[0], 0);
        assert_eq!(viterbi_path[3], 30);
        assert_eq!(dwell_path[3], 30);

        // Dwell penalty path should have no base with extremely short dwell
        for w in dwell_path.windows(2) {
            let dwell = w[1] - w[0];
            assert!(
                dwell >= 2,
                "dwell penalty produced dwell of {} at {:?}",
                dwell,
                dwell_path
            );
        }
    }

    #[test]
    fn test_banded_dp_viterbi_vs_dwell_penalty_agree_on_clean_signal() {
        // With perfectly clean signal (each base exactly at expected level),
        // both algorithms should produce similar paths.
        let n_bases = 8;
        let spb = 10;
        let signal_len = n_bases * spb;

        let levels: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.3, -0.3, 0.8];
        let mut signal = vec![0.0f32; signal_len];
        for (i, &level) in levels.iter().enumerate() {
            for j in 0..spb {
                signal[i * spb + j] = level;
            }
        }

        let half_bw = 3;
        let start: Vec<usize> = (0..n_bases)
            .map(|i| (i * spb).saturating_sub(half_bw))
            .collect();
        let end: Vec<usize> = (0..n_bases)
            .map(|i| ((i + 1) * spb + half_bw).min(signal_len))
            .collect();
        let band = Band::new(start, end, true);

        let viterbi_path = banded_dp(&signal, &levels, &band, &RefineAlgo::Viterbi);
        let dwell_path = banded_dp(
            &signal,
            &levels,
            &band,
            &RefineAlgo::DwellPenalty {
                target: spb as f32,
                weight: 0.5,
            },
        );

        // Paths should agree within 2 samples at each boundary
        for i in 0..=n_bases {
            let diff = (viterbi_path[i] as i64 - dwell_path[i] as i64).unsigned_abs() as usize;
            assert!(
                diff <= 2,
                "paths diverge at boundary {}: viterbi={}, dwell={}",
                i,
                viterbi_path[i],
                dwell_path[i]
            );
        }
    }
}

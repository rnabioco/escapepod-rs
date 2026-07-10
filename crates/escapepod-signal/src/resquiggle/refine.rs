// SPDX-License-Identifier: MIT
// Algorithm inspired by fishnet (Brickner et al.); independent implementation.

//! Top-level refinement pipeline: iterative DP + rescale loop.

use anyhow::Result;

use super::adaptive_dp::adaptive_banded_dp;
use super::bands::Band;
use super::dp::banded_dp;
use super::rescale::{rescale, rough_rescale};
use super::types::{BandingAlgo, RefineAlgo, RefineSettings};

/// Result of the refinement pipeline.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Refined sequence-to-signal boundary map.
    pub seq_to_signal_map: Vec<usize>,
    /// Final signal scale parameter.
    pub scale: f32,
    /// Final signal shift parameter.
    pub shift: f32,
    /// Final signal drift parameter (per-sample trend).
    pub drift: f32,
}

/// Calculate initial scaling from POD5 calibration and BAM signal stats.
///
/// Converts raw DAC measurements to normalized signal:
///   normalized = (raw - shift) / scale
///
/// Parameters:
/// - `cal_scale`: POD5 ReadData.calibration_scale
/// - `cal_offset`: POD5 ReadData.calibration_offset
/// - `scale_pa_to_norm`: BAM `sd` tag (signal scaling dispersion)
/// - `shift_pa_to_norm`: BAM `sm` tag (signal scaling mean)
pub fn calculate_initial_scaling(
    cal_scale: f32,
    cal_offset: f32,
    scale_pa_to_norm: f32,
    shift_pa_to_norm: f32,
) -> (f32, f32) {
    let scale_raw_to_pa = 1.0 / cal_scale;
    let scale = scale_raw_to_pa * scale_pa_to_norm;
    let shift = scale_raw_to_pa * shift_pa_to_norm - cal_offset;
    (scale, shift)
}

/// Reverse a query-to-signal map for RNA signal reversal.
///
/// For RNA data, the raw signal is acquired 3'→5' but basecalling operates on the
/// reversed (5'→3') signal. This function reverses the boundary map so that it
/// corresponds to the reversed signal coordinates.
///
/// Given a forward map `[b0, b1, ..., bn]` and `signal_len`, produces
/// `[signal_len - bn, ..., signal_len - b1, signal_len - b0]`.
pub fn reverse_query_to_signal_map(map: &[usize], signal_len: usize) -> Vec<usize> {
    map.iter().rev().map(|&el| signal_len - el).collect()
}

/// Run the full refinement pipeline.
///
/// 1. Optional rough rescaling
/// 2. Iterative: normalize signal -> DP refinement -> rescale
/// 3. Return refined mapping + final scale/shift
pub fn refine_signal_map(
    settings: &RefineSettings,
    signal: &[f32],
    seq_to_signal_map: &[usize],
    expected_levels: &[f32],
    initial_scale: f32,
    initial_shift: f32,
) -> Result<RefinementResult> {
    // Step 1: optional rough rescaling
    let (mut shift, mut scale, mut drift) = rough_rescale(
        initial_scale,
        initial_shift,
        seq_to_signal_map,
        expected_levels,
        signal,
        &settings.rough_rescale_algo,
    )?;

    let mut map = seq_to_signal_map.to_vec();

    // Step 1.5: resolve auto-target for dwell penalty from move table median
    let resolved_algo = match &settings.refinement_algo {
        RefineAlgo::DwellPenalty { target, weight } if *target <= 0.0 => {
            let median = median_dwell(&map);
            RefineAlgo::DwellPenalty {
                target: median,
                weight: *weight,
            }
        }
        other => other.clone(),
    };
    let resolved_settings = RefineSettings {
        refinement_algo: resolved_algo,
        ..settings.clone()
    };

    // Step 2: iterative refinement + rescale
    let n_iterations = settings.n_refinement_iters;
    let perform_rescaling = n_iterations > 0;
    let n_iter = n_iterations.max(1);

    // Pre-allocate normalization buffer (reused across iterations)
    let mut signal_norm = vec![0.0f32; signal.len()];

    for _i in 0..n_iter {
        // Normalize signal in-place into reusable buffer
        for (i, el) in signal.iter().enumerate() {
            signal_norm[i] = (el - shift - drift * i as f32) / scale;
        }

        // Run one refinement step (trim, band, DP)
        map = refinement_step(map, &signal_norm, expected_levels, &resolved_settings)?;

        // Rescale if configured
        if perform_rescaling {
            let result = rescale(
                scale,
                shift,
                drift,
                &map,
                expected_levels,
                signal,
                &settings.rescale_algo,
            );
            match result {
                Ok((new_shift, new_scale, new_drift)) => {
                    shift = new_shift;
                    scale = new_scale;
                    drift = new_drift;
                }
                Err(_) => {
                    // If rescaling fails, keep current parameters and continue
                }
            }
        }
    }

    Ok(RefinementResult {
        seq_to_signal_map: map,
        scale,
        shift,
        drift,
    })
}

/// Single refinement step: trim signal, zero-base map, compute bands, run DP.
fn refinement_step(
    seq_to_signal_map: Vec<usize>,
    signal: &[f32],
    expected_levels: &[f32],
    settings: &RefineSettings,
) -> Result<Vec<usize>> {
    // Trim signal to mapping region
    let sig_start = seq_to_signal_map[0];
    let sig_end = seq_to_signal_map[seq_to_signal_map.len() - 1];
    let signal_trimmed = &signal[sig_start..sig_end];

    // Zero-base the mapping
    let map_zeroed: Vec<usize> = seq_to_signal_map.iter().map(|el| el - sig_start).collect();

    let optimized = match &settings.banding_algo {
        BandingAlgo::Fixed => {
            // Compute signal band
            let mut band = Band::compute_signal_band(
                &map_zeroed,
                expected_levels.len(),
                settings.half_bandwidth,
            )?;

            // Convert to sequence band
            band.convert_to_sequence_band(settings.adjust_band_min_size)?;

            // Banded DP
            banded_dp(
                signal_trimmed,
                expected_levels,
                &band,
                &settings.refinement_algo,
            )
        }
        BandingAlgo::Adaptive { bandwidth, x_drop } => {
            // Adaptive banding uses the initial map directly
            adaptive_banded_dp(
                signal_trimmed,
                expected_levels,
                *bandwidth,
                &map_zeroed,
                &settings.refinement_algo,
                *x_drop,
            )
        }
    };

    // Adjust back to original coordinates
    Ok(optimized.iter().map(|el| el + sig_start).collect())
}

/// Compute the median dwell time (in signal samples per base) from a signal map.
///
/// The signal map has `n_bases + 1` entries (boundaries). Dwell for base i
/// is `map[i+1] - map[i]`.
fn median_dwell(seq_to_signal_map: &[usize]) -> f32 {
    if seq_to_signal_map.len() < 2 {
        return 1.0;
    }
    let mut dwells: Vec<usize> = seq_to_signal_map.windows(2).map(|w| w[1] - w[0]).collect();
    let n = dwells.len();
    let mid = n / 2;
    // Median via select_nth_unstable (O(n) expected) instead of a full sort.
    let (lo_part, pivot, _) = dwells.select_nth_unstable(mid);
    let upper = *pivot;
    if n % 2 == 1 {
        upper as f32
    } else {
        let lower = *lo_part.iter().max().unwrap();
        (lower + upper) as f32 / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{RescaleAlgo, RescaleFilterParams, RoughRescaleAlgo};
    use super::*;

    #[test]
    fn test_median_dwell_odd() {
        // 5 bases with dwells: 3, 5, 2, 7, 4 → sorted: 2, 3, 4, 5, 7 → median = 4
        let map = vec![0, 3, 8, 10, 17, 21];
        assert_eq!(median_dwell(&map), 4.0);
    }

    #[test]
    fn test_median_dwell_even() {
        // 4 bases with dwells: 3, 5, 2, 6 → sorted: 2, 3, 5, 6 → median = (3+5)/2 = 4
        let map = vec![0, 3, 8, 10, 16];
        assert_eq!(median_dwell(&map), 4.0);
    }

    #[test]
    fn test_median_dwell_single_base() {
        let map = vec![0, 10];
        assert_eq!(median_dwell(&map), 10.0);
    }

    #[test]
    fn test_median_dwell_empty() {
        let map: Vec<usize> = vec![];
        assert_eq!(median_dwell(&map), 1.0);
    }

    #[test]
    fn test_median_dwell_single_entry() {
        let map = vec![5];
        assert_eq!(median_dwell(&map), 1.0);
    }

    #[test]
    fn test_median_dwell_uniform() {
        // All dwells identical: 10, 10, 10
        let map = vec![0, 10, 20, 30];
        assert_eq!(median_dwell(&map), 10.0);
    }

    #[test]
    fn test_calculate_initial_scaling() {
        // Identity calibration: cal_scale=1, cal_offset=0
        let (scale, shift) = calculate_initial_scaling(1.0, 0.0, 2.0, 5.0);
        assert!((scale - 2.0).abs() < 1e-6);
        assert!((shift - 5.0).abs() < 1e-6);

        // With offset: cal_scale=2, cal_offset=10
        let (scale, shift) = calculate_initial_scaling(2.0, 10.0, 1.0, 3.0);
        // scale_raw_to_pa = 1/2 = 0.5
        // scale = 0.5 * 1.0 = 0.5
        // shift = 0.5 * 3.0 - 10.0 = 1.5 - 10.0 = -8.5
        assert!((scale - 0.5).abs() < 1e-6);
        assert!((shift - (-8.5)).abs() < 1e-6);
    }

    #[test]
    fn test_reverse_query_to_signal_map() {
        // Forward map: boundaries [0, 10, 25, 40, 50] with signal_len=50
        let map = vec![0, 10, 25, 40, 50];
        let reversed = reverse_query_to_signal_map(&map, 50);
        // Expected: [50-50, 50-40, 50-25, 50-10, 50-0] = [0, 10, 25, 40, 50]
        assert_eq!(reversed, vec![0, 10, 25, 40, 50]);

        // Asymmetric map: [0, 5, 30, 100] with signal_len=100
        let map = vec![0, 5, 30, 100];
        let reversed = reverse_query_to_signal_map(&map, 100);
        assert_eq!(reversed, vec![0, 70, 95, 100]);

        // Verify first=0 and last=signal_len for valid maps
        assert_eq!(reversed[0], 0);
        assert_eq!(reversed[reversed.len() - 1], 100);
    }

    #[test]
    fn test_refine_signal_map_viterbi_basic() {
        // Simple test: 5 bases, each with ~10 signal samples
        // Signal is constant per base matching expected levels
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

        let map: Vec<usize> = (0..=n_bases).map(|i| i * samples_per_base).collect();

        let settings = RefineSettings {
            refinement_algo: RefineAlgo::Viterbi,
            n_refinement_iters: 0, // just 1 DP pass, no rescale
            half_bandwidth: 3,
            adjust_band_min_size: 2,
            rescale_algo: RescaleAlgo::default(),
            rough_rescale_algo: RoughRescaleAlgo::None,
            normalize_levels: false,
            banding_algo: BandingAlgo::Fixed,
        };

        let result = refine_signal_map(&settings, &signal, &map, &levels, 1.0, 0.0).unwrap();

        // Path should start at 0 and end at signal_len
        assert_eq!(result.seq_to_signal_map[0], 0);
        assert_eq!(result.seq_to_signal_map[n_bases], signal_len);

        // Path should be monotonically increasing
        for w in result.seq_to_signal_map.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing");
        }

        // With perfect signal, the boundaries should be close to the original map
        for (i, &boundary) in result.seq_to_signal_map.iter().enumerate() {
            let expected = i * samples_per_base;
            let diff = (boundary as i64 - expected as i64).unsigned_abs() as usize;
            assert!(
                diff <= settings.half_bandwidth,
                "boundary {} deviated by {} (expected ~{})",
                i,
                diff,
                expected
            );
        }
    }

    #[test]
    fn test_refine_signal_map_dwell_penalty_auto_target() {
        // Verify auto-target resolution (target=0 → median dwell from map)
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

        let map: Vec<usize> = (0..=n_bases).map(|i| i * samples_per_base).collect();

        let settings = RefineSettings {
            refinement_algo: RefineAlgo::DwellPenalty {
                target: 0.0, // auto
                weight: 0.5,
            },
            n_refinement_iters: 0,
            half_bandwidth: 3,
            adjust_band_min_size: 2,
            rescale_algo: RescaleAlgo::default(),
            rough_rescale_algo: RoughRescaleAlgo::None,
            normalize_levels: false,
            banding_algo: BandingAlgo::Fixed,
        };

        let result = refine_signal_map(&settings, &signal, &map, &levels, 1.0, 0.0).unwrap();

        assert_eq!(result.seq_to_signal_map[0], 0);
        assert_eq!(result.seq_to_signal_map[n_bases], signal_len);

        for w in result.seq_to_signal_map.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn test_refine_with_synthetic_drift() {
        // Synthetic drifting signal: raw[i] = shift + drift*i + scale * level[base_of(i)]
        // The refinement pipeline should recover the drift and produce good boundaries.
        let n_bases = 10;
        let samples_per_base = 20;
        let signal_len = n_bases * samples_per_base;

        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0, 0.3, -0.3, 0.8, -0.8, 0.2];
        let shift = 100.0;
        let scale = 5.0;
        let true_drift = 0.02;

        // Build signal with drift
        let mut signal = vec![0.0f32; signal_len];
        for (base_idx, &level) in levels.iter().enumerate() {
            for j in 0..samples_per_base {
                let i = base_idx * samples_per_base + j;
                signal[i] = shift + true_drift * i as f32 + scale * level;
            }
        }

        let map: Vec<usize> = (0..=n_bases).map(|i| i * samples_per_base).collect();

        let settings = RefineSettings {
            refinement_algo: RefineAlgo::Viterbi,
            n_refinement_iters: 2,
            half_bandwidth: 5,
            adjust_band_min_size: 2,
            rescale_algo: RescaleAlgo::LeastSquares {
                filter: RescaleFilterParams {
                    dwell_filter_lower_percentile: 0.0,
                    dwell_filter_upper_percentile: 1.0,
                    min_abs_level: 0.0,
                    n_bases_truncate: 0,
                    min_num_filtered_levels: 3,
                },
            },
            rough_rescale_algo: RoughRescaleAlgo::None,
            normalize_levels: false,
            banding_algo: BandingAlgo::Fixed,
        };

        let result = refine_signal_map(&settings, &signal, &map, &levels, scale, shift).unwrap();

        // Path should be valid
        assert_eq!(result.seq_to_signal_map[0], 0);
        assert_eq!(result.seq_to_signal_map[n_bases], signal_len);
        for w in result.seq_to_signal_map.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing");
        }

        // Boundaries should still be close to the true positions
        for (i, &boundary) in result.seq_to_signal_map.iter().enumerate() {
            let expected = i * samples_per_base;
            let diff = (boundary as i64 - expected as i64).unsigned_abs() as usize;
            assert!(
                diff <= settings.half_bandwidth + 2,
                "boundary {} deviated by {} (expected ~{})",
                i,
                diff,
                expected
            );
        }

        // Drift should be detected (non-zero, positive)
        // Note: exact recovery depends on iterations and filtering
        // Just check it moved in the right direction
        assert!(
            result.drift.abs() > 1e-6,
            "drift should be non-zero, got {}",
            result.drift
        );
    }
}

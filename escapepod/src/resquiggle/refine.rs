//! Top-level refinement pipeline: iterative DP + rescale loop.

use anyhow::Result;

use super::bands::Band;
use super::dp::banded_dp;
use super::rescale::{rescale, rough_rescale};
use super::types::RefineSettings;

/// Result of the refinement pipeline.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Refined sequence-to-signal boundary map.
    pub seq_to_signal_map: Vec<usize>,
    /// Final signal scale parameter.
    pub scale: f32,
    /// Final signal shift parameter.
    pub shift: f32,
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
    let (mut shift, mut scale) = rough_rescale(
        initial_scale,
        initial_shift,
        seq_to_signal_map,
        expected_levels,
        signal,
        &settings.rough_rescale_algo,
    )?;

    let mut map = seq_to_signal_map.to_vec();

    // Step 2: iterative refinement + rescale
    let n_iterations = settings.n_refinement_iters;
    let perform_rescaling = n_iterations > 0;
    let n_iter = n_iterations.max(1);

    // Pre-allocate normalization buffer (reused across iterations)
    let mut signal_norm = vec![0.0f32; signal.len()];

    for _i in 0..n_iter {
        // Normalize signal in-place into reusable buffer
        for (i, el) in signal.iter().enumerate() {
            signal_norm[i] = (el - shift) / scale;
        }

        // Run one refinement step (trim, band, DP)
        map = refinement_step(map, &signal_norm, expected_levels, settings)?;

        // Rescale if configured
        if perform_rescaling {
            let result = rescale(
                scale,
                shift,
                &map,
                expected_levels,
                signal,
                &settings.rescale_algo,
            );
            match result {
                Ok((new_shift, new_scale)) => {
                    shift = new_shift;
                    scale = new_scale;
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

    // Compute signal band
    let mut band =
        Band::compute_signal_band(&map_zeroed, expected_levels.len(), settings.half_bandwidth)?;

    // Convert to sequence band
    band.convert_to_sequence_band(settings.adjust_band_min_size)?;

    // Banded DP
    let optimized = banded_dp(
        signal_trimmed,
        expected_levels,
        &band,
        &settings.refinement_algo,
    );

    // Adjust back to original coordinates
    Ok(optimized.iter().map(|el| el + sig_start).collect())
}

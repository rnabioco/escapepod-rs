//! Signal rescaling algorithms (least squares and Theil-Sen).

use anyhow::{bail, Result};
use rand::seq::IteratorRandom;

use super::types::{RescaleAlgo, RoughRescaleAlgo};

/// Rough rescale using the specified algorithm.
pub fn rough_rescale(
    scale: f32,
    shift: f32,
    seq_to_signal_map: &[usize],
    levels: &[f32],
    signal: &[f32],
    algo: &RoughRescaleAlgo,
) -> Result<(f32, f32)> {
    let (quantiles, clip_bases, use_base_center) = match algo {
        RoughRescaleAlgo::None => return Ok((shift, scale)),
        RoughRescaleAlgo::LeastSquares {
            quantiles,
            clip_bases,
            use_base_center,
        } => (quantiles.as_slice(), *clip_bases, *use_base_center),
        RoughRescaleAlgo::TheilSen {
            quantiles,
            clip_bases,
            use_base_center,
        } => (quantiles.as_slice(), *clip_bases, *use_base_center),
    };

    // Compute clipping bounds
    let (clip_start, clip_end) = if clip_bases > 0 && levels.len() > clip_bases * 2 {
        (clip_bases, levels.len() - clip_bases)
    } else {
        (0, signal.len())
    };

    // Compute normalized signal values
    let norm_signal = if use_base_center {
        seq_to_signal_map
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2)
            .filter(|&idx| idx < signal.len())
            .map(|idx| (signal[idx] - shift) / scale)
            .skip(clip_start)
            .take(clip_end - clip_start)
            .collect::<Vec<f32>>()
    } else if !seq_to_signal_map.is_empty() {
        let start = seq_to_signal_map[0];
        let end = seq_to_signal_map[seq_to_signal_map.len() - 1].min(signal.len());
        signal[start..end]
            .iter()
            .map(|&val| (val - shift) / scale)
            .skip(clip_start)
            .take(clip_end - clip_start)
            .collect::<Vec<f32>>()
    } else {
        bail!("empty seq_to_signal_map");
    };

    let clipped_levels = &levels[clip_start.min(levels.len())..clip_end.min(levels.len())];

    let norm_signal_quantiles = calculate_quantiles(&norm_signal, quantiles)?;
    let level_quantiles = calculate_quantiles(clipped_levels, quantiles)?;

    match algo {
        RoughRescaleAlgo::LeastSquares { .. } => {
            least_squares(&norm_signal_quantiles, &level_quantiles, shift, scale)
        }
        RoughRescaleAlgo::TheilSen { .. } => {
            // max_points=0 to prevent subsetting (only a handful of quantile values)
            theil_sen(&norm_signal_quantiles, &level_quantiles, shift, scale, 0)
        }
        RoughRescaleAlgo::None => unreachable!(),
    }
}

/// Calculate quantile values from data.
fn calculate_quantiles(data: &[f32], quantiles: &[f32]) -> Result<Vec<f32>> {
    if data.is_empty() {
        bail!("empty data for quantile calculation");
    }
    if quantiles.is_empty() {
        bail!("empty quantiles vector");
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    quantiles
        .iter()
        .map(|&q| {
            if !(0.0..=1.0).contains(&q) {
                bail!("invalid quantile value: {}", q);
            }
            let pos = q * (sorted.len() - 1) as f32;
            let idx_floor = pos.floor() as usize;
            let idx_ceil = pos.ceil() as usize;
            if idx_floor == idx_ceil {
                Ok(sorted[idx_floor])
            } else {
                let w = pos - idx_floor as f32;
                Ok((1.0 - w) * sorted[idx_floor] + w * sorted[idx_ceil])
            }
        })
        .collect()
}

/// Least-squares linear regression: y = shift_est + scale_est * x.
/// Returns updated (shift, scale) parameters.
fn least_squares(x: &[f32], y: &[f32], shift: f32, scale: f32) -> Result<(f32, f32)> {
    if x.len() != y.len() {
        bail!("least_squares: length mismatch {} vs {}", x.len(), y.len());
    }
    let n = x.len();
    let x_mean: f32 = x.iter().sum::<f32>() / n as f32;
    let y_mean: f32 = y.iter().sum::<f32>() / n as f32;

    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;

    for i in 0..n {
        let xd = x[i] - x_mean;
        numerator += xd * (y[i] - y_mean);
        denominator += xd * xd;
    }

    let scale_est = if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        numerator / denominator
    };

    if scale_est == 0.0 {
        return Ok((shift, scale));
    }

    let shift_est = y_mean - scale_est * x_mean;
    let new_shift = shift - (scale * shift_est / scale_est);
    let new_scale = scale / scale_est;

    Ok((new_shift, new_scale))
}

/// Theil-Sen robust regression.
/// Returns updated (shift, scale) parameters.
fn theil_sen(
    x: &[f32],
    y: &[f32],
    shift: f32,
    scale: f32,
    max_points: usize,
) -> Result<(f32, f32)> {
    if x.len() != y.len() {
        bail!("theil_sen: length mismatch {} vs {}", x.len(), y.len());
    }
    let n = x.len();

    let mut slopes = Vec::new();

    if max_points > 0 && n > max_points {
        let indices = random_subset(n, max_points);
        for i in 0..max_points {
            let xi = x[indices[i]];
            let yi = y[indices[i]];
            for j in (i + 1)..max_points {
                let dx = x[indices[j]] - xi;
                if dx != 0.0 {
                    slopes.push((y[indices[j]] - yi) / dx);
                }
            }
        }
    } else {
        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            for j in (i + 1)..n {
                let dx = x[j] - xi;
                if dx != 0.0 {
                    slopes.push((y[j] - yi) / dx);
                }
            }
        }
    }

    if slopes.is_empty() {
        bail!("theil_sen: all slopes are zero");
    }

    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_slope = median_sorted(&slopes)?;

    if median_slope == 0.0 {
        bail!("theil_sen: median slope is zero");
    }

    let mut intercepts: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| yi - median_slope * xi)
        .collect();
    intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_intercept = median_sorted(&intercepts)?;

    let shift_est = -median_intercept / median_slope;
    let scale_est = 1.0 / median_slope;

    let new_shift = shift + (shift_est * scale);
    let new_scale = scale * scale_est;

    Ok((new_shift, new_scale))
}

/// Random subset of indices.
fn random_subset(vec_len: usize, downsampled_len: usize) -> Vec<usize> {
    (0..vec_len).choose_multiple(&mut rand::thread_rng(), downsampled_len)
}

/// Median of a sorted slice.
fn median_sorted(sorted: &[f32]) -> Result<f32> {
    if sorted.is_empty() {
        bail!("median of empty slice");
    }
    let len = sorted.len();
    Ok(if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    })
}

/// Precise rescale using filtered base-level statistics.
pub fn rescale(
    scale: f32,
    shift: f32,
    seq_to_signal_map: &[usize],
    levels: &[f32],
    signal: &[f32],
    rescale_algo: &RescaleAlgo,
) -> Result<(f32, f32)> {
    let map_len = seq_to_signal_map.len();
    if map_len == 0 {
        bail!("rescale: empty seq_to_signal_map");
    }
    if levels.len() != map_len - 1 {
        bail!(
            "rescale: levels length {} != map_len - 1 ({})",
            levels.len(),
            map_len - 1
        );
    }

    let (dwell_lower, dwell_upper, min_abs_level, n_trunc, min_filtered, max_points) =
        match *rescale_algo {
            RescaleAlgo::TheilSen {
                dwell_filter_lower_percentile,
                dwell_filter_upper_percentile,
                min_abs_level,
                n_bases_truncate,
                min_num_filtered_levels,
                max_points,
            } => (
                dwell_filter_lower_percentile,
                dwell_filter_upper_percentile,
                min_abs_level,
                n_bases_truncate,
                min_num_filtered_levels,
                max_points,
            ),
            RescaleAlgo::LeastSquares {
                dwell_filter_lower_percentile,
                dwell_filter_upper_percentile,
                min_abs_level,
                n_bases_truncate,
                min_num_filtered_levels,
            } => (
                dwell_filter_lower_percentile,
                dwell_filter_upper_percentile,
                min_abs_level,
                n_bases_truncate,
                min_num_filtered_levels,
                0,
            ),
        };

    // Calculate dwell times
    let dwells: Vec<f32> = seq_to_signal_map
        .windows(2)
        .map(|w| (w[1] - w[0]) as f32)
        .collect();

    let n_bases = dwells.len();
    if n_bases < min_filtered {
        bail!(
            "rescale: too few bases ({}) for min_filtered ({})",
            n_bases,
            min_filtered
        );
    }
    if 2 * n_trunc > n_bases {
        bail!(
            "rescale: too few bases ({}) for truncation ({})",
            n_bases,
            n_trunc
        );
    }
    if n_bases - 2 * n_trunc < min_filtered {
        bail!(
            "rescale: too few bases after truncation ({}) for min_filtered ({})",
            n_bases - 2 * n_trunc,
            min_filtered
        );
    }

    // Dwell quantiles for filtering
    let (dwell_lower_val, dwell_upper_val) = {
        let mut sorted_dwells = dwells.clone();
        sorted_dwells.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let low = calculate_quantiles(&sorted_dwells, &[dwell_lower])?[0];
        let high = calculate_quantiles(&sorted_dwells, &[dwell_upper])?[0];
        (low, high)
    };

    let levels_mean = levels.iter().sum::<f32>() / levels.len() as f32;

    let (start_idx, end_idx) = if n_trunc == 0 {
        (0, n_bases)
    } else {
        (n_trunc, n_bases - n_trunc)
    };

    let mut mean_signal_filtered = Vec::new();
    let mut levels_filtered = Vec::new();

    for base_idx in start_idx..end_idx {
        let dwell = dwells[base_idx];
        if dwell <= dwell_lower_val || dwell >= dwell_upper_val {
            continue;
        }

        let expected = levels[base_idx];
        if (expected - levels_mean).abs() <= min_abs_level {
            continue;
        }

        let mean_sig = signal[seq_to_signal_map[base_idx]..seq_to_signal_map[base_idx + 1]]
            .iter()
            .sum::<f32>()
            / dwell;
        mean_signal_filtered.push(mean_sig);
        levels_filtered.push(expected);
    }

    if mean_signal_filtered.len() < min_filtered {
        bail!(
            "rescale: too few bases passed filtering ({}) for min_filtered ({})",
            mean_signal_filtered.len(),
            min_filtered
        );
    }

    let norm_signal: Vec<f32> = mean_signal_filtered
        .iter()
        .map(|el| (el - shift) / scale)
        .collect();

    match rescale_algo {
        RescaleAlgo::TheilSen { .. } => {
            theil_sen(&norm_signal, &levels_filtered, shift, scale, max_points)
        }
        RescaleAlgo::LeastSquares { .. } => {
            least_squares(&norm_signal, &levels_filtered, shift, scale)
        }
    }
}

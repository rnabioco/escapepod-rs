//! Signal normalization utilities for nanopore signal processing.
//!
//! Provides MAD (Median Absolute Deviation) normalization, dwell time normalization,
//! and downscaling operations.

/// Compute the median of a sorted slice of values.
///
/// # Panics
/// Panics if the slice is empty.
#[inline]
fn median_sorted(sorted: &[f32]) -> f32 {
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Compute median and MAD together with only 2 sorts (the minimum required).
///
/// Returns (median, MAD) tuple.
///
/// # Panics
/// Panics if the slice is empty.
fn median_and_mad(data: &[f32]) -> (f32, f32) {
    // Sort once to get median
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = median_sorted(&sorted);

    // Compute absolute deviations and sort for MAD
    let mut abs_devs: Vec<f32> = data.iter().map(|&x| (x - med).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad_val = median_sorted(&abs_devs);

    (med, mad_val)
}

/// Normalize signal using MAD (Median Absolute Deviation).
///
/// Transforms the signal to: `(signal - median) / MAD`
///
/// This is a robust normalization method that is less sensitive to outliers
/// than standard z-score normalization.
///
/// # Arguments
/// * `signal` - The raw signal values to normalize
///
/// # Returns
/// A new vector containing the normalized signal values.
///
/// # Panics
/// Panics if the signal is empty or if MAD is zero.
///
/// # Example
/// ```
/// use escapepod::segmentation::mad_normalize;
///
/// let signal = vec![100.0, 102.0, 98.0, 101.0, 99.0];
/// let normalized = mad_normalize(&signal);
/// ```
pub fn mad_normalize(signal: &[f32]) -> Vec<f32> {
    let (med, mad_val) = median_and_mad(signal);

    if mad_val == 0.0 {
        panic!("MAD is zero - cannot normalize signal with no variation");
    }

    signal.iter().map(|&x| (x - med) / mad_val).collect()
}

/// Normalize signal using MAD with optional outlier clipping.
///
/// Clips values beyond `clip_sigma` MAD units from the median before normalizing.
///
/// # Arguments
/// * `signal` - The raw signal values to normalize
/// * `clip_sigma` - Number of MAD units to clip at (e.g., 5.0)
///
/// # Returns
/// A new vector containing the normalized and clipped signal values.
///
/// # Panics
/// Panics if the signal is empty or if MAD is zero.
pub fn mad_normalize_with_clipping(signal: &[f32], clip_sigma: f32) -> Vec<f32> {
    let (med, mad_val) = median_and_mad(signal);

    if mad_val == 0.0 {
        panic!("MAD is zero - cannot normalize signal with no variation");
    }

    let lower_bound = med - clip_sigma * mad_val;
    let upper_bound = med + clip_sigma * mad_val;

    signal
        .iter()
        .map(|&x| {
            let clipped = x.max(lower_bound).min(upper_bound);
            (clipped - med) / mad_val
        })
        .collect()
}

/// Clip signal outliers using median ± threshold × MAD, without normalizing.
///
/// This matches WarpDemuX's outlier clipping behavior where extreme values are
/// clamped before segmentation. The signal values are clipped but not rescaled.
///
/// # Arguments
/// * `signal` - The raw signal values to clip
/// * `clip_sigma` - Number of MADs from the median for clipping bounds
///
/// # Returns
/// A new vector with outlier values clipped. Returns the original signal
/// unchanged if MAD is zero (constant signal).
pub fn clip_outliers(signal: &[f32], clip_sigma: f32) -> Vec<f32> {
    if signal.len() < 2 {
        return signal.to_vec();
    }

    let (med, mad_val) = median_and_mad(signal);

    if mad_val == 0.0 {
        return signal.to_vec();
    }

    let lower_bound = med - clip_sigma * mad_val;
    let upper_bound = med + clip_sigma * mad_val;

    signal
        .iter()
        .map(|&x| x.max(lower_bound).min(upper_bound))
        .collect()
}

/// Normalize dwell times using log-transform followed by z-score normalization.
///
/// Dwell times are inherently right-skewed (most are short, few are long),
/// so log transformation before normalization is recommended. This creates
/// a more Gaussian-like distribution suitable for distance-based classification.
///
/// The transformation is: `(log(dwell) - mean(log(dwell))) / std(log(dwell))`
///
/// # Arguments
/// * `dwell_times` - Raw dwell times in samples
///
/// # Returns
/// A new vector containing the normalized dwell times.
/// Returns empty vector if input is empty.
/// If all dwells are identical, returns zeros.
///
/// # Example
/// ```
/// use escapepod::segmentation::normalize_dwell_times;
///
/// let dwells = vec![30.0, 45.0, 32.0, 100.0, 28.0];
/// let normalized = normalize_dwell_times(&dwells);
/// ```
pub fn normalize_dwell_times(dwell_times: &[f32]) -> Vec<f32> {
    if dwell_times.is_empty() {
        return Vec::new();
    }

    // Log transform (add small epsilon to avoid log(0))
    let log_dwells: Vec<f32> = dwell_times.iter().map(|&d| (d.max(1.0)).ln()).collect();

    // Compute mean and std of log-transformed values
    let n = log_dwells.len() as f32;
    let mean = log_dwells.iter().sum::<f32>() / n;
    let variance = log_dwells.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    if std < 1e-6 {
        // All dwells are essentially identical
        return vec![0.0; dwell_times.len()];
    }

    // Z-score normalize
    log_dwells.iter().map(|&x| (x - mean) / std).collect()
}

/// Normalize dwell times using MAD (robust to outliers).
///
/// Uses log-transform followed by MAD normalization instead of z-score.
/// This is more robust to extreme dwell time outliers.
///
/// # Arguments
/// * `dwell_times` - Raw dwell times in samples
///
/// # Returns
/// A new vector containing the normalized dwell times.
/// Returns empty vector if input is empty.
/// If MAD is zero, returns zeros.
///
/// # Example
/// ```
/// use escapepod::segmentation::normalize_dwell_times_mad;
///
/// let dwells = vec![30.0, 45.0, 32.0, 500.0, 28.0];  // 500 is outlier
/// let normalized = normalize_dwell_times_mad(&dwells);
/// ```
pub fn normalize_dwell_times_mad(dwell_times: &[f32]) -> Vec<f32> {
    if dwell_times.is_empty() {
        return Vec::new();
    }

    // Log transform (add small epsilon to avoid log(0))
    let log_dwells: Vec<f32> = dwell_times.iter().map(|&d| (d.max(1.0)).ln()).collect();

    // Compute median and MAD
    let (med, mad_val) = median_and_mad(&log_dwells);

    if mad_val < 1e-6 {
        // All dwells are essentially identical
        return vec![0.0; dwell_times.len()];
    }

    // MAD normalize
    log_dwells.iter().map(|&x| (x - med) / mad_val).collect()
}

/// Downscale signal by averaging consecutive samples.
///
/// Reduces the signal length by a factor using mean pooling.
/// Each output sample is the average of `factor` consecutive input samples.
///
/// # Arguments
/// * `signal` - The signal to downscale
/// * `factor` - The downscaling factor (e.g., 2 for halving the length)
///
/// # Returns
/// A new vector containing the downscaled signal. The length will be
/// `signal.len() / factor`, with any remaining samples averaged into the last element.
///
/// # Panics
/// Panics if factor is 0.
///
/// # Example
/// ```
/// use escapepod::segmentation::downscale;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let downscaled = downscale(&signal, 2);
/// assert_eq!(downscaled, vec![1.5, 3.5, 5.5]);
/// ```
pub fn downscale(signal: &[f32], factor: usize) -> Vec<f32> {
    if factor == 0 {
        panic!("Downscaling factor must be greater than 0");
    }

    if factor == 1 {
        return signal.to_vec();
    }

    let mut result = Vec::with_capacity(signal.len() / factor + 1);

    for chunk in signal.chunks(factor) {
        let sum: f32 = chunk.iter().sum();
        result.push(sum / chunk.len() as f32);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_odd_length() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let (med, _) = median_and_mad(&data);
        assert_eq!(med, 3.0);
    }

    #[test]
    fn test_median_even_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (med, _) = median_and_mad(&data);
        assert_eq!(med, 2.5);
    }

    #[test]
    fn test_mad_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, mad_val) = median_and_mad(&data);
        // median is 3.0, deviations are [2.0, 1.0, 0.0, 1.0, 2.0], median of abs devs is 1.0
        assert_eq!(mad_val, 1.0);
    }

    #[test]
    fn test_mad_normalize() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = mad_normalize(&signal);

        // median = 3.0, MAD = 1.0
        // expected: [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert_eq!(normalized.len(), 5);
        assert!((normalized[0] - (-2.0)).abs() < 1e-6);
        assert!((normalized[2] - 0.0).abs() < 1e-6);
        assert!((normalized[4] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mad_normalize_with_clipping() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // 100.0 is an outlier
        let normalized = mad_normalize_with_clipping(&signal, 2.0);

        // The outlier should be clipped before normalization
        assert_eq!(normalized.len(), 5);
        // All values should be within [-2, 2] range
        for &val in &normalized {
            assert!((-2.0..=2.0).contains(&val));
        }
    }

    #[test]
    fn test_downscale_factor_2() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let downscaled = downscale(&signal, 2);

        assert_eq!(downscaled.len(), 3);
        assert_eq!(downscaled[0], 1.5); // avg of 1.0, 2.0
        assert_eq!(downscaled[1], 3.5); // avg of 3.0, 4.0
        assert_eq!(downscaled[2], 5.5); // avg of 5.0, 6.0
    }

    #[test]
    fn test_downscale_factor_3() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let downscaled = downscale(&signal, 3);

        assert_eq!(downscaled.len(), 2);
        assert_eq!(downscaled[0], 2.0); // avg of 1.0, 2.0, 3.0
        assert_eq!(downscaled[1], 4.5); // avg of 4.0, 5.0 (remaining)
    }

    #[test]
    fn test_downscale_factor_1() {
        let signal = vec![1.0, 2.0, 3.0];
        let downscaled = downscale(&signal, 1);

        assert_eq!(downscaled, signal);
    }

    #[test]
    #[should_panic(expected = "Downscaling factor must be greater than 0")]
    fn test_downscale_factor_0() {
        let signal = vec![1.0, 2.0, 3.0];
        downscale(&signal, 0);
    }

    #[test]
    #[should_panic(expected = "MAD is zero")]
    fn test_mad_normalize_constant_signal() {
        let signal = vec![5.0, 5.0, 5.0, 5.0];
        mad_normalize(&signal);
    }

    #[test]
    fn test_normalize_dwell_times() {
        // Typical dwell times in samples (right-skewed distribution)
        let dwells = vec![30.0, 45.0, 32.0, 100.0, 28.0];
        let normalized = normalize_dwell_times(&dwells);

        assert_eq!(normalized.len(), 5);

        // After z-score normalization, mean should be ~0
        let mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 1e-5);

        // The outlier (100) should have the highest normalized value
        let max_idx = normalized
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 3); // Index of 100.0
    }

    #[test]
    fn test_normalize_dwell_times_empty() {
        let dwells: Vec<f32> = vec![];
        let normalized = normalize_dwell_times(&dwells);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_normalize_dwell_times_constant() {
        // All identical dwell times
        let dwells = vec![50.0, 50.0, 50.0, 50.0];
        let normalized = normalize_dwell_times(&dwells);

        // Should return zeros for constant values
        assert_eq!(normalized.len(), 4);
        for &val in &normalized {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_normalize_dwell_times_mad() {
        // Include an outlier
        let dwells = vec![30.0, 45.0, 32.0, 500.0, 28.0]; // 500 is extreme outlier
        let normalized = normalize_dwell_times_mad(&dwells);

        assert_eq!(normalized.len(), 5);

        // The outlier should still have the highest value
        let max_idx = normalized
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 3); // Index of 500.0
    }

    #[test]
    fn test_normalize_dwell_times_preserves_order() {
        let dwells = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let normalized = normalize_dwell_times(&dwells);

        // Monotonically increasing dwells should still be monotonically increasing
        for i in 1..normalized.len() {
            assert!(normalized[i] > normalized[i - 1]);
        }
    }
}

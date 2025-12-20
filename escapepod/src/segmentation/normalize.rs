//! Signal normalization utilities for nanopore signal processing.
//!
//! Provides MAD (Median Absolute Deviation) normalization and downscaling operations.

/// Compute the median of a sorted slice of values.
///
/// # Panics
/// Panics if the slice is empty.
#[inline]
fn median_sorted(sorted: &[f32]) -> f32 {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
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
}

//! Barcode fingerprint utilities for nanopore sequencing.
//!
//! A barcode fingerprint is a normalized sequence of signal features (e.g., event means)
//! that can be used for barcode identification via DTW distance computation.

use crate::Uuid;

/// Normalization method for fingerprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMethod {
    /// Z-score normalization: (x - mean) / std
    ZScore,
    /// Min-max normalization: (x - min) / (max - min)
    MinMax,
    /// Median normalization: (x - median) / mad (median absolute deviation)
    Median,
    /// No normalization
    None,
}

/// A barcode fingerprint consisting of a normalized sequence of features.
///
/// Typically these are event-level statistics (e.g., mean current) from the
/// barcode region of a nanopore read.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// Feature values (e.g., event means)
    pub values: Vec<f32>,
    /// Read ID this fingerprint belongs to
    pub read_id: Uuid,
}

impl Fingerprint {
    /// Create a new fingerprint.
    ///
    /// # Arguments
    ///
    /// * `values` - Feature values
    /// * `read_id` - Read ID
    pub fn new(values: Vec<f32>, read_id: Uuid) -> Self {
        Self { values, read_id }
    }

    /// Get the length of the fingerprint.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the fingerprint is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Normalize this fingerprint in-place.
    pub fn normalize(&mut self, method: NormMethod) {
        normalize_fingerprint(self, method);
    }
}

/// Normalize a fingerprint using the specified method.
///
/// # Arguments
///
/// * `fp` - Fingerprint to normalize (modified in-place)
/// * `method` - Normalization method to use
///
/// # Example
///
/// ```
/// use escapepod::dtw::{Fingerprint, normalize_fingerprint, NormMethod};
/// use uuid::Uuid;
///
/// let mut fp = Fingerprint::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Uuid::nil());
/// normalize_fingerprint(&mut fp, NormMethod::ZScore);
///
/// // After z-score normalization, mean should be ~0 and std ~1
/// let mean: f32 = fp.values.iter().sum::<f32>() / fp.values.len() as f32;
/// assert!(mean.abs() < 1e-5);
/// ```
pub fn normalize_fingerprint(fp: &mut Fingerprint, method: NormMethod) {
    if fp.values.is_empty() {
        return;
    }

    match method {
        NormMethod::ZScore => {
            let mean = compute_mean(&fp.values);
            let std = compute_std(&fp.values, mean);

            if std > 0.0 {
                for val in &mut fp.values {
                    *val = (*val - mean) / std;
                }
            }
        }
        NormMethod::MinMax => {
            let min = fp.values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = fp.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;

            if range > 0.0 {
                for val in &mut fp.values {
                    *val = (*val - min) / range;
                }
            }
        }
        NormMethod::Median => {
            let median = compute_median(&fp.values);
            let mad = compute_mad(&fp.values, median);

            if mad > 0.0 {
                for val in &mut fp.values {
                    *val = (*val - median) / mad;
                }
            }
        }
        NormMethod::None => {
            // No normalization
        }
    }
}

/// Compute the mean of a slice.
fn compute_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute the standard deviation of a slice.
fn compute_std(values: &[f32], mean: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

/// Compute the median of a slice.
fn compute_median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

/// Compute the median absolute deviation (MAD).
fn compute_mad(values: &[f32], median: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let deviations: Vec<f32> = values.iter().map(|&x| (x - median).abs()).collect();
    compute_median(&deviations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_creation() {
        let fp = Fingerprint::new(vec![1.0, 2.0, 3.0], Uuid::nil());
        assert_eq!(fp.len(), 3);
        assert!(!fp.is_empty());
    }

    #[test]
    fn test_normalize_zscore() {
        let mut fp = Fingerprint::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Uuid::nil());
        normalize_fingerprint(&mut fp, NormMethod::ZScore);

        // Mean should be approximately 0
        let mean = compute_mean(&fp.values);
        assert!(mean.abs() < 1e-5);

        // Standard deviation should be approximately 1
        let std = compute_std(&fp.values, mean);
        assert!((std - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_minmax() {
        let mut fp = Fingerprint::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Uuid::nil());
        normalize_fingerprint(&mut fp, NormMethod::MinMax);

        // Min should be 0
        let min = fp.values.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(min.abs() < 1e-5);

        // Max should be 1
        let max = fp.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((max - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_median() {
        let mut fp = Fingerprint::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Uuid::nil());
        normalize_fingerprint(&mut fp, NormMethod::Median);

        // After median normalization, the median should be approximately 0
        let median = compute_median(&fp.values);
        assert!(median.abs() < 0.5); // Relaxed tolerance since MAD normalization
    }

    #[test]
    fn test_normalize_none() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut fp = Fingerprint::new(original.clone(), Uuid::nil());
        normalize_fingerprint(&mut fp, NormMethod::None);

        // Values should be unchanged
        assert_eq!(fp.values, original);
    }

    #[test]
    fn test_normalize_empty() {
        let mut fp = Fingerprint::new(vec![], Uuid::nil());
        normalize_fingerprint(&mut fp, NormMethod::ZScore);
        assert!(fp.is_empty());
    }

    #[test]
    fn test_normalize_constant() {
        let mut fp = Fingerprint::new(vec![5.0, 5.0, 5.0, 5.0], Uuid::nil());
        normalize_fingerprint(&mut fp, NormMethod::ZScore);

        // With zero std, values should remain unchanged
        assert_eq!(fp.values, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_compute_mean() {
        assert_eq!(compute_mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(compute_mean(&[2.0, 2.0, 2.0]), 2.0);
        assert_eq!(compute_mean(&[]), 0.0);
    }

    #[test]
    fn test_compute_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = compute_mean(&values);
        let std = compute_std(&values, mean);

        // Std of 1,2,3,4,5 is sqrt(2) ≈ 1.414
        assert!((std - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_compute_median() {
        assert_eq!(compute_median(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(compute_median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(compute_median(&[5.0]), 5.0);
        assert_eq!(compute_median(&[]), 0.0);
    }

    #[test]
    fn test_compute_mad() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = compute_median(&values);
        let mad = compute_mad(&values, median);

        // MAD of 1,2,3,4,5 (median=3) is median of |1-3|,|2-3|,|3-3|,|4-3|,|5-3| = median of 2,1,0,1,2 = 1
        assert!((mad - 1.0).abs() < 1e-5);
    }
}

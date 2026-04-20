//! Barcode fingerprint utilities for nanopore sequencing.
//!
//! A barcode fingerprint is a normalized sequence of signal features (e.g., event means)
//! that can be used for barcode identification via DTW distance computation.

use uuid::Uuid;

/// Normalization method for fingerprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMethod {
    /// Z-score normalization: (x - mean) / std
    ZScore,
    /// Min-max normalization: (x - min) / (max - min)
    MinMax,
    /// Median normalization: (x - median) / mad (median absolute deviation)
    Median,
    /// Mean normalization: (x - mean), no scaling
    Mean,
    /// No normalization
    None,
}

/// A barcode fingerprint consisting of a normalized sequence of features.
///
/// Typically these are event-level statistics (e.g., mean current) from the
/// barcode region of a nanopore read. Optionally includes dwell times for
/// enhanced classification.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// Feature values (e.g., event means)
    pub values: Vec<f32>,
    /// Dwell times per event (in samples), normalized
    /// DNA barcodes translocate ~3-4% faster than RNA, providing
    /// independent discriminative information.
    pub dwell_times: Option<Vec<f32>>,
    /// Read ID this fingerprint belongs to
    pub read_id: Uuid,
}

impl Fingerprint {
    /// Create a new fingerprint with only event means.
    ///
    /// # Arguments
    ///
    /// * `values` - Feature values (event means)
    /// * `read_id` - Read ID
    pub fn new(values: Vec<f32>, read_id: Uuid) -> Self {
        Self {
            values,
            dwell_times: None,
            read_id,
        }
    }

    /// Create a new fingerprint with both event means and dwell times.
    ///
    /// # Arguments
    ///
    /// * `values` - Feature values (event means)
    /// * `dwell_times` - Dwell times per event (in samples)
    /// * `read_id` - Read ID
    pub fn with_dwell_times(values: Vec<f32>, dwell_times: Vec<f32>, read_id: Uuid) -> Self {
        debug_assert_eq!(
            values.len(),
            dwell_times.len(),
            "values and dwell_times must have same length"
        );
        Self {
            values,
            dwell_times: Some(dwell_times),
            read_id,
        }
    }

    /// Get the length of the fingerprint.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the fingerprint is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Check if this fingerprint has dwell time information.
    pub fn has_dwell_times(&self) -> bool {
        self.dwell_times.is_some()
    }

    /// Normalize this fingerprint in-place.
    pub fn normalize(&mut self, method: NormMethod) {
        normalize_fingerprint(self, method);
    }

    /// Convert to a combined feature vector by concatenating event means and dwell times.
    ///
    /// If dwell times are present, the output is `[means..., dwells...]` (2N features).
    /// If no dwell times, returns just the event means (N features).
    ///
    /// # Arguments
    ///
    /// * `dwell_weight` - Optional weight for dwell time features (default 1.0).
    ///   Use values < 1.0 to down-weight dwells relative to means.
    pub fn to_feature_vector(&self, dwell_weight: Option<f32>) -> Vec<f32> {
        match &self.dwell_times {
            Some(dwells) => {
                let weight = dwell_weight.unwrap_or(1.0);
                let mut features = self.values.clone();
                features.extend(dwells.iter().map(|&d| d * weight));
                features
            }
            None => self.values.clone(),
        }
    }

    /// Create an interleaved feature vector: [(mean_1, dwell_1), (mean_2, dwell_2), ...].
    ///
    /// This may be better for DTW as it keeps local event information together.
    /// Returns None if no dwell times are present.
    pub fn to_interleaved_features(&self, dwell_weight: Option<f32>) -> Option<Vec<f32>> {
        self.dwell_times.as_ref().map(|dwells| {
            let weight = dwell_weight.unwrap_or(1.0);
            self.values
                .iter()
                .zip(dwells.iter())
                .flat_map(|(&m, &d)| [m, d * weight])
                .collect()
        })
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
            // Sort once for median, reuse the buffer for MAD to avoid a
            // second to_vec allocation that compute_mad→compute_median would do.
            let mut buf = fp.values.clone();
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = median_of_sorted(&buf);

            for (v, slot) in fp.values.iter().zip(buf.iter_mut()) {
                *slot = (*v - median).abs();
            }
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = median_of_sorted(&buf);

            if mad > 0.0 {
                for val in &mut fp.values {
                    *val = (*val - median) / mad;
                }
            }
        }
        NormMethod::Mean => {
            let mean = compute_mean(&fp.values);
            for val in &mut fp.values {
                *val -= mean;
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

fn median_of_sorted(sorted: &[f32]) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len().is_multiple_of(2) {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

#[cfg(test)]
fn compute_median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    median_of_sorted(&sorted)
}

#[cfg(test)]
fn compute_mad(values: &[f32], median: f32) -> f32 {
    let mut deviations: Vec<f32> = values.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    median_of_sorted(&deviations)
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

    #[test]
    fn test_fingerprint_with_dwell_times() {
        let values = vec![1.0, 2.0, 3.0];
        let dwells = vec![30.0, 45.0, 35.0];
        let fp = Fingerprint::with_dwell_times(values.clone(), dwells.clone(), Uuid::nil());

        assert_eq!(fp.len(), 3);
        assert!(fp.has_dwell_times());
        assert_eq!(fp.values, values);
        assert_eq!(fp.dwell_times, Some(dwells));
    }

    #[test]
    fn test_fingerprint_no_dwell_times() {
        let fp = Fingerprint::new(vec![1.0, 2.0, 3.0], Uuid::nil());

        assert!(!fp.has_dwell_times());
        assert_eq!(fp.dwell_times, None);
    }

    #[test]
    fn test_to_feature_vector_without_dwells() {
        let fp = Fingerprint::new(vec![1.0, 2.0, 3.0], Uuid::nil());
        let features = fp.to_feature_vector(None);

        // Without dwells, should just return values
        assert_eq!(features, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_feature_vector_with_dwells() {
        let values = vec![1.0, 2.0, 3.0];
        let dwells = vec![10.0, 20.0, 30.0];
        let fp = Fingerprint::with_dwell_times(values, dwells, Uuid::nil());
        let features = fp.to_feature_vector(None);

        // Should concatenate: [values..., dwells...]
        assert_eq!(features, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_to_feature_vector_with_weight() {
        let values = vec![1.0, 2.0, 3.0];
        let dwells = vec![10.0, 20.0, 30.0];
        let fp = Fingerprint::with_dwell_times(values, dwells, Uuid::nil());
        let features = fp.to_feature_vector(Some(0.5));

        // Dwells should be scaled by 0.5
        assert_eq!(features, vec![1.0, 2.0, 3.0, 5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_to_interleaved_features() {
        let values = vec![1.0, 2.0, 3.0];
        let dwells = vec![10.0, 20.0, 30.0];
        let fp = Fingerprint::with_dwell_times(values, dwells, Uuid::nil());
        let features = fp.to_interleaved_features(None);

        // Should interleave: [(mean1, dwell1), (mean2, dwell2), ...]
        assert_eq!(features, Some(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]));
    }

    #[test]
    fn test_to_interleaved_features_without_dwells() {
        let fp = Fingerprint::new(vec![1.0, 2.0, 3.0], Uuid::nil());
        let features = fp.to_interleaved_features(None);

        // Should return None if no dwells
        assert_eq!(features, None);
    }

    #[test]
    fn test_to_interleaved_features_with_weight() {
        let values = vec![1.0, 2.0];
        let dwells = vec![10.0, 20.0];
        let fp = Fingerprint::with_dwell_times(values, dwells, Uuid::nil());
        let features = fp.to_interleaved_features(Some(0.5));

        assert_eq!(features, Some(vec![1.0, 5.0, 2.0, 10.0]));
    }
}

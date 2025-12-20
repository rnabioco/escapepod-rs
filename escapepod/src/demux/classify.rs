//! Classification logic for barcode demultiplexing.

use super::model::WarpDemuxModel;
use crate::dtw::dtw_distance;

/// Result of classifying a read by barcode.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Assigned barcode name (e.g., "BC01", "BC02", or "unclassified")
    pub barcode: String,

    /// Confidence score for the classification.
    /// For ratio-based: 1.0 - (best_distance / second_best_distance)
    /// For kernel-based: the kernel similarity value
    pub confidence: f64,

    /// DTW distance to the nearest training fingerprint
    pub best_distance: f64,

    /// DTW distance to the second-nearest training fingerprint
    pub second_best_distance: f64,

    /// Whether the classification passed the confidence threshold
    pub is_confident: bool,

    /// Index of the best matching training sample
    pub best_match_index: usize,
}

impl ClassificationResult {
    /// Create an unclassified result (no confident match).
    pub fn unclassified(best_distance: f64, second_best_distance: f64) -> Self {
        Self {
            barcode: "unclassified".to_string(),
            confidence: 0.0,
            best_distance,
            second_best_distance,
            is_confident: false,
            best_match_index: 0,
        }
    }
}

/// Classify a read fingerprint using a trained WarpDemuX model.
///
/// This function:
/// 1. Computes DTW distance from the query fingerprint to all training fingerprints
/// 2. Finds the nearest and second-nearest training samples
/// 3. Applies the model's threshold to determine if the classification is confident
/// 4. Returns the barcode assignment and confidence metrics
///
/// # Arguments
///
/// * `model` - The trained WarpDemuX model
/// * `fingerprint` - The query fingerprint to classify (normalized feature vector)
///
/// # Returns
///
/// A `ClassificationResult` with the assigned barcode and confidence metrics.
///
/// # Example
///
/// ```no_run
/// use escapepod::demux::{load_model, classify_read};
/// use std::path::Path;
///
/// let model = load_model(Path::new("model.json"))?;
/// let fingerprint = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let result = classify_read(&model, &fingerprint);
///
/// if result.is_confident {
///     println!("Classified as {} with confidence {:.3}", result.barcode, result.confidence);
/// } else {
///     println!("Unclassified (low confidence)");
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn classify_read(model: &WarpDemuxModel, fingerprint: &[f64]) -> ClassificationResult {
    // Convert query fingerprint to f32 for DTW
    let query_f32: Vec<f32> = fingerprint.iter().map(|&x| x as f32).collect();

    // Compute DTW distances to all training fingerprints
    let mut distances: Vec<(usize, f32)> = model
        .training_fingerprints
        .iter()
        .enumerate()
        .map(|(i, train_fp)| {
            let train_f32: Vec<f32> = train_fp.iter().map(|&x| x as f32).collect();
            let dist = dtw_distance(&query_f32, &train_f32, None);
            (i, dist)
        })
        .collect();

    // Sort by distance (ascending)
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Get best and second-best matches
    let (best_idx, best_dist) = distances[0];
    let second_best_dist = if distances.len() > 1 {
        distances[1].1
    } else {
        f32::INFINITY
    };

    let best_dist_f64 = best_dist as f64;
    let second_best_dist_f64 = second_best_dist as f64;

    // Get the label for the best match
    let best_label = model.training_labels[best_idx];
    let barcode_name = model.get_barcode_name(best_label);

    // Determine confidence based on threshold type
    let (is_confident, confidence) = match model.threshold_type.as_str() {
        "ratio" => {
            // Ratio of best to second-best distance
            // Lower ratio = more confident (best is much closer than second-best)
            let ratio = if second_best_dist_f64 > 0.0 {
                best_dist_f64 / second_best_dist_f64
            } else {
                1.0
            };

            let confident = ratio <= model.threshold;
            let confidence_score = if confident {
                1.0 - ratio // Higher is better
            } else {
                0.0
            };

            (confident, confidence_score)
        }
        "kernel" => {
            // Convert distance to kernel similarity using RBF
            let gamma = model.kernel_params.gamma;
            let power = model.kernel_params.power;
            let kernel_value = (-gamma * best_dist_f64.powf(power)).exp();

            let confident = kernel_value >= model.threshold;
            (confident, kernel_value)
        }
        _ => {
            // Default to ratio if unknown type
            let ratio = if second_best_dist_f64 > 0.0 {
                best_dist_f64 / second_best_dist_f64
            } else {
                1.0
            };

            let confident = ratio <= model.threshold;
            let confidence_score = if confident { 1.0 - ratio } else { 0.0 };

            (confident, confidence_score)
        }
    };

    if is_confident {
        ClassificationResult {
            barcode: barcode_name,
            confidence,
            best_distance: best_dist_f64,
            second_best_distance: second_best_dist_f64,
            is_confident: true,
            best_match_index: best_idx,
        }
    } else {
        ClassificationResult::unclassified(best_dist_f64, second_best_dist_f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::model::KernelParams;
    use std::collections::HashMap;

    fn create_test_model() -> WarpDemuxModel {
        let mut label_map = HashMap::new();
        label_map.insert("BC01".to_string(), 0);
        label_map.insert("BC02".to_string(), 1);

        WarpDemuxModel {
            training_fingerprints: vec![
                vec![0.0, 0.0, 0.0], // BC01
                vec![1.0, 1.0, 1.0], // BC02
                vec![0.1, 0.1, 0.1], // BC01
            ],
            training_labels: vec![0, 1, 0],
            kernel_params: KernelParams {
                gamma: 1.0,
                power: 1.0,
            },
            label_map,
            threshold: 0.5, // ratio threshold
            threshold_type: "ratio".to_string(),
        }
    }

    #[test]
    fn test_classify_exact_match() {
        let model = create_test_model();
        let fingerprint = vec![0.0, 0.0, 0.0]; // Exact match to BC01

        let result = classify_read(&model, &fingerprint);

        assert_eq!(result.barcode, "BC01");
        assert!(result.is_confident);
        assert!(result.best_distance < 0.01); // Should be nearly zero
    }

    #[test]
    fn test_classify_close_match() {
        let model = create_test_model();
        let fingerprint = vec![0.02, 0.02, 0.02]; // Very close to BC01 [0.0, 0.0, 0.0]
                                                    // Much farther from BC02 [1.0, 1.0, 1.0]

        let result = classify_read(&model, &fingerprint);

        // Should be classified as BC01
        assert_eq!(result.barcode, "BC01");
        assert!(result.is_confident);
    }

    #[test]
    fn test_classify_ambiguous() {
        let model = create_test_model();
        let fingerprint = vec![0.5, 0.5, 0.5]; // Midway between BC01 and BC02

        let result = classify_read(&model, &fingerprint);

        // Should be unclassified due to poor ratio
        assert!(!result.is_confident);
        assert_eq!(result.barcode, "unclassified");
    }

    #[test]
    fn test_classify_kernel_threshold() {
        let mut model = create_test_model();
        model.threshold_type = "kernel".to_string();
        model.threshold = 0.5; // Kernel similarity threshold

        let fingerprint = vec![0.0, 0.0, 0.0]; // Exact match

        let result = classify_read(&model, &fingerprint);

        assert_eq!(result.barcode, "BC01");
        assert!(result.is_confident);
        // Kernel value for distance=0 should be 1.0
        assert!((result.confidence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_classify_confidence_metrics() {
        let model = create_test_model();
        let fingerprint = vec![0.0, 0.0, 0.0];

        let result = classify_read(&model, &fingerprint);

        // Should have valid distance metrics
        assert!(result.best_distance >= 0.0);
        assert!(result.second_best_distance >= result.best_distance);
        assert!(result.best_match_index < model.num_samples());
    }

    #[test]
    fn test_unclassified_result() {
        let result = ClassificationResult::unclassified(1.0, 1.5);

        assert_eq!(result.barcode, "unclassified");
        assert!(!result.is_confident);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.best_distance, 1.0);
        assert_eq!(result.second_best_distance, 1.5);
    }
}

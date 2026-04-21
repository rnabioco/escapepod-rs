//! Probability processing utilities for barcode classification.
//!
//! This module provides functions for processing raw model outputs into
//! probability distributions and confidence scores, compatible with WarpDemuX.

use std::collections::HashMap;

/// Result of processing probabilities for a single sample.
#[derive(Debug, Clone)]
pub struct ProbabilityResult {
    /// Predicted barcode ID (-1 for unclassified)
    pub predicted_barcode: i32,

    /// Confidence score (margin between top 2 probabilities)
    pub confidence: f64,

    /// Index of the predicted class in the probability vector
    pub predicted_index: usize,

    /// Whether the prediction passed the confidence threshold
    pub is_confident: bool,
}

/// Apply softmax to convert raw scores to probabilities.
///
/// Uses the stable softmax algorithm: subtract max before exp to prevent overflow.
///
/// # Arguments
///
/// * `scores` - Raw decision function scores
///
/// # Returns
///
/// Probability distribution that sums to 1.0
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }

    // Find max for numerical stability
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(score - max) for each score
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();

    // Normalize
    let sum: f64 = exp_scores.iter().sum();
    if sum > 0.0 {
        exp_scores.iter().map(|&e| e / sum).collect()
    } else {
        // Fallback: uniform distribution
        let n = scores.len() as f64;
        vec![1.0 / n; scores.len()]
    }
}

/// Compute confidence margin between top 2 probabilities.
///
/// This matches WarpDemuX's `confidence_margin` function:
/// confidence = P(top_class) - P(second_class)
///
/// Higher confidence means the model is more certain about its prediction.
///
/// # Arguments
///
/// * `probs` - Probability distribution over classes
///
/// # Returns
///
/// Confidence margin in [0, 1]
pub fn confidence_margin(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return if probs.is_empty() { 0.0 } else { 1.0 };
    }

    // Find top two probabilities. We only need the top two, so a full sort
    // is overkill — two max-passes are O(n) vs O(n log n).
    let (mut best, mut second) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &p in probs {
        if p > best {
            second = best;
            best = p;
        } else if p > second {
            second = p;
        }
    }
    best - second
}

/// Process probabilities to predictions and confidence scores.
///
/// This matches WarpDemuX's `process_probs` function:
/// 1. Find argmax for prediction
/// 2. Compute confidence as margin between top 2 probabilities
/// 3. Apply per-class thresholds to filter low-confidence predictions
///
/// # Arguments
///
/// * `probs` - Probability distribution over classes
/// * `label_mapper` - Maps class index to barcode ID
/// * `thresholds` - Optional per-class confidence thresholds
///
/// # Returns
///
/// `ProbabilityResult` with prediction and confidence
pub fn process_probabilities(
    probs: &[f64],
    label_mapper: &HashMap<usize, i32>,
    thresholds: Option<&[f64]>,
) -> ProbabilityResult {
    if probs.is_empty() {
        return ProbabilityResult {
            predicted_barcode: -1,
            confidence: 0.0,
            predicted_index: 0,
            is_confident: false,
        };
    }

    // Find argmax
    let (pred_idx, _max_prob) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Map to barcode ID
    let barcode_id = label_mapper.get(&pred_idx).copied().unwrap_or(-1);

    // Compute confidence margin
    let confidence = confidence_margin(probs);

    // Apply threshold if provided
    let is_confident = if let Some(thresh) = thresholds {
        let threshold = thresh.get(pred_idx).copied().unwrap_or(0.0);
        confidence >= threshold
    } else {
        true
    };

    let final_barcode = if is_confident { barcode_id } else { -1 };

    ProbabilityResult {
        predicted_barcode: final_barcode,
        confidence,
        predicted_index: pred_idx,
        is_confident,
    }
}

/// Convert probability vector to formatted output columns.
///
/// Creates column names like "p00", "p01", ..., "p12" matching WarpDemuX output.
///
/// # Arguments
///
/// * `probs` - Probability distribution
/// * `label_mapper` - Maps class index to barcode ID
///
/// # Returns
///
/// Vector of (column_name, probability) pairs
pub fn format_probability_columns(
    probs: &[f64],
    label_mapper: &HashMap<usize, i32>,
) -> Vec<(String, f64)> {
    probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let barcode_id = label_mapper.get(&i).copied().unwrap_or(i as i32);
            (format!("p{:02}", barcode_id), p)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax(&scores);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Higher score = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_stability() {
        // Large values that could overflow without stability fix
        let scores = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&scores);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_confidence_margin_clear_winner() {
        let probs = vec![0.9, 0.05, 0.05];
        let conf = confidence_margin(&probs);
        assert!((conf - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_margin_ambiguous() {
        let probs = vec![0.4, 0.35, 0.25];
        let conf = confidence_margin(&probs);
        assert!((conf - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_margin_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let conf = confidence_margin(&probs);
        assert!(conf.abs() < 1e-10);
    }

    #[test]
    fn test_process_probabilities_basic() {
        let probs = vec![0.1, 0.7, 0.2];
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 4);
        label_mapper.insert(1, 5);
        label_mapper.insert(2, 6);

        let result = process_probabilities(&probs, &label_mapper, None);

        assert_eq!(result.predicted_barcode, 5);
        assert_eq!(result.predicted_index, 1);
        assert!((result.confidence - 0.5).abs() < 1e-10);
        assert!(result.is_confident);
    }

    #[test]
    fn test_process_probabilities_with_threshold() {
        let probs = vec![0.4, 0.35, 0.25];
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 4);
        label_mapper.insert(1, 5);
        label_mapper.insert(2, 6);

        // Threshold is higher than confidence (0.05)
        let thresholds = vec![0.1, 0.1, 0.1];
        let result = process_probabilities(&probs, &label_mapper, Some(&thresholds));

        assert_eq!(result.predicted_barcode, -1); // Unclassified
        assert!(!result.is_confident);
        assert_eq!(result.predicted_index, 0); // Still tracks the argmax
    }

    #[test]
    fn test_format_probability_columns() {
        let probs = vec![0.1, 0.7, 0.2];
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 4);
        label_mapper.insert(1, 5);
        label_mapper.insert(2, 6);

        let cols = format_probability_columns(&probs, &label_mapper);

        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].0, "p04");
        assert_eq!(cols[1].0, "p05");
        assert_eq!(cols[2].0, "p06");
        assert!((cols[1].1 - 0.7).abs() < 1e-10);
    }
}

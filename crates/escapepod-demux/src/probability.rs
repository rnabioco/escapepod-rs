//! Probability post-processing for multiclass SVM classifiers: numerically
//! stable softmax, margin-based confidence, and per-class threshold gating.
//!
//! These are standard building blocks for probabilistic classifiers — none of
//! this module is specific to any particular upstream tool. The margin score
//! is the classical top-1-minus-top-2 quantity used in margin-based active
//! learning (Scheffer, Decomain & Wrobel, 2001; see also Schapire & Singer
//! 1999 on margins for confidence-rated predictions). The threshold-gated
//! argmax head is the textbook decision rule for a Platt-scaled SVM.

use std::collections::HashMap;

/// Outcome of running [`process_probabilities`] on one sample.
#[derive(Debug, Clone)]
pub struct ProbabilityResult {
    /// Predicted barcode ID after threshold gating. `-1` means the
    /// prediction was rejected (low confidence) or no mapping exists.
    pub predicted_barcode: i32,

    /// Margin between the top-1 and top-2 probabilities, in `[0.0, 1.0]`.
    pub confidence: f64,

    /// Index of the argmax class in the input probability vector. Always
    /// populated, even when the prediction is rejected.
    pub predicted_index: usize,

    /// `true` iff `confidence` was at or above the gating threshold for
    /// the predicted class.
    pub is_confident: bool,
}

/// Numerically stable softmax: subtracts the maximum input before
/// exponentiating to avoid overflow. Returns a uniform distribution as a
/// graceful fallback if every input is `-inf` (sum underflows to 0).
///
/// # Arguments
///
/// * `scores` - Raw decision-function outputs (any real range).
///
/// # Returns
///
/// A vector that sums to `1.0`, or an empty vector if the input was empty.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }

    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();

    let sum: f64 = exp_scores.iter().sum();
    if sum > 0.0 {
        exp_scores.iter().map(|&e| e / sum).collect()
    } else {
        let n = scores.len() as f64;
        vec![1.0 / n; scores.len()]
    }
}

/// Margin between the top-1 and top-2 probabilities — `p[best] - p[second]`.
///
/// Standard uncertainty score in margin-based active learning. Larger values
/// mean the classifier is committing more strongly to a single class; values
/// near `0.0` indicate the top two classes are tied. Single-class input
/// returns `1.0`; empty input returns `0.0`.
///
/// Two linear passes (one to find best, one tracking second) — `O(n)`, no
/// allocation, faster than sorting for the common case where there are only
/// a handful of classes.
///
/// # Arguments
///
/// * `probs` - Class probabilities (need not sum to 1 — only the relative
///   ordering of the two largest entries matters).
///
/// # Returns
///
/// `p[best] - p[second]`, clamped to `[0.0, 1.0]` for well-formed inputs.
pub fn confidence_margin(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return if probs.is_empty() { 0.0 } else { 1.0 };
    }

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

/// Classify one sample given a class-probability vector: argmax to pick a
/// class, map to the caller's external label space, score with the top-1
/// vs. top-2 margin, and gate on a per-class threshold.
///
/// This is the textbook decision rule for a probabilistic multiclass
/// classifier with a reject option: classes whose margin falls below their
/// individual threshold are returned as `-1` (rejected) instead of the
/// argmax label.
///
/// # Arguments
///
/// * `probs` - Class probability vector (one entry per training class).
/// * `label_mapper` - Maps internal class index → external barcode ID. An
///   index missing from the map maps to `-1`.
/// * `thresholds` - Optional per-class margin thresholds, indexed the same
///   way as `probs`. If `None`, no rejection is applied.
///
/// # Returns
///
/// A [`ProbabilityResult`] describing the decision. `predicted_index` is
/// always the argmax position even when rejected, so callers can recover
/// the most likely class if they want to override the gate.
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

    let (pred_idx, _max_prob) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let barcode_id = label_mapper.get(&pred_idx).copied().unwrap_or(-1);

    let confidence = confidence_margin(probs);

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

/// Lay out per-class probabilities as `(column_name, value)` pairs for CSV
/// output, with column names of the form `p<NN>` where `NN` is the
/// zero-padded external barcode ID. Two-digit padding is for stable column
/// ordering when sorted lexically (`p00 … p99`).
///
/// This is a serialization helper — it produces no decisions and consumes
/// no decisions; it just names columns.
///
/// # Arguments
///
/// * `probs` - Class probability vector.
/// * `label_mapper` - Maps internal class index → external barcode ID.
///   Indices missing from the map fall back to the index itself.
///
/// # Returns
///
/// One `(name, probability)` pair per input entry, in the input's order.
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

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_stability() {
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

        let thresholds = vec![0.1, 0.1, 0.1];
        let result = process_probabilities(&probs, &label_mapper, Some(&thresholds));

        assert_eq!(result.predicted_barcode, -1);
        assert!(!result.is_confident);
        assert_eq!(result.predicted_index, 0);
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

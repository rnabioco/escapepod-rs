//! CatBoost model support for WarpDemuX Fpt_Boost models.
//!
//! This module provides support for loading and using CatBoost models
//! exported from WarpDemuX for barcode classification.
//!
//! ## Model Export
//!
//! WarpDemuX Fpt_Boost models can be exported using:
//! ```python
//! import joblib
//! model = joblib.load("model.joblib")
//! model.model.save_model("model.cbm")
//! ```
//!
//! The metadata (label_mapper, thresholds) must be saved separately as JSON.

use anyhow::{Context, Result};
use catboost_rust::{Model, ObjectsOrderFeatures};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Metadata for a WarpDemuX CatBoost model.
///
/// This contains the label mapping and thresholds that are stored
/// separately from the CatBoost model itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatBoostMeta {
    /// Maps class index (0, 1, 2, ...) to barcode ID (4, 5, 7, ...).
    pub label_mapper: HashMap<usize, i32>,

    /// Per-class confidence thresholds.
    pub thresholds: Vec<f64>,

    /// Number of classes (barcodes).
    pub n_classes: usize,

    /// Whether the model has a dedicated noise class.
    #[serde(default)]
    pub noise_class: bool,
}

impl CatBoostMeta {
    /// Load metadata from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open metadata file '{}'", path.display()))?;
        let reader = BufReader::new(file);
        let meta: CatBoostMeta = serde_json::from_reader(reader)
            .with_context(|| "Failed to parse CatBoost metadata JSON")?;
        Ok(meta)
    }

    /// Get barcode name for a class index.
    pub fn get_barcode_name(&self, class_idx: usize) -> String {
        self.label_mapper
            .get(&class_idx)
            .map(|id| format!("BC{:02}", id))
            .unwrap_or_else(|| format!("unknown_{}", class_idx))
    }

    /// Get barcode ID for a class index.
    pub fn get_barcode_id(&self, class_idx: usize) -> Option<i32> {
        self.label_mapper.get(&class_idx).copied()
    }
}

/// A CatBoost classifier for barcode demultiplexing.
///
/// This wraps a CatBoost model loaded from a CBM file along with
/// the WarpDemuX metadata for label mapping and thresholds.
pub struct CatBoostClassifier {
    /// The underlying CatBoost model.
    model: Model,

    /// Model metadata (label mapping, thresholds).
    pub meta: CatBoostMeta,
}

impl CatBoostClassifier {
    /// Load a CatBoost classifier from model and metadata files.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the CatBoost model file (.cbm)
    /// * `meta_path` - Path to the metadata JSON file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use escapepod::demux::CatBoostClassifier;
    /// use std::path::Path;
    ///
    /// let classifier = CatBoostClassifier::load(
    ///     Path::new("model.cbm"),
    ///     Path::new("model_meta.json"),
    /// )?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn load(model_path: &Path, meta_path: &Path) -> Result<Self> {
        let model = Model::load(model_path.to_str().ok_or_else(|| {
            anyhow::anyhow!("Invalid model path: {}", model_path.display())
        })?)
        .with_context(|| format!("Failed to load CatBoost model from '{}'", model_path.display()))?;

        let meta = CatBoostMeta::load(meta_path)?;

        Ok(Self { model, meta })
    }

    /// Get the number of classes in the model.
    pub fn n_classes(&self) -> usize {
        self.meta.n_classes
    }

    /// Classify a single fingerprint.
    ///
    /// Returns the predicted class probabilities.
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The input fingerprint (feature vector)
    ///
    /// # Returns
    ///
    /// A vector of probabilities for each class.
    pub fn predict_proba(&self, fingerprint: &[f64]) -> Result<Vec<f64>> {
        // Convert to f32 for CatBoost
        let features: Vec<f32> = fingerprint.iter().map(|&x| x as f32).collect();
        let features_slice: &[f32] = &features;
        let features_arr = [features_slice];

        // Create feature matrix (single row)
        let float_features = ObjectsOrderFeatures::new()
            .with_float_features(&features_arr);

        // Get raw predictions
        let predictions = self
            .model
            .predict(float_features)
            .with_context(|| "CatBoost prediction failed")?;

        // CatBoost returns raw scores, apply softmax for probabilities
        let probs = softmax_vec(&predictions);

        Ok(probs)
    }

    /// Classify multiple fingerprints in batch.
    ///
    /// # Arguments
    ///
    /// * `fingerprints` - Vector of fingerprints to classify
    ///
    /// # Returns
    ///
    /// A vector of probability vectors, one per input fingerprint.
    pub fn predict_proba_batch(&self, fingerprints: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if fingerprints.is_empty() {
            return Ok(vec![]);
        }

        // Convert to f32
        let features: Vec<Vec<f32>> = fingerprints
            .iter()
            .map(|fp| fp.iter().map(|&x| x as f32).collect())
            .collect();

        // Create references for the API
        let feature_refs: Vec<&[f32]> = features.iter().map(|v| v.as_slice()).collect();

        let float_features = ObjectsOrderFeatures::new()
            .with_float_features(&feature_refs);

        // Get raw predictions
        let predictions = self
            .model
            .predict(float_features)
            .with_context(|| "CatBoost batch prediction failed")?;

        // Reshape predictions (flat array to 2D)
        let n_classes = self.meta.n_classes;
        let batch_probs: Vec<Vec<f64>> = predictions
            .chunks(n_classes)
            .map(|chunk| softmax_vec(chunk))
            .collect();

        Ok(batch_probs)
    }

    /// Classify a fingerprint and return the predicted barcode.
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The input fingerprint
    ///
    /// # Returns
    ///
    /// A tuple of (barcode_name, confidence, class_probabilities).
    pub fn classify(&self, fingerprint: &[f64]) -> Result<CatBoostResult> {
        let probs = self.predict_proba(fingerprint)?;

        // Find the class with highest probability
        let (best_idx, &best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Check threshold
        let threshold = self.meta.thresholds.get(best_idx).copied().unwrap_or(0.0);
        let passes_threshold = best_prob >= threshold;

        let barcode = if passes_threshold {
            self.meta.get_barcode_name(best_idx)
        } else {
            "unclassified".to_string()
        };

        let barcode_id = if passes_threshold {
            self.meta.get_barcode_id(best_idx).unwrap_or(-1)
        } else {
            -1
        };

        Ok(CatBoostResult {
            barcode,
            barcode_id,
            confidence: best_prob,
            class_idx: best_idx,
            probabilities: probs,
            passes_threshold,
        })
    }
}

/// Result from CatBoost classification.
#[derive(Debug, Clone)]
pub struct CatBoostResult {
    /// Predicted barcode name (e.g., "BC04") or "unclassified".
    pub barcode: String,

    /// Predicted barcode ID (e.g., 4) or -1 for unclassified.
    pub barcode_id: i32,

    /// Confidence (probability of predicted class).
    pub confidence: f64,

    /// Index of the predicted class.
    pub class_idx: usize,

    /// Probabilities for all classes.
    pub probabilities: Vec<f64>,

    /// Whether the prediction passes the threshold.
    pub passes_threshold: bool,
}

/// Apply softmax to convert raw scores to probabilities.
fn softmax_vec(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }

    // Find max for numerical stability
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(x - max)
    let exp_scores: Vec<f64> = scores.iter().map(|&x| (x - max_score).exp()).collect();

    // Normalize
    let sum: f64 = exp_scores.iter().sum();
    exp_scores.iter().map(|&x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax_vec(&scores);

        // Check they sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check ordering preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax_vec(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_meta_barcode_name() {
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 4);
        label_mapper.insert(1, 5);

        let meta = CatBoostMeta {
            label_mapper,
            thresholds: vec![0.5, 0.5],
            n_classes: 2,
            noise_class: false,
        };

        assert_eq!(meta.get_barcode_name(0), "BC04");
        assert_eq!(meta.get_barcode_name(1), "BC05");
        assert!(meta.get_barcode_name(99).starts_with("unknown"));
    }
}

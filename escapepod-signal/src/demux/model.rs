//! WarpDemuX model loading and data structures.
//!
//! This module provides two model types:
//! - `WarpDemuxModel`: Legacy distance-based nearest-neighbor classifier
//! - `DtwSvmModel`: Full SVM classifier with probability output

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// RBF kernel parameters for DTW distance conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelParams {
    /// Gamma parameter: controls the width of the RBF kernel.
    /// The kernel is computed as: K = exp(-gamma * distance^power)
    pub gamma: f64,

    /// Power to raise the distance to before applying exponential.
    /// Typically 1.0 or 2.0.
    pub power: f64,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            power: 1.0,
        }
    }
}

/// DTW-SVM model with full SVM parameters for barcode classification.
///
/// This model stores everything needed for SVM inference:
/// - Training fingerprints for DTW distance computation
/// - SVM dual coefficients, support vectors, and intercepts
/// - Kernel parameters and thresholds
///
/// Models can be exported from WarpDemuX using `export_warpdemux_models.py`
/// or trained natively in Rust with the `train` feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtwSvmModel {
    /// Model format version for compatibility checking.
    #[serde(default = "default_version")]
    pub version: String,

    /// Training fingerprints (reference barcodes).
    /// Shape: [n_training_samples, n_features]
    pub training_fingerprints: Vec<Vec<f64>>,

    /// Training labels for each fingerprint.
    pub training_labels: Vec<i32>,

    /// Indices of support vectors in the training data.
    pub support_indices: Vec<usize>,

    /// Dual coefficients for the decision function.
    /// For OvO multiclass: shape [n_classes-1, n_support_vectors]
    pub dual_coef: Vec<Vec<f64>>,

    /// Intercept (bias) terms for each binary classifier.
    /// For OvO: length = n_classes * (n_classes - 1) / 2
    pub intercept: Vec<f64>,

    /// Class labels (barcode IDs).
    pub classes: Vec<i32>,

    /// RBF kernel parameters.
    pub kernel_params: KernelParams,

    /// DTW window constraint (Sakoe-Chiba band).
    #[serde(default)]
    pub window: Option<usize>,

    /// Maps class index (0, 1, 2, ...) to barcode ID (4, 5, 6, ...).
    pub label_mapper: HashMap<usize, i32>,

    /// Per-class confidence thresholds.
    #[serde(default)]
    pub thresholds: Option<Vec<f64>>,

    /// Platt scaling parameter A for probability calibration.
    /// P = 1 / (1 + exp(A * f + B))
    #[serde(default)]
    pub prob_a: Option<Vec<f64>>,

    /// Platt scaling parameter B for probability calibration.
    #[serde(default)]
    pub prob_b: Option<Vec<f64>>,

    /// Number of classes (barcodes).
    pub n_classes: usize,

    /// Whether the model has a dedicated noise class.
    #[serde(default)]
    pub noise_class: bool,

    /// Whether to use kernel-weighted voting instead of SVM dual coefficients.
    /// Set to true for models trained with placeholder coefficients.
    #[serde(default)]
    pub use_kernel_weighted: bool,
}

fn default_version() -> String {
    "1.0".to_string()
}

impl DtwSvmModel {
    /// Get the number of training samples.
    pub fn n_samples(&self) -> usize {
        self.training_fingerprints.len()
    }

    /// Get the feature dimension.
    pub fn n_features(&self) -> usize {
        self.training_fingerprints
            .first()
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get barcode name for a class index.
    pub fn get_barcode_name(&self, class_idx: usize) -> String {
        self.label_mapper
            .get(&class_idx)
            .map(|id| format!("BC{:02}", id))
            .unwrap_or_else(|| format!("unknown_{}", class_idx))
    }

    /// Validate the model structure.
    pub fn validate(&self) -> Result<(), String> {
        // Check training data consistency
        if self.training_fingerprints.len() != self.training_labels.len() {
            return Err(format!(
                "Mismatch: {} fingerprints but {} labels",
                self.training_fingerprints.len(),
                self.training_labels.len()
            ));
        }

        // Check feature dimensions
        if let Some(first_fp) = self.training_fingerprints.first() {
            let expected_dim = first_fp.len();
            for (i, fp) in self.training_fingerprints.iter().enumerate() {
                if fp.len() != expected_dim {
                    return Err(format!(
                        "Fingerprint {} has {} features, expected {}",
                        i,
                        fp.len(),
                        expected_dim
                    ));
                }
            }
        }

        // Check support vector indices are valid
        let n_samples = self.training_fingerprints.len();
        for &idx in &self.support_indices {
            if idx >= n_samples {
                return Err(format!(
                    "Support vector index {} out of range (n_samples={})",
                    idx, n_samples
                ));
            }
        }

        // Check dual_coef dimensions
        if self.dual_coef.is_empty() {
            return Err("dual_coef is empty".to_string());
        }

        let n_sv = self.support_indices.len();
        for (i, row) in self.dual_coef.iter().enumerate() {
            if row.len() != n_sv {
                return Err(format!(
                    "dual_coef row {} has {} elements, expected {}",
                    i,
                    row.len(),
                    n_sv
                ));
            }
        }

        // Check intercept length
        let n_pairs = self.n_classes * (self.n_classes - 1) / 2;
        if self.intercept.len() != n_pairs {
            return Err(format!(
                "intercept has {} elements, expected {} for {} classes",
                self.intercept.len(),
                n_pairs,
                self.n_classes
            ));
        }

        // Check kernel params
        if self.kernel_params.gamma <= 0.0 {
            return Err(format!("Invalid gamma: {}", self.kernel_params.gamma));
        }

        if self.kernel_params.power <= 0.0 {
            return Err(format!("Invalid power: {}", self.kernel_params.power));
        }

        Ok(())
    }

    /// Save model to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), anyhow::Error> {
        use anyhow::Context;

        let file = File::create(path)
            .with_context(|| format!("Failed to create model file '{}'", path.display()))?;

        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, self)
            .with_context(|| "Failed to serialize model to JSON")?;

        Ok(())
    }
}

/// Load a DTW-SVM model from a JSON file.
///
/// # Arguments
///
/// * `path` - Path to the JSON model file
///
/// # Returns
///
/// The loaded model, or an error if loading/parsing fails.
pub fn load_svm_model(path: &Path) -> Result<DtwSvmModel, anyhow::Error> {
    use anyhow::Context;

    let file = File::open(path)
        .with_context(|| format!("Failed to open model file '{}'", path.display()))?;

    let reader = BufReader::new(file);

    let model: DtwSvmModel =
        serde_json::from_reader(reader).with_context(|| "Failed to parse SVM model JSON")?;

    // Validate the model
    model
        .validate()
        .map_err(|e| anyhow::anyhow!("Invalid model: {}", e))?;

    Ok(model)
}

/// A trained WarpDemuX model for barcode classification.
///
/// This model uses DTW distance to find the nearest training fingerprint,
/// then converts the distance to a kernel similarity score and applies
/// a threshold for classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpDemuxModel {
    /// Training fingerprints (reference barcodes).
    /// Each inner vector is a fingerprint (normalized feature vector).
    pub training_fingerprints: Vec<Vec<f64>>,

    /// Training labels corresponding to each fingerprint.
    /// These are integer IDs that map to barcode names via label_map.
    pub training_labels: Vec<i32>,

    /// RBF kernel parameters for distance-to-similarity conversion.
    pub kernel_params: KernelParams,

    /// Mapping from barcode names to integer label IDs.
    pub label_map: HashMap<String, i32>,

    /// Classification threshold.
    /// For "ratio" type: max_distance_ratio for confident classification.
    /// For "kernel" type: minimum kernel similarity.
    pub threshold: f64,

    /// Type of threshold: "ratio" or "kernel".
    pub threshold_type: String,
}

impl WarpDemuxModel {
    /// Get the barcode name for a given label ID.
    ///
    /// # Arguments
    ///
    /// * `label_id` - Integer label ID
    ///
    /// # Returns
    ///
    /// The barcode name, or "unknown" if not found.
    pub fn get_barcode_name(&self, label_id: i32) -> String {
        // Build reverse map inline - for frequent lookups, consider caching
        self.label_map
            .iter()
            .find(|&(_, &id)| id == label_id)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| format!("unknown_{}", label_id))
    }

    /// Build a reverse label map (label_id -> barcode_name) for O(1) lookups.
    ///
    /// Use this when you need to look up many barcode names.
    pub fn build_reverse_label_map(&self) -> HashMap<i32, String> {
        self.label_map
            .iter()
            .map(|(name, &id)| (id, name.clone()))
            .collect()
    }

    /// Get the number of training samples.
    pub fn num_samples(&self) -> usize {
        self.training_fingerprints.len()
    }

    /// Get the fingerprint dimension (number of features).
    pub fn feature_dim(&self) -> usize {
        self.training_fingerprints
            .first()
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Validate the model structure.
    ///
    /// Checks that:
    /// - training_fingerprints and training_labels have the same length
    /// - all fingerprints have the same dimension
    /// - threshold is valid
    ///
    /// # Returns
    ///
    /// Ok(()) if valid, Err with description if invalid.
    pub fn validate(&self) -> Result<(), String> {
        // Check matching lengths
        if self.training_fingerprints.len() != self.training_labels.len() {
            return Err(format!(
                "Mismatch: {} fingerprints but {} labels",
                self.training_fingerprints.len(),
                self.training_labels.len()
            ));
        }

        // Check consistent feature dimensions
        if let Some(first_fp) = self.training_fingerprints.first() {
            let expected_dim = first_fp.len();
            for (i, fp) in self.training_fingerprints.iter().enumerate() {
                if fp.len() != expected_dim {
                    return Err(format!(
                        "Fingerprint {} has {} features, expected {}",
                        i,
                        fp.len(),
                        expected_dim
                    ));
                }
            }
        }

        // Check threshold
        if self.threshold <= 0.0 {
            return Err(format!("Invalid threshold: {}", self.threshold));
        }

        // Check kernel params
        if self.kernel_params.gamma <= 0.0 {
            return Err(format!("Invalid gamma: {}", self.kernel_params.gamma));
        }

        if self.kernel_params.power <= 0.0 {
            return Err(format!("Invalid power: {}", self.kernel_params.power));
        }

        Ok(())
    }
}

/// Load a WarpDemuX model from a JSON file.
///
/// The JSON file should be exported using the `export_warpdemux_model.py` script.
///
/// # Arguments
///
/// * `path` - Path to the JSON model file
///
/// # Returns
///
/// The loaded model, or an error if loading/parsing fails.
///
/// # Example
///
/// ```no_run
/// use escapepod_signal::demux::load_model;
/// use std::path::Path;
///
/// let model = load_model(Path::new("model.json"))?;
/// println!("Loaded model with {} training samples", model.num_samples());
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_model(path: &Path) -> Result<WarpDemuxModel, anyhow::Error> {
    use anyhow::Context;

    let file = File::open(path)
        .with_context(|| format!("Failed to open model file '{}'", path.display()))?;

    let reader = BufReader::new(file);

    let model: WarpDemuxModel =
        serde_json::from_reader(reader).with_context(|| "Failed to parse model JSON")?;

    // Validate the model
    model
        .validate()
        .map_err(|e| anyhow::anyhow!("Invalid model: {}", e))?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_model() -> WarpDemuxModel {
        let mut label_map = HashMap::new();
        label_map.insert("BC01".to_string(), 0);
        label_map.insert("BC02".to_string(), 1);

        WarpDemuxModel {
            training_fingerprints: vec![
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
                vec![0.7, 0.8, 0.9],
            ],
            training_labels: vec![0, 1, 0],
            kernel_params: KernelParams {
                gamma: 1.0,
                power: 1.0,
            },
            label_map,
            threshold: 0.8,
            threshold_type: "ratio".to_string(),
        }
    }

    #[test]
    fn test_model_validation() {
        let model = create_test_model();
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_validation_length_mismatch() {
        let mut model = create_test_model();
        model.training_labels.push(2); // Extra label
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_model_validation_dimension_mismatch() {
        let mut model = create_test_model();
        model.training_fingerprints.push(vec![1.0, 2.0]); // Different dimension
        model.training_labels.push(1);
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_get_barcode_name() {
        let model = create_test_model();
        assert_eq!(model.get_barcode_name(0), "BC01");
        assert_eq!(model.get_barcode_name(1), "BC02");
        assert!(model.get_barcode_name(99).starts_with("unknown"));
    }

    #[test]
    fn test_model_properties() {
        let model = create_test_model();
        assert_eq!(model.num_samples(), 3);
        assert_eq!(model.feature_dim(), 3);
    }

    #[test]
    fn test_load_model_json() {
        let model = create_test_model();

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&model).unwrap();

        // Write to temp file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(json.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        // Load the model
        let loaded_model = load_model(temp_file.path()).unwrap();

        assert_eq!(loaded_model.num_samples(), model.num_samples());
        assert_eq!(loaded_model.feature_dim(), model.feature_dim());
        assert_eq!(loaded_model.threshold, model.threshold);
        assert_eq!(loaded_model.threshold_type, model.threshold_type);
    }

    // Tests for DtwSvmModel

    fn create_test_svm_model() -> DtwSvmModel {
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 4);
        label_mapper.insert(1, 5);
        label_mapper.insert(2, 6);

        DtwSvmModel {
            version: "1.0".to_string(),
            training_fingerprints: vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![2.0, 2.0, 2.0],
            ],
            training_labels: vec![4, 5, 6],
            support_indices: vec![0, 1, 2],
            dual_coef: vec![vec![1.0, -1.0, 0.5], vec![-0.5, 0.5, 1.0]],
            intercept: vec![0.0, 0.0, 0.0], // 3 pairs for 3 classes
            classes: vec![4, 5, 6],
            kernel_params: KernelParams::default(),
            window: None,
            label_mapper,
            thresholds: None,
            prob_a: None,
            prob_b: None,
            n_classes: 3,
            noise_class: false,
            use_kernel_weighted: false,
        }
    }

    #[test]
    fn test_svm_model_validation() {
        let model = create_test_svm_model();
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_svm_model_validation_sv_out_of_range() {
        let mut model = create_test_svm_model();
        model.support_indices.push(100); // Out of range
        model.dual_coef[0].push(0.0);
        model.dual_coef[1].push(0.0);
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_svm_model_validation_intercept_mismatch() {
        let mut model = create_test_svm_model();
        model.intercept.push(0.0); // Too many intercepts
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_svm_model_save_load() {
        let model = create_test_svm_model();

        // Save to temp file
        let temp_file = NamedTempFile::new().unwrap();
        model.save(temp_file.path()).unwrap();

        // Load back
        let loaded = load_svm_model(temp_file.path()).unwrap();

        assert_eq!(loaded.n_samples(), model.n_samples());
        assert_eq!(loaded.n_features(), model.n_features());
        assert_eq!(loaded.n_classes, model.n_classes);
        assert_eq!(loaded.classes, model.classes);
    }

    #[test]
    fn test_svm_model_get_barcode_name() {
        let model = create_test_svm_model();
        assert_eq!(model.get_barcode_name(0), "BC04");
        assert_eq!(model.get_barcode_name(1), "BC05");
        assert_eq!(model.get_barcode_name(2), "BC06");
        assert!(model.get_barcode_name(99).starts_with("unknown"));
    }
}

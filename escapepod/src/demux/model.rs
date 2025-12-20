//! WarpDemuX model loading and data structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
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
        for (name, &id) in &self.label_map {
            if id == label_id {
                return name.clone();
            }
        }
        format!("unknown_{}", label_id)
    }

    /// Get the number of training samples.
    pub fn num_samples(&self) -> usize {
        self.training_fingerprints.len()
    }

    /// Get the fingerprint dimension (number of features).
    pub fn feature_dim(&self) -> usize {
        self.training_fingerprints.first().map(|v| v.len()).unwrap_or(0)
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
/// use escapepod::demux::load_model;
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

    let model: WarpDemuxModel = serde_json::from_reader(reader)
        .with_context(|| "Failed to parse model JSON")?;

    // Validate the model
    model.validate()
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
}

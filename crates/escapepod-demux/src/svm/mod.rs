//! SVM prediction for barcode classification.
//!
//! This module implements SVM prediction with precomputed kernel,
//! compatible with WarpDemuX's DTW-SVM models.
//!
//! The workflow is:
//! 1. Compute DTW distances from query to training fingerprints
//! 2. Convert distances to RBF kernel: K = exp(-gamma * dist^power)
//! 3. Apply SVM decision function using dual coefficients
//! 4. Convert decision values to probabilities via Platt scaling
//!
//! The implementation is split across submodules:
//! - [`kernel`]: RBF kernel computation and DTW distance helpers.
//! - [`workspace`]: reusable scratch buffers ([`SvmWorkspace`]).
//! - [`predictor`]: OvO decision function, Platt scaling, probability coupling.
//! - [`gpu`] (feature `gpu`): batched on-device SVM classification.

mod gpu;
mod kernel;
mod predictor;
mod workspace;

pub use kernel::{compute_distances, distances_to_kernel};
pub use predictor::{SvmModel, SvmPredictor, classify_with_svm};
pub use workspace::SvmWorkspace;

pub use gpu::DEFAULT_GPU_CHUNK_CELLS;

#[cfg(feature = "gpu")]
pub use gpu::{classify_with_svm_batch_gpu, classify_with_svm_batch_gpu_with_ctx};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DtwSvmModel, KernelParams};
    use std::collections::HashMap;

    fn create_test_model() -> DtwSvmModel {
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 4);
        label_mapper.insert(1, 5);
        label_mapper.insert(2, 6);

        DtwSvmModel {
            version: "1.0".to_string(),
            training_fingerprints: vec![
                vec![0.0, 0.0, 0.0], // Class 0
                vec![1.0, 1.0, 1.0], // Class 1
                vec![2.0, 2.0, 2.0], // Class 2
            ],
            training_labels: vec![4, 5, 6],
            support_indices: vec![0, 1, 2],
            dual_coef: vec![vec![1.0, -1.0, 0.5], vec![-0.5, 0.5, 1.0]],
            intercept: vec![0.0, 0.0, 0.0],
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
    fn test_distances_to_kernel() {
        let distances = vec![0.0, 1.0, 2.0];
        let params = KernelParams::default();
        let kernel = distances_to_kernel(&distances, &params);

        assert!((kernel[0] - 1.0).abs() < 1e-10); // exp(0) = 1
        assert!((kernel[1] - (-1.0f64).exp()).abs() < 1e-10);
        assert!((kernel[2] - (-2.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_distances() {
        let query = vec![0.0, 0.0, 0.0];
        let training = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];

        let distances = compute_distances(&query, &training, None);

        assert!(distances[0] < 0.1); // Same vector, distance ~0
        assert!(distances[1] > distances[0]); // Different vector, larger distance
    }

    #[test]
    fn test_svm_predictor_decision() {
        let model = create_test_model();
        let predictor = SvmPredictor::new(&model);

        // Kernel values that strongly favor class 0
        let kernel_values = vec![1.0, 0.1, 0.05];
        let decisions = predictor.decision_function(&kernel_values);

        // Should have n_classes * (n_classes - 1) / 2 = 3 decision values
        assert_eq!(decisions.len(), 3);
    }

    #[test]
    fn test_svm_predictor_probabilities() {
        let model = create_test_model();
        let predictor = SvmPredictor::new(&model);

        let kernel_values = vec![1.0, 0.1, 0.05];
        let decisions = predictor.decision_function(&kernel_values);
        let probs = predictor.decision_to_probabilities(&decisions);

        // Should have n_classes probabilities
        assert_eq!(probs.len(), 3);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_with_svm() {
        let model = create_test_model();

        // Query close to class 0 training point
        let query = vec![0.1, 0.1, 0.1];
        let (probs, result) = classify_with_svm(&model, &query);

        assert_eq!(probs.len(), 3);
        assert!(result.is_confident);
    }

    #[test]
    fn test_kernel_weighted_scores() {
        let model = create_test_model();
        let predictor = SvmPredictor::new(&model);

        // Kernel values: high for class 0 (index 0), low for others
        // Training: sample 0 = class 4, sample 1 = class 5, sample 2 = class 6
        let kernel_values = vec![0.9, 0.1, 0.05];
        let scores = predictor.kernel_weighted_scores(&kernel_values);

        println!("Kernel values: {:?}", kernel_values);
        println!("Scores: {:?}", scores);
        println!("Training labels: {:?}", model.training_labels);
        println!("Classes: {:?}", model.classes);

        // Class 0 (barcode 4) should have highest score
        assert!(scores[0] > scores[1], "Class 0 should beat class 1");
        assert!(scores[0] > scores[2], "Class 0 should beat class 2");
    }

    #[test]
    fn test_classify_different_classes() {
        // Create a model with clear class separation
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 0);
        label_mapper.insert(1, 1);
        label_mapper.insert(2, 2);

        let model = DtwSvmModel {
            version: "1.0".to_string(),
            training_fingerprints: vec![
                vec![0.1, 0.1, 0.1], // Class 0
                vec![0.9, 0.9, 0.9], // Class 1
                vec![0.5, 0.5, 0.5], // Class 2
            ],
            training_labels: vec![0, 1, 2],
            support_indices: vec![0, 1, 2],
            dual_coef: vec![vec![0.333, 0.333, 0.333], vec![0.333, 0.333, 0.333]],
            intercept: vec![0.0, 0.0, 0.0],
            classes: vec![0, 1, 2],
            kernel_params: KernelParams::default(),
            window: None,
            label_mapper,
            thresholds: None,
            prob_a: None,
            prob_b: None,
            n_classes: 3,
            noise_class: false,
            use_kernel_weighted: true, // Use kernel-weighted voting
        };

        // Query close to class 0
        let query0 = vec![0.12, 0.12, 0.12];
        let (probs0, result0) = classify_with_svm(&model, &query0);
        println!(
            "Query near class 0: probs={:?}, predicted={}",
            probs0, result0.predicted_barcode
        );
        assert_eq!(result0.predicted_barcode, 0, "Should predict class 0");

        // Query close to class 1
        let query1 = vec![0.88, 0.88, 0.88];
        let (probs1, result1) = classify_with_svm(&model, &query1);
        println!(
            "Query near class 1: probs={:?}, predicted={}",
            probs1, result1.predicted_barcode
        );
        assert_eq!(result1.predicted_barcode, 1, "Should predict class 1");

        // Query close to class 2
        let query2 = vec![0.52, 0.52, 0.52];
        let (probs2, result2) = classify_with_svm(&model, &query2);
        println!(
            "Query near class 2: probs={:?}, predicted={}",
            probs2, result2.predicted_barcode
        );
        assert_eq!(result2.predicted_barcode, 2, "Should predict class 2");
    }
}

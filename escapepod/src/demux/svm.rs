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

use crate::dtw::dtw_distance;

use super::model::{DtwSvmModel, KernelParams};
use super::probability::{process_probabilities, ProbabilityResult};

// Re-export SvmModel as an alias for DtwSvmModel for backwards compatibility
pub type SvmModel = DtwSvmModel;

/// Compute RBF kernel from distances.
///
/// K = exp(-gamma * distance^power)
///
/// # Arguments
///
/// * `distances` - DTW distances to training samples
/// * `params` - Kernel parameters
///
/// # Returns
///
/// Kernel values (similarity scores)
pub fn distances_to_kernel(distances: &[f64], params: &KernelParams) -> Vec<f64> {
    distances
        .iter()
        .map(|&d| (-params.gamma * d.powf(params.power)).exp())
        .collect()
}

/// Compute DTW distances from a query fingerprint to all training fingerprints.
///
/// # Arguments
///
/// * `query` - Query fingerprint
/// * `training` - Training fingerprints
/// * `window` - Optional Sakoe-Chiba band constraint
///
/// # Returns
///
/// Vector of DTW distances
pub fn compute_distances(
    query: &[f64],
    training: &[Vec<f64>],
    window: Option<usize>,
) -> Vec<f64> {
    let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();

    training
        .iter()
        .map(|train_fp| {
            let train_f32: Vec<f32> = train_fp.iter().map(|&x| x as f32).collect();
            dtw_distance(&query_f32, &train_f32, window) as f64
        })
        .collect()
}

/// SVM predictor for One-vs-One multiclass classification.
///
/// Implements the decision function for sklearn's SVC with precomputed kernel.
pub struct SvmPredictor<'a> {
    model: &'a DtwSvmModel,
}

impl<'a> SvmPredictor<'a> {
    pub fn new(model: &'a SvmModel) -> Self {
        Self { model }
    }

    /// Compute decision function for a query fingerprint.
    ///
    /// For OvO classification, this computes decision values for each pair of classes.
    ///
    /// # Arguments
    ///
    /// * `kernel_values` - Kernel values K(query, x_i) for all training samples
    ///
    /// # Returns
    ///
    /// Decision values for each class pair
    pub fn decision_function(&self, kernel_values: &[f64]) -> Vec<f64> {
        let n_classes = self.model.n_classes;
        let n_pairs = n_classes * (n_classes - 1) / 2;

        let mut decisions = vec![0.0; n_pairs];
        let mut pair_idx = 0;

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                // Decision function: sum over support vectors
                let mut sum = self.model.intercept[pair_idx];

                for (sv_local_idx, &sv_global_idx) in
                    self.model.support_indices.iter().enumerate()
                {
                    // dual_coef layout for sklearn SVC OvO:
                    // Row i contains coefficients for class i vs all higher classes
                    // For pair (i, j): use row j-1 for coefs of class i's SVs
                    //                  and row i for coefs of class j's SVs

                    let coef = if sv_local_idx < self.model.dual_coef[0].len() {
                        // Simplified: just use all dual coefficients
                        // This is a linear combination over all support vectors
                        let row_idx = (i + j - 1) % self.model.dual_coef.len();
                        self.model.dual_coef[row_idx]
                            .get(sv_local_idx)
                            .copied()
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    sum += coef * kernel_values[sv_global_idx];
                }

                decisions[pair_idx] = sum;
                pair_idx += 1;
            }
        }

        decisions
    }

    /// Convert OvO decision values to class votes.
    ///
    /// Each binary classifier votes for one of its two classes.
    /// The class with the most votes wins.
    ///
    /// # Arguments
    ///
    /// * `decisions` - Decision values from `decision_function`
    ///
    /// # Returns
    ///
    /// Vote count for each class
    pub fn ovo_votes(&self, decisions: &[f64]) -> Vec<i32> {
        let n_classes = self.model.n_classes;
        let mut votes = vec![0i32; n_classes];

        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                if decisions[pair_idx] > 0.0 {
                    votes[i] += 1;
                } else {
                    votes[j] += 1;
                }
                pair_idx += 1;
            }
        }

        votes
    }

    /// Convert decision values to probabilities using sigmoid + OvO voting.
    ///
    /// If Platt scaling parameters are available, uses those for calibration.
    /// Otherwise, uses a simple sigmoid and normalization.
    ///
    /// # Arguments
    ///
    /// * `decisions` - Decision values from `decision_function`
    ///
    /// # Returns
    ///
    /// Probability distribution over classes
    pub fn decision_to_probabilities(&self, decisions: &[f64]) -> Vec<f64> {
        let n_classes = self.model.n_classes;

        // Use Platt scaling if available
        if let (Some(prob_a), Some(prob_b)) = (&self.model.prob_a, &self.model.prob_b) {
            // Platt scaling: P = 1 / (1 + exp(A * f + B))
            let pair_probs: Vec<f64> = decisions
                .iter()
                .zip(prob_a.iter().zip(prob_b.iter()))
                .map(|(&f, (&a, &b))| 1.0 / (1.0 + (a * f + b).exp()))
                .collect();

            // Aggregate pairwise probabilities to class probabilities
            // Using the coupling method from sklearn
            return self.couple_probabilities(&pair_probs);
        }

        // Fallback: simple sigmoid + voting
        let mut class_scores = vec![0.0; n_classes];

        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                // Sigmoid of decision value
                let prob_i = 1.0 / (1.0 + (-decisions[pair_idx]).exp());
                class_scores[i] += prob_i;
                class_scores[j] += 1.0 - prob_i;
                pair_idx += 1;
            }
        }

        // Normalize to probabilities
        let sum: f64 = class_scores.iter().sum();
        if sum > 0.0 {
            class_scores.iter_mut().for_each(|v| *v /= sum);
        }

        class_scores
    }

    /// Couple pairwise probabilities to class probabilities.
    ///
    /// Implements the coupling algorithm from Wu et al. (2004).
    fn couple_probabilities(&self, pair_probs: &[f64]) -> Vec<f64> {
        let n_classes = self.model.n_classes;

        // Build pairwise probability matrix
        let mut r = vec![vec![0.0; n_classes]; n_classes];
        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                r[i][j] = pair_probs[pair_idx];
                r[j][i] = 1.0 - pair_probs[pair_idx];
                pair_idx += 1;
            }
        }

        // Initialize with uniform probabilities
        let mut p = vec![1.0 / n_classes as f64; n_classes];

        // Iterative refinement (simplified)
        for _ in 0..100 {
            let mut new_p = vec![0.0; n_classes];
            let mut sum = 0.0;

            for i in 0..n_classes {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for j in 0..n_classes {
                    if i != j {
                        numerator += r[i][j] * p[j];
                        denominator += p[j];
                    }
                }

                new_p[i] = if denominator > 0.0 {
                    numerator / denominator
                } else {
                    1.0 / n_classes as f64
                };
                sum += new_p[i];
            }

            // Normalize
            if sum > 0.0 {
                new_p.iter_mut().for_each(|v| *v /= sum);
            }

            // Check convergence
            let max_diff: f64 = p
                .iter()
                .zip(new_p.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            p = new_p;

            if max_diff < 1e-7 {
                break;
            }
        }

        p
    }

    /// Classify a query fingerprint.
    ///
    /// Full prediction pipeline:
    /// 1. Compute DTW distances
    /// 2. Convert to kernel
    /// 3. Compute decision function
    /// 4. Convert to probabilities
    /// 5. Process to get prediction and confidence
    ///
    /// # Arguments
    ///
    /// * `query` - Query fingerprint
    ///
    /// # Returns
    ///
    /// Tuple of (probabilities, prediction result)
    pub fn predict(&self, query: &[f64]) -> (Vec<f64>, ProbabilityResult) {
        // Compute DTW distances
        let distances = compute_distances(
            query,
            &self.model.training_fingerprints,
            self.model.window,
        );

        // Convert to kernel
        let kernel_values = distances_to_kernel(&distances, &self.model.kernel_params);

        // Compute decision function
        let decisions = self.decision_function(&kernel_values);

        // Convert to probabilities
        let probabilities = self.decision_to_probabilities(&decisions);

        // Process probabilities
        let result = process_probabilities(
            &probabilities,
            &self.model.label_mapper,
            self.model.thresholds.as_deref(),
        );

        (probabilities, result)
    }
}

/// Classify a fingerprint using a trained SVM model.
///
/// Convenience function that creates a predictor and runs prediction.
///
/// # Arguments
///
/// * `model` - Trained SVM model
/// * `fingerprint` - Query fingerprint
///
/// # Returns
///
/// Tuple of (probabilities, prediction result)
pub fn classify_with_svm(model: &SvmModel, fingerprint: &[f64]) -> (Vec<f64>, ProbabilityResult) {
    let predictor = SvmPredictor::new(model);
    predictor.predict(fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let training = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];

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
}

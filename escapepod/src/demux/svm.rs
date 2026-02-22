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
pub fn compute_distances(query: &[f64], training: &[Vec<f64>], window: Option<usize>) -> Vec<f64> {
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

    /// Compute class scores using kernel-weighted voting.
    ///
    /// For each class, sum the kernel similarities to all training samples of that class.
    /// This provides a soft voting mechanism where closer samples contribute more.
    ///
    /// # Arguments
    ///
    /// * `kernel_values` - Kernel values K(query, x_i) for all training samples
    ///
    /// # Returns
    ///
    /// Score for each class (higher = more similar to that class)
    pub fn kernel_weighted_scores(&self, kernel_values: &[f64]) -> Vec<f64> {
        let n_classes = self.model.n_classes;
        let mut class_scores = vec![0.0; n_classes];
        let mut class_counts = vec![0usize; n_classes];

        // Map labels to class indices
        let label_to_class: std::collections::HashMap<i32, usize> = self
            .model
            .classes
            .iter()
            .enumerate()
            .map(|(idx, &label)| (label, idx))
            .collect();

        // Accumulate kernel-weighted votes for each class
        for (i, &label) in self.model.training_labels.iter().enumerate() {
            if let Some(&class_idx) = label_to_class.get(&label) {
                class_scores[class_idx] += kernel_values[i];
                class_counts[class_idx] += 1;
            }
        }

        // Normalize by class size to avoid bias toward larger classes
        for (score, count) in class_scores.iter_mut().zip(class_counts.iter()) {
            if *count > 0 {
                *score /= *count as f64;
            }
        }

        class_scores
    }

    /// Compute decision function for a query fingerprint.
    ///
    /// For OvO classification, this computes decision values for each pair of classes.
    /// Uses kernel-weighted voting if model.use_kernel_weighted is true.
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

        // Use kernel-weighted voting if specified in model
        if self.model.use_kernel_weighted {
            // Use kernel-weighted scores to generate OvO decisions
            let class_scores = self.kernel_weighted_scores(kernel_values);
            let mut decisions = vec![0.0; n_pairs];
            let mut pair_idx = 0;

            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    // Decision: positive if class i wins, negative if class j wins
                    decisions[pair_idx] = class_scores[i] - class_scores[j];
                    pair_idx += 1;
                }
            }

            return decisions;
        }

        // Use real SVM dual coefficients (libsvm OvO layout)
        //
        // For pairwise classifier (class_i, class_j) with i < j:
        //   - SVs of class i: use dual_coef[j-1][sv_idx]
        //   - SVs of class j: use dual_coef[i][sv_idx]
        //   - Other SVs: don't contribute
        let mut decisions = vec![0.0; n_pairs];

        // Build class index lookup: label -> class index
        let label_to_class: std::collections::HashMap<i32, usize> = self
            .model
            .classes
            .iter()
            .enumerate()
            .map(|(idx, &label)| (label, idx))
            .collect();

        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let mut sum = self.model.intercept[pair_idx];

                for (sv_local_idx, &sv_global_idx) in
                    self.model.support_indices.iter().enumerate()
                {
                    // Determine which class this SV belongs to
                    let sv_label = self.model.training_labels[sv_global_idx];
                    let sv_class = label_to_class.get(&sv_label).copied();

                    let coef = match sv_class {
                        Some(c) if c == i => {
                            // SV of class i: use dual_coef[j-1]
                            self.model.dual_coef[j - 1]
                                .get(sv_local_idx)
                                .copied()
                                .unwrap_or(0.0)
                        }
                        Some(c) if c == j => {
                            // SV of class j: use dual_coef[i]
                            self.model.dual_coef[i]
                                .get(sv_local_idx)
                                .copied()
                                .unwrap_or(0.0)
                        }
                        _ => 0.0, // SV of other class: doesn't contribute
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
    /// Implements libsvm's `multiclass_probability` (Wu et al. 2004).
    /// This matches sklearn's internal probability coupling exactly.
    #[allow(clippy::needless_range_loop)]
    fn couple_probabilities(&self, pair_probs: &[f64]) -> Vec<f64> {
        let k = self.model.n_classes;

        // Build pairwise probability matrix r[i][j]
        let mut r = vec![vec![0.0; k]; k];
        let mut pair_idx = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                r[i][j] = pair_probs[pair_idx].max(1e-7).min(1.0 - 1e-7);
                r[j][i] = 1.0 - r[i][j];
                pair_idx += 1;
            }
        }

        // Build Q matrix: Q[t][t] = sum_{j!=t} r[j][t]^2
        //                  Q[t][j] = -r[j][t] * r[t][j]  (j != t)
        let mut q = vec![vec![0.0; k]; k];
        for t in 0..k {
            q[t][t] = 0.0;
            for j in 0..k {
                if j != t {
                    q[t][t] += r[j][t] * r[j][t];
                    q[t][j] = -r[j][t] * r[t][j];
                }
            }
        }

        // Initialize uniform probabilities
        let mut p = vec![1.0 / k as f64; k];
        let eps = 0.005 / k as f64;
        let max_iter = 100.max(k);

        for _ in 0..max_iter {
            // Compute Qp and pQp
            let mut qp = vec![0.0; k];
            let mut p_qp = 0.0;
            for t in 0..k {
                for j in 0..k {
                    qp[t] += q[t][j] * p[j];
                }
                p_qp += p[t] * qp[t];
            }

            // Check convergence
            let max_error = (0..k)
                .map(|t| (qp[t] - p_qp).abs())
                .fold(0.0_f64, f64::max);
            if max_error < eps {
                break;
            }

            // Update each p[t]
            for t in 0..k {
                let diff = (-qp[t] + p_qp) / q[t][t];
                p[t] += diff;
                p_qp = (p_qp + diff * (diff * q[t][t] + 2.0 * qp[t]))
                    / (1.0 + diff)
                    / (1.0 + diff);
                for j in 0..k {
                    qp[j] = (qp[j] + diff * q[t][j]) / (1.0 + diff);
                    p[j] /= 1.0 + diff;
                }
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
        let distances =
            compute_distances(query, &self.model.training_fingerprints, self.model.window);

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

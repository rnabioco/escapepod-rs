//! SVM training for barcode classification.
//!
//! This module provides training functionality for DTW-SVM models,
//! only available with the `train` feature.
//!
//! The training process:
//! 1. Compute DTW distance matrix between all training fingerprints
//! 2. Convert distances to RBF kernel: K = exp(-gamma * dist^power)
//! 3. Train SVM on the kernel matrix
//! 4. Export model parameters for inference

use std::collections::HashMap;

use ndarray::Array2;
use rayon::prelude::*;

use crate::model::{DtwSvmModel, KernelParams};
use escapepod_signal::dtw::dtw_distance;

/// Training configuration for DTW-SVM.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// RBF kernel gamma parameter.
    pub gamma: f64,

    /// Power to raise distances before exponential.
    pub power: f64,

    /// SVM regularization parameter C.
    pub c: f64,

    /// DTW window constraint (Sakoe-Chiba band).
    pub window: Option<usize>,

    /// Per-class confidence thresholds (optional).
    pub thresholds: Option<Vec<f64>>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            power: 1.0,
            c: 1.0,
            window: None,
            thresholds: None,
        }
    }
}

/// Compute the full DTW distance matrix between all fingerprints.
///
/// Returns a symmetric matrix D where D[i,j] = DTW(fp_i, fp_j).
pub fn compute_distance_matrix(fingerprints: &[Vec<f64>], window: Option<usize>) -> Array2<f64> {
    let n = fingerprints.len();

    // Pre-convert each fingerprint to f32 once (was previously done per
    // DTW call, O(n^2) extra allocations).
    let f32_fps: Vec<Vec<f32>> = fingerprints
        .par_iter()
        .map(|fp| fp.iter().map(|&x| x as f32).collect())
        .collect();

    // Compute upper triangle in parallel across rows. Each row returns
    // only the strict upper part `(j > i)`; diagonal is zero.
    let upper: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let fi = &f32_fps[i];
            (i + 1..n)
                .map(|j| dtw_distance(fi, &f32_fps[j], window) as f64)
                .collect()
        })
        .collect();

    let mut distances = Array2::<f64>::zeros((n, n));
    for (i, row) in upper.into_iter().enumerate() {
        for (k, d) in row.into_iter().enumerate() {
            let j = i + 1 + k;
            distances[[i, j]] = d;
            distances[[j, i]] = d;
        }
    }

    distances
}

/// Convert distance matrix to kernel matrix using RBF.
///
/// K[i,j] = exp(-gamma * D[i,j]^power)
pub fn distance_to_kernel_matrix(distances: &Array2<f64>, gamma: f64, power: f64) -> Array2<f64> {
    distances.mapv(|d| (-gamma * d.powf(power)).exp())
}

/// GPU variant of [`compute_distance_matrix`].
///
/// Runs a single all-pairs DTW batch on the device. The result is
/// mathematically symmetric up to floating-point summation order; we
/// symmetrize the upper triangle before returning so downstream code sees
/// the same bit-exact symmetry the CPU path produces.
#[cfg(feature = "gpu")]
pub fn compute_distance_matrix_gpu(
    fingerprints: &[Vec<f64>],
    window: Option<usize>,
) -> Result<Array2<f64>, escapepod_signal::dtw::GpuDtwError> {
    let ctx = escapepod_signal::dtw::GpuDtwContext::new()?;
    compute_distance_matrix_gpu_with_ctx(&ctx, fingerprints, window)
}

/// Same as [`compute_distance_matrix_gpu`] but reuses an existing
/// [`escapepod_signal::dtw::GpuDtwContext`].
#[cfg(feature = "gpu")]
pub fn compute_distance_matrix_gpu_with_ctx(
    ctx: &escapepod_signal::dtw::GpuDtwContext,
    fingerprints: &[Vec<f64>],
    window: Option<usize>,
) -> Result<Array2<f64>, escapepod_signal::dtw::GpuDtwError> {
    let n = fingerprints.len();
    if n == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }

    let f32_fps: Vec<Vec<f32>> = fingerprints
        .iter()
        .map(|fp| fp.iter().map(|&x| x as f32).collect())
        .collect();

    let dist_f32 = ctx.distance_matrix(&f32_fps, &f32_fps, window)?;

    // Promote to f64 and symmetrize: use the upper triangle as canonical,
    // zero the diagonal. CPU `compute_distance_matrix` does the same.
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist_f32[[i, j]] as f64;
            out[[i, j]] = d;
            out[[j, i]] = d;
        }
    }
    Ok(out)
}

/// Train a DTW-SVM model.
///
/// This trains an SVM using DTW distances as the kernel.
/// The approach treats the kernel matrix K as feature vectors and
/// trains a linear SVM on it.
///
/// # Arguments
///
/// * `fingerprints` - Training fingerprints
/// * `labels` - Integer labels for each fingerprint
/// * `config` - Training configuration
///
/// # Returns
///
/// A trained `DtwSvmModel` ready for inference.
pub fn train_svm(
    fingerprints: Vec<Vec<f64>>,
    labels: Vec<i32>,
    config: &TrainConfig,
) -> Result<DtwSvmModel, anyhow::Error> {
    let distance_matrix = compute_distance_matrix(&fingerprints, config.window);
    train_svm_from_distances(fingerprints, labels, distance_matrix, config)
}

/// GPU variant of [`train_svm`]: computes the DTW distance matrix on the GPU,
/// then runs the existing CPU kernel + SVM training logic.
#[cfg(feature = "gpu")]
pub fn train_svm_gpu(
    fingerprints: Vec<Vec<f64>>,
    labels: Vec<i32>,
    config: &TrainConfig,
) -> Result<DtwSvmModel, anyhow::Error> {
    let distance_matrix = compute_distance_matrix_gpu(&fingerprints, config.window)
        .map_err(|e| anyhow::anyhow!("GPU DTW failed: {e}"))?;
    train_svm_from_distances(fingerprints, labels, distance_matrix, config)
}

/// Shared back half of training: given a precomputed DTW distance matrix,
/// build the RBF kernel, fit the SVM (binary or OvO multiclass), and package
/// the model. Used by both [`train_svm`] (CPU DTW) and [`train_svm_gpu`].
pub fn train_svm_from_distances(
    fingerprints: Vec<Vec<f64>>,
    labels: Vec<i32>,
    distance_matrix: Array2<f64>,
    config: &TrainConfig,
) -> Result<DtwSvmModel, anyhow::Error> {
    if fingerprints.len() != labels.len() {
        anyhow::bail!(
            "Mismatch: {} fingerprints but {} labels",
            fingerprints.len(),
            labels.len()
        );
    }

    if fingerprints.is_empty() {
        anyhow::bail!("No training data provided");
    }

    if distance_matrix.nrows() != fingerprints.len()
        || distance_matrix.ncols() != fingerprints.len()
    {
        anyhow::bail!(
            "Distance matrix shape {:?} does not match {} fingerprints",
            distance_matrix.shape(),
            fingerprints.len()
        );
    }

    // Get unique classes
    let mut unique_labels: Vec<i32> = labels.clone();
    unique_labels.sort();
    unique_labels.dedup();
    let n_classes = unique_labels.len();

    if n_classes < 2 {
        anyhow::bail!("Need at least 2 classes for training, found {}", n_classes);
    }

    // Create label mapper: class_index -> barcode_id
    let label_mapper: HashMap<usize, i32> = unique_labels
        .iter()
        .enumerate()
        .map(|(idx, &label)| (idx, label))
        .collect();

    // Reverse mapping for training: barcode_id -> class_index
    let label_to_idx: HashMap<i32, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(idx, &label)| (label, idx))
        .collect();

    // Convert to kernel matrix
    let kernel_matrix = distance_to_kernel_matrix(&distance_matrix, config.gamma, config.power);

    // Convert labels to class indices for linfa
    let target_indices: Vec<usize> = labels
        .iter()
        .map(|&l| *label_to_idx.get(&l).unwrap())
        .collect();

    // For binary classification, use linfa-svm directly
    // For multiclass, we'll use One-vs-Rest
    if n_classes == 2 {
        train_binary_svm(
            fingerprints,
            labels,
            &kernel_matrix,
            &target_indices,
            &label_mapper,
            unique_labels,
            config,
        )
    } else {
        train_multiclass_svm(
            fingerprints,
            labels,
            &kernel_matrix,
            &target_indices,
            &label_mapper,
            unique_labels,
            config,
        )
    }
}

/// Train binary SVM classifier.
///
/// Same stub status as `train_multiclass_svm`: returns uniform-weighted
/// kernel voting, not a real SVM. The previous SMO-fit-then-discard dance
/// is removed here too; see that function for the full explanation and
/// the `TODO(svm-real-fit)` tracking note.
fn train_binary_svm(
    fingerprints: Vec<Vec<f64>>,
    labels: Vec<i32>,
    _kernel_matrix: &Array2<f64>,
    _target_indices: &[usize],
    label_mapper: &HashMap<usize, i32>,
    classes: Vec<i32>,
    config: &TrainConfig,
) -> Result<DtwSvmModel, anyhow::Error> {
    let n_samples = fingerprints.len();
    let support_indices: Vec<usize> = (0..n_samples).collect();
    let n_classes = 2;
    let dual_coef = vec![vec![1.0 / n_samples as f64; n_samples]];
    let intercept = vec![0.0];

    Ok(DtwSvmModel {
        version: "1.0".to_string(),
        training_fingerprints: fingerprints,
        training_labels: labels,
        support_indices,
        dual_coef,
        intercept,
        classes,
        kernel_params: KernelParams {
            gamma: config.gamma,
            power: config.power,
        },
        window: config.window,
        label_mapper: label_mapper.clone(),
        thresholds: config.thresholds.clone(),
        prob_a: None,
        prob_b: None,
        n_classes,
        noise_class: false,
        use_kernel_weighted: true, // Use kernel-weighted voting since we can't extract real dual coefficients
    })
}

/// Train multiclass SVM using One-vs-One decomposition.
///
/// *Known limitation.* This path does not currently produce a real SVM — it
/// emits uniform per-sample weights and relies on the predictor's
/// kernel-weighted voting fallback (`use_kernel_weighted: true`). Proper
/// dual-coefficient extraction from linfa-svm requires either vendoring the
/// SMO solver or a Cholesky factorization of each kernel submatrix; see
/// `TODO(svm-real-fit)` in `train_svm_gpu` for the tracking note. The old
/// implementation here called `Svm::params().gaussian_kernel(1.0).fit(...)`
/// which re-applied RBF on top of the already-RBF-transformed kernel matrix
/// (meaningless) and then threw the solver's output away; removed because
/// it cost hours of single-threaded SMO for a result that was discarded.
fn train_multiclass_svm(
    fingerprints: Vec<Vec<f64>>,
    labels: Vec<i32>,
    _kernel_matrix: &Array2<f64>,
    target_indices: &[usize],
    label_mapper: &HashMap<usize, i32>,
    classes: Vec<i32>,
    config: &TrainConfig,
) -> Result<DtwSvmModel, anyhow::Error> {
    let n_samples = fingerprints.len();
    let n_classes = classes.len();
    let n_pairs = n_classes * (n_classes - 1) / 2;

    // Kernel-weighted voting weights: every sample in the OvO pair gets a
    // uniform contribution. Because `svm::decision_function` adds a
    // class-scores subtraction `scores[i] - scores[j]` in kernel-weighted
    // mode, these weights translate into a nearest-neighbour-like
    // classifier on the DTW-RBF kernel. Works passably when classes are
    // well-separated in kernel space, which is the regime DNA barcodes
    // typically live in; not a substitute for a real SVM.
    let mut all_dual_coef: Vec<Vec<f64>> = vec![vec![0.0; n_samples]; n_classes - 1];
    let intercepts = vec![0.0; n_pairs];
    for i in 0..n_classes {
        for j in (i + 1)..n_classes {
            let pair_indices: Vec<usize> = target_indices
                .iter()
                .enumerate()
                .filter_map(|(idx, &c)| (c == i || c == j).then_some(idx))
                .collect();
            if pair_indices.len() < 2 {
                continue;
            }
            let n_pair = pair_indices.len();
            for &idx in &pair_indices {
                let class_idx = target_indices[idx];
                if class_idx < n_classes - 1 {
                    all_dual_coef[class_idx][idx] += 1.0 / n_pair as f64;
                }
            }
        }
    }

    let support_indices: Vec<usize> = (0..n_samples).collect();

    Ok(DtwSvmModel {
        version: "1.0".to_string(),
        training_fingerprints: fingerprints,
        training_labels: labels,
        support_indices,
        dual_coef: all_dual_coef,
        intercept: intercepts,
        classes,
        kernel_params: KernelParams {
            gamma: config.gamma,
            power: config.power,
        },
        window: config.window,
        label_mapper: label_mapper.clone(),
        thresholds: config.thresholds.clone(),
        prob_a: None,
        prob_b: None,
        n_classes,
        noise_class: false,
        use_kernel_weighted: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_distance_matrix() {
        let fps = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
        ];

        let dm = compute_distance_matrix(&fps, None);

        // Diagonal should be zero
        assert!(dm[[0, 0]].abs() < 1e-10);
        assert!(dm[[1, 1]].abs() < 1e-10);
        assert!(dm[[2, 2]].abs() < 1e-10);

        // Should be symmetric
        assert!((dm[[0, 1]] - dm[[1, 0]]).abs() < 1e-10);
        assert!((dm[[0, 2]] - dm[[2, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_kernel_matrix() {
        let distances = Array2::<f64>::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let kernel = distance_to_kernel_matrix(&distances, 1.0, 1.0);

        // K(0,0) = exp(0) = 1
        assert!((kernel[[0, 0]] - 1.0).abs() < 1e-10);
        // K(0,1) = exp(-1)
        assert!((kernel[[0, 1]] - (-1.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_train_svm_basic() {
        let fingerprints = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1],
            vec![1.0, 1.0, 1.0],
            vec![1.1, 1.1, 1.1],
        ];
        let labels = vec![0, 0, 1, 1];

        let config = TrainConfig::default();
        let model = train_svm(fingerprints, labels, &config).unwrap();

        assert_eq!(model.n_classes, 2);
        assert_eq!(model.n_samples(), 4);
    }

    #[test]
    fn test_train_svm_multiclass() {
        let fingerprints = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![2.0, 2.0],
            vec![2.1, 2.1],
        ];
        let labels = vec![0, 0, 1, 1, 2, 2];

        let config = TrainConfig::default();
        let model = train_svm(fingerprints, labels, &config).unwrap();

        assert_eq!(model.n_classes, 3);
        assert_eq!(model.n_samples(), 6);
    }
}

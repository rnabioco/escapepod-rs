//! Kernel conversion from DTW distances for classification.
//!
//! This module provides utilities to convert DTW distance matrices into
//! kernel matrices using the RBF (Radial Basis Function) kernel.

use ndarray::Array2;

/// Convert a distance matrix to an RBF kernel matrix.
///
/// The RBF kernel is computed as:
/// `K[i, j] = exp(-gamma * D[i, j]^power)`
///
/// where `D[i, j]` is the distance between sequences i and j.
///
/// # Arguments
///
/// * `distances` - Distance matrix where `distances[i, j]` is the distance between sequences i and j
/// * `gamma` - Kernel coefficient (controls the width of the RBF kernel)
/// * `power` - Power to raise distances before applying the exponential (typically 1 or 2)
///
/// # Returns
///
/// A kernel matrix where `K[i, j] = exp(-gamma * D[i, j]^power)`
///
/// # Example
///
/// ```
/// use escapepod::dtw::{dtw_distance_matrix, distance_to_kernel};
///
/// let queries = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let references = vec![vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0]];
///
/// let distances = dtw_distance_matrix(&queries, &references, None);
/// let kernel = distance_to_kernel(&distances, 1.0, 1.0);
///
/// // Kernel values should be in range (0, 1]
/// for &value in kernel.iter() {
///     assert!(value > 0.0 && value <= 1.0);
/// }
/// ```
pub fn distance_to_kernel(distances: &Array2<f32>, gamma: f32, power: f32) -> Array2<f32> {
    distances.mapv(|d| {
        let powered = d.powf(power);
        (-gamma * powered).exp()
    })
}

/// Convert a distance matrix to an RBF kernel matrix with automatic gamma estimation.
///
/// This uses the median heuristic to automatically determine gamma:
/// `gamma = 1 / (2 * median(distances)^2)`
///
/// # Arguments
///
/// * `distances` - Distance matrix
/// * `power` - Power to raise distances before applying the exponential
///
/// # Returns
///
/// A tuple of `(kernel_matrix, gamma)` where gamma was automatically estimated
pub fn distance_to_kernel_auto(distances: &Array2<f32>, power: f32) -> (Array2<f32>, f32) {
    // Compute median of all non-zero distances
    let mut dist_vec: Vec<f32> = distances
        .iter()
        .copied()
        .filter(|&d| d > 0.0 && d.is_finite())
        .collect();

    if dist_vec.is_empty() {
        // Default gamma if no valid distances
        let gamma = 1.0;
        return (distance_to_kernel(distances, gamma, power), gamma);
    }

    dist_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if dist_vec.len() % 2 == 0 {
        let mid = dist_vec.len() / 2;
        (dist_vec[mid - 1] + dist_vec[mid]) / 2.0
    } else {
        dist_vec[dist_vec.len() / 2]
    };

    // Median heuristic for gamma
    let gamma = 1.0 / (2.0 * median.powi(2));

    (distance_to_kernel(distances, gamma, power), gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_distance_to_kernel() {
        let distances = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
        let kernel = distance_to_kernel(&distances, 1.0, 1.0);

        // exp(-1.0 * 0.0^1) = exp(0) = 1.0
        assert!((kernel[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((kernel[[1, 1]] - 1.0).abs() < 1e-6);

        // exp(-1.0 * 1.0^1) = exp(-1) ≈ 0.368
        let expected = (-1.0_f32).exp();
        assert!((kernel[[0, 1]] - expected).abs() < 1e-6);
        assert!((kernel[[1, 0]] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_distance_to_kernel_power() {
        let distances = arr2(&[[0.0, 2.0], [2.0, 0.0]]);

        // With power=1: exp(-1.0 * 2.0) = exp(-2)
        let kernel1 = distance_to_kernel(&distances, 1.0, 1.0);
        let expected1 = (-2.0_f32).exp();
        assert!((kernel1[[0, 1]] - expected1).abs() < 1e-6);

        // With power=2: exp(-1.0 * 2.0^2) = exp(-4)
        let kernel2 = distance_to_kernel(&distances, 1.0, 2.0);
        let expected2 = (-4.0_f32).exp();
        assert!((kernel2[[0, 1]] - expected2).abs() < 1e-6);
    }

    #[test]
    fn test_distance_to_kernel_gamma() {
        let distances = arr2(&[[0.0, 1.0], [1.0, 0.0]]);

        // Larger gamma -> smaller kernel values (faster decay)
        let kernel1 = distance_to_kernel(&distances, 0.5, 1.0);
        let kernel2 = distance_to_kernel(&distances, 2.0, 1.0);

        // For distance=1: exp(-0.5*1) > exp(-2.0*1)
        assert!(kernel1[[0, 1]] > kernel2[[0, 1]]);
    }

    #[test]
    fn test_kernel_values_in_range() {
        let distances = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);
        let kernel = distance_to_kernel(&distances, 1.0, 1.0);

        // All kernel values should be in (0, 1]
        for &value in kernel.iter() {
            assert!(
                value > 0.0 && value <= 1.0,
                "Kernel value {} out of range",
                value
            );
        }

        // Diagonal should be 1.0 (distance=0)
        for i in 0..3 {
            assert!((kernel[[i, i]] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_distance_to_kernel_auto() {
        let distances = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);
        let (kernel, gamma) = distance_to_kernel_auto(&distances, 1.0);

        // Gamma should be positive
        assert!(gamma > 0.0);

        // Kernel should have valid values
        for &value in kernel.iter() {
            assert!(value > 0.0 && value <= 1.0);
        }

        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((kernel[[i, i]] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kernel_symmetry() {
        let distances = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);
        let kernel = distance_to_kernel(&distances, 1.0, 1.0);

        // Kernel should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((kernel[[i, j]] - kernel[[j, i]]).abs() < 1e-6);
            }
        }
    }
}

//! Core DTW distance computation with Sakoe-Chiba band constraint.

use ndarray::Array2;
use rayon::prelude::*;

/// Compute DTW distance between two sequences.
///
/// Uses the classic DTW recurrence relation:
/// `D[i,j] = dist(a[i], b[j]) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])`
///
/// # Arguments
///
/// * `a` - First sequence
/// * `b` - Second sequence
/// * `window` - Optional Sakoe-Chiba band width. If `Some(w)`, only compute
///   distances where `|i - j| <= w`. This restricts the warping path
///   to a diagonal band and improves performance.
///
/// # Returns
///
/// The DTW distance between the two sequences.
///
/// # Example
///
/// ```
/// use escapepod::dtw::dtw_distance;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![1.0, 2.0, 3.0, 4.0];
/// let distance = dtw_distance(&a, &b, None);
/// assert_eq!(distance, 0.0);
/// ```
pub fn dtw_distance(a: &[f32], b: &[f32], window: Option<usize>) -> f32 {
    dtw_distance_bounded(a, b, window, f32::INFINITY)
}

/// Compute DTW distance with early abandonment.
///
/// If the minimum distance in any row exceeds `upper_bound`, returns `f32::INFINITY`
/// early without completing the full computation. This is useful when searching for
/// the best match and we can skip candidates that can't beat the current best.
///
/// # Arguments
///
/// * `a` - First sequence
/// * `b` - Second sequence
/// * `window` - Optional Sakoe-Chiba band width
/// * `upper_bound` - If all values in a row exceed this, return early
///
/// # Returns
///
/// The DTW distance, or `f32::INFINITY` if early abandonment occurred.
pub fn dtw_distance_bounded(a: &[f32], b: &[f32], window: Option<usize>, upper_bound: f32) -> f32 {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return f32::INFINITY;
    }

    // Use two rows for memory efficiency (current and previous)
    let mut prev = vec![f32::INFINITY; m + 1];
    let mut curr = vec![f32::INFINITY; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f32::INFINITY;

        // Determine the valid column range based on Sakoe-Chiba band
        let j_start = if let Some(w) = window {
            1.max(i.saturating_sub(w))
        } else {
            1
        };

        let j_end = if let Some(w) = window {
            m.min(i + w)
        } else {
            m
        };

        // Track minimum value in this row for early abandonment
        let mut row_min = f32::INFINITY;

        let ai = a[i - 1];
        for j in j_start..=j_end {
            let diff = ai - b[j - 1];
            let cost = diff * diff;
            // Split the chained min so LLVM can schedule the two
            // independent pair-mins in parallel (shorter critical path).
            let m1 = prev[j - 1].min(prev[j]);
            let min_prev = m1.min(curr[j - 1]);
            let v = cost + min_prev;
            curr[j] = v;
            row_min = row_min.min(v);
        }

        // Early abandonment: if minimum in row exceeds bound, can't beat best
        if row_min > upper_bound {
            return f32::INFINITY;
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m].sqrt()
}

/// Compute the full DTW distance matrix between query sequences and reference sequences.
///
/// This function computes pairwise DTW distances between all query sequences and all
/// reference sequences in parallel using rayon.
///
/// # Arguments
///
/// * `queries` - Slice of query sequences
/// * `references` - Slice of reference sequences
/// * `window` - Optional Sakoe-Chiba band width
///
/// # Returns
///
/// A 2D array where `result[i, j]` is the DTW distance between `queries[i]` and `references[j]`.
///
/// # Example
///
/// ```
/// use escapepod::dtw::dtw_distance_matrix;
///
/// let queries = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let references = vec![vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0]];
/// let matrix = dtw_distance_matrix(&queries, &references, None);
/// assert_eq!(matrix.shape(), &[2, 2]);
/// ```
pub fn dtw_distance_matrix(
    queries: &[Vec<f32>],
    references: &[Vec<f32>],
    window: Option<usize>,
) -> Array2<f32> {
    let n_queries = queries.len();
    let n_refs = references.len();

    // Compute distances in parallel
    let distances: Vec<f32> = (0..n_queries)
        .into_par_iter()
        .flat_map(|i| {
            (0..n_refs)
                .map(|j| dtw_distance(&queries[i], &references[j], window))
                .collect::<Vec<_>>()
        })
        .collect();

    // Reshape into matrix
    Array2::from_shape_vec((n_queries, n_refs), distances)
        .expect("Failed to create distance matrix")
}

/// Compute DTW distance matrix with block-based parallelization.
///
/// This divides the distance matrix into blocks and computes them in parallel,
/// which can be more efficient for very large matrices.
///
/// # Arguments
///
/// * `queries` - Slice of query sequences
/// * `references` - Slice of reference sequences
/// * `window` - Optional Sakoe-Chiba band width
/// * `block_size` - Size of blocks for parallel computation
///
/// # Returns
///
/// A 2D array where `result[i, j]` is the DTW distance between `queries[i]` and `references[j]`.
pub fn dtw_distance_matrix_blocked(
    queries: &[Vec<f32>],
    references: &[Vec<f32>],
    window: Option<usize>,
    block_size: usize,
) -> Array2<f32> {
    let n_queries = queries.len();
    let n_refs = references.len();

    let mut result = Array2::zeros((n_queries, n_refs));

    // Generate block indices
    let blocks: Vec<_> = (0..n_queries)
        .step_by(block_size)
        .flat_map(|i| {
            (0..n_refs).step_by(block_size).map(move |j| {
                let i_end = (i + block_size).min(n_queries);
                let j_end = (j + block_size).min(n_refs);
                (i, i_end, j, j_end)
            })
        })
        .collect();

    // Compute blocks in parallel
    let block_results: Vec<_> = blocks
        .par_iter()
        .map(|&(i_start, i_end, j_start, j_end)| {
            let mut block = Array2::zeros((i_end - i_start, j_end - j_start));
            for i in i_start..i_end {
                for j in j_start..j_end {
                    block[[i - i_start, j - j_start]] =
                        dtw_distance(&queries[i], &references[j], window);
                }
            }
            (i_start, i_end, j_start, j_end, block)
        })
        .collect();

    // Assemble result matrix
    for (i_start, i_end, j_start, j_end, block) in block_results {
        for i in i_start..i_end {
            for j in j_start..j_end {
                result[[i, j]] = block[[i - i_start, j - j_start]];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_identical_sequences() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let distance = dtw_distance(&a, &b, None);
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_symmetric() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let d1 = dtw_distance(&a, &b, None);
        let d2 = dtw_distance(&b, &a, None);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_dtw_known_distance() {
        // Simple case: [0] vs [1] should give distance of 1
        let a = vec![0.0];
        let b = vec![1.0];
        let distance = dtw_distance(&a, &b, None);
        assert_eq!(distance, 1.0);

        // [0, 0] vs [1, 1]: accumulated squared cost = 1+1 = 2, sqrt(2)
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let distance = dtw_distance(&a, &b, None);
        assert!((distance - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_dtw_with_window() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // With window constraint, identical sequences should still have distance 0
        let distance = dtw_distance(&a, &b, Some(2));
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_empty_sequences() {
        let a: Vec<f32> = vec![];
        let b = vec![1.0, 2.0];
        let distance = dtw_distance(&a, &b, None);
        assert!(distance.is_infinite());

        let a = vec![1.0, 2.0];
        let b: Vec<f32> = vec![];
        let distance = dtw_distance(&a, &b, None);
        assert!(distance.is_infinite());
    }

    #[test]
    fn test_dtw_distance_matrix() {
        let queries = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let references = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let matrix = dtw_distance_matrix(&queries, &references, None);

        assert_eq!(matrix.shape(), &[2, 2]);
        // Diagonal should be zero (identical sequences)
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[1, 1]], 0.0);
        // Matrix should be symmetric
        assert_eq!(matrix[[0, 1]], matrix[[1, 0]]);
    }

    #[test]
    fn test_dtw_distance_matrix_blocked() {
        let queries = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let references = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let matrix1 = dtw_distance_matrix(&queries, &references, None);
        let matrix2 = dtw_distance_matrix_blocked(&queries, &references, None, 2);

        // Both methods should produce the same result
        assert_eq!(matrix1.shape(), matrix2.shape());
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix1[[i, j]], matrix2[[i, j]]);
            }
        }
    }

    #[test]
    fn test_dtw_alignment_stretch() {
        // Test that DTW can handle sequences of different lengths
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.5, 2.0, 2.5, 3.0];

        let distance = dtw_distance(&a, &b, None);
        // Should be able to align despite length difference
        assert!(distance < f32::INFINITY);
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_dtw_bounded_no_abandonment() {
        // With high upper bound, should return same result as unbounded
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let unbounded = dtw_distance(&a, &b, None);
        let bounded = dtw_distance_bounded(&a, &b, None, 100.0);
        assert_eq!(unbounded, bounded);
    }

    #[test]
    fn test_dtw_bounded_early_abandonment() {
        // With very low upper bound, should abandon early
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![10.0, 10.0, 10.0, 10.0];

        // The actual distance would be 40, so bound of 5 should cause abandonment
        let bounded = dtw_distance_bounded(&a, &b, None, 5.0);
        assert!(bounded.is_infinite());
    }

    #[test]
    fn test_dtw_bounded_exact_bound() {
        // Bound equal to actual distance should not abandon
        let a = vec![0.0];
        let b = vec![1.0];

        // Distance is exactly 1.0
        let bounded = dtw_distance_bounded(&a, &b, None, 1.0);
        assert_eq!(bounded, 1.0);

        // Just below should abandon
        let bounded_low = dtw_distance_bounded(&a, &b, None, 0.5);
        assert!(bounded_low.is_infinite());
    }
}

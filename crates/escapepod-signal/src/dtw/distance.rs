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
/// use escapepod_signal::dtw::dtw_distance;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![1.0, 2.0, 3.0, 4.0];
/// let distance = dtw_distance(&a, &b, None);
/// assert_eq!(distance, 0.0);
/// ```
pub fn dtw_distance(a: &[f32], b: &[f32], window: Option<usize>) -> f32 {
    dtw_distance_bounded_penalty(a, b, window, f32::INFINITY, 0.0)
}

/// Compute DTW distance with a warping penalty.
///
/// Matches `dtaidistance`'s `penalty` parameter (used by WarpDemuX): a fixed
/// cost added to the two non-diagonal (expansion / compression) transitions of
/// the DP recurrence, so the alignment is biased toward the diagonal:
///
/// `D[i,j] = (a_i - b_j)^2 + min(D[i-1,j-1], D[i-1,j] + penalty^2, D[i,j-1] + penalty^2)`
///
/// `penalty` matches dtaidistance's parameter: it is given in the *non-squared*
/// distance space, so each warping step contributes `penalty^2` to this DP's
/// squared cumulative cost (the final distance is `sqrt(D[n,m])`). Verified
/// against `dtaidistance.dtw.distance(..., penalty=p, use_c=True)`. `penalty ==
/// 0.0` is bit-identical to [`dtw_distance`].
pub fn dtw_distance_penalty(a: &[f32], b: &[f32], window: Option<usize>, penalty: f32) -> f32 {
    dtw_distance_bounded_penalty(a, b, window, f32::INFINITY, penalty)
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
    dtw_distance_bounded_penalty(a, b, window, upper_bound, 0.0)
}

/// Core DTW with both early abandonment and a warping `penalty`. See
/// [`dtw_distance_bounded`] and [`dtw_distance_penalty`]. The penalty is added
/// to the expansion (`prev[j]`) and compression (`curr[j-1]`) neighbors before
/// the `min`; with `penalty == 0.0` (`x + 0.0 == x` for finite and ±INF) the
/// arithmetic is identical to the no-penalty path.
pub fn dtw_distance_bounded_penalty(
    a: &[f32],
    b: &[f32],
    window: Option<usize>,
    upper_bound: f32,
    penalty: f32,
) -> f32 {
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return f32::INFINITY;
    }

    // dtaidistance's `penalty` is expressed in the *non-squared* distance space,
    // but this DP accumulates squared local costs (`(a-b)^2`) and takes a single
    // `sqrt` at the end. To match, each warping step adds `penalty^2` to the
    // squared cumulative (verified: dtaidistance `distance([0,0,0],[0], penalty=p)`
    // == sqrt(2 * p^2)). `penalty == 0.0` keeps `pen_sq == 0.0`, a no-op.
    let pen_sq = penalty * penalty;

    // Classical Sakoe-Chiba: the endpoint `(n, m)` itself has to lie in the
    // band. If `|n - m| > w` no alignment is possible and the DP would
    // otherwise return a stale value left over from an earlier in-band row.
    if let Some(w) = window
        && n.abs_diff(m) > w
    {
        return f32::INFINITY;
    }

    // Two rows for memory efficiency (current and previous) plus two scratch
    // buffers that hold per-row precomputed values. The inner loop is split
    // into a vectorizable "precompute" pass (no loop-carried deps, LLVM
    // auto-vectorizes to AVX2) and a short serial "chain" pass that applies
    // the `curr[j-1]` left-neighbor dependency.
    let mut prev = vec![f32::INFINITY; m + 1];
    let mut curr = vec![f32::INFINITY; m + 1];
    let mut cost_buf = vec![0.0f32; m + 1];
    let mut prev_min_buf = vec![0.0f32; m + 1];
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

        // Classical Sakoe-Chiba treats cells outside the band as unreachable
        // (INF). `curr[j_start - 1]` still holds whatever a prior row wrote,
        // which can be a finite value and would let the DP cheat by using an
        // out-of-band predecessor. Re-seed it to INF before we compute the
        // row so the `curr[j-1]` read below is correct for j == j_start.
        // Only needed when the inner loop actually runs and is reading from
        // a non-zero-indexed boundary.
        if j_start <= j_end && j_start > 1 {
            curr[j_start - 1] = f32::INFINITY;
        }

        if j_start > j_end {
            std::mem::swap(&mut prev, &mut curr);
            continue;
        }

        let ai = a[i - 1];
        let len = j_end - j_start + 1;

        // Pass 1 (vectorizable): cost[k] = (ai - b[j-1])^2,
        //                      prev_min[k] = min(prev[j-1], prev[j])
        // where k = j - j_start. No loop-carried deps on writes; LLVM
        // auto-vectorizes to AVX2 8-wide with -C target-cpu=x86-64-v3.
        {
            let b_slice = &b[j_start - 1..j_end];
            let prev_left = &prev[j_start - 1..j_end];
            let prev_right = &prev[j_start..=j_end];
            let cost = &mut cost_buf[..len];
            let pm = &mut prev_min_buf[..len];
            for k in 0..len {
                let diff = ai - b_slice[k];
                cost[k] = diff * diff;
                // Diagonal (prev_left) takes no penalty; expansion (prev_right)
                // does. `+ 0.0` is a no-op for the default penalty.
                pm[k] = prev_left[k].min(prev_right[k] + pen_sq);
            }
        }

        // Pass 2 (serial): apply the `curr[j-1]` left-chain and update curr.
        //   curr[j] = cost[k] + min(prev_min[k], curr[j-1])
        // Only three fp ops per iteration: min, add, row_min update.
        let mut row_min = f32::INFINITY;
        let mut left = curr[j_start - 1];
        let cost = &cost_buf[..len];
        let pm = &prev_min_buf[..len];
        let out = &mut curr[j_start..=j_end];
        for k in 0..len {
            // Compression (left neighbor, `curr[j-1]`) takes the penalty too.
            let v = cost[k] + pm[k].min(left + pen_sq);
            out[k] = v;
            left = v;
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
/// use escapepod_signal::dtw::dtw_distance_matrix;
///
/// let queries = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let references = vec![vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0]];
/// let matrix = dtw_distance_matrix(&queries, &references, None);
/// assert_eq!(matrix.shape(), &[2, 2]);
/// ```
pub fn dtw_distance_matrix<Q, R>(
    queries: &[Q],
    references: &[R],
    window: Option<usize>,
) -> Array2<f32>
where
    Q: AsRef<[f32]> + Sync,
    R: AsRef<[f32]> + Sync,
{
    let n_queries = queries.len();
    let n_refs = references.len();

    // Flat row-major buffer; rayon writes each row in parallel. The previous
    // `flat_map(... .collect::<Vec<_>>())` allocated n_queries temporary Vecs
    // just to be flattened and dropped.
    let mut distances = vec![0.0f32; n_queries * n_refs];
    distances
        .par_chunks_mut(n_refs)
        .enumerate()
        .for_each(|(i, row)| {
            let q = queries[i].as_ref();
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = dtw_distance(q, references[j].as_ref(), window);
            }
        });

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
    fn test_dtw_penalty_zero_matches_unpenalized() {
        // penalty == 0.0 must be bit-identical to the no-penalty path.
        let a = vec![1.0, 2.0, 3.0, 2.5, 4.0];
        let b = vec![1.0, 1.5, 3.0, 4.0];
        assert_eq!(
            dtw_distance(&a, &b, None),
            dtw_distance_penalty(&a, &b, None, 0.0)
        );
        assert_eq!(
            dtw_distance(&a, &b, Some(2)),
            dtw_distance_penalty(&a, &b, Some(2), 0.0)
        );
    }

    #[test]
    fn test_dtw_penalty_forced_warp() {
        // Aligning [0,0,0] to [0] forces two compression steps, each charged
        // `penalty^2` in the squared cumulative: D = 2*penalty^2, distance =
        // sqrt(2)*penalty. Matches dtaidistance: distance([0,0,0],[0],penalty=p).
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0];
        assert!(dtw_distance_penalty(&a, &b, None, 0.0).abs() < 1e-6);
        // dtaidistance penalty=0.1 -> sqrt(2 * 0.1^2) = 0.14142...
        let d = dtw_distance_penalty(&a, &b, None, 0.1);
        let expected = (2.0f32 * 0.1 * 0.1).sqrt();
        assert!((d - expected).abs() < 1e-6, "expected {expected}, got {d}");
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

//! DTW subsequence alignment for finding best match of query within a longer series.
//!
//! This implements the subsequence matching algorithm used by WarpDemuX to locate
//! the barcode region within the segmented adapter signal.

/// Result of a subsequence alignment search.
#[derive(Debug, Clone)]
pub struct SubsequenceMatch {
    /// Start index in the series (inclusive).
    pub start: usize,
    /// End index in the series (exclusive).
    pub end: usize,
    /// DTW distance of this match.
    pub distance: f32,
}

/// Find the best matching subsequence of `series` that aligns with `query`.
///
/// This uses a DTW-based approach with open-end matching, allowing the query
/// to match any contiguous subsequence of the series.
///
/// # Arguments
///
/// * `query` - The short sequence to find (e.g., consensus model)
/// * `series` - The longer sequence to search in (e.g., segmented adapter events)
/// * `penalty` - Penalty for skipping elements (default: 1.5)
///
/// # Returns
///
/// The best matching subsequence with its start/end indices and distance.
///
/// # Algorithm
///
/// Uses open-end DTW where:
/// - First row initialized to 0 (can start matching at any position)
/// - Last row gives scores for ending at each position
/// - Backtrack to find where match started
pub fn dtw_subsequence_match(query: &[f32], series: &[f32], penalty: f32) -> Option<SubsequenceMatch> {
    let n = query.len(); // rows
    let m = series.len(); // cols

    if n == 0 || m == 0 || n > m {
        return None;
    }

    // DTW matrix: D[i][j] = cost to align query[0..i] with some subsequence ending at series[j]
    // First row initialized to 0: matching can start anywhere
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];

    // Initialize: can start matching at any position in series
    for j in 0..=m {
        dp[0][j] = 0.0;
    }

    // Fill the DP matrix
    for i in 1..=n {
        for j in 1..=m {
            let cost = (query[i - 1] - series[j - 1]).abs();

            // Standard DTW transitions with penalty for insertions/deletions
            let match_cost = dp[i - 1][j - 1] + cost;
            let insert_cost = dp[i - 1][j] + cost + penalty; // skip series element
            let delete_cost = dp[i][j - 1] + penalty; // skip query element

            dp[i][j] = match_cost.min(insert_cost).min(delete_cost);
        }
    }

    // Find best end position (minimum in last row of query)
    let mut best_end = 0;
    let mut best_dist = f32::INFINITY;
    for j in 1..=m {
        if dp[n][j] < best_dist {
            best_dist = dp[n][j];
            best_end = j;
        }
    }

    if best_dist.is_infinite() {
        return None;
    }

    // Backtrack to find start position
    let mut i = n;
    let mut j = best_end;
    let mut start = j;

    while i > 0 && j > 0 {
        let current = dp[i][j];
        let cost = (query[i - 1] - series[j - 1]).abs();

        // Check which transition we came from
        if i > 0 && j > 0 && (dp[i - 1][j - 1] + cost - current).abs() < 1e-6 {
            i -= 1;
            j -= 1;
            start = j;
        } else if i > 0 && (dp[i - 1][j] + cost + penalty - current).abs() < 1e-6 {
            i -= 1;
            start = j;
        } else if j > 0 {
            j -= 1;
        } else {
            break;
        }
    }

    Some(SubsequenceMatch {
        start,
        end: best_end,
        distance: best_dist,
    })
}

/// Simplified subsequence search using sliding window DTW.
///
/// For each starting position in the series, compute DTW distance to query
/// and find the position with minimum distance.
///
/// This is simpler but may be slower for long series.
pub fn dtw_subsequence_sliding(
    query: &[f32],
    series: &[f32],
    window_expansion: usize,
) -> Option<SubsequenceMatch> {
    let n = query.len();
    let m = series.len();

    if n == 0 || m == 0 || n > m {
        return None;
    }

    let mut best_start = 0;
    let mut best_dist = f32::INFINITY;

    // Slide window across series
    let max_start = m.saturating_sub(n.saturating_sub(window_expansion));
    for start in 0..=max_start {
        let end = (start + n + window_expansion).min(m);
        let window = &series[start..end];

        let dist = crate::dtw::dtw_distance(
            &query.iter().map(|&x| x).collect::<Vec<_>>(),
            &window.iter().map(|&x| x).collect::<Vec<_>>(),
            None,
        );

        if dist < best_dist {
            best_dist = dist;
            best_start = start;
        }
    }

    if best_dist.is_infinite() {
        return None;
    }

    Some(SubsequenceMatch {
        start: best_start,
        end: (best_start + n + window_expansion).min(m),
        distance: best_dist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subsequence_exact_match() {
        let query = vec![1.0, 2.0, 3.0];
        let series = vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0];

        let result = dtw_subsequence_match(&query, &series, 1.5).unwrap();
        assert_eq!(result.start, 2);
        assert_eq!(result.end, 5);
        assert!(result.distance < 0.1); // Should be near 0
    }

    #[test]
    fn test_subsequence_at_start() {
        let query = vec![1.0, 2.0, 3.0];
        let series = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];

        let result = dtw_subsequence_match(&query, &series, 1.5).unwrap();
        assert_eq!(result.start, 0);
        assert_eq!(result.end, 3);
    }

    #[test]
    fn test_subsequence_at_end() {
        let query = vec![1.0, 2.0, 3.0];
        let series = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];

        let result = dtw_subsequence_match(&query, &series, 1.5).unwrap();
        assert_eq!(result.start, 3);
        assert_eq!(result.end, 6);
    }

    #[test]
    fn test_subsequence_empty() {
        let query: Vec<f32> = vec![];
        let series = vec![1.0, 2.0, 3.0];

        assert!(dtw_subsequence_match(&query, &series, 1.5).is_none());
    }

    #[test]
    fn test_subsequence_query_longer() {
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series = vec![1.0, 2.0, 3.0];

        assert!(dtw_subsequence_match(&query, &series, 1.5).is_none());
    }
}

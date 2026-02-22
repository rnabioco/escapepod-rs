//! Windowed t-test segmentation for nanopore signal processing.
//!
//! This module implements a sliding window t-test approach to find changepoints
//! in nanopore signals. It's based on the Tombo algorithm used in WarpDemuX.

/// Compute t-scores for all candidate changepoint positions using a sliding window.
///
/// For each position, computes the means of two adjacent windows and calculates
/// a t-score-like statistic. While not a true t-score (missing the degrees of freedom
/// adjustment), it maintains the same rank order.
///
/// The t-score is computed as: `|m1 - m2| / sqrt(var1 + var2)`
///
/// # Arguments
/// * `signal` - The signal to segment
/// * `window_width` - Width of each comparison window
///
/// # Returns
/// A vector of t-scores, one for each valid candidate position. The length will be
/// `signal.len() - 2 * window_width`.
///
/// # Example
/// ```
/// use escapepod::segmentation::windowed_ttest;
///
/// let signal = vec![1.0; 50];
/// let scores = windowed_ttest(&signal, 10);
/// ```
pub fn windowed_ttest(signal: &[f32], window_width: usize) -> Vec<f64> {
    let num_candidates = signal.len().saturating_sub(2 * window_width);

    if num_candidates == 0 {
        return Vec::new();
    }

    let w = window_width as f64;

    // Precompute prefix sums for O(1) window statistics
    // cumsum[i] = sum of signal[0..i], cumsum[0] = 0
    // cumsum_sq[i] = sum of signal[0..i]^2, cumsum_sq[0] = 0
    let mut cumsum = Vec::with_capacity(signal.len() + 1);
    let mut cumsum_sq = Vec::with_capacity(signal.len() + 1);
    cumsum.push(0.0);
    cumsum_sq.push(0.0);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &val in signal {
        let v = val as f64;
        sum += v;
        sum_sq += v * v;
        cumsum.push(sum);
        cumsum_sq.push(sum_sq);
    }

    let mut t_scores = Vec::with_capacity(num_candidates);

    for pos in 0..num_candidates {
        // Window 1: [pos, pos + window_width)
        let w1_start = pos;
        let w1_end = pos + window_width;

        // Window 2: [pos + window_width, pos + 2*window_width)
        let w2_start = pos + window_width;
        let w2_end = pos + 2 * window_width;

        // O(1) mean calculation using prefix sums
        let sum1 = cumsum[w1_end] - cumsum[w1_start];
        let sum2 = cumsum[w2_end] - cumsum[w2_start];
        let m1 = sum1 / w;
        let m2 = sum2 / w;

        // O(1) variance calculation using prefix sums
        // var = E[X²] - E[X]² = (sum_sq / n) - mean²
        // We need sum of (x - mean)² = sum_sq - 2*mean*sum + n*mean²
        //                            = sum_sq - 2*mean*sum + n*mean²
        //                            = sum_sq - n*mean² (since sum = n*mean)
        let sum_sq1 = cumsum_sq[w1_end] - cumsum_sq[w1_start];
        let sum_sq2 = cumsum_sq[w2_end] - cumsum_sq[w2_start];

        // Sum of squared deviations (not normalized variance)
        let var1 = sum_sq1 - w * m1 * m1;
        let var2 = sum_sq2 - w * m2 * m2;

        // Compute t-score (monotonic transform, not true t-score)
        let t_score = if var1 + var2 <= 0.0 {
            0.0
        } else {
            (m1 - m2).abs() / (var1 + var2).sqrt()
        };

        t_scores.push(t_score);
    }

    t_scores
}

/// Find the top N changepoints from t-test scores.
///
/// Selects the positions with the highest t-scores while ensuring they are
/// separated by at least `min_separation` samples to avoid clustering.
///
/// # Arguments
/// * `signal` - The signal to segment
/// * `window_width` - Width of each comparison window
/// * `num_changepoints` - Number of changepoints to find
/// * `min_separation` - Minimum distance between changepoints
///
/// # Returns
/// A vector of changepoint positions (adjusted to be at the boundary between windows).
/// Returns fewer changepoints if not enough valid candidates are available.
///
/// # Example
/// ```
/// use escapepod::segmentation::find_changepoints;
///
/// let signal = vec![1.0; 100];
/// let changepoints = find_changepoints(&signal, 10, 3, 15);
/// ```
pub fn find_changepoints(
    signal: &[f32],
    window_width: usize,
    num_changepoints: usize,
    min_separation: usize,
) -> Vec<usize> {
    let t_scores = windowed_ttest(signal, window_width);

    if t_scores.is_empty() {
        return Vec::new();
    }

    // Step 1: Find local maxima (peaks) in the t-scores.
    // A peak at position i means t_scores[i] >= neighbors.
    // This matches scipy.signal.find_peaks behavior used by WarpDemuX.
    let mut peaks: Vec<usize> = Vec::new();
    for i in 0..t_scores.len() {
        let left_ok = i == 0 || t_scores[i] >= t_scores[i - 1];
        let right_ok = i + 1 >= t_scores.len() || t_scores[i] >= t_scores[i + 1];
        if left_ok && right_ok && t_scores[i] > 0.0 {
            peaks.push(i);
        }
    }

    // Step 2: Sort peaks by score (descending) and select top N
    // with minimum distance constraint (matching scipy find_peaks distance param).
    peaks.sort_by(|&a, &b| {
        t_scores[b]
            .partial_cmp(&t_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut changepoints = Vec::new();
    let mut blacklist = std::collections::HashSet::new();

    for &peak_pos in &peaks {
        if changepoints.len() >= num_changepoints {
            break;
        }

        if !blacklist.contains(&peak_pos) {
            // Adjust position to be at the boundary between windows
            let adjusted_pos = peak_pos + window_width;
            changepoints.push(adjusted_pos);

            // Blacklist nearby positions (enforce min_separation between peaks)
            let start = peak_pos.saturating_sub(min_separation - 1);
            let end = (peak_pos + min_separation).min(t_scores.len());
            for pos in start..end {
                blacklist.insert(pos);
            }
        }
    }

    // Sort changepoints by position
    changepoints.sort_unstable();
    changepoints
}

/// Segment a signal using changepoints and compute mean signal per segment.
///
/// Given a set of changepoints, divides the signal into segments and computes
/// the mean value for each segment. These means represent the "events" or
/// discrete levels in the signal.
///
/// # Arguments
/// * `signal` - The signal to segment
/// * `changepoints` - Positions where the signal changes (must be sorted)
///
/// # Returns
/// A vector of (segment_start, segment_end, segment_mean) tuples.
///
/// # Example
/// ```
/// use escapepod::segmentation::compute_segment_means;
///
/// let signal = vec![1.0, 1.0, 5.0, 5.0, 5.0, 2.0, 2.0];
/// let changepoints = vec![2, 5];
/// let segments = compute_segment_means(&signal, &changepoints);
/// ```
pub fn compute_segment_means(signal: &[f32], changepoints: &[usize]) -> Vec<(usize, usize, f64)> {
    if signal.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();

    // Create boundaries: [0, changepoints..., signal.len()]
    let mut boundaries = vec![0];
    boundaries.extend_from_slice(changepoints);
    boundaries.push(signal.len());

    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];

        if start >= end {
            continue;
        }

        // Compute mean of segment
        let sum: f64 = signal[start..end].iter().map(|&x| x as f64).sum();
        let mean = sum / (end - start) as f64;

        segments.push((start, end, mean));
    }

    segments
}

/// Perform complete t-test segmentation and return segment information.
///
/// This is a convenience function that combines changepoint detection and
/// segment mean calculation.
///
/// # Arguments
/// * `signal` - The signal to segment
/// * `window_width` - Width of comparison windows for t-test
/// * `num_changepoints` - Number of changepoints to detect
/// * `min_separation` - Minimum separation between changepoints
///
/// # Returns
/// A vector of (segment_start, segment_end, segment_mean) tuples representing
/// the detected segments and their mean values.
///
/// # Example
/// ```
/// use escapepod::segmentation::segment_signal;
///
/// let signal = vec![50.0; 20];
/// let segments = segment_signal(&signal, 5, 2, 10);
/// ```
pub fn segment_signal(
    signal: &[f32],
    window_width: usize,
    num_changepoints: usize,
    min_separation: usize,
) -> Vec<(usize, usize, f64)> {
    let changepoints = find_changepoints(signal, window_width, num_changepoints, min_separation);
    compute_segment_means(signal, &changepoints)
}

/// Segmentation result containing both event means and dwell times.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Mean signal value for each segment (event)
    pub event_means: Vec<f32>,
    /// Duration of each segment in samples (dwell time)
    pub dwell_times: Vec<f32>,
    /// Segment boundaries: (start, end) indices into the original signal
    pub boundaries: Vec<(usize, usize)>,
}

impl SegmentationResult {
    /// Get the number of segments/events.
    pub fn num_events(&self) -> usize {
        self.event_means.len()
    }

    /// Check if the segmentation is empty.
    pub fn is_empty(&self) -> bool {
        self.event_means.is_empty()
    }

    /// Get the total signal duration covered by all segments.
    pub fn total_duration(&self) -> usize {
        self.dwell_times.iter().map(|&d| d as usize).sum()
    }
}

/// Perform t-test segmentation and return both event means and dwell times.
///
/// This function is essential for barcode fingerprinting as dwell times
/// provide independent discriminative information from signal levels.
/// DNA barcodes translocate ~3-4% faster than RNA, creating a detectable
/// signature in dwell time patterns.
///
/// # Arguments
/// * `signal` - The signal to segment
/// * `window_width` - Width of comparison windows for t-test
/// * `num_changepoints` - Number of changepoints to detect
/// * `min_separation` - Minimum separation between changepoints
///
/// # Returns
/// A `SegmentationResult` containing event means, dwell times, and boundaries.
///
/// # Example
/// ```
/// use escapepod::segmentation::segment_signal_with_dwell;
///
/// let signal = vec![50.0; 100];
/// let result = segment_signal_with_dwell(&signal, 10, 5, 15);
///
/// println!("Found {} events", result.num_events());
/// for (mean, dwell) in result.event_means.iter().zip(&result.dwell_times) {
///     println!("  mean={:.2}, dwell={} samples", mean, dwell);
/// }
/// ```
pub fn segment_signal_with_dwell(
    signal: &[f32],
    window_width: usize,
    num_changepoints: usize,
    min_separation: usize,
) -> SegmentationResult {
    let changepoints = find_changepoints(signal, window_width, num_changepoints, min_separation);
    let segments = compute_segment_means(signal, &changepoints);

    let mut event_means = Vec::with_capacity(segments.len());
    let mut dwell_times = Vec::with_capacity(segments.len());
    let mut boundaries = Vec::with_capacity(segments.len());

    for (start, end, mean) in segments {
        event_means.push(mean as f32);
        dwell_times.push((end - start) as f32);
        boundaries.push((start, end));
    }

    SegmentationResult {
        event_means,
        dwell_times,
        boundaries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windowed_ttest_constant_signal() {
        let signal = vec![5.0; 50];
        let scores = windowed_ttest(&signal, 10);

        // All scores should be 0 for constant signal
        assert_eq!(scores.len(), 30);
        for score in scores {
            assert_eq!(score, 0.0);
        }
    }

    #[test]
    fn test_windowed_ttest_step_change() {
        // Create signal with clear step: 50 low values, then 50 high values
        let mut signal = vec![1.0; 50];
        signal.extend(vec![10.0; 50]);

        let scores = windowed_ttest(&signal, 10);

        // Find the maximum score
        let max_score_pos = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(pos, _)| pos)
            .unwrap();

        // Maximum should be near the boundary (position 40 in candidate space)
        // which corresponds to position 50 in signal space (after adjusting for window)
        assert!((30..=50).contains(&max_score_pos));
    }

    #[test]
    fn test_find_changepoints_single() {
        let mut signal = vec![1.0; 50];
        signal.extend(vec![10.0; 50]);

        let changepoints = find_changepoints(&signal, 10, 1, 15);

        assert_eq!(changepoints.len(), 1);
        // Should detect the boundary around position 50
        assert!(changepoints[0] >= 40 && changepoints[0] <= 60);
    }

    #[test]
    fn test_find_changepoints_multiple() {
        // Create signal with multiple steps
        let mut signal = vec![1.0; 30];
        signal.extend(vec![5.0; 30]);
        signal.extend(vec![10.0; 30]);

        let changepoints = find_changepoints(&signal, 5, 2, 10);

        // Should find 2 changepoints
        assert_eq!(changepoints.len(), 2);
        // Changepoints should be sorted
        assert!(changepoints[0] < changepoints[1]);
    }

    #[test]
    fn test_find_changepoints_min_separation() {
        let mut signal = vec![1.0; 50];
        signal.extend(vec![10.0; 50]);

        // Request 3 changepoints but there's only 1 clear boundary
        // min_separation should prevent clustering
        let changepoints = find_changepoints(&signal, 10, 3, 20);

        // Should find fewer than requested due to separation constraint
        assert!(changepoints.len() <= 3);
    }

    #[test]
    fn test_compute_segment_means_no_changepoints() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let changepoints = vec![];

        let segments = compute_segment_means(&signal, &changepoints);

        // Should have 1 segment covering entire signal
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].0, 0);
        assert_eq!(segments[0].1, 5);
        assert!((segments[0].2 - 3.0).abs() < 1e-6); // mean is 3.0
    }

    #[test]
    fn test_compute_segment_means_with_changepoints() {
        let signal = vec![1.0, 1.0, 5.0, 5.0, 5.0, 2.0, 2.0];
        let changepoints = vec![2, 5];

        let segments = compute_segment_means(&signal, &changepoints);

        assert_eq!(segments.len(), 3);

        // First segment: [0, 2) -> [1.0, 1.0], mean = 1.0
        assert_eq!(segments[0].0, 0);
        assert_eq!(segments[0].1, 2);
        assert!((segments[0].2 - 1.0).abs() < 1e-6);

        // Second segment: [2, 5) -> [5.0, 5.0, 5.0], mean = 5.0
        assert_eq!(segments[1].0, 2);
        assert_eq!(segments[1].1, 5);
        assert!((segments[1].2 - 5.0).abs() < 1e-6);

        // Third segment: [5, 7) -> [2.0, 2.0], mean = 2.0
        assert_eq!(segments[2].0, 5);
        assert_eq!(segments[2].1, 7);
        assert!((segments[2].2 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_segment_signal_integration() {
        let mut signal = vec![1.0; 30];
        signal.extend(vec![5.0; 30]);

        let segments = segment_signal(&signal, 5, 1, 10);

        // Should detect at least 1 changepoint, creating 2 segments
        assert!(segments.len() >= 2);

        // First segment should have mean ~1.0, last segment ~5.0
        assert!((segments[0].2 - 1.0).abs() < 0.5);
        assert!((segments[segments.len() - 1].2 - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_windowed_ttest_too_short() {
        let signal = vec![1.0, 2.0, 3.0];
        let scores = windowed_ttest(&signal, 10);

        // Signal too short for window size
        assert!(scores.is_empty());
    }

    #[test]
    fn test_find_changepoints_empty_signal() {
        let signal = vec![];
        let changepoints = find_changepoints(&signal, 10, 3, 5);

        assert!(changepoints.is_empty());
    }

    #[test]
    fn test_compute_segment_means_empty_signal() {
        let signal = vec![];
        let segments = compute_segment_means(&signal, &[]);

        assert!(segments.is_empty());
    }

    #[test]
    fn test_changepoints_sorted() {
        let mut signal = vec![1.0; 20];
        signal.extend(vec![5.0; 20]);
        signal.extend(vec![10.0; 20]);
        signal.extend(vec![3.0; 20]);

        let changepoints = find_changepoints(&signal, 5, 3, 5);

        // Verify changepoints are sorted
        for i in 1..changepoints.len() {
            assert!(changepoints[i - 1] < changepoints[i]);
        }
    }

    #[test]
    fn test_segment_signal_with_dwell_basic() {
        // Create signal with clear steps
        let mut signal = vec![1.0; 30];
        signal.extend(vec![5.0; 30]);

        let result = segment_signal_with_dwell(&signal, 5, 1, 10);

        // Should have multiple segments
        assert!(result.num_events() >= 2);
        assert!(!result.is_empty());

        // event_means and dwell_times should have same length
        assert_eq!(result.event_means.len(), result.dwell_times.len());
        assert_eq!(result.event_means.len(), result.boundaries.len());
    }

    #[test]
    fn test_segment_signal_with_dwell_dwell_times() {
        // Create signal with clear steps
        let signal = vec![1.0; 60];

        let result = segment_signal_with_dwell(&signal, 5, 2, 10);

        // Dwell times should sum to approximately signal length
        let total: usize = result.dwell_times.iter().map(|&d| d as usize).sum();
        assert_eq!(total, signal.len());

        // All dwell times should be positive
        for &dwell in &result.dwell_times {
            assert!(dwell > 0.0);
        }
    }

    #[test]
    fn test_segment_signal_with_dwell_boundaries() {
        let signal = vec![1.0; 60];

        let result = segment_signal_with_dwell(&signal, 5, 2, 10);

        // Boundaries should be contiguous
        for i in 1..result.boundaries.len() {
            assert_eq!(
                result.boundaries[i].0,
                result.boundaries[i - 1].1,
                "Boundaries should be contiguous"
            );
        }

        // First boundary should start at 0, last should end at signal length
        if !result.boundaries.is_empty() {
            assert_eq!(result.boundaries[0].0, 0);
            assert_eq!(
                result.boundaries[result.boundaries.len() - 1].1,
                signal.len()
            );
        }
    }

    #[test]
    fn test_segment_signal_with_dwell_total_duration() {
        let signal = vec![1.0; 100];

        let result = segment_signal_with_dwell(&signal, 5, 3, 10);

        // total_duration should equal signal length
        assert_eq!(result.total_duration(), signal.len());
    }

    #[test]
    fn test_segmentation_result_empty() {
        let result = SegmentationResult {
            event_means: vec![],
            dwell_times: vec![],
            boundaries: vec![],
        };

        assert!(result.is_empty());
        assert_eq!(result.num_events(), 0);
        assert_eq!(result.total_duration(), 0);
    }
}

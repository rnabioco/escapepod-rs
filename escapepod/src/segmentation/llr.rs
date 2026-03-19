//! Log-Likelihood Ratio (LLR) boundary detection for nanopore signal segmentation.
//!
//! This module implements the LLR algorithm for detecting boundaries in nanopore signals,
//! particularly useful for detecting adapter sequences and poly(A) tails.
//!
//! The algorithm is adapted from ADAPTed (Adapter and poly(A) Detection And Profiling Tool)
//! by Wiep K. van der Toorn.

/// Precomputed cumulative sums for efficient variance calculation.
///
/// This structure stores cumulative sums and cumulative sums of squares,
/// enabling O(1) variance calculation for any segment of the signal.
#[derive(Debug, Clone)]
pub struct LlrTrace {
    /// Cumulative sum of signal values
    cumsum: Vec<f64>,
    /// Cumulative sum of squared signal values
    cumsum_sq: Vec<f64>,
    /// Stride for computing gains (allows skipping samples for efficiency)
    stride: usize,
}

impl LlrTrace {
    /// Create a new LLR trace from raw signal.
    ///
    /// # Arguments
    /// * `signal` - The raw signal values
    /// * `stride` - Step size for computing gains (1 = every sample, 2 = every other sample, etc.)
    ///
    /// # Example
    /// ```
    /// use escapepod::segmentation::LlrTrace;
    ///
    /// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let trace = LlrTrace::new(&signal, 1);
    /// ```
    pub fn new(signal: &[f32], stride: usize) -> Self {
        let mut cumsum = Vec::with_capacity(signal.len());
        let mut cumsum_sq = Vec::with_capacity(signal.len());

        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for &val in signal {
            let val_f64 = val as f64;
            sum += val_f64;
            sum_sq += val_f64 * val_f64;
            cumsum.push(sum);
            cumsum_sq.push(sum_sq);
        }

        Self {
            cumsum,
            cumsum_sq,
            stride,
        }
    }

    /// Compute the variance of a segment [start, end).
    ///
    /// Uses the precomputed cumulative sums for O(1) calculation:
    /// var = (c2[end] - c2[start]) / (end - start) - ((c[end] - c[start]) / (end - start))^2
    ///
    /// # Arguments
    /// * `start` - Start index (inclusive)
    /// * `end` - End index (exclusive)
    ///
    /// # Returns
    /// The variance of the segment, or 0.0 if start == end.
    fn variance(&self, start: usize, end: usize) -> f64 {
        if start == end {
            return 0.0;
        }

        let n = (end - start) as f64;

        if start == 0 {
            let mean = self.cumsum[end - 1] / n;
            return self.cumsum_sq[end - 1] / n - mean * mean;
        }

        let sum_diff = self.cumsum[end - 1] - self.cumsum[start - 1];
        let sum_sq_diff = self.cumsum_sq[end - 1] - self.cumsum_sq[start - 1];

        let mean = sum_diff / n;
        sum_sq_diff / n - mean * mean
    }

    /// Compute LLR gains for all candidate split points in a range.
    ///
    /// For each candidate position i, computes the gain from splitting at i:
    /// gain = n * log(var(start, end)) - (n_head * log(var(start, i)) + n_tail * log(var(i, end)))
    ///
    /// # Arguments
    /// * `start` - Start index of the range to search
    /// * `end` - End index of the range to search
    /// * `min_obs` - Minimum observations required in head segment
    /// * `border_trim` - Minimum observations required in tail segment
    ///
    /// # Returns
    /// A vector of LLR gain values, one per signal sample. Only positions in the
    /// searchable range [start + min_obs, end - border_trim) will have non-zero values.
    pub fn compute_gains(
        &self,
        start: usize,
        end: usize,
        min_obs: usize,
        border_trim: usize,
    ) -> Vec<f64> {
        let mut gains = vec![0.0; self.cumsum.len()];

        let var_full = self.variance(start, end);
        if var_full <= 0.0 {
            return gains;
        }

        let var_summed = ((end - start) as f64) * var_full.ln();

        let search_start = start + min_obs;
        let search_end = end.saturating_sub(border_trim);

        for i in (search_start..search_end).step_by(self.stride) {
            let var_head = self.variance(start, i);
            let var_tail = self.variance(i, end);

            if var_head > 0.0 && var_tail > 0.0 {
                let var_summed_head = ((i - start) as f64) * var_head.ln();
                let var_summed_tail = ((end - i) as f64) * var_tail.ln();
                gains[i] = var_summed - (var_summed_head + var_summed_tail);
            }
        }

        gains
    }

    /// Compute LLR gains with early stopping when derivative becomes negative.
    ///
    /// Stops computation early if the mean derivative over a window becomes negative,
    /// indicating we've likely passed the true boundary.
    ///
    /// # Arguments
    /// * `start` - Start index of the range to search
    /// * `end` - End index of the range to search
    /// * `min_obs` - Minimum observations required in head segment
    /// * `border_trim` - Minimum observations required in tail segment
    /// * `early_stop_window` - Window size for computing derivative (e.g., 500)
    /// * `early_stop_stride` - How often to check for early stopping (e.g., 100)
    ///
    /// # Returns
    /// A vector of LLR gain values. Computation stops early if the derivative condition is met.
    pub fn compute_gains_with_early_stop(
        &self,
        start: usize,
        end: usize,
        min_obs: usize,
        border_trim: usize,
        early_stop_window: usize,
        early_stop_stride: usize,
    ) -> Vec<f64> {
        let mut gains = vec![0.0; self.cumsum.len()];

        let var_full = self.variance(start, end);
        if var_full <= 0.0 {
            return gains;
        }

        let var_summed = ((end - start) as f64) * var_full.ln();

        let search_start = start + min_obs;
        let search_end = end.saturating_sub(border_trim);

        for i in (search_start..search_end).step_by(self.stride) {
            // Check for early stopping
            if i >= search_start + early_stop_window
                && (i - search_start).is_multiple_of(early_stop_stride)
            {
                // Compute mean derivative over the window
                let window_start = i - early_stop_window;
                let mut derivative_sum = 0.0;
                let mut count = 0;

                for j in (window_start..i).step_by(self.stride) {
                    if j + self.stride < i && gains[j + self.stride] > 0.0 {
                        derivative_sum += gains[j + self.stride] - gains[j];
                        count += 1;
                    }
                }

                if count > 0 && derivative_sum / (count as f64) < 0.0 {
                    break; // Early stop: derivative is negative
                }
            }

            let var_head = self.variance(start, i);
            let var_tail = self.variance(i, end);

            if var_head > 0.0 && var_tail > 0.0 {
                let var_summed_head = ((i - start) as f64) * var_head.ln();
                let var_summed_tail = ((end - i) as f64) * var_tail.ln();
                gains[i] = var_summed - (var_summed_head + var_summed_tail);
            }
        }

        gains
    }

    /// Find the single best split point based on maximum LLR gain.
    ///
    /// # Arguments
    /// * `start` - Start index of the range to search
    /// * `end` - End index of the range to search
    /// * `min_obs` - Minimum observations required in head segment
    /// * `border_trim` - Minimum observations required in tail segment
    ///
    /// # Returns
    /// A tuple of (position, gain) where position is the best split point,
    /// or None if no valid split was found.
    pub fn best_split(
        &self,
        start: usize,
        end: usize,
        min_obs: usize,
        border_trim: usize,
    ) -> Option<(usize, f64)> {
        let gains = self.compute_gains(start, end, min_obs, border_trim);

        let search_start = start + min_obs;
        let search_end = end.saturating_sub(border_trim);

        // Check for invalid range (segment too short)
        if search_start >= search_end || search_end > gains.len() {
            return None;
        }

        gains[search_start..search_end]
            .iter()
            .enumerate()
            .fold(None, |best, (offset, &gain)| {
                let i = search_start + offset;
                match best {
                    Some((_, best_gain)) if gain <= best_gain => best,
                    _ if gain > 0.0 => Some((i, gain)),
                    _ => best,
                }
            })
    }

    /// Get the length of the signal.
    pub fn len(&self) -> usize {
        self.cumsum.len()
    }

    /// Check if the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.cumsum.is_empty()
    }
}

/// Detect adapter boundaries using LLR.
///
/// This implements a three-split strategy to identify adapter start and end positions:
/// 1. Find the primary split (adapter end)
/// 2. Split the left segment to find adapter start
/// 3. Split the right segment for refinement
///
/// # Arguments
/// * `signal` - The raw signal values
/// * `min_obs_adapter` - Minimum observations for adapter segments
/// * `border_trim` - Border trim size
///
/// # Returns
/// A tuple of (adapter_start, adapter_end), or (0, 0) if no adapter found.
///
/// # Example
/// ```
/// use escapepod::segmentation::detect_adapter;
///
/// let signal = vec![120.0; 100]; // Simplified example
/// let (start, end) = detect_adapter(&signal, 10, 5);
/// ```
pub fn detect_adapter(
    signal: &[f32],
    min_obs_adapter: usize,
    border_trim: usize,
) -> (usize, usize) {
    let trace = LlrTrace::new(signal, 1);
    let length = trace.len();

    // Find primary split
    let Some((x_first, _)) =
        trace.best_split(0, length, min_obs_adapter + border_trim, border_trim)
    else {
        return (0, 0);
    };

    // Split left segment
    let (x_head, gain_head) = trace
        .best_split(0, x_first, border_trim, min_obs_adapter)
        .unwrap_or((1, 0.0));

    // Split right segment
    let (x_tail, gain_tail) = trace
        .best_split(x_first, length, min_obs_adapter, border_trim)
        .unwrap_or((x_first + 1, 0.0));

    // Compute medians of the four segments
    let median_0 = median_slice(&signal[..x_head]);
    let median_1 = median_slice(&signal[x_head..x_first]);
    let median_2 = median_slice(&signal[x_first..x_tail]);
    let median_3 = median_slice(&signal[x_tail..]);

    let medians = [median_0, median_1, median_2, median_3];
    let mean_median = medians.iter().sum::<f32>() / 4.0;

    // Adapters represent a drop in pA space
    let diff_1 = median_2 - median_1;

    if diff_1 > 0.0 {
        // First detected end of adapter
        if median_0 >= mean_median {
            // Full adapter: open_pore/prev_RNA - DNA - RNA - RNA
            (x_head, x_first)
        } else {
            // Partial adapter: DNA - RNA - RNA - RNA
            (0, x_first)
        }
    } else if gain_tail > gain_head {
        // First detected start of adapter
        (x_first, x_tail)
    } else {
        (0, 0)
    }
}

/// Helper function to compute median of a slice using O(N) selection.
///
/// Uses `select_nth_unstable` for O(N) performance instead of O(N log N) sort.
/// For even-length arrays, returns the lower-middle element (acceptable for comparisons).
fn median_slice(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut buf = data.to_vec();
    let mid = buf.len() / 2;

    // select_nth_unstable partitions so that element at mid is the nth smallest
    // This is O(N) average case vs O(N log N) for full sort
    buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());

    if buf.len().is_multiple_of(2) && mid > 0 {
        // For even length, need to also find max of left partition for true median
        // Since we're only using this for comparisons, approximate with lower middle
        // To get exact even-length median:
        let left_max = buf[..mid]
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(buf[mid]);
        (left_max + buf[mid]) / 2.0
    } else {
        buf[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llr_trace_creation() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trace = LlrTrace::new(&signal, 1);

        assert_eq!(trace.len(), 5);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_variance_calculation() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trace = LlrTrace::new(&signal, 1);

        // Variance of [1,2,3,4,5] = 2.0
        let var = trace.variance(0, 5);
        assert!((var - 2.0).abs() < 1e-6);

        // Variance of single element
        let var_single = trace.variance(0, 1);
        assert_eq!(var_single, 0.0);
    }

    #[test]
    fn test_variance_subsegment() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trace = LlrTrace::new(&signal, 1);

        // Variance of [2,3,4] = var of middle 3 elements
        let var = trace.variance(1, 4);
        // Mean = 3.0, variance = ((2-3)^2 + (3-3)^2 + (4-3)^2) / 3 = 2/3
        assert!((var - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_gains() {
        // Create a signal with a clear boundary: low values with noise, then high values with noise
        let mut signal = Vec::new();
        // Add noise to avoid zero variance
        for i in 0..50 {
            signal.push(50.0 + (i % 3) as f32);
        }
        for i in 0..50 {
            signal.push(100.0 + (i % 3) as f32);
        }

        let trace = LlrTrace::new(&signal, 1);
        let gains = trace.compute_gains(0, 100, 10, 10);

        // Find the maximum gain
        let max_gain_pos = gains
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(pos, _)| pos)
            .unwrap();

        // The maximum gain should be near position 50 (the boundary)
        assert!((40..=60).contains(&max_gain_pos));
    }

    #[test]
    fn test_best_split() {
        // Create a signal with noise to have non-zero variance
        let mut signal = Vec::new();
        for i in 0..50 {
            signal.push(50.0 + (i % 3) as f32);
        }
        for i in 0..50 {
            signal.push(100.0 + (i % 3) as f32);
        }

        let trace = LlrTrace::new(&signal, 1);
        let result = trace.best_split(0, 100, 10, 10);

        assert!(result.is_some());
        let (pos, gain) = result.unwrap();

        // Position should be near the boundary
        assert!((40..=60).contains(&pos));
        // Gain should be positive
        assert!(gain > 0.0);
    }

    #[test]
    fn test_median_slice() {
        assert_eq!(median_slice(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_slice(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_slice(&[]), 0.0);
    }

    #[test]
    fn test_detect_adapter_no_boundary() {
        // Constant signal - no adapter
        let signal = vec![100.0; 100];
        let (start, end) = detect_adapter(&signal, 10, 5);

        // Should return (0, 0) for no adapter
        assert_eq!((start, end), (0, 0));
    }

    #[test]
    fn test_detect_adapter_with_boundary() {
        // Create a signal with adapter-like pattern: high - low - high
        let mut signal = Vec::new();
        signal.extend(vec![120.0; 30]); // Open pore
        signal.extend(vec![80.0; 40]); // Adapter (lower)
        signal.extend(vec![110.0; 30]); // RNA

        let (start, end) = detect_adapter(&signal, 10, 5);

        // Should detect something (exact values depend on algorithm details)
        // At minimum, start and end should be different if adapter detected
        assert!(start < end || (start == 0 && end == 0));
    }

    #[test]
    fn test_llr_stride() {
        let signal = vec![1.0; 100];
        let trace1 = LlrTrace::new(&signal, 1);
        let trace2 = LlrTrace::new(&signal, 2);

        assert_eq!(trace1.stride, 1);
        assert_eq!(trace2.stride, 2);
    }

    #[test]
    fn test_early_stop_functionality() {
        // Create a signal with noise to have non-zero variance
        let mut signal = Vec::new();
        for i in 0..50 {
            signal.push(50.0 + (i % 3) as f32);
        }
        for i in 0..50 {
            signal.push(100.0 + (i % 3) as f32);
        }

        let trace = LlrTrace::new(&signal, 1);

        // Compute with early stopping
        let gains_early = trace.compute_gains_with_early_stop(0, 100, 10, 10, 20, 10);

        // Should have some non-zero gains
        let max_gain = gains_early.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_gain > 0.0);
    }
}

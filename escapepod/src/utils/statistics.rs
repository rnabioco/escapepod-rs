//! Statistical computation utilities.

/// Summary statistics for a collection of values.
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Minimum value.
    pub min: u64,
    /// Maximum value.
    pub max: u64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Median value.
    pub median: u64,
    /// N50 value (length at which 50% of total is in sequences >= this length).
    pub n50: u64,
    /// Total sum of all values.
    pub total: u64,
    /// Number of values.
    pub count: u64,
}

/// Compute statistics from a slice of values.
///
/// This function computes common summary statistics including min, max, mean,
/// median, and N50. The input slice will be sorted in place for efficient
/// computation.
///
/// # Arguments
///
/// * `values` - Mutable slice of values (will be sorted)
///
/// # Returns
///
/// A `Statistics` struct containing all computed values, or default values
/// if the input is empty.
///
/// # Example
///
/// ```
/// use escapepod::utils::compute_statistics;
///
/// let mut lengths = vec![100, 200, 300, 400, 500];
/// let stats = compute_statistics(&mut lengths);
///
/// assert_eq!(stats.min, 100);
/// assert_eq!(stats.max, 500);
/// assert_eq!(stats.median, 300);
/// assert_eq!(stats.count, 5);
/// ```
pub fn compute_statistics(values: &mut [u64]) -> Statistics {
    if values.is_empty() {
        return Statistics::default();
    }

    // Sort for median and N50
    values.sort_unstable();

    let total: u64 = values.iter().sum();
    let count = values.len() as u64;

    let min = *values.first().unwrap_or(&0);
    let max = *values.last().unwrap_or(&0);
    let mean = total as f64 / count as f64;

    let median = if values.len() % 2 == 0 {
        let mid = values.len() / 2;
        (values[mid - 1] + values[mid]) / 2
    } else {
        values[values.len() / 2]
    };

    let n50 = compute_n50(values);

    Statistics {
        min,
        max,
        mean,
        median,
        n50,
        total,
        count,
    }
}

/// Compute N50 from a sorted slice of lengths.
///
/// N50 is defined as the length at which 50% of the total sequence length
/// is contained in sequences of length N50 or longer.
///
/// # Arguments
///
/// * `sorted_lengths` - Slice of lengths, must be sorted in ascending order
///
/// # Returns
///
/// The N50 value, or 0 if the input is empty.
///
/// # Example
///
/// ```
/// use escapepod::utils::compute_n50;
///
/// let lengths = vec![100, 200, 300, 400, 500];
/// let n50 = compute_n50(&lengths);
/// // Total = 1500, half = 750
/// // Iterating from 500: 500 (cumsum=500), 400 (cumsum=900 >= 750) -> N50 = 400
/// assert_eq!(n50, 400);
/// ```
pub fn compute_n50(sorted_lengths: &[u64]) -> u64 {
    if sorted_lengths.is_empty() {
        return 0;
    }

    let total: u64 = sorted_lengths.iter().sum();
    let half = total / 2;
    let mut cumsum = 0u64;

    // N50 requires reverse iteration (longest to shortest)
    for &len in sorted_lengths.iter().rev() {
        cumsum += len;
        if cumsum >= half {
            return len;
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_statistics_empty() {
        let mut values: Vec<u64> = vec![];
        let stats = compute_statistics(&mut values);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_compute_statistics_single() {
        let mut values = vec![100];
        let stats = compute_statistics(&mut values);
        assert_eq!(stats.min, 100);
        assert_eq!(stats.max, 100);
        assert_eq!(stats.median, 100);
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_compute_statistics_odd() {
        let mut values = vec![100, 200, 300, 400, 500];
        let stats = compute_statistics(&mut values);
        assert_eq!(stats.min, 100);
        assert_eq!(stats.max, 500);
        assert_eq!(stats.median, 300);
        assert_eq!(stats.mean, 300.0);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_compute_statistics_even() {
        let mut values = vec![100, 200, 300, 400];
        let stats = compute_statistics(&mut values);
        assert_eq!(stats.median, 250);
    }

    #[test]
    fn test_compute_n50() {
        // Total = 1500, half = 750
        // From longest: 500, 400 -> cumsum = 900 >= 750
        let lengths = vec![100, 200, 300, 400, 500];
        assert_eq!(compute_n50(&lengths), 400);
    }

    #[test]
    fn test_compute_n50_empty() {
        assert_eq!(compute_n50(&[]), 0);
    }
}

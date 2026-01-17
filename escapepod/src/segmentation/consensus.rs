//! Consensus-guided segmentation for barcode fingerprint extraction.
//!
//! This module implements the WarpDemuX-style consensus-guided segmentation algorithm
//! that uses a reference consensus model to locate the barcode region within the
//! segmented adapter signal.
//!
//! # Algorithm
//!
//! 1. First pass: Segment the full adapter signal with t-test → ~120 events
//! 2. DTW subsequence alignment: Match the consensus model to find barcode start
//! 3. Second pass: Segment only the barcode region → 25 events
//! 4. Normalize: Normalize barcode means using adapter statistics

use crate::dtw::dtw_subsequence_match;
use crate::segmentation::ttest;

/// Consensus model for RNA004 130bps adapter (from WarpDemuX).
///
/// This 84-element array represents the expected signal levels for the constant
/// adapter sequence, used to locate where the barcode region begins.
pub const CONSENSUS_RNA004_130BPS_V1_0: [f32; 84] = [
    -1.5183, -1.8727, -1.9543, -1.9302, -1.8015, -1.6293, -1.0795, 1.2055, 1.4142, 2.7910, 3.0117,
    3.1124, 1.3108, 0.1161, 0.0464, -0.0383, -0.1137, -0.1224, -0.1762, -0.2305, -0.2356, -0.2878,
    -1.1652, 0.9497, -0.0844, -0.2241, -0.3000, -0.3241, -0.3042, -0.2528, -0.2289, -0.2457,
    -1.9121, -0.6398, -0.3237, -0.3340, -0.4090, -0.8197, -1.0163, -1.3572, -1.5977, -1.7627,
    -1.9416, 0.4983, -1.4384, -0.0037, 0.2078, 0.2972, 0.1884, 0.1117, 0.0596, 0.0223, -0.0145,
    -0.0300, -0.0020, 0.0167, 0.0310, 0.0985, 0.7983, 0.8181, 0.6753, 0.5921, -0.7926, 1.3769,
    1.0878, 0.8852, 0.9476, 0.1551, 0.4764, 0.0895, -0.1273, -0.2631, 1.1313, 0.4736, 0.3932,
    0.2558, -0.4511, -0.6029, -0.7543, -1.5670, -1.9172, -0.1928, 0.0437, -0.6502,
];

/// Configuration for consensus-guided segmentation.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Number of events for initial adapter segmentation (default: 120)
    pub num_events: usize,
    /// Minimum observations per event (default: 9)
    pub min_obs_per_base: usize,
    /// Running window width for t-test (default: 18)
    pub running_stat_width: usize,
    /// Number of events for barcode segmentation (default: 25)
    pub barcode_num_events: usize,
    /// DTW penalty for subsequence matching (default: 1.5)
    pub subseq_penalty: f32,
    /// Upper bound for consensus query start position (default: 18)
    pub query_start_upper_bound: usize,
    /// Lower bound for consensus query end position (default: 69)
    pub query_end_lower_bound: usize,
    /// Upper bound for consensus query end position (default: 97)
    pub query_end_upper_bound: usize,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            num_events: 120,
            min_obs_per_base: 9,
            running_stat_width: 18,
            barcode_num_events: 25,
            subseq_penalty: 1.5,
            query_start_upper_bound: 18,
            query_end_lower_bound: 69,
            query_end_upper_bound: 97,
        }
    }
}

/// Result of consensus-guided segmentation.
#[derive(Debug, Clone)]
pub struct ConsensusSegmentationResult {
    /// Barcode fingerprint values (normalized segment means)
    pub fingerprint: Vec<f32>,
    /// Dwell times for barcode segments
    pub dwell_times: Vec<usize>,
    /// Start index of consensus match in adapter events
    pub query_start: usize,
    /// End index of consensus match in adapter events
    pub query_end: usize,
    /// Signal index where barcode region starts
    pub barcode_signal_start: usize,
}

/// Segment adapter signal using consensus-guided approach.
///
/// This implements the WarpDemuX algorithm:
/// 1. Segment full adapter to get event means
/// 2. Find consensus match via DTW subsequence alignment
/// 3. Segment barcode region after consensus match
/// 4. Normalize barcode means relative to adapter statistics
///
/// # Arguments
///
/// * `signal` - The adapter signal (should be normalized)
/// * `consensus` - The consensus model to match against
/// * `config` - Segmentation configuration
///
/// # Returns
///
/// The segmentation result with fingerprint, or None if segmentation fails.
pub fn segment_with_consensus(
    signal: &[f32],
    consensus: &[f32],
    config: &ConsensusConfig,
) -> Option<ConsensusSegmentationResult> {
    if signal.len() < config.num_events * config.min_obs_per_base {
        return None;
    }

    // Adaptive parameters based on signal length
    let min_obs = config
        .min_obs_per_base
        .min(signal.len() / config.num_events / 2);
    let running_width = config
        .running_stat_width
        .min(signal.len() / config.num_events);

    // First pass: segment full adapter using t-test
    let changepoints = ttest::find_changepoints(signal, running_width, config.num_events, min_obs);
    if changepoints.len() < 2 {
        return None;
    }

    // Get segment means (returns Vec<(start, end, mean)>)
    let adapter_segments = ttest::compute_segment_means(signal, &changepoints);
    if adapter_segments.is_empty() {
        return None;
    }

    // Extract just the means as f32
    let adapter_means: Vec<f32> = adapter_segments.iter().map(|(_, _, m)| *m as f32).collect();

    // Mean-normalize the adapter event means for consensus matching
    let adapter_mean: f32 = adapter_means.iter().sum::<f32>() / adapter_means.len() as f32;
    let adapter_std: f32 = (adapter_means
        .iter()
        .map(|x| (x - adapter_mean).powi(2))
        .sum::<f32>()
        / adapter_means.len() as f32)
        .sqrt();

    if adapter_std < 1e-6 {
        return None;
    }

    let norm_adapter: Vec<f32> = adapter_means
        .iter()
        .map(|x| (x - adapter_mean) / adapter_std)
        .collect();

    // Find consensus match via DTW subsequence alignment
    let subseq_match = dtw_subsequence_match(consensus, &norm_adapter, config.subseq_penalty)?;

    // Validate consensus match position
    if subseq_match.start > config.query_start_upper_bound
        || subseq_match.end < config.query_end_lower_bound
        || subseq_match.end > config.query_end_upper_bound
    {
        return None;
    }

    // Calculate barcode signal start position from segment end positions
    let barcode_signal_start = if subseq_match.end < adapter_segments.len() {
        adapter_segments[subseq_match.end].0 // start of segment after consensus match
    } else {
        adapter_segments.last().map(|(_, end, _)| *end).unwrap_or(0)
    };

    // Second pass: segment barcode region
    let barcode_signal = &signal[barcode_signal_start..];
    if barcode_signal.len() < config.barcode_num_events * config.min_obs_per_base {
        return None;
    }

    let barcode_cpts =
        ttest::find_changepoints(barcode_signal, running_width, config.barcode_num_events, min_obs);
    if barcode_cpts.len() < 2 {
        return None;
    }

    let barcode_segments = ttest::compute_segment_means(barcode_signal, &barcode_cpts);
    if barcode_segments.len() < config.barcode_num_events {
        return None;
    }

    // Normalize barcode means using adapter statistics
    let fingerprint: Vec<f32> = barcode_segments
        .iter()
        .map(|(_, _, m)| (*m as f32 - adapter_mean) / adapter_std)
        .collect();

    let barcode_dwell: Vec<usize> = barcode_segments
        .iter()
        .map(|(start, end, _)| end - start)
        .collect();

    // Take last N values (matching WarpDemuX behavior)
    let n = config.barcode_num_events.min(fingerprint.len());
    let fingerprint = fingerprint[fingerprint.len() - n..].to_vec();
    let dwell_times = barcode_dwell[barcode_dwell.len() - n..].to_vec();

    Some(ConsensusSegmentationResult {
        fingerprint,
        dwell_times,
        query_start: subseq_match.start,
        query_end: subseq_match.end,
        barcode_signal_start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_model_length() {
        assert_eq!(CONSENSUS_RNA004_130BPS_V1_0.len(), 84);
    }

    #[test]
    fn test_default_config() {
        let config = ConsensusConfig::default();
        assert_eq!(config.num_events, 120);
        assert_eq!(config.barcode_num_events, 25);
    }
}

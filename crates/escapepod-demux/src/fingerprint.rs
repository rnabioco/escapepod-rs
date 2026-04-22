//! Barcode fingerprint types and extraction.
//!
//! The pipeline:
//! 1. `extract_fingerprint_from_signal` — given a read's raw signal and its
//!    detected adapter boundaries, segment the adapter region with
//!    t-test changepoints and take each segment's mean as a fingerprint
//!    feature.
//! 2. `compute_consensus_fingerprint` / `compute_std_dev_fingerprint` —
//!    aggregate per-read fingerprints into a reference barcode
//!    fingerprint at training time.

use escapepod_signal::dtw::{Fingerprint, NormMethod, normalize_fingerprint};
use escapepod_signal::segmentation::{clip_outliers, mad_normalize, segment_signal};
use uuid::Uuid;

/// A fingerprint extracted from a single read's adapter region.
#[derive(Debug, Clone)]
pub struct ReadFingerprint {
    /// The read identifier
    pub read_id: Uuid,
    /// The fingerprint feature values (segment means, normalized)
    pub values: Vec<f64>,
}

impl ReadFingerprint {
    /// Create a new read fingerprint.
    pub fn new(read_id: Uuid, values: Vec<f64>) -> Self {
        Self { read_id, values }
    }
}

/// A reference barcode fingerprint for classification.
#[derive(Debug, Clone)]
pub struct BarcodeFingerprint {
    /// The barcode name (e.g., "BC01")
    pub barcode: String,
    /// The fingerprint feature values
    pub values: Vec<f32>,
}

impl BarcodeFingerprint {
    /// Create a new barcode fingerprint.
    pub fn new(barcode: String, values: Vec<f32>) -> Self {
        Self { barcode, values }
    }
}

/// Extract a fingerprint from an adapter region of a signal.
///
/// Returns `None` if the region is too small or segmentation fails.
///
/// When `keep_last` is set (WarpDemuX-compat mode), normalization is applied
/// to ALL segment means before truncation, matching WarpDemuX's behavior where
/// z-score is computed over all 110 events then the last 25 are retained.
/// In this mode, the signal is NOT pre-normalized (WarpDemuX segments raw pA values).
#[allow(clippy::too_many_arguments)]
pub fn extract_fingerprint_from_signal(
    signal: &[i16],
    adapter_start: usize,
    adapter_end: usize,
    num_segments: usize,
    window_width: usize,
    norm_method: NormMethod,
    read_id: Uuid,
    min_separation: Option<usize>,
    keep_last: Option<usize>,
) -> Option<ReadFingerprint> {
    // In WarpDemuX-compat mode, extend the adapter region by
    // `sig_extract.padding = 100` samples on each side before clipping and
    // segmenting (matches `extract_adapter` in WarpDemuX's `sig_proc.py`).
    // This is load-bearing — changes to this offset shift every changepoint
    // and the final 25 segment means retained for the fingerprint.
    let (slice_start, slice_end) = if keep_last.is_some() {
        const WARPDEMUX_PADDING: usize = 100;
        let ss = adapter_start.saturating_sub(WARPDEMUX_PADDING);
        let se = adapter_end
            .saturating_add(WARPDEMUX_PADDING)
            .min(signal.len());
        (ss, se)
    } else {
        (adapter_start, adapter_end.min(signal.len()))
    };

    if slice_end <= slice_start || slice_end - slice_start < window_width * 2 {
        return None;
    }

    // Convert to f32. When keep_last is set (WarpDemuX-compat), don't pre-normalize
    // — WarpDemuX segments raw pA values and normalizes event means afterwards.
    // For the default mode, MAD-normalize for consistency with existing behavior.
    let adapter_signal: Vec<f32> = if keep_last.is_some() {
        let raw: Vec<f32> = signal[slice_start..slice_end]
            .iter()
            .map(|&s| s as f32)
            .collect();
        // WarpDemuX clips outliers (median ± 5*MAD) before t-test segmentation
        // to prevent extreme values from distorting changepoint detection.
        clip_outliers(&raw, 5.0)
    } else {
        let raw: Vec<f32> = signal[slice_start..slice_end]
            .iter()
            .map(|&s| s as f32)
            .collect();
        if raw.len() > 10 {
            mad_normalize(&raw)
        } else {
            raw
        }
    };

    let sep = min_separation.unwrap_or(window_width);

    // Segment the adapter region
    let segments = segment_signal(
        &adapter_signal,
        window_width,
        num_segments.saturating_sub(1),
        sep,
    );

    if segments.is_empty() {
        return None;
    }

    // Extract segment means as fingerprint
    let mut fingerprint_values: Vec<f32> =
        segments.iter().map(|(_, _, mean)| *mean as f32).collect();

    if let Some(n) = keep_last {
        // WarpDemuX-compat: normalize ALL event means first, then truncate.
        // WarpDemuX's "mean" normalization is actually z-score (mean/std).
        let mut all_fp = Fingerprint::new(fingerprint_values, read_id);
        normalize_fingerprint(&mut all_fp, norm_method);
        fingerprint_values = all_fp.values;

        if fingerprint_values.len() > n {
            fingerprint_values = fingerprint_values[fingerprint_values.len() - n..].to_vec();
        }

        return Some(ReadFingerprint::new(
            read_id,
            fingerprint_values.iter().map(|&v| v as f64).collect(),
        ));
    }

    let mut fp = Fingerprint::new(fingerprint_values, read_id);
    normalize_fingerprint(&mut fp, norm_method);

    Some(ReadFingerprint::new(
        read_id,
        fp.values.iter().map(|&v| v as f64).collect(),
    ))
}

/// Compute consensus fingerprint as element-wise median.
///
/// Filters fingerprints to only include those with the most common length,
/// so that mixed-width reference sets don't pollute the median.
pub fn compute_consensus_fingerprint(fingerprints: &[Vec<f32>]) -> Vec<f32> {
    if fingerprints.is_empty() {
        return Vec::new();
    }

    let mut length_counts: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for fp in fingerprints {
        *length_counts.entry(fp.len()).or_insert(0) += 1;
    }
    let target_length = length_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(len, _)| len)
        .unwrap_or(0);

    let filtered: Vec<&Vec<f32>> = fingerprints
        .iter()
        .filter(|fp| fp.len() == target_length)
        .collect();
    if filtered.is_empty() {
        return Vec::new();
    }

    let mut consensus = Vec::with_capacity(target_length);

    for i in 0..target_length {
        let mut values: Vec<f32> = filtered.iter().map(|fp| fp[i]).collect();
        values.sort_unstable_by(|a, b| a.total_cmp(b));

        let median = if values.len().is_multiple_of(2) {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };

        consensus.push(median);
    }

    consensus
}

/// Compute element-wise standard deviation of a set of fingerprints relative
/// to a consensus. Only uses fingerprints that match the consensus length.
pub fn compute_std_dev_fingerprint(fingerprints: &[Vec<f32>], consensus: &[f32]) -> Vec<f32> {
    if fingerprints.is_empty() || consensus.is_empty() {
        return Vec::new();
    }

    let length = consensus.len();

    let filtered: Vec<&Vec<f32>> = fingerprints
        .iter()
        .filter(|fp| fp.len() == length)
        .collect();
    if filtered.is_empty() {
        return vec![0.0; length];
    }

    let mut std_dev = Vec::with_capacity(length);

    for i in 0..length {
        let mean = consensus[i];
        let variance = filtered
            .iter()
            .map(|fp| {
                let d = fp[i] - mean;
                d * d
            })
            .sum::<f32>()
            / filtered.len() as f32;
        std_dev.push(variance.sqrt());
    }

    std_dev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_consensus_fingerprint_empty() {
        let result = compute_consensus_fingerprint(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_consensus_fingerprint_single() {
        let fingerprints = vec![vec![1.0, 2.0, 3.0]];
        let result = compute_consensus_fingerprint(&fingerprints);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compute_consensus_fingerprint_multiple_odd() {
        let fingerprints = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let result = compute_consensus_fingerprint(&fingerprints);
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_compute_consensus_fingerprint_multiple_even() {
        let fingerprints = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        let result = compute_consensus_fingerprint(&fingerprints);
        assert_eq!(result, vec![2.5, 5.0]);
    }

    #[test]
    fn test_compute_std_dev_fingerprint_empty() {
        let result = compute_std_dev_fingerprint(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_std_dev_fingerprint_single() {
        let fingerprints = vec![vec![1.0, 2.0, 3.0]];
        let consensus = vec![1.0, 2.0, 3.0];
        let result = compute_std_dev_fingerprint(&fingerprints, &consensus);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_std_dev_fingerprint_multiple() {
        let fingerprints = vec![vec![1.0, 0.0], vec![3.0, 0.0]];
        let consensus = vec![2.0, 0.0];
        let result = compute_std_dev_fingerprint(&fingerprints, &consensus);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn test_extract_fingerprint_from_signal_too_small() {
        let signal: Vec<i16> = vec![100, 200, 300];
        let read_id = Uuid::new_v4();
        let result = extract_fingerprint_from_signal(
            &signal,
            0,
            3,
            10,
            5,
            NormMethod::ZScore,
            read_id,
            None,
            None,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_fingerprint_from_signal_valid() {
        let signal: Vec<i16> = (0..1000).map(|i| (i as i16) % 1000).collect();
        let read_id = Uuid::new_v4();
        let result = extract_fingerprint_from_signal(
            &signal,
            0,
            500,
            10,
            5,
            NormMethod::None,
            read_id,
            None,
            None,
        );
        assert!(result.is_some());
        let fp = result.unwrap();
        assert_eq!(fp.read_id, read_id);
        assert!(!fp.values.is_empty());
    }
}

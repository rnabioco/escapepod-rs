//! Shared utilities for demux subcommands.

use super::types::{BarcodeFingerprint, ReadBoundaries, ReadFingerprint};
use escapepod_signal::Reader;
use escapepod_signal::dtw::{Fingerprint, NormMethod, normalize_fingerprint};
use escapepod_signal::segmentation::{clip_outliers, mad_normalize, segment_signal};
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Parse a normalization method string into a NormMethod enum.
///
/// Valid values: "zscore", "minmax", "median", "none"
pub fn parse_norm_method(s: &str) -> anyhow::Result<NormMethod> {
    match s.to_lowercase().as_str() {
        "zscore" => Ok(NormMethod::ZScore),
        "minmax" => Ok(NormMethod::MinMax),
        "median" => Ok(NormMethod::Median),
        "mean" => Ok(NormMethod::Mean),
        "none" => Ok(NormMethod::None),
        _ => anyhow::bail!(
            "Invalid normalization method: {}. Use zscore, minmax, median, mean, or none",
            s
        ),
    }
}

/// Configure the rayon thread pool with the specified number of threads.
///
/// Note: This should only be called once per process. Subsequent calls are ignored.
pub fn configure_thread_pool(num_threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok();
}

/// Parse a boundaries CSV file into a map of read_id -> ReadBoundaries.
///
/// Auto-detects the format by examining the header line:
/// - escapepod format: `read_id,num_samples,adapter_start,adapter_end`
/// - WarpDemuX format: `read_id,signal_len,...,adapter_start,adapter_end,...` (55+ columns)
///
/// Supports both plain CSV and gzip-compressed (.gz) files.
pub fn parse_boundaries_csv(path: &Path) -> anyhow::Result<HashMap<Uuid, ReadBoundaries>> {
    let reader: Box<dyn BufRead> = open_csv_reader(path)?;
    let mut lines = reader.lines();

    // Read header to detect format
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty boundaries file"))??;

    // Skip WarpDemuX comment header (starts with #)
    let header = header
        .strip_prefix('#')
        .map(|s| s.to_string())
        .unwrap_or(header);

    let columns: Vec<&str> = header.split(',').collect();

    // Detect format by finding column indices
    let (read_id_col, num_samples_col, adapter_start_col, adapter_end_col) =
        detect_boundary_columns(&columns)?;

    let mut boundaries_map = HashMap::new();

    for line in lines {
        let line = line?;
        // Skip comment lines
        if line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        let max_col = [
            read_id_col,
            num_samples_col,
            adapter_start_col,
            adapter_end_col,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        if parts.len() > max_col {
            if let Ok(read_id) = Uuid::parse_str(parts[read_id_col]) {
                let num_samples = parts[num_samples_col].parse::<u64>().unwrap_or(0);
                let adapter_start = parts[adapter_start_col]
                    .parse::<f64>()
                    .map(|v| v as usize)
                    .unwrap_or(0);
                let adapter_end = parts[adapter_end_col]
                    .parse::<f64>()
                    .map(|v| v as usize)
                    .unwrap_or(0);

                let boundaries = ReadBoundaries {
                    read_id,
                    num_samples,
                    adapter_start,
                    adapter_end,
                };

                if boundaries.has_valid_adapter() {
                    boundaries_map.insert(read_id, boundaries);
                }
            }
        }
    }

    Ok(boundaries_map)
}

/// Open a CSV file, auto-detecting gzip compression from the .gz extension.
fn open_csv_reader(path: &Path) -> anyhow::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        let decoder = GzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Detect column indices for boundary fields by examining the CSV header.
///
/// Returns (read_id_col, num_samples_col, adapter_start_col, adapter_end_col).
fn detect_boundary_columns(columns: &[&str]) -> anyhow::Result<(usize, usize, usize, usize)> {
    // Try to find columns by name
    let find_col = |names: &[&str]| -> Option<usize> {
        for name in names {
            if let Some(idx) = columns.iter().position(|c| c.trim() == *name) {
                return Some(idx);
            }
        }
        None
    };

    let read_id_col = find_col(&["read_id"])
        .ok_or_else(|| anyhow::anyhow!("No 'read_id' column found in boundaries CSV"))?;

    let num_samples_col = find_col(&["num_samples", "signal_len"]).unwrap_or(1);

    let adapter_start_col = find_col(&["adapter_start"])
        .ok_or_else(|| anyhow::anyhow!("No 'adapter_start' column found in boundaries CSV"))?;

    let adapter_end_col = find_col(&["adapter_end"])
        .ok_or_else(|| anyhow::anyhow!("No 'adapter_end' column found in boundaries CSV"))?;

    Ok((
        read_id_col,
        num_samples_col,
        adapter_start_col,
        adapter_end_col,
    ))
}

/// Parse a reference fingerprints CSV into barcode fingerprints.
pub fn parse_reference_csv(path: &PathBuf) -> anyhow::Result<Vec<BarcodeFingerprint>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut fingerprints = Vec::new();
    let mut header_seen = false;

    for line in reader.lines() {
        let line = line?;
        if !header_seen {
            header_seen = true;
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let barcode_name = parts[0].to_string();
            let values: Vec<f32> = parts[1..]
                .iter()
                .filter_map(|s| s.parse::<f32>().ok())
                .collect();
            if !values.is_empty() {
                fingerprints.push(BarcodeFingerprint::new(barcode_name, values));
            }
        }
    }

    Ok(fingerprints)
}

/// Collect reads with signal data from POD5 files.
///
/// Returns a vector of (read_id, num_samples, signal) tuples.
pub fn collect_reads_with_signals(
    input_files: &[PathBuf],
) -> anyhow::Result<Vec<(Uuid, u64, Vec<i16>)>> {
    let mut all_reads = Vec::new();

    for path in input_files {
        let reader = Reader::open(path)?;
        if let Ok(reads) = reader.reads() {
            for read_result in reads {
                let read = read_result?;
                if !read.signal_rows.is_empty() {
                    if let Ok(signal) = reader.get_signal(&read.signal_rows) {
                        all_reads.push((read.read_id, read.num_samples, signal));
                    }
                }
            }
        }
    }

    Ok(all_reads)
}

/// Process raw signal for adapter detection.
///
/// Converts i16 to f32 and applies MAD normalization.
pub fn normalize_signal(signal: &[i16]) -> Vec<f32> {
    let signal_f32: Vec<f32> = signal.iter().map(|&s| s as f32).collect();

    if signal_f32.len() > 10 {
        mad_normalize(&signal_f32)
    } else {
        signal_f32
    }
}

/// Downscale a signal by averaging consecutive samples.
///
/// Similar to WarpDemuX/ADAPTed's average pooling for faster LLR computation.
/// A factor of 10 matches WarpDemuX's default behavior.
pub fn downscale_signal(signal: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return signal.to_vec();
    }

    let n_output = signal.len() / factor;
    let mut result = Vec::with_capacity(n_output);

    for i in 0..n_output {
        let start = i * factor;
        let end = start + factor;
        let sum: f32 = signal[start..end].iter().sum();
        result.push(sum / factor as f32);
    }

    result
}

/// Extract a fingerprint from an adapter region of a signal.
///
/// Returns None if the region is too small or segmentation fails.
///
/// When `keep_last` is set (WarpDemuX-compat mode), normalization is applied
/// to ALL segment means before truncation, matching WarpDemuX's behavior where
/// z-score is computed over all 110 events then last 25 are retained.
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
    let end = adapter_end.min(signal.len());

    if end <= adapter_start || end - adapter_start < window_width * 2 {
        return None;
    }

    // Convert to f32. When keep_last is set (WarpDemuX-compat), don't pre-normalize
    // — WarpDemuX segments raw pA values and normalizes event means afterwards.
    // For the default mode, MAD-normalize for consistency with existing behavior.
    let adapter_signal: Vec<f32> = if keep_last.is_some() {
        let raw: Vec<f32> = signal[adapter_start..end]
            .iter()
            .map(|&s| s as f32)
            .collect();
        // WarpDemuX clips outliers (median ± 5*MAD) before t-test segmentation
        // to prevent extreme values from distorting changepoint detection.
        clip_outliers(&raw, 5.0)
    } else {
        let raw: Vec<f32> = signal[adapter_start..end]
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

        // Keep only the last N segments
        if fingerprint_values.len() > n {
            fingerprint_values = fingerprint_values[fingerprint_values.len() - n..].to_vec();
        }

        return Some(ReadFingerprint::new(
            read_id,
            fingerprint_values.iter().map(|&v| v as f64).collect(),
        ));
    }

    // Default mode: normalize the fingerprint after extraction
    let mut fp = Fingerprint::new(fingerprint_values, read_id);
    normalize_fingerprint(&mut fp, norm_method);

    Some(ReadFingerprint::new(
        read_id,
        fp.values.iter().map(|&v| v as f64).collect(),
    ))
}

/// Compute consensus fingerprint as element-wise median.
///
/// Filters fingerprints to only include those with the most common length.
pub fn compute_consensus_fingerprint(fingerprints: &[Vec<f32>]) -> Vec<f32> {
    if fingerprints.is_empty() {
        return Vec::new();
    }

    // Find the most common length
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

    // Filter to fingerprints with target length
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
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };

        consensus.push(median);
    }

    consensus
}

/// Compute element-wise standard deviation.
///
/// Only uses fingerprints that match the consensus length.
pub fn compute_std_dev_fingerprint(fingerprints: &[Vec<f32>], consensus: &[f32]) -> Vec<f32> {
    if fingerprints.is_empty() || consensus.is_empty() {
        return Vec::new();
    }

    let length = consensus.len();

    // Filter to fingerprints matching consensus length
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
            .map(|fp| (fp[i] - mean).powi(2))
            .sum::<f32>()
            / filtered.len() as f32;
        std_dev.push(variance.sqrt());
    }

    std_dev
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_norm_method_valid() {
        assert!(matches!(
            parse_norm_method("zscore").unwrap(),
            NormMethod::ZScore
        ));
        assert!(matches!(
            parse_norm_method("ZSCORE").unwrap(),
            NormMethod::ZScore
        ));
        assert!(matches!(
            parse_norm_method("minmax").unwrap(),
            NormMethod::MinMax
        ));
        assert!(matches!(
            parse_norm_method("median").unwrap(),
            NormMethod::Median
        ));
        assert!(matches!(
            parse_norm_method("mean").unwrap(),
            NormMethod::Mean
        ));
        assert!(matches!(
            parse_norm_method("none").unwrap(),
            NormMethod::None
        ));
    }

    #[test]
    fn test_parse_norm_method_invalid() {
        assert!(parse_norm_method("invalid").is_err());
        assert!(parse_norm_method("").is_err());
    }

    #[test]
    fn test_parse_boundaries_csv_valid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,num_samples,adapter_start,adapter_end").unwrap();
        writeln!(
            temp_file,
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890,1000,100,500"
        )
        .unwrap();
        writeln!(
            temp_file,
            "b2c3d4e5-f6a7-8901-bcde-f12345678901,2000,200,600"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let result = parse_boundaries_csv(&temp_file.path().to_path_buf()).unwrap();

        assert_eq!(result.len(), 2);

        let uuid1 = Uuid::parse_str("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        assert!(result.contains_key(&uuid1));
        let b1 = result.get(&uuid1).unwrap();
        assert_eq!(b1.num_samples, 1000);
        assert_eq!(b1.adapter_start, 100);
        assert_eq!(b1.adapter_end, 500);
    }

    #[test]
    fn test_parse_boundaries_csv_skips_invalid_adapter() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,num_samples,adapter_start,adapter_end").unwrap();
        // Invalid: adapter_end <= adapter_start
        writeln!(
            temp_file,
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890,1000,500,100"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let result = parse_boundaries_csv(&temp_file.path().to_path_buf()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_parse_reference_csv_valid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "barcode,fp_0,fp_1,fp_2").unwrap();
        writeln!(temp_file, "BC01,0.1,0.2,0.3").unwrap();
        writeln!(temp_file, "BC02,0.4,0.5,0.6").unwrap();
        temp_file.flush().unwrap();

        let result = parse_reference_csv(&temp_file.path().to_path_buf()).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].barcode, "BC01");
        assert_eq!(result[0].values, vec![0.1, 0.2, 0.3]);
        assert_eq!(result[1].barcode, "BC02");
        assert_eq!(result[1].values, vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_normalize_signal_short() {
        let signal: Vec<i16> = vec![100, 200, 300];
        let result = normalize_signal(&signal);
        // Short signals are just converted, not normalized
        assert_eq!(result, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn test_normalize_signal_long() {
        // Create a signal with enough samples for normalization
        let signal: Vec<i16> = (0..100).map(|i| (i as i16) * 10 + 200).collect();
        let result = normalize_signal(&signal);
        // MAD normalization should be applied
        assert_eq!(result.len(), 100);
        // The result should be different from simple conversion
        assert!(result[0] != 200.0);
    }

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
        // Median of [1,2,3], [2,3,4], [3,4,5]
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
        // Median of [1,2,3,4] = (2+3)/2 = 2.5
        // Median of [2,4,6,8] = (4+6)/2 = 5.0
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
        // Std dev of single value is 0
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_std_dev_fingerprint_multiple() {
        let fingerprints = vec![vec![1.0, 0.0], vec![3.0, 0.0]];
        let consensus = vec![2.0, 0.0];
        let result = compute_std_dev_fingerprint(&fingerprints, &consensus);
        // Variance = ((1-2)^2 + (3-2)^2) / 2 = (1 + 1) / 2 = 1
        // Std dev = sqrt(1) = 1
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn test_downscale_signal_factor_1() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let result = downscale_signal(&signal, 1);
        assert_eq!(result, signal);
    }

    #[test]
    fn test_downscale_signal_factor_2() {
        let signal = vec![1.0, 3.0, 5.0, 7.0];
        let result = downscale_signal(&signal, 2);
        // (1+3)/2=2, (5+7)/2=6
        assert_eq!(result, vec![2.0, 6.0]);
    }

    #[test]
    fn test_downscale_signal_factor_10() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = downscale_signal(&signal, 10);
        assert_eq!(result.len(), 10);
        // First chunk: 0+1+...+9 = 45, avg = 4.5
        assert!((result[0] - 4.5).abs() < 0.001);
        // Last chunk: 90+91+...+99 = 945, avg = 94.5
        assert!((result[9] - 94.5).abs() < 0.001);
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
            5, // window_width larger than signal
            NormMethod::ZScore,
            read_id,
            None,
            None,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_fingerprint_from_signal_valid() {
        // Create a signal with enough samples
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

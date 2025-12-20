//! Shared utilities for demux subcommands.

use super::types::{BarcodeFingerprint, ReadBoundaries, ReadFingerprint};
use escapepod::dtw::{normalize_fingerprint, Fingerprint, NormMethod};
use escapepod::segmentation::{mad_normalize, segment_signal};
use escapepod::Reader;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use uuid::Uuid;

/// Parse a normalization method string into a NormMethod enum.
///
/// Valid values: "zscore", "minmax", "median", "none"
pub fn parse_norm_method(s: &str) -> anyhow::Result<NormMethod> {
    match s.to_lowercase().as_str() {
        "zscore" => Ok(NormMethod::ZScore),
        "minmax" => Ok(NormMethod::MinMax),
        "median" => Ok(NormMethod::Median),
        "none" => Ok(NormMethod::None),
        _ => anyhow::bail!(
            "Invalid normalization method: {}. Use zscore, minmax, median, or none",
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
pub fn parse_boundaries_csv(path: &PathBuf) -> anyhow::Result<HashMap<Uuid, ReadBoundaries>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut boundaries_map = HashMap::new();
    let mut line_count = 0;

    for line in reader.lines() {
        let line = line?;
        line_count += 1;

        // Skip header
        if line_count == 1 {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            if let Ok(read_id) = Uuid::parse_str(parts[0]) {
                let num_samples = parts[1].parse::<u64>().unwrap_or(0);
                let adapter_start = parts[2].parse::<usize>().unwrap_or(0);
                let adapter_end = parts[3].parse::<usize>().unwrap_or(0);

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

/// Extract a fingerprint from an adapter region of a signal.
///
/// Returns None if the region is too small or segmentation fails.
pub fn extract_fingerprint_from_signal(
    signal: &[i16],
    adapter_start: usize,
    adapter_end: usize,
    num_segments: usize,
    window_width: usize,
    norm_method: NormMethod,
    read_id: Uuid,
) -> Option<ReadFingerprint> {
    let end = adapter_end.min(signal.len());

    if end <= adapter_start || end - adapter_start < window_width * 2 {
        return None;
    }

    // Convert to f32
    let adapter_signal: Vec<f32> = signal[adapter_start..end]
        .iter()
        .map(|&s| s as f32)
        .collect();

    // Segment the adapter region
    let segments = segment_signal(
        &adapter_signal,
        window_width,
        num_segments.saturating_sub(1),
        window_width,
    );

    if segments.is_empty() {
        return None;
    }

    // Extract segment means as fingerprint
    let fingerprint_values: Vec<f32> = segments.iter().map(|(_, _, mean)| *mean as f32).collect();

    // Normalize the fingerprint
    let mut fp = Fingerprint::new(fingerprint_values, read_id);
    normalize_fingerprint(&mut fp, norm_method);

    Some(ReadFingerprint::new(
        read_id,
        fp.values.iter().map(|&v| v as f64).collect(),
    ))
}

/// Compute consensus fingerprint as element-wise median.
pub fn compute_consensus_fingerprint(fingerprints: &[Vec<f32>]) -> Vec<f32> {
    if fingerprints.is_empty() {
        return Vec::new();
    }

    let length = fingerprints[0].len();
    let mut consensus = Vec::with_capacity(length);

    for i in 0..length {
        let mut values: Vec<f32> = fingerprints.iter().map(|fp| fp[i]).collect();
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
pub fn compute_std_dev_fingerprint(fingerprints: &[Vec<f32>], consensus: &[f32]) -> Vec<f32> {
    if fingerprints.is_empty() {
        return Vec::new();
    }

    let length = consensus.len();
    let mut std_dev = Vec::with_capacity(length);

    for i in 0..length {
        let mean = consensus[i];
        let variance = fingerprints
            .iter()
            .map(|fp| (fp[i] - mean).powi(2))
            .sum::<f32>()
            / fingerprints.len() as f32;
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
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_fingerprint_from_signal_valid() {
        // Create a signal with enough samples
        let signal: Vec<i16> = (0..1000).map(|i| (i as i16) % 1000).collect();
        let read_id = Uuid::new_v4();
        let result =
            extract_fingerprint_from_signal(&signal, 0, 500, 10, 5, NormMethod::None, read_id);
        assert!(result.is_some());
        let fp = result.unwrap();
        assert_eq!(fp.read_id, read_id);
        assert!(!fp.values.is_empty());
    }
}

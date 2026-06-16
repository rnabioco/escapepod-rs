//! Shared utilities for demux subcommands.

use escapepod_demux::{BarcodeFingerprint, ReadBoundaries};
use escapepod_signal::Reader;
use escapepod_signal::dtw::NormMethod;
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
/// `None` leaves rayon's default (all available CPUs). Only the first call
/// per process takes effect; subsequent calls are ignored.
pub fn configure_thread_pool(num_threads: Option<usize>) {
    if let Some(n) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }
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

        if parts.len() > max_col
            && let Ok(read_id) = Uuid::parse_str(parts[read_id_col])
        {
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

/// Sum of reads across a list of POD5 files (metadata-only scan, no signal I/O).
///
/// `Reader::read_count` only touches the reads Arrow table, so this is cheap
/// enough to run up-front to size a progress bar.
pub fn total_read_count(input_files: &[PathBuf]) -> usize {
    use rayon::prelude::*;
    input_files
        .par_iter()
        .map(|p| Reader::open(p).and_then(|r| r.read_count()).unwrap_or(0))
        .sum()
}

/// Fan out across POD5 files and across reads within each file, processing
/// each read in-place so signal buffers never outlive the closure.
///
/// Nested rayon: the outer `par_iter` pairs one worker per file; within each
/// file, read metadata is collected cheaply, then a second `par_iter` fans
/// signal decompression + the user closure across workers using the
/// thread-safe `SignalExtractor`. Rayon's work-stealing flows across both
/// levels, so on fixtures with few files and many reads we stop bottlenecking
/// on file count — all CPUs stay busy. Peak signal RAM is still bounded by
/// `rayon::current_num_threads()` reads (not total reads).
pub fn process_reads_par<F, T>(
    input_files: &[PathBuf],
    progress: Option<&indicatif::ProgressBar>,
    process: F,
) -> anyhow::Result<Vec<T>>
where
    F: Fn(Uuid, u64, &[i16]) -> T + Send + Sync,
    T: Send,
{
    use rayon::prelude::*;

    // Batch progress-bar updates: under N rayon workers, `pb.inc(1)` per
    // read is an atomic add contended across all cores. Chunking at 64
    // cuts the contended writes 64× while keeping the bar visibly
    // responsive (64 reads is well under 1s of wall time in practice).
    const PROGRESS_BATCH: usize = 64;

    let per_file: anyhow::Result<Vec<Vec<T>>> = input_files
        .par_iter()
        .map(|path| -> anyhow::Result<Vec<T>> {
            let reader = Reader::open(path)?;
            let reads: Vec<_> = reader
                .reads()?
                .filter_map(Result::ok)
                .filter(|r| !r.signal_rows.is_empty())
                .collect();
            let extractor = reader.signal_extractor()?;

            let out: Vec<T> = reads
                .par_chunks(PROGRESS_BATCH)
                .flat_map(|chunk| {
                    let results: Vec<T> = chunk
                        .iter()
                        .filter_map(|r| {
                            let signal = extractor.get_signal(&r.signal_rows).ok()?;
                            Some(process(r.read_id, r.num_samples, &signal))
                        })
                        .collect();
                    if let Some(pb) = progress {
                        // Count the reads we *attempted* (chunk.len()), not
                        // just the successful ones, so the bar reflects
                        // work done even when signals fail to decompress.
                        pb.inc(chunk.len() as u64);
                    }
                    results
                })
                .collect();
            Ok(out)
        })
        .collect();

    Ok(per_file?.into_iter().flatten().collect())
}

/// Like [`process_reads_par`] but hands the closure a whole *batch* of decoded
/// reads at once, so it can run one batched inference per call rather than one
/// per read.
///
/// Reads within each file are sorted by `num_samples` before chunking, so each
/// fixed-size batch spans a narrow length range — keeping any zero-padding to
/// the batch's max length tight. Peak signal RAM is bounded by `batch_size ×
/// rayon workers` reads. Output is per-file in length-sorted order, so keep
/// results keyed by `read_id` (the row order is not the input order).
pub fn process_read_batches_par<F, T>(
    input_files: &[PathBuf],
    batch_size: usize,
    progress: Option<&indicatif::ProgressBar>,
    process_batch: F,
) -> anyhow::Result<Vec<T>>
where
    F: Fn(&[(Uuid, u64, Vec<i16>)]) -> Vec<T> + Send + Sync,
    T: Send,
{
    use rayon::prelude::*;
    let batch_size = batch_size.max(1);

    let per_file: anyhow::Result<Vec<Vec<T>>> = input_files
        .par_iter()
        .map(|path| -> anyhow::Result<Vec<T>> {
            let reader = Reader::open(path)?;
            let mut reads: Vec<_> = reader
                .reads()?
                .filter_map(Result::ok)
                .filter(|r| !r.signal_rows.is_empty())
                .collect();
            // Length-bucket so each chunk pads minimally to its own max length.
            reads.sort_by_key(|r| r.num_samples);
            let extractor = reader.signal_extractor()?;

            let out: Vec<T> = reads
                .par_chunks(batch_size)
                .flat_map(|chunk| {
                    let batch: Vec<(Uuid, u64, Vec<i16>)> = chunk
                        .iter()
                        .filter_map(|r| {
                            let signal = extractor.get_signal(&r.signal_rows).ok()?;
                            Some((r.read_id, r.num_samples, signal))
                        })
                        .collect();
                    let results = process_batch(&batch);
                    if let Some(pb) = progress {
                        pb.inc(chunk.len() as u64);
                    }
                    results
                })
                .collect();
            Ok(out)
        })
        .collect();

    Ok(per_file?.into_iter().flatten().collect())
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

        let result = parse_boundaries_csv(temp_file.path()).unwrap();

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

        let result = parse_boundaries_csv(temp_file.path()).unwrap();
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
}

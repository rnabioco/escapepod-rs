//! Filter operation for POD5 files.

use crate::utils::{
    add_run_infos_deduplicated, map_run_info_index, parse_uuid_flexible, scan_dictionary_values,
};
use crate::{Error, PredefinedDictionaries, Reader, Result, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use uuid::Uuid;

/// Options for the filter operation.
#[derive(Debug, Clone)]
pub struct FilterOptions {
    /// Signal chunks per batch.
    pub signal_batch_size: u32,
    /// Reads per batch.
    pub read_batch_size: u32,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            signal_batch_size: 1_000,
            read_batch_size: 10_000,
        }
    }
}

/// Result of a filter operation.
#[derive(Debug, Clone, Default)]
pub struct FilterResult {
    /// Total number of reads processed.
    pub total_reads: u64,
    /// Number of reads that matched the filter.
    pub matched_reads: u64,
    /// Number of read errors encountered.
    pub read_errors: u64,
    /// Number of signal errors encountered.
    pub signal_errors: u64,
}

impl FilterResult {
    /// Returns the percentage of reads that matched.
    pub fn match_percentage(&self) -> f64 {
        if self.total_reads > 0 {
            (self.matched_reads as f64 / self.total_reads as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Callback for reporting progress during filtering.
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send>;

/// Filter reads from POD5 files based on a set of read IDs.
///
/// This function reads from multiple input files and writes matching reads
/// to a single output file. It uses lazy signal loading and block-level
/// copying for maximum performance.
///
/// # Arguments
///
/// * `input_files` - Slice of input POD5 file paths
/// * `output_path` - Path to the output POD5 file
/// * `filter_ids` - Set of read IDs to extract
/// * `options` - Filter options (batch sizes, etc.)
/// * `progress` - Optional callback for progress reporting (current, total)
///
/// # Returns
///
/// A `FilterResult` with statistics about the operation.
///
/// # Example
///
/// ```no_run
/// use podfive_core::operations::{filter_files, FilterOptions};
/// use std::collections::HashSet;
/// use std::path::PathBuf;
/// use uuid::Uuid;
///
/// let files = vec![PathBuf::from("input.pod5")];
/// let mut ids = HashSet::new();
/// ids.insert(Uuid::parse_str("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap());
///
/// let result = filter_files(
///     &files,
///     "output.pod5",
///     &ids,
///     FilterOptions::default(),
///     None,
/// )?;
///
/// println!("Matched {} of {} reads", result.matched_reads, result.total_reads);
/// # Ok::<(), podfive_core::Error>(())
/// ```
pub fn filter_files<P: AsRef<Path>>(
    input_files: &[P],
    output_path: impl AsRef<Path>,
    filter_ids: &HashSet<Uuid>,
    options: FilterOptions,
    progress: Option<ProgressCallback>,
) -> Result<FilterResult> {
    let mut result = FilterResult::default();

    // Pre-scan files to collect unique dictionary values
    let scanned = scan_dictionary_values(input_files, Some(filter_ids));
    result.total_reads = scanned.total_read_count;

    // Create writer with predefined dictionaries for consistent multi-batch writes
    let writer_options = WriterOptions {
        signal_batch_size: options.signal_batch_size,
        read_batch_size: options.read_batch_size,
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(scanned.pore_types.into_iter().collect()),
            end_reasons: Some(scanned.end_reasons.into_iter().collect()),
        }),
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(output_path, writer_options)?;

    // Track run infos across all files
    let mut run_info_map: HashMap<String, u32> = HashMap::new();
    let mut processed = 0u64;

    for file_path in input_files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(_) => continue, // Skip unreadable files
        };

        // Add run infos (deduplicated by acquisition_id)
        add_run_infos_deduplicated(&reader, &mut writer, &mut run_info_map)?;

        let reads_iter = match reader.reads() {
            Ok(iter) => iter,
            Err(_) => continue, // Skip files with read errors
        };

        for read_result in reads_iter {
            let read = match read_result {
                Ok(r) => r,
                Err(_) => {
                    result.read_errors += 1;
                    continue;
                }
            };

            processed += 1;
            if let Some(ref cb) = progress {
                cb(processed, result.total_reads);
            }

            // Check if this read's ID is in the filter list
            if filter_ids.contains(&read.read_id) {
                // Lazy load: only fetch signal for matching reads
                let compressed_signal = match reader.get_compressed_signal_for_rows(&read.signal_rows)
                {
                    Ok(s) => s,
                    Err(_) => {
                        result.signal_errors += 1;
                        continue;
                    }
                };

                // Map run_info index
                let new_run_info_idx =
                    map_run_info_index(&reader, read.run_info_index, &run_info_map);

                // Create new read data for writing
                let new_read = read.for_writing(new_run_info_idx);

                writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;
                result.matched_reads += 1;
            }
        }
    }

    writer.finish()?;
    Ok(result)
}

/// Read read IDs from a text file (one per line).
///
/// Supports UUIDs in various formats:
/// - Standard: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
/// - No dashes: `a1b2c3d4e5f67890abcdef1234567890`
///
/// Lines starting with `#` are treated as comments and skipped.
/// Empty lines are also skipped.
///
/// # Example
///
/// ```no_run
/// use podfive_core::operations::read_ids_from_file;
///
/// let ids = read_ids_from_file("ids.txt")?;
/// println!("Loaded {} IDs", ids.len());
/// # Ok::<(), podfive_core::Error>(())
/// ```
pub fn read_ids_from_file(path: impl AsRef<Path>) -> Result<HashSet<Uuid>> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut ids = HashSet::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Try to parse as UUID (supports both standard and compact formats)
        match parse_uuid_flexible(line) {
            Ok(uuid) => {
                ids.insert(uuid);
            }
            Err(_) => {
                return Err(Error::Parse(format!(
                    "Invalid UUID on line {}: '{}'",
                    line_num + 1,
                    line
                )));
            }
        }
    }

    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_ids_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        writeln!(temp_file, "# This is a comment").unwrap();
        writeln!(temp_file).unwrap(); // Empty line
        writeln!(temp_file, "b2c3d4e5f6a78901bcdef12345678901").unwrap(); // No dashes
        temp_file.flush().unwrap();

        let ids = read_ids_from_file(temp_file.path()).unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_read_ids_invalid_uuid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "not-a-uuid").unwrap();
        temp_file.flush().unwrap();

        let result = read_ids_from_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_result_percentage() {
        let result = FilterResult {
            total_reads: 100,
            matched_reads: 25,
            read_errors: 0,
            signal_errors: 0,
        };
        assert!((result.match_percentage() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_result_empty() {
        let result = FilterResult::default();
        assert_eq!(result.match_percentage(), 0.0);
    }
}

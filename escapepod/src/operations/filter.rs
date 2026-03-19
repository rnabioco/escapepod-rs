//! High-performance filter operation for POD5 files.
//!
//! Uses raw byte extraction from mmap without Arrow deserialization.

use crate::arrow_ipc::{ArrowIpcFooter, BatchBlock};
use crate::error::{Error, Result};
use crate::reader::Reader;
use crate::types::{EndReason, FOOTER_MAGIC, POD5_SIGNATURE, ReadData, RunInfoData, Uuid};
use crate::utils::parse_uuid_flexible;
use crate::utils::table_builders::{
    SchemaMetadata, build_pod5_footer, build_reads_table, build_run_info_table,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, Write};
use std::path::Path;
use std::sync::Arc;

/// Criteria for filtering reads.
#[derive(Debug, Clone, Default)]
pub struct FilterCriteria {
    /// Only include reads with these IDs.
    pub read_ids: Option<HashSet<Uuid>>,
    /// Minimum number of samples (inclusive).
    pub min_samples: Option<u64>,
    /// Maximum number of samples (inclusive).
    pub max_samples: Option<u64>,
    /// Only include reads with these end reasons.
    pub include_end_reasons: Option<HashSet<EndReason>>,
    /// Exclude reads with these end reasons.
    pub exclude_end_reasons: Option<HashSet<EndReason>>,
}

impl FilterCriteria {
    /// Check if a read matches all the filter criteria.
    pub fn matches(&self, read: &ReadData) -> bool {
        // Check read ID filter
        if let Some(ref ids) = self.read_ids
            && !ids.contains(&read.read_id) {
                return false;
            }

        // Check min samples
        if let Some(min) = self.min_samples
            && read.num_samples < min {
                return false;
            }

        // Check max samples
        if let Some(max) = self.max_samples
            && read.num_samples > max {
                return false;
            }

        // Check include end reasons (if specified, read must have one of these)
        if let Some(ref include) = self.include_end_reasons
            && !include.contains(&read.end_reason) {
                return false;
            }

        // Check exclude end reasons
        if let Some(ref exclude) = self.exclude_end_reasons
            && exclude.contains(&read.end_reason) {
                return false;
            }

        true
    }

    /// Returns true if the only active criterion is a read-ID set
    /// (no sample-count or end-reason filters). When true, the
    /// accelerated `reads_by_ids()` path can be used.
    pub fn is_uuid_only(&self) -> bool {
        self.read_ids.is_some()
            && self.min_samples.is_none()
            && self.max_samples.is_none()
            && self.include_end_reasons.is_none()
            && self.exclude_end_reasons.is_none()
    }

    /// Returns true if no criteria are set (matches all reads).
    pub fn is_empty(&self) -> bool {
        self.read_ids.is_none()
            && self.min_samples.is_none()
            && self.max_samples.is_none()
            && self.include_end_reasons.is_none()
            && self.exclude_end_reasons.is_none()
    }
}

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
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Collected metadata from a single file for filtering.
struct FileMetadata {
    reader: Reader,
    signal_footer: ArrowIpcFooter,
    run_infos: Vec<RunInfoData>,
    /// Only reads that match the filter criteria.
    matching_reads: Vec<ReadData>,
    /// Total number of reads in the file (for statistics).
    total_read_count: usize,
}

/// Filter reads from POD5 files based on a set of read IDs.
///
/// This function uses raw byte extraction from mmap - no Arrow deserialization.
/// It extracts only the specific signal rows needed for matching reads.
///
/// This is a convenience wrapper around `filter_files_with_criteria` for
/// backwards compatibility.
pub fn filter_files<P: AsRef<Path> + Sync>(
    input_files: &[P],
    output_path: impl AsRef<Path>,
    filter_ids: &HashSet<Uuid>,
    options: FilterOptions,
    progress: Option<ProgressCallback>,
) -> Result<FilterResult> {
    let criteria = FilterCriteria {
        read_ids: Some(filter_ids.clone()),
        ..Default::default()
    };
    filter_files_with_criteria(input_files, output_path, &criteria, options, progress)
}

/// Filter reads from POD5 files based on filter criteria.
///
/// This function uses raw byte extraction from mmap - no Arrow deserialization.
/// It extracts only the specific signal rows needed for matching reads.
pub fn filter_files_with_criteria<P: AsRef<Path> + Sync>(
    input_files: &[P],
    output_path: impl AsRef<Path>,
    criteria: &FilterCriteria,
    options: FilterOptions,
    progress: Option<ProgressCallback>,
) -> Result<FilterResult> {
    if input_files.is_empty() {
        return Err(Error::InvalidState("No input files specified".into()));
    }

    if criteria.is_empty() {
        return Err(Error::InvalidState("No filter criteria specified".into()));
    }

    let num_files = input_files.len();

    // Phase 1: Open files and identify matching reads in parallel
    let input_paths: Vec<&Path> = input_files.iter().map(|p| p.as_ref()).collect();

    let metadata_results: Vec<Result<FileMetadata>> = input_paths
        .par_iter()
        .map(|path| {
            let reader = Reader::open(path)?;
            let signal_bytes = reader.signal_table_bytes()?;
            let signal_footer = ArrowIpcFooter::parse(signal_bytes)?;
            let run_infos = reader.run_infos().to_vec();

            // Fast path: when only filtering by UUID, use reads_by_ids()
            // to skip non-matching batches entirely.
            let (matching_reads, total_read_count) = if criteria.is_uuid_only() {
                let total = reader.read_count()?;
                let target_ids = criteria.read_ids.as_ref().unwrap();
                let matching = reader.reads_by_ids(target_ids)?;
                (matching, total)
            } else {
                let all_reads: Vec<ReadData> = reader
                    .reads()?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let total = all_reads.len();
                let matching = all_reads
                    .into_iter()
                    .filter(|read| criteria.matches(read))
                    .collect();
                (matching, total)
            };

            Ok(FileMetadata {
                reader,
                signal_footer,
                run_infos,
                matching_reads,
                total_read_count,
            })
        })
        .collect();

    let file_metadata: Vec<FileMetadata> =
        metadata_results.into_iter().collect::<Result<Vec<_>>>()?;

    // Count total reads for statistics
    let total_read_count: u64 = file_metadata
        .iter()
        .map(|m| m.total_read_count as u64)
        .sum();
    let matching_count: u64 = file_metadata
        .iter()
        .map(|m| m.matching_reads.len() as u64)
        .sum();

    // Phase 2: Extract signal data in parallel across files
    // Each file's mmap is independent, so extraction can be parallelized
    struct FileSignalExtraction {
        /// Extracted signal chunks (Arc to avoid copying)
        chunks: Vec<(Arc<[u8]>, u32)>, // (signal_data, samples)
    }

    let extractions: Vec<Result<FileSignalExtraction>> = file_metadata
        .par_iter()
        .map(|metadata| {
            if metadata.matching_reads.is_empty() {
                return Ok(FileSignalExtraction { chunks: Vec::new() });
            }

            let signal_bytes = metadata.reader.signal_table_bytes()?;

            // Collect all signal rows needed from this file
            let signal_row_indices: Vec<u64> = metadata
                .matching_reads
                .iter()
                .flat_map(|read| read.signal_rows.iter().copied())
                .collect();

            // Extract all signal rows in batch
            let raw_chunks = metadata
                .signal_footer
                .extract_signal_rows(&signal_row_indices, signal_bytes)?;

            // Store as Arc to avoid copying signal bytes
            let chunks: Vec<(Arc<[u8]>, u32)> = raw_chunks
                .into_iter()
                .map(|chunk| (Arc::from(chunk.signal), chunk.samples))
                .collect();

            Ok(FileSignalExtraction { chunks })
        })
        .collect();

    // Unwrap results and combine sequentially (preserves file order)
    let file_extractions: Vec<FileSignalExtraction> =
        extractions.into_iter().collect::<Result<Vec<_>>>()?;

    // Combine all signal chunks with sequential index assignment
    let mut all_signal_chunks: Vec<(usize, u64, u32)> = Vec::new();
    let mut signal_data_arcs: Vec<Arc<[u8]>> = Vec::new();
    let mut current_signal_row: u64 = 0;

    for (file_idx, extraction) in file_extractions.iter().enumerate() {
        for (signal_data, samples) in &extraction.chunks {
            signal_data_arcs.push(signal_data.clone());
            all_signal_chunks.push((signal_data_arcs.len() - 1, current_signal_row, *samples));
            current_signal_row += 1;
        }

        if let Some(ref cb) = progress {
            cb(file_idx as u64 + 1, num_files as u64);
        }
    }

    // Phase 3: Build output file
    // We need to write:
    // 1. Signal table with extracted chunks
    // 2. Run info table (deduplicated)
    // 3. Reads table with remapped signal rows

    let schema_meta = SchemaMetadata::new();

    let file = File::create(output_path.as_ref())?;
    let mut file = BufWriter::with_capacity(16 * 1024 * 1024, file);

    // Write POD5 header
    file.write_all(&POD5_SIGNATURE)?;
    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;

    // Build signal table - we need to create new Arrow IPC batches
    // For simplicity, put all signal in one batch
    let _signal_table_start = file.stream_position()? as i64;

    // Build the signal table using raw bytes
    let (signal_table_bytes, _signal_batches) = build_raw_signal_table(
        &signal_data_arcs,
        &all_signal_chunks,
        options.signal_batch_size,
        &schema_meta,
    )?;
    file.write_all(&signal_table_bytes)?;

    let signal_end = file.stream_position()? as usize;

    // Pad to 8-byte alignment
    let padding_needed = (8 - (signal_end % 8)) % 8;
    for _ in 0..padding_needed {
        file.write_all(&[0u8])?;
    }

    // Write section marker (reuse same UUID throughout file)
    file.write_all(section_marker.as_bytes())?;

    // Build deduplicated run_info table
    let mut run_info_map: HashMap<String, u32> = HashMap::new();
    let mut all_run_infos: Vec<RunInfoData> = Vec::new();

    for metadata in &file_metadata {
        for run_info in &metadata.run_infos {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = all_run_infos.len() as u32;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
                all_run_infos.push(run_info.clone());
            }
        }
    }

    let run_info_offset = file.stream_position()? as i64;
    let run_info_bytes = build_run_info_table(&all_run_infos, &schema_meta)?;
    file.write_all(&run_info_bytes)?;
    let run_info_length = run_info_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    file.write_all(section_marker.as_bytes())?;

    // Build reads table with new signal row indices
    let reads_offset = file.stream_position()? as i64;

    let mut processed_reads: Vec<(ReadData, Vec<u64>)> = Vec::new();
    let mut signal_row_cursor: u64 = 0;

    for metadata in &file_metadata {
        for read in &metadata.matching_reads {
            // New signal rows are sequential starting from signal_row_cursor
            let num_signal_rows = read.signal_rows.len();
            let new_signal_rows: Vec<u64> =
                (signal_row_cursor..signal_row_cursor + num_signal_rows as u64).collect();
            signal_row_cursor += num_signal_rows as u64;

            // Update run_info index
            let original_run_info = metadata.run_infos.get(read.run_info_index as usize);
            let new_run_info_idx = if let Some(ri) = original_run_info {
                *run_info_map.get(&ri.acquisition_id).unwrap_or(&0)
            } else {
                0
            };

            let new_read = read.for_writing(new_run_info_idx);
            processed_reads.push((new_read, new_signal_rows));
        }
    }

    let reads_bytes = build_reads_table(&processed_reads, &all_run_infos, &schema_meta)?;
    file.write_all(&reads_bytes)?;
    let reads_length = reads_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    file.write_all(section_marker.as_bytes())?;

    // Write POD5 footer
    file.write_all(&FOOTER_MAGIC)?;

    let signal_offset_val = 24i64; // POD5 header size
    let signal_length = signal_end as i64 - 24;

    let pod5_footer = build_pod5_footer(
        signal_offset_val,
        signal_length,
        run_info_offset,
        run_info_length,
        reads_offset,
        reads_length,
        &schema_meta,
    )?;
    file.write_all(&pod5_footer)?;

    let footer_len = pod5_footer.len() as i64;
    file.write_all(&footer_len.to_le_bytes())?;

    file.write_all(section_marker.as_bytes())?;
    file.write_all(&POD5_SIGNATURE)?;

    file.flush()?;

    Ok(FilterResult {
        total_reads: total_read_count,
        matched_reads: matching_count,
        read_errors: 0,
        signal_errors: 0,
    })
}

/// Build a signal table from raw signal chunks.
/// Returns the complete IPC bytes and batch metadata.
fn build_raw_signal_table(
    signal_data: &[Arc<[u8]>],
    chunks: &[(usize, u64, u32)], // (data_idx, _signal_row, samples)
    batch_size: u32,
    meta: &SchemaMetadata,
) -> Result<(Vec<u8>, Vec<BatchBlock>)> {
    use crate::schema::signal_schema;
    use arrow::array::{ArrayRef, FixedSizeBinaryBuilder, LargeBinaryBuilder, UInt32Builder};
    use arrow::ipc::writer::FileWriter;
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(meta.apply(signal_schema()));

    let mut output = Vec::new();
    let mut writer = FileWriter::try_new(&mut output, &schema)?;

    // Build batches
    let total_rows = chunks.len();
    let mut offset = 0;

    while offset < total_rows {
        let end = std::cmp::min(offset + batch_size as usize, total_rows);
        let batch_chunks = &chunks[offset..end];

        let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(batch_chunks.len(), 16);
        let mut signal_builder = LargeBinaryBuilder::new();
        let mut samples_builder = UInt32Builder::with_capacity(batch_chunks.len());

        for (data_idx, _, samples) in batch_chunks {
            // Use a placeholder read_id (the actual read_id is in the reads table)
            read_id_builder.append_value([0u8; 16])?;
            signal_builder.append_value(&signal_data[*data_idx]);
            samples_builder.append_value(*samples);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(read_id_builder.finish()) as ArrayRef,
                Arc::new(signal_builder.finish()) as ArrayRef,
                Arc::new(samples_builder.finish()) as ArrayRef,
            ],
        )?;

        writer.write(&batch)?;
        offset = end;
    }

    writer.finish()?;
    drop(writer);

    // Parse the output to get batch metadata
    let footer = ArrowIpcFooter::parse(&output)?;
    let batches = footer.record_batches.clone();

    Ok((output, batches))
}

/// Read read IDs from a text file or stdin (one per line).
///
/// If `path` is "-" or "stdin" (case-insensitive), reads from stdin.
/// Otherwise reads from the specified file.
///
/// Supports UUIDs in various formats:
/// - Standard: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
/// - No dashes: `a1b2c3d4e5f67890abcdef1234567890`
///
/// Lines starting with `#` are treated as comments and skipped.
/// Empty lines are also skipped.
pub fn read_ids_from_file(path: impl AsRef<Path>) -> Result<HashSet<Uuid>> {
    let path_str = path.as_ref().to_string_lossy();

    if path_str == "-" || path_str.eq_ignore_ascii_case("stdin") {
        read_ids_from_reader(std::io::stdin().lock())
    } else {
        let file = File::open(path.as_ref())?;
        read_ids_from_reader(BufReader::new(file))
    }
}

/// Read read IDs from any `BufRead` source.
fn read_ids_from_reader<R: BufRead>(reader: R) -> Result<HashSet<Uuid>> {
    let mut ids = HashSet::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

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
    use crate::types::ReadData;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_test_read(num_samples: u64, end_reason: EndReason) -> ReadData {
        ReadData {
            read_id: Uuid::new_v4(),
            num_samples,
            end_reason,
            ..Default::default()
        }
    }

    #[test]
    fn test_filter_criteria_is_empty() {
        let criteria = FilterCriteria::default();
        assert!(criteria.is_empty());

        let criteria = FilterCriteria {
            min_samples: Some(100),
            ..Default::default()
        };
        assert!(!criteria.is_empty());
    }

    #[test]
    fn test_filter_criteria_is_uuid_only() {
        // UUID-only: should be true
        let criteria = FilterCriteria {
            read_ids: Some(HashSet::new()),
            ..Default::default()
        };
        assert!(criteria.is_uuid_only());

        // No criteria at all: not UUID-only
        assert!(!FilterCriteria::default().is_uuid_only());

        // UUID + min_samples: not UUID-only
        let criteria = FilterCriteria {
            read_ids: Some(HashSet::new()),
            min_samples: Some(100),
            ..Default::default()
        };
        assert!(!criteria.is_uuid_only());

        // Only min_samples, no UUIDs: not UUID-only
        let criteria = FilterCriteria {
            min_samples: Some(100),
            ..Default::default()
        };
        assert!(!criteria.is_uuid_only());
    }

    #[test]
    fn test_filter_criteria_matches_read_ids() {
        let read = make_test_read(1000, EndReason::SignalPositive);
        let mut ids = HashSet::new();
        ids.insert(read.read_id);

        let criteria = FilterCriteria {
            read_ids: Some(ids),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Different read ID should not match
        let other_read = make_test_read(1000, EndReason::SignalPositive);
        assert!(!criteria.matches(&other_read));
    }

    #[test]
    fn test_filter_criteria_matches_min_samples() {
        let read = make_test_read(5000, EndReason::SignalPositive);

        // Read with 5000 samples should match min_samples=4000
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should match min_samples=5000 (inclusive)
        let criteria = FilterCriteria {
            min_samples: Some(5000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should not match min_samples=6000
        let criteria = FilterCriteria {
            min_samples: Some(6000),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_max_samples() {
        let read = make_test_read(5000, EndReason::SignalPositive);

        // Read with 5000 samples should match max_samples=6000
        let criteria = FilterCriteria {
            max_samples: Some(6000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should match max_samples=5000 (inclusive)
        let criteria = FilterCriteria {
            max_samples: Some(5000),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Read with 5000 samples should not match max_samples=4000
        let criteria = FilterCriteria {
            max_samples: Some(4000),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_include_end_reasons() {
        let read = make_test_read(1000, EndReason::SignalPositive);

        // Should match when end_reason is in the include set
        let mut include = HashSet::new();
        include.insert(EndReason::SignalPositive);
        include.insert(EndReason::SignalNegative);
        let criteria = FilterCriteria {
            include_end_reasons: Some(include),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Should not match when end_reason is not in the include set
        let mut include = HashSet::new();
        include.insert(EndReason::MuxChange);
        let criteria = FilterCriteria {
            include_end_reasons: Some(include),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_exclude_end_reasons() {
        let read = make_test_read(1000, EndReason::SignalPositive);

        // Should match when end_reason is not in the exclude set
        let mut exclude = HashSet::new();
        exclude.insert(EndReason::MuxChange);
        exclude.insert(EndReason::UnblockMuxChange);
        let criteria = FilterCriteria {
            exclude_end_reasons: Some(exclude),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Should not match when end_reason is in the exclude set
        let mut exclude = HashSet::new();
        exclude.insert(EndReason::SignalPositive);
        let criteria = FilterCriteria {
            exclude_end_reasons: Some(exclude),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_combined() {
        let read = make_test_read(5000, EndReason::SignalPositive);

        // Should match when all criteria are satisfied
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            max_samples: Some(6000),
            include_end_reasons: Some([EndReason::SignalPositive].into_iter().collect()),
            ..Default::default()
        };
        assert!(criteria.matches(&read));

        // Should not match when one criterion fails (samples too low)
        let criteria = FilterCriteria {
            min_samples: Some(6000),
            max_samples: Some(10000),
            include_end_reasons: Some([EndReason::SignalPositive].into_iter().collect()),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));

        // Should not match when one criterion fails (wrong end reason)
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            max_samples: Some(6000),
            include_end_reasons: Some([EndReason::MuxChange].into_iter().collect()),
            ..Default::default()
        };
        assert!(!criteria.matches(&read));
    }

    #[test]
    fn test_filter_criteria_matches_sample_range() {
        // Test combined min and max samples
        let criteria = FilterCriteria {
            min_samples: Some(4000),
            max_samples: Some(6000),
            ..Default::default()
        };

        assert!(!criteria.matches(&make_test_read(3999, EndReason::SignalPositive)));
        assert!(criteria.matches(&make_test_read(4000, EndReason::SignalPositive)));
        assert!(criteria.matches(&make_test_read(5000, EndReason::SignalPositive)));
        assert!(criteria.matches(&make_test_read(6000, EndReason::SignalPositive)));
        assert!(!criteria.matches(&make_test_read(6001, EndReason::SignalPositive)));
    }

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

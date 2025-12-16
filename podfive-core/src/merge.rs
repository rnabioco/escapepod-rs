//! High-performance POD5 file merging.
//!
//! This module provides functionality to merge multiple POD5 files into one,
//! using raw byte copying to avoid Arrow deserialization overhead.

use crate::arrow_ipc::{ArrowIpcFooter, BatchBlock};
use crate::error::{Error, Result};
use crate::reader::Reader;
use crate::types::{ReadData, RunInfoData, Uuid, FOOTER_MAGIC, POD5_SIGNATURE};
use crate::utils::table_builders::{
    build_arrow_ipc_footer, build_pod5_footer, build_reads_table, build_run_info_table,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Options for merge operations.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Allow duplicate read IDs (default: false, skip duplicates).
    pub duplicate_ok: bool,
    /// Number of reads per batch in output file.
    pub read_batch_size: u32,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            duplicate_ok: false,
            read_batch_size: 100_000,
        }
    }
}

/// Result of a merge operation.
#[derive(Debug)]
pub struct MergeResult {
    /// Number of reads written.
    pub reads_written: u64,
    /// Number of duplicate reads skipped.
    pub duplicates_skipped: u64,
    /// Number of signal rows written.
    pub signal_rows: u64,
    /// Number of files processed.
    pub files_processed: usize,
}

/// Merge multiple POD5 files into a single output file.
///
/// This function uses zero-copy async I/O with scoped threads to overlap
/// reading and writing, passing mmap slices directly to the writer thread.
///
/// # Arguments
/// * `inputs` - Slice of input file paths
/// * `output` - Output file path
/// * `options` - Merge options
/// * `progress_callback` - Optional callback for progress updates (file_idx, total_files)
///
/// # Returns
/// A `MergeResult` with statistics about the merge operation.
pub fn merge_files<P: AsRef<Path>, Q: AsRef<Path>>(
    inputs: &[P],
    output: Q,
    options: &MergeOptions,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeResult> {
    if inputs.is_empty() {
        return Err(Error::InvalidState("No input files specified".into()));
    }

    merge_impl(inputs, output, options, progress_callback)
}

/// Collected metadata from a single file for merging.
struct FileMetadata {
    reader: Reader,
    footer: ArrowIpcFooter,
    run_infos: Vec<RunInfoData>,
    reads: Vec<ReadData>,
}

/// Main merge implementation using zero-copy async I/O.
/// Uses scoped threads to pass mmap slices directly to writer thread.
fn merge_impl<P: AsRef<Path>, Q: AsRef<Path>>(
    inputs: &[P],
    output: Q,
    options: &MergeOptions,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeResult> {
    use std::sync::mpsc;
    use std::thread;

    let num_files = inputs.len();

    // Convert to owned paths for parallel processing
    let input_paths: Vec<&Path> = inputs.iter().map(|p| p.as_ref()).collect();

    // Phase 1: Open files and collect metadata in parallel (single open per file)
    let metadata_results: Vec<Result<FileMetadata>> = input_paths
        .par_iter()
        .map(|path| {
            let reader = Reader::open(path)?;
            let signal_bytes = reader.signal_table_bytes()?;
            let footer = ArrowIpcFooter::parse(signal_bytes)?;
            let run_infos = reader.run_infos().to_vec();
            let reads: Vec<ReadData> = reader
                .reads()?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(FileMetadata {
                reader,
                footer,
                run_infos,
                reads,
            })
        })
        .collect();

    // Unwrap results and count reads
    let file_metadata: Vec<FileMetadata> =
        metadata_results.into_iter().collect::<Result<Vec<_>>>()?;
    let total_read_count: u64 = file_metadata.iter().map(|m| m.reads.len() as u64).sum();

    // Phase 2: Write signal data using scoped thread (zero-copy from mmap)
    let mut all_batches: Vec<BatchBlock> = Vec::new();
    let mut current_offset: usize = 0;
    let mut current_signal_row: u64 = 0;
    let mut signal_offsets: Vec<u64> = Vec::with_capacity(num_files);

    // Use scoped thread to allow borrowing mmap slices without copying
    let (file, signal_end, signal_rows) = thread::scope(|scope| -> Result<(File, usize, u64)> {
        // Channel for sending byte slices to writer thread
        let (tx, rx) = mpsc::sync_channel::<&[u8]>(4); // Small buffer for backpressure

        // Spawn writer thread within scope - can borrow from parent
        let output_path = output.as_ref();
        let writer_handle = scope.spawn(move || -> std::io::Result<(File, usize)> {
            let file = File::create(output_path)?;
            let mut file = BufWriter::with_capacity(16 * 1024 * 1024, file);

            // Write POD5 header
            file.write_all(&POD5_SIGNATURE)?;
            let section_marker = Uuid::new_v4();
            file.write_all(section_marker.as_bytes())?;

            // Write all signal data from channel
            for bytes in rx {
                file.write_all(bytes)?;
            }

            let pos = file.stream_position()? as usize;
            file.flush()?;
            Ok((file.into_inner()?, pos))
        });

        // Main thread: send signal bytes to writer
        let mut header_written = false;

        for (file_idx, metadata) in file_metadata.iter().enumerate() {
            let signal_bytes = metadata.reader.signal_table_bytes()?;

            // Record signal row offset for this file
            signal_offsets.push(current_signal_row);

            // Write header from first file only
            if !header_written {
                let header_bytes = metadata.footer.header_bytes(signal_bytes);
                tx.send(header_bytes)
                    .map_err(|_| Error::Io(std::io::Error::other("Writer thread closed")))?;
                current_offset = header_bytes.len();
                header_written = true;
            }

            // Send batch bytes directly (zero-copy from mmap)
            let batches_bytes = metadata.footer.batches_bytes(signal_bytes);
            tx.send(batches_bytes)
                .map_err(|_| Error::Io(std::io::Error::other("Writer thread closed")))?;

            // Adjust batch offsets for the combined output
            for batch in &metadata.footer.record_batches {
                let relative_offset = batch.offset as usize - metadata.footer.batches_start_offset;
                let new_offset = current_offset + relative_offset;

                all_batches.push(BatchBlock {
                    offset: new_offset as i64,
                    metadata_length: batch.metadata_length,
                    body_length: batch.body_length,
                    row_count: batch.row_count,
                });
            }

            current_offset += batches_bytes.len();
            current_signal_row += metadata.footer.total_rows;

            if let Some(cb) = progress_callback {
                cb(file_idx + 1, num_files);
            }
        }

        // Close channel - must happen before footer_bytes is created
        // to ensure all mmap slices are consumed
        drop(tx);

        // Wait for writer to finish with mmap data
        let (mut file, _signal_end) = writer_handle
            .join()
            .map_err(|_| Error::Io(std::io::Error::other("Writer thread panicked")))?
            .map_err(Error::Io)?;

        // Write IPC footer directly (small data, no need for async)
        let footer_bytes = build_arrow_ipc_footer(&all_batches)?;
        file.write_all(&footer_bytes).map_err(Error::Io)?;

        let footer_len = footer_bytes.len() as i32;
        file.write_all(&footer_len.to_le_bytes())
            .map_err(Error::Io)?;
        file.write_all(b"ARROW1").map_err(Error::Io)?;
        file.flush().map_err(Error::Io)?;

        let final_pos = file.stream_position().map_err(Error::Io)? as usize;

        Ok((file, final_pos, current_signal_row))
    })?;

    // Phase 3: Write remaining sections using BufWriter
    let mut file = BufWriter::with_capacity(16 * 1024 * 1024, file);
    file.seek(SeekFrom::Start(signal_end as u64))?;

    // Pad to 8-byte alignment
    let padding_needed = (8 - (signal_end % 8)) % 8;
    for _ in 0..padding_needed {
        file.write_all(&[0u8])?;
    }

    // Write section marker
    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;

    // Build and write run_info table
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
    let run_info_bytes = build_run_info_table(&all_run_infos)?;
    file.write_all(&run_info_bytes)?;
    let run_info_length = run_info_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;

    // Build and write reads table
    let reads_offset = file.stream_position()? as i64;

    let mut seen_reads: HashSet<Uuid> = if options.duplicate_ok {
        HashSet::new()
    } else {
        HashSet::with_capacity(total_read_count as usize)
    };

    let mut processed_reads: Vec<(ReadData, Vec<u64>)> = Vec::new();
    let mut total_reads = 0u64;
    let mut duplicate_count = 0u64;

    for (metadata, &signal_offset) in file_metadata.iter().zip(signal_offsets.iter()) {
        for read in &metadata.reads {
            if !options.duplicate_ok {
                if seen_reads.contains(&read.read_id) {
                    duplicate_count += 1;
                    continue;
                }
                seen_reads.insert(read.read_id);
            }

            let original_run_info = metadata.run_infos.get(read.run_info_index as usize);
            let new_run_info_idx = if let Some(ri) = original_run_info {
                *run_info_map.get(&ri.acquisition_id).unwrap_or(&0)
            } else {
                0
            };

            let new_signal_rows: Vec<u64> = read
                .signal_rows
                .iter()
                .map(|&row| row + signal_offset)
                .collect();

            let new_read = read.for_writing(new_run_info_idx);
            processed_reads.push((new_read, new_signal_rows));
            total_reads += 1;
        }
    }

    let reads_bytes = build_reads_table(&processed_reads, &all_run_infos)?;
    file.write_all(&reads_bytes)?;
    let reads_length = reads_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    let section_marker = Uuid::new_v4();
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
    )?;
    file.write_all(&pod5_footer)?;

    let footer_len = pod5_footer.len() as i64;
    file.write_all(&footer_len.to_le_bytes())?;

    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;
    file.write_all(&POD5_SIGNATURE)?;

    file.flush()?;

    Ok(MergeResult {
        reads_written: total_reads,
        duplicates_skipped: duplicate_count,
        signal_rows,
        files_processed: file_metadata.len(),
    })
}

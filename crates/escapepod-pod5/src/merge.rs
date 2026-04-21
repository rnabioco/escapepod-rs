//! High-performance POD5 file merging.
//!
//! This module provides functionality to merge multiple POD5 files into one,
//! using raw byte copying to avoid Arrow deserialization overhead.

use crate::arrow_ipc::{ArrowIpcFooter, BatchBlock};
use crate::error::{Error, Result};
use crate::reader::Reader;
use crate::types::{POD5_SIGNATURE, ReadData, RunInfoData, Uuid};
use crate::utils::pod5_assembler::{
    SourceFileMetadata, deduplicate_run_infos, write_post_signal_sections,
};
use crate::utils::table_builders::{SchemaMetadata, build_arrow_ipc_footer};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Write buffer size for the merge output file (128 MiB).
const MERGE_WRITE_BUFFER_SIZE: usize = 128 * 1024 * 1024;

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

/// Phase of the merge operation for progress reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergePhase {
    /// Loading metadata from input files (parallel).
    LoadingMetadata,
    /// Writing signal data to output file.
    WritingSignal,
    /// Writing reads table.
    WritingReads,
}

/// Progress information for merge operations.
#[derive(Debug, Clone)]
pub struct MergeProgress {
    /// Current phase of the merge.
    pub phase: MergePhase,
    /// Current item being processed (file index or similar).
    pub current: usize,
    /// Total items in this phase.
    pub total: usize,
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
/// * `progress_callback` - Optional callback for progress updates with phase info
///
/// # Returns
/// A `MergeResult` with statistics about the merge operation.
pub fn merge_files<P: AsRef<Path>, Q: AsRef<Path>>(
    inputs: &[P],
    output: Q,
    options: &MergeOptions,
    progress_callback: Option<&(dyn Fn(MergeProgress) + Sync + Send)>,
) -> Result<MergeResult> {
    if inputs.is_empty() {
        return Err(Error::InvalidState("No input files specified".into()));
    }

    merge_impl(inputs, output, options, progress_callback)
}

/// Collected metadata from a single file for merging.
struct FileMetadata {
    footer: ArrowIpcFooter,
    run_infos: Vec<RunInfoData>,
    reads: Vec<ReadData>,
    /// Pre-read signal header bytes (Arrow schema/magic).
    signal_header: Vec<u8>,
    /// Pre-read signal batch bytes (all record batches).
    signal_batches: Vec<u8>,
}

/// Main merge implementation using zero-copy async I/O.
/// Uses scoped threads to pass mmap slices directly to writer thread.
fn merge_impl<P: AsRef<Path>, Q: AsRef<Path>>(
    inputs: &[P],
    output: Q,
    options: &MergeOptions,
    progress_callback: Option<&(dyn Fn(MergeProgress) + Sync + Send)>,
) -> Result<MergeResult> {
    let num_files = inputs.len();

    // Convert to owned paths for parallel processing
    let input_paths: Vec<&Path> = inputs.iter().map(|p| p.as_ref()).collect();

    // Progress counter for parallel metadata loading
    let files_loaded = AtomicUsize::new(0);

    // Phase 1: Open files and collect metadata in parallel
    // Pre-reads signal bytes into memory so Phase 2 only writes from RAM
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

            // Pre-read signal bytes into memory (moves I/O to parallel phase)
            let signal_header = footer.header_bytes(signal_bytes).to_vec();
            let signal_batches = footer.batches_bytes(signal_bytes).to_vec();

            // Update progress after successfully loading a file
            let loaded = files_loaded.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(cb) = &progress_callback {
                cb(MergeProgress {
                    phase: MergePhase::LoadingMetadata,
                    current: loaded,
                    total: num_files,
                });
            }

            Ok(FileMetadata {
                footer,
                run_infos,
                reads,
                signal_header,
                signal_batches,
            })
        })
        .collect();

    // Unwrap results and count reads
    let file_metadata: Vec<FileMetadata> =
        metadata_results.into_iter().collect::<Result<Vec<_>>>()?;
    let total_read_count: u64 = file_metadata.iter().map(|m| m.reads.len() as u64).sum();

    // Phase 2: Write signal data using scoped thread (from pre-read memory)
    let mut all_batches: Vec<BatchBlock> = Vec::new();
    let mut current_offset: usize = 0;
    let mut current_signal_row: u64 = 0;
    let mut signal_offsets: Vec<u64> = Vec::with_capacity(num_files);

    use std::sync::mpsc;
    use std::thread;

    // Create a single section marker UUID to reuse at all section boundaries
    let section_marker = Uuid::new_v4();
    let schema_meta = SchemaMetadata::new();

    let (mut file, signal_end, signal_rows) =
        thread::scope(|scope| -> Result<(File, usize, u64)> {
            // Channel for sending byte slices to writer thread
            // Buffer of 32 allows sender to stay ahead of writer
            let (tx, rx) = mpsc::sync_channel::<&[u8]>(32);

            // Spawn writer thread within scope - can borrow from parent
            let output_path = output.as_ref();
            let writer_handle = scope.spawn(move || -> std::io::Result<(File, usize)> {
                let file = File::create(output_path)?;
                let mut file = BufWriter::with_capacity(MERGE_WRITE_BUFFER_SIZE, file);

                // Write POD5 header
                file.write_all(&POD5_SIGNATURE)?;
                file.write_all(section_marker.as_bytes())?;

                // Write all signal data from channel
                for bytes in rx {
                    file.write_all(bytes)?;
                }

                let pos = file.stream_position()? as usize;
                file.flush()?;
                Ok((file.into_inner()?, pos))
            });

            // Main thread: send pre-read signal bytes to writer
            let mut header_written = false;

            for (file_idx, metadata) in file_metadata.iter().enumerate() {
                // Record signal row offset for this file
                signal_offsets.push(current_signal_row);

                // Write header from first file only
                if !header_written {
                    tx.send(&metadata.signal_header)
                        .map_err(|_| Error::Io(std::io::Error::other("Writer thread closed")))?;
                    current_offset = metadata.signal_header.len();
                    header_written = true;
                }

                // Send pre-read batch bytes (no mmap access here)
                tx.send(&metadata.signal_batches)
                    .map_err(|_| Error::Io(std::io::Error::other("Writer thread closed")))?;

                // Adjust batch offsets for the combined output
                for batch in &metadata.footer.record_batches {
                    let relative_offset =
                        batch.offset as usize - metadata.footer.batches_start_offset;
                    let new_offset = current_offset + relative_offset;

                    all_batches.push(BatchBlock {
                        offset: new_offset as i64,
                        metadata_length: batch.metadata_length,
                        body_length: batch.body_length,
                        row_count: batch.row_count,
                    });
                }

                current_offset += metadata.signal_batches.len();
                current_signal_row += metadata.footer.total_rows;

                if let Some(cb) = progress_callback {
                    cb(MergeProgress {
                        phase: MergePhase::WritingSignal,
                        current: file_idx + 1,
                        total: num_files,
                    });
                }
            }

            // Close channel
            drop(tx);

            // Wait for writer to finish
            let (mut file, _signal_end) = writer_handle
                .join()
                .map_err(|_| Error::Io(std::io::Error::other("Writer thread panicked")))?
                .map_err(Error::Io)?;

            // Write IPC footer directly (small data, no need for async).
            // The footer must embed the real signal schema — Arrow's reader
            // trusts the footer's schema when decoding batches, so an empty
            // one silently strips every column.
            let signal_schema = schema_meta.apply(crate::schema::signal_schema());
            let footer_bytes = build_arrow_ipc_footer(&all_batches, &signal_schema)?;
            file.write_all(&footer_bytes).map_err(Error::Io)?;

            let footer_len = footer_bytes.len() as i32;
            file.write_all(&footer_len.to_le_bytes())
                .map_err(Error::Io)?;
            file.write_all(b"ARROW1").map_err(Error::Io)?;
            file.flush().map_err(Error::Io)?;

            let final_pos = file.stream_position().map_err(Error::Io)? as usize;

            Ok((file, final_pos, current_signal_row))
        })?;

    // Phase 3: Write remaining sections (run info, reads, footer)

    // Build source metadata for run info dedup
    let source_metadata: Vec<SourceFileMetadata> = file_metadata
        .iter()
        .map(|m| SourceFileMetadata {
            run_infos: m.run_infos.clone(),
        })
        .collect();

    let (_all_run_infos, run_info_map) = deduplicate_run_infos(&source_metadata);

    // Notify start of reads phase
    if let Some(cb) = progress_callback {
        cb(MergeProgress {
            phase: MergePhase::WritingReads,
            current: 0,
            total: total_read_count as usize,
        });
    }

    // Transform reads in parallel (signal row adjustment, run_info remapping)
    let per_file_reads: Vec<Vec<(Uuid, ReadData, Vec<u64>)>> = file_metadata
        .par_iter()
        .zip(signal_offsets.par_iter())
        .map(|(metadata, &signal_offset)| {
            metadata
                .reads
                .iter()
                .map(|read| {
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
                    (read.read_id, new_read, new_signal_rows)
                })
                .collect()
        })
        .collect();

    // Sequential duplicate filtering (requires ordered access to seen_reads)
    let mut seen_reads: HashSet<Uuid> = if options.duplicate_ok {
        HashSet::new()
    } else {
        HashSet::with_capacity(total_read_count as usize)
    };

    let mut processed_reads: Vec<(ReadData, Vec<u64>)> =
        Vec::with_capacity(total_read_count as usize);
    let mut duplicate_count = 0u64;

    for file_reads in per_file_reads {
        for (read_id, new_read, new_signal_rows) in file_reads {
            if !options.duplicate_ok {
                if seen_reads.contains(&read_id) {
                    duplicate_count += 1;
                    continue;
                }
                seen_reads.insert(read_id);
            }
            processed_reads.push((new_read, new_signal_rows));
        }
    }

    let total_reads = processed_reads.len() as u64;

    // Write post-signal sections (run info, reads, footer)
    write_post_signal_sections(
        &mut file,
        &section_marker,
        &schema_meta,
        signal_end,
        &source_metadata,
        &processed_reads,
    )?;

    Ok(MergeResult {
        reads_written: total_reads,
        duplicates_skipped: duplicate_count,
        signal_rows,
        files_processed: file_metadata.len(),
    })
}

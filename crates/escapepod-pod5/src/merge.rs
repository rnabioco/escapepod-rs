//! High-performance POD5 file merging.
//!
//! This module provides functionality to merge multiple POD5 files into one.
//!
//! The signal table is rebuilt at a uniform batch stride rather than
//! concatenating each input's own Arrow batches as-is. A read's `signal`
//! column holds GLOBAL row indices, and readers resolve one to a position by
//! assuming a constant stride between batches (`escpod` itself does not —
//! see [`crate::reader::file_reader::Reader::nonuniform_signal_batch`] — but
//! the official `pod5` library and dorado do). Almost every POD5 file's own
//! *last* batch is short (its read count rarely divides evenly by the batch
//! size), so copying N files' batches back-to-back puts a short batch from
//! file `k` immediately before a full one from file `k+1` at every file
//! boundary but the last — breaking the stride dorado assumes at every one
//! of those points. `filter`/`subset` already avoid this by flattening every
//! source's compressed signal chunks into one list and re-batching it
//! uniformly (`write_raw_signal_table`, in `utils::table_builders`); `merge`
//! now does the same, reusing that writer rather than duplicating it.
//!
//! Signal bytes are still copied, not decompressed/recompressed: each row is
//! one independently VBZ-compressed chunk, so only the Arrow batch grouping
//! (which rows land in which batch) is rebuilt — the compressed bytes
//! themselves are borrowed straight out of each source's mmap.

use crate::arrow_ipc::ArrowIpcFooter;
use crate::error::Result;
use crate::reader::Reader;
use crate::types::{POD5_SIGNATURE, ReadData, RunInfoData, SECTION_MARKER_LENGTH, Uuid};
use crate::utils::pod5_assembler::{
    ProcessedRead, deduplicate_run_infos, write_post_signal_sections,
};
use crate::utils::table_builders::{
    SchemaMetadata, SignalRow, build_reads_table, write_raw_signal_table,
};
use crate::writer::atomic::{AtomicFile, Durability};
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::{BufWriter, Write};
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
    /// Number of signal chunks per batch in the output file's signal table.
    pub signal_batch_size: u32,
    /// How hard to push bytes to stable storage before renaming into place.
    pub durability: Durability,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            duplicate_ok: false,
            read_batch_size: 1_000,
            // Matches `FilterOptions`'s default — the two now share a writer.
            signal_batch_size: 1_000,
            durability: Durability::default(),
        }
    }
}

/// Phase of the merge operation for progress reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergePhase {
    /// Loading metadata from input files (parallel).
    LoadingMetadata,
    /// Extracting and flattening signal chunks, file by file.
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
        return Err(crate::error::Error::InvalidState(
            "No input files specified".into(),
        ));
    }

    merge_impl(inputs, output, options, progress_callback)
}

/// Collected metadata from a single file for merging.
///
/// Holds the live `Reader` so that the signal footer/chunk slices into its
/// mmap stay valid for the duration of the signal-flattening pass. Dropping
/// the reader would invalidate the mmap and force us to copy the
/// (potentially 20+ GB) signal bytes into owned `Vec<u8>` on the heap, which
/// doesn't fit when merging production-scale runs.
struct FileMetadata {
    reader: Reader,
    footer: ArrowIpcFooter,
    run_infos: Vec<RunInfoData>,
    reads: Vec<ReadData>,
}

/// A read carried from its source file through to the merged output: its
/// original UUID (for cross-file duplicate detection), the `ReadData` with
/// its `run_info_index` already remapped onto the deduplicated run-info
/// list, and its **original, file-local** signal row indices (not yet
/// renumbered — that happens only for reads that survive dedup, during the
/// flattening pass below).
type RemappedRead = (Uuid, ReadData, Vec<u64>);

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

    // Phase 1: Open files and collect metadata in parallel. We keep each
    // `Reader` alive (and therefore its mmap) so the next phase can borrow
    // signal bytes directly from the mmap rather than copying them into heap
    // Vecs.
    let metadata_results: Vec<Result<FileMetadata>> = input_paths
        .par_iter()
        .map(|path| {
            let reader = Reader::open(path)?;
            let signal_bytes = reader.signal_table_bytes()?;
            let footer = reader.signal_footer_for_bulk(signal_bytes)?.into_owned();
            let run_infos = reader.run_infos().to_vec();
            // collect_all_reads resolves columns once per batch (see
            // ReadsBatchView), versus reads()'s per-row resolution. This
            // is the merge metadata-load hot path.
            let reads: Vec<ReadData> = reader.collect_all_reads()?;

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

    // Borrow each file's run_infos for dedup — no clone of the (heavy)
    // RunInfoData entries. The deduped Vec is reused for the writer below.
    let per_file_run_infos: Vec<&[RunInfoData]> = file_metadata
        .iter()
        .map(|m| m.run_infos.as_slice())
        .collect();
    let (all_run_infos, run_info_map) = deduplicate_run_infos(&per_file_run_infos);

    // Transform reads in parallel (run_info remapping only — signal rows
    // stay file-local until the flattening pass below knows which reads
    // survive dedup).
    let per_file_reads: Vec<Vec<RemappedRead>> = file_metadata
        .par_iter()
        .map(|metadata| {
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
                    let new_read = read.for_writing(new_run_info_idx);
                    (read.read_id, new_read, read.signal_rows.clone())
                })
                .collect()
        })
        .collect();

    // Phase 2: filter duplicates and flatten surviving reads' compressed
    // signal chunks into one list, assigning fresh global row indices from
    // the flattened order rather than from each file's original numbering.
    // One `extract_signal_rows` call per file (not per read) so the lookup
    // amortizes its batch-grouping work across every row it is asked for,
    // matching `filter`'s approach.
    let mut seen_reads: HashSet<Uuid> = if options.duplicate_ok {
        HashSet::new()
    } else {
        HashSet::with_capacity(total_read_count as usize)
    };

    let mut processed_reads: Vec<ProcessedRead> = Vec::with_capacity(total_read_count as usize);
    let mut signal_chunks: Vec<SignalRow<'_>> = Vec::new();
    let mut duplicate_count = 0u64;
    let mut signal_row_cursor: u64 = 0;

    for (file_idx, file_reads) in per_file_reads.into_iter().enumerate() {
        let metadata = &file_metadata[file_idx];

        let mut survivors: Vec<(ReadData, Vec<u64>)> = Vec::with_capacity(file_reads.len());
        for (read_id, new_read, original_signal_rows) in file_reads {
            if !options.duplicate_ok {
                if seen_reads.contains(&read_id) {
                    duplicate_count += 1;
                    continue;
                }
                seen_reads.insert(read_id);
            }
            survivors.push((new_read, original_signal_rows));
        }

        if !survivors.is_empty() {
            let signal_bytes = metadata.reader.signal_table_bytes()?;
            let flat_row_indices: Vec<u64> = survivors
                .iter()
                .flat_map(|(_, rows)| rows.iter().copied())
                .collect();
            let raw_chunks = metadata
                .footer
                .extract_signal_rows(&flat_row_indices, signal_bytes)?;

            let mut chunk_iter = raw_chunks.into_iter();
            for (new_read, original_rows) in survivors {
                let n = original_rows.len();
                let new_signal_rows: Vec<u64> =
                    (signal_row_cursor..signal_row_cursor + n as u64).collect();
                signal_row_cursor += n as u64;

                for chunk in chunk_iter.by_ref().take(n) {
                    signal_chunks.push(SignalRow {
                        read_id: chunk.read_id,
                        data: chunk.signal,
                        samples: chunk.samples,
                    });
                }

                processed_reads.push((new_read, new_signal_rows));
            }
        }

        if let Some(cb) = progress_callback {
            cb(MergeProgress {
                phase: MergePhase::WritingSignal,
                current: file_idx + 1,
                total: num_files,
            });
        }
    }

    let total_reads = processed_reads.len() as u64;
    let signal_rows_written = signal_row_cursor;

    // Phase 3: write the output file — header, the freshly uniform-stride
    // signal table, then run info / reads / footer.

    let schema_meta = SchemaMetadata::new();
    let section_marker = Uuid::new_v4();

    // Stage the output alongside its destination. Any `?` between here and
    // the commit at the end drops this guard, which unlinks the partial file
    // on unwind and leaves an existing destination untouched. This is also
    // what makes an in-place merge safe: the inputs stay mapped on their
    // original inode and only a directory entry is swapped at the end.
    let atomic = AtomicFile::with_durability(output.as_ref(), options.durability)?;
    let mut file = BufWriter::with_capacity(MERGE_WRITE_BUFFER_SIZE, atomic.reopen()?);

    file.write_all(&POD5_SIGNATURE)?;
    file.write_all(section_marker.as_bytes())?;

    let signal_table_bytes_written = write_raw_signal_table(
        &mut file,
        &signal_chunks,
        options.signal_batch_size,
        &schema_meta,
    )?;
    let signal_end = POD5_SIGNATURE.len() + SECTION_MARKER_LENGTH + signal_table_bytes_written;

    if let Some(cb) = progress_callback {
        cb(MergeProgress {
            phase: MergePhase::WritingReads,
            current: 0,
            total: total_reads as usize,
        });
    }

    let reads_table_bytes = build_reads_table(
        &processed_reads,
        &all_run_infos,
        &schema_meta,
        options.read_batch_size as usize,
    )?;
    write_post_signal_sections(
        &mut file,
        &section_marker,
        &schema_meta,
        signal_end,
        &all_run_infos,
        &reads_table_bytes,
    )?;

    // Everything is written; release our handle and move the file into place.
    file.flush()?;
    drop(file);
    atomic.commit()?;

    Ok(MergeResult {
        reads_written: total_reads,
        duplicates_skipped: duplicate_count,
        signal_rows: signal_rows_written,
        files_processed: file_metadata.len(),
    })
}

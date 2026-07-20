//! Repack operation for POD5 files.
//!
//! Repacks POD5 files using block-level signal copying (no decompression/recompression).

use crate::{Durability, Reader, ReadsBatchView, Result, Writer, WriterOptions};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Options for the repack operation.
#[derive(Debug, Clone)]
pub struct RepackOptions {
    /// Signal chunks per batch.
    pub signal_batch_size: u32,
    /// Reads per batch.
    pub read_batch_size: u32,
    /// Force overwrite existing files.
    pub force: bool,
    /// How hard to push bytes to stable storage before renaming into place.
    pub durability: Durability,
}

impl Default for RepackOptions {
    fn default() -> Self {
        Self {
            signal_batch_size: 1_000,
            read_batch_size: 10_000,
            force: false,
            durability: Durability::default(),
        }
    }
}

/// Result of a repack operation.
#[derive(Debug, Clone, Default)]
pub struct RepackResult {
    /// Total number of reads repacked.
    pub total_reads: u64,
    /// Number of files processed.
    pub files_processed: usize,
    /// Number of files skipped because the output already existed and
    /// `force` was not set. These were never written.
    pub files_skipped: usize,
    /// Files that failed partway, as (input path, error message). Distinct
    /// from `files_skipped`: these were attempted and did not produce output.
    pub failures: Vec<(PathBuf, String)>,
}

use crate::progress::{Progress, ProgressCallback};

/// Repack a single POD5 file using block-level signal copying.
///
/// This is much faster than decompressing and recompressing signal data.
fn repack_single_file(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &RepackOptions,
) -> Result<u64> {
    let input = input.as_ref();
    let output = output.as_ref();

    // Repacking in place needs no special case: the writer stages into a temp
    // file and renames, and the reader's mmap keeps the original inode alive
    // until it is dropped below.
    let reader = Reader::open(input)?;

    let writer_options = WriterOptions {
        signal_batch_size: options.signal_batch_size,
        read_batch_size: options.read_batch_size,
        durability: options.durability,
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(output, writer_options)?;

    // Copy run infos
    for run_info in reader.run_infos() {
        writer.add_run_info(run_info.clone())?;
    }

    let mut count = 0u64;

    // Copy reads with COMPRESSED signals (no decompression/recompression).
    // Iterate by Arrow batch and resolve columns once per batch via
    // ReadsBatchView — much faster than reader.reads()'s per-row column
    // lookup loop.
    for batch_result in reader.read_batches()? {
        let batch = batch_result?;
        let view = ReadsBatchView::new(&batch, false)?;
        for row in 0..view.num_rows() {
            let read = view.read(row)?;

            // Get compressed signal blocks directly (no decompression)
            let compressed_signal = reader.get_compressed_signal_for_rows(&read.signal_rows)?;

            // Write with compressed signal (no recompression)
            let new_read = read.for_writing_same_run();
            writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;

            count += 1;
        }
    }

    // Release the mapping before the rename. Harmless on Unix, where the old
    // inode outlives the directory entry, but Windows refuses to replace a
    // file that is still mapped.
    drop(reader);
    writer.finish()?;

    Ok(count)
}

/// Repack multiple POD5 files in parallel using rayon.
///
/// Each file is repacked independently using block-level signal copying
/// (no decompression/recompression). Files are processed in parallel.
///
/// # Arguments
///
/// * `inputs` - Slice of (input_path, output_path) tuples
/// * `options` - Repack options
/// * `progress` - Optional callback for progress reporting (files_done, total_files)
///
/// # Returns
///
/// A `RepackResult` with statistics about the operation.
pub fn repack_files<P: AsRef<Path> + Sync, Q: AsRef<Path> + Sync>(
    file_pairs: &[(P, Q)],
    options: RepackOptions,
    progress: Option<ProgressCallback>,
) -> RepackResult {
    let total_files = file_pairs.len();
    let total_reads = Arc::new(AtomicU64::new(0));
    let files_done = Arc::new(AtomicU64::new(0));
    let files_skipped = Arc::new(AtomicU64::new(0));
    let failures = Mutex::new(Vec::new());

    file_pairs.par_iter().for_each(|(input, output)| {
        let output_path = output.as_ref();

        // Skip if output exists and force is not set
        if output_path.exists() && !options.force {
            files_skipped.fetch_add(1, Ordering::Relaxed);
            if let Some(ref cb) = progress {
                let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
                cb(Progress {
                    current: done,
                    total: total_files as u64,
                });
            }
            return;
        }

        match repack_single_file(input, output, &options) {
            Ok(reads) => {
                total_reads.fetch_add(reads, Ordering::Relaxed);
            }
            Err(e) => {
                // A file that failed mid-write is not the same as one we chose
                // to skip, and the caller can't act on a bare count.
                if let Ok(mut f) = failures.lock() {
                    f.push((input.as_ref().to_path_buf(), e.to_string()));
                }
            }
        }

        if let Some(ref cb) = progress {
            let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
            cb(Progress {
                current: done,
                total: total_files as u64,
            });
        }
    });

    let skipped = files_skipped.load(Ordering::Relaxed) as usize;
    let mut failures = failures.into_inner().unwrap_or_default();
    failures.sort_by(|a, b| a.0.cmp(&b.0));

    RepackResult {
        total_reads: total_reads.load(Ordering::Relaxed),
        files_processed: total_files - skipped - failures.len(),
        files_skipped: skipped,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repack_options_default() {
        let options = RepackOptions::default();
        assert_eq!(options.signal_batch_size, 1_000);
        assert_eq!(options.read_batch_size, 10_000);
        assert!(!options.force);
    }
}

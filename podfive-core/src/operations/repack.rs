//! Repack operation for POD5 files.
//!
//! Repacks POD5 files using block-level signal copying (no decompression/recompression).

use crate::{Reader, Result, Writer, WriterOptions};
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tempfile::NamedTempFile;

/// Options for the repack operation.
#[derive(Debug, Clone)]
pub struct RepackOptions {
    /// Signal chunks per batch.
    pub signal_batch_size: u32,
    /// Reads per batch.
    pub read_batch_size: u32,
    /// Force overwrite existing files.
    pub force: bool,
}

impl Default for RepackOptions {
    fn default() -> Self {
        Self {
            signal_batch_size: 1_000,
            read_batch_size: 10_000,
            force: false,
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
    /// Number of files skipped (errors).
    pub files_skipped: usize,
}

/// Callback for reporting progress during repacking.
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

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

    // Check if input and output resolve to the same file
    let input_canonical = std::fs::canonicalize(input)?;
    let same_file = output.exists()
        && std::fs::canonicalize(output)
            .map(|o| o == input_canonical)
            .unwrap_or(false);

    // Use a temp file if writing to the same location as input
    let (actual_output, temp_file): (std::path::PathBuf, Option<NamedTempFile>) = if same_file {
        let temp = NamedTempFile::new_in(output.parent().unwrap_or(std::path::Path::new(".")))?;
        (temp.path().to_path_buf(), Some(temp))
    } else {
        (output.to_path_buf(), None)
    };

    let reader = Reader::open(input)?;

    let writer_options = WriterOptions {
        signal_batch_size: options.signal_batch_size,
        read_batch_size: options.read_batch_size,
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&actual_output, writer_options)?;

    // Copy run infos
    for run_info in reader.run_infos() {
        writer.add_run_info(run_info.clone())?;
    }

    let mut count = 0u64;

    // Copy reads with COMPRESSED signals (no decompression/recompression)
    for read_result in reader.reads()? {
        let read = read_result?;

        // Get compressed signal blocks directly (no decompression)
        let compressed_signal = reader.get_compressed_signal_for_rows(&read.signal_rows)?;

        // Write with compressed signal (no recompression)
        let new_read = read.for_writing_same_run();
        writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;

        count += 1;
    }

    writer.finish()?;

    // If we used a temp file, rename it to the output
    if let Some(temp) = temp_file {
        drop(reader); // Release the memory map
        temp.persist(output).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    }

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

    file_pairs.par_iter().for_each(|(input, output)| {
        let output_path = output.as_ref();

        // Skip if output exists and force is not set
        if output_path.exists() && !options.force {
            files_skipped.fetch_add(1, Ordering::Relaxed);
            if let Some(ref cb) = progress {
                let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
                cb(done as usize, total_files);
            }
            return;
        }

        match repack_single_file(input, output, &options) {
            Ok(reads) => {
                total_reads.fetch_add(reads, Ordering::Relaxed);
            }
            Err(_) => {
                files_skipped.fetch_add(1, Ordering::Relaxed);
            }
        }

        if let Some(ref cb) = progress {
            let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
            cb(done as usize, total_files);
        }
    });

    let skipped = files_skipped.load(Ordering::Relaxed) as usize;

    RepackResult {
        total_reads: total_reads.load(Ordering::Relaxed),
        files_processed: total_files - skipped,
        files_skipped: skipped,
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

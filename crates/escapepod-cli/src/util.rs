//! Utility functions for the CLI.

use escapepod_signal::Reader;
use noodles_bam as bam;
#[cfg(feature = "experimental")]
use noodles_csi::BinningIndex;
#[cfg(feature = "experimental")]
use noodles_csi::binning_index::ReferenceSequence as _;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use tracing::{info, warn};

use crate::style;

/// Resolve input path to a list of POD5 files.
///
/// - If path is a file, return it as a single-element vector
/// - If path is a directory, find all *.pod5 files recursively
pub fn resolve_pod5_inputs(path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if path.is_dir() {
        let mut files = Vec::new();
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let p = entry.path();
            if p.is_file() && p.extension().is_some_and(|e| e == "pod5") {
                files.push(p.to_path_buf());
            }
        }

        if files.is_empty() {
            anyhow::bail!("No POD5 files found in directory: {}", path.display());
        }

        files.sort();
        return Ok(files);
    }

    // Unquoted shell glob that didn't expand leaks through as a literal path
    // (e.g. `*.pod5` when nullglob is off). Detect and explain.
    if path_looks_like_glob(path) {
        anyhow::bail!(
            "No files matched the pattern: {}. \
             If the pattern is quoted or your shell didn't expand it, \
             check the directory contents or pass a directory instead.",
            path.display()
        );
    }

    anyhow::bail!("Path does not exist: {}", path.display())
}

/// Heuristic: does this path look like an unexpanded shell glob?
fn path_looks_like_glob(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.contains('*') || s.contains('?') || s.contains('['))
        .unwrap_or(false)
}

/// Validate that we can write to `output` before running a long operation.
///
/// Fails if the parent directory is missing, or if the file already exists
/// and `force` is false. Call this before the expensive work so users don't
/// have to wait for a merge/filter to finish just to hit a write error.
pub fn check_output_writable(output: &Path, force: bool) -> anyhow::Result<()> {
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        anyhow::bail!("Output directory does not exist: {}", parent.display());
    }
    if output.exists() && !force {
        anyhow::bail!(
            "Output file {} already exists. Use --force to overwrite.",
            output.display()
        );
    }
    Ok(())
}

/// Stable identity for a path that may not exist yet.
///
/// An output path usually has no file to canonicalize, so resolve its parent
/// and re-attach the filename. Returns `None` when even the parent is
/// unresolvable, in which case the caller should not claim a collision.
fn resolve_for_compare(path: &Path) -> Option<PathBuf> {
    if let Ok(resolved) = std::fs::canonicalize(path) {
        return Some(resolved);
    }
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let name = path.file_name()?;
    std::fs::canonicalize(parent).ok().map(|p| p.join(name))
}

/// Reject an output path that is also one of the inputs.
///
/// Writing over an input is survivable — output is staged and renamed, so the
/// inputs stay readable on their original inode for the whole run — but it
/// consumes a source file the user probably meant to keep, and it is almost
/// always a typo or an over-broad glob rather than an intent.
pub fn check_output_not_input(output: &Path, inputs: &[PathBuf]) -> anyhow::Result<()> {
    let Some(resolved_output) = resolve_for_compare(output) else {
        return Ok(());
    };
    for input in inputs {
        if resolve_for_compare(input).as_deref() == Some(resolved_output.as_path()) {
            anyhow::bail!(
                "Output {} is also an input. Choose a different output path \
                 (or drop it from the inputs) so the source file is not replaced.",
                output.display()
            );
        }
    }
    Ok(())
}

/// Resolve multiple input paths to a flat list of POD5 files.
///
/// Each input can be a file or directory. Directories are expanded recursively.
/// Returns an error if no POD5 files are found.
pub fn collect_pod5_inputs(inputs: &[PathBuf]) -> anyhow::Result<Vec<PathBuf>> {
    if inputs.is_empty() {
        anyhow::bail!("No input files specified");
    }

    let mut all_files = Vec::new();
    for input in inputs {
        all_files.extend(resolve_pod5_inputs(input)?);
    }

    if all_files.is_empty() {
        anyhow::bail!("No POD5 files found in specified inputs");
    }

    Ok(all_files)
}

/// Format a byte count as a human-readable string (e.g., "1.2 GB").
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format a number with thousands separators.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();

    // Use rchunks to group digits from the right
    bytes
        .rchunks(3)
        .rev()
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect::<Vec<_>>()
        .join(",")
}

/// Format duration in hours from sample count and sample rate.
pub fn format_duration_hours(samples: u64, sample_rate: u16) -> String {
    if sample_rate == 0 {
        return "N/A".to_string();
    }
    let seconds = samples as f64 / sample_rate as f64;
    let hours = seconds / 3600.0;
    format!("{:.1} hrs", hours)
}

/// Result of opening a POD5 file, with handling for directory mode.
pub enum OpenResult<T> {
    /// Successfully opened.
    Ok(T),
    /// Failed but should continue (directory mode).
    Skip,
    /// Failed and should abort.
    Err(anyhow::Error),
}

/// Open a POD5 file with appropriate error handling for directory mode.
///
/// In directory mode, file open errors result in a warning and `Skip`.
/// In single-file mode, errors are propagated.
pub fn open_reader_with_warning(file_path: &PathBuf, is_directory: bool) -> OpenResult<Reader> {
    match Reader::open(file_path) {
        Ok(r) => OpenResult::Ok(r),
        Err(e) => {
            if is_directory {
                warn!(
                    "skipping {} ({})",
                    file_path.file_name().unwrap_or_default().to_string_lossy(),
                    e
                );
                OpenResult::Skip
            } else {
                OpenResult::Err(e.into())
            }
        }
    }
}

/// Ensure a BAI index exists for the given BAM file, creating one if necessary.
///
/// Returns the path to the BAI file (either existing or newly created).
pub fn ensure_bai_index(bam_path: &Path) -> anyhow::Result<PathBuf> {
    // noodles expects the index at path.bam.bai
    let bai_path = bam_path.with_extension("bam.bai");

    if bai_path.exists() {
        return Ok(bai_path);
    }

    // Also check for path.bai (alternative naming convention)
    let alt_bai_path = bam_path.with_extension("bai");
    if alt_bai_path.exists() {
        info!(
            "found index at {} but noodles expects {}",
            style::path(alt_bai_path.display()),
            style::path(bai_path.display())
        );
    }

    info!(
        "BAI index not found, creating {}...",
        style::path(bai_path.display())
    );

    // Build the index from the BAM file
    let index = bam::fs::index(bam_path)?;

    // Write the index to file
    bam::bai::fs::write(&bai_path, &index)?;

    info!("created BAI index: {}", style::path(bai_path.display()));

    Ok(bai_path)
}

/// Count total records in a BAM file using its BAI index.
///
/// Creates the BAI index if it does not already exist.
/// Returns the total number of records (mapped + unmapped).
#[cfg(feature = "experimental")]
pub fn count_bam_records(bam_path: &Path) -> anyhow::Result<u64> {
    let bai_path = ensure_bai_index(bam_path)?;
    let index = bam::bai::fs::read(&bai_path)?;

    let mut total: u64 = 0;
    for ref_seq in index.reference_sequences() {
        if let Some(metadata) = ref_seq.metadata() {
            total += metadata.mapped_record_count();
            total += metadata.unmapped_record_count();
        }
    }
    if let Some(count) = index.unplaced_unmapped_record_count() {
        total += count;
    }

    Ok(total)
}

/// Get a reads iterator with appropriate error handling for directory mode.
pub fn get_reads_iter_with_warning<'a>(
    reader: &'a Reader,
    file_path: &Path,
    is_directory: bool,
) -> OpenResult<impl Iterator<Item = escapepod_signal::Result<escapepod_signal::ReadData>> + 'a> {
    match reader.reads() {
        Ok(iter) => OpenResult::Ok(iter),
        Err(e) => {
            if is_directory {
                warn!(
                    "cannot read {} ({})",
                    file_path.file_name().unwrap_or_default().to_string_lossy(),
                    e
                );
                OpenResult::Skip
            } else {
                OpenResult::Err(e.into())
            }
        }
    }
}

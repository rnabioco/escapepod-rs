//! Utility functions for the CLI.

use escapepod::Reader;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

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

    anyhow::bail!("Path does not exist: {}", path.display())
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
                eprintln!(
                    "Warning: skipping {} ({})",
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

/// Get a reads iterator with appropriate error handling for directory mode.
pub fn get_reads_iter_with_warning<'a>(
    reader: &'a Reader,
    file_path: &Path,
    is_directory: bool,
) -> OpenResult<impl Iterator<Item = escapepod::Result<escapepod::ReadData>> + 'a> {
    match reader.reads() {
        Ok(iter) => OpenResult::Ok(iter),
        Err(e) => {
            if is_directory {
                eprintln!(
                    "Warning: cannot read {} ({})",
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

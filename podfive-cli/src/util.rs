//! Utility functions for the CLI.

use podfive_core::{Reader, Writer};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use uuid::Uuid;
use walkdir::WalkDir;

/// Default batch sizes for write operations.
pub mod batch_sizes {
    /// Signal chunks per batch for filter operations.
    pub const SIGNAL_BATCH_SIZE: u32 = 1_000;
    /// Reads per batch for filter operations.
    pub const READ_BATCH_SIZE: u32 = 10_000;
    /// Reads per batch for merge operations (large to avoid dictionary issues).
    pub const MERGE_READ_BATCH_SIZE: u32 = 1_000_000;
}

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

/// Parse a UUID from various formats.
///
/// Supports:
/// - Standard format with dashes: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
/// - Compact format without dashes: `a1b2c3d4e5f67890abcdef1234567890`
pub fn parse_uuid_flexible(s: &str) -> anyhow::Result<Uuid> {
    // Try standard format first
    if let Ok(uuid) = Uuid::parse_str(s) {
        return Ok(uuid);
    }

    // Try without dashes (32 hex characters)
    if s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        let with_dashes = format!(
            "{}-{}-{}-{}-{}",
            &s[0..8],
            &s[8..12],
            &s[12..16],
            &s[16..20],
            &s[20..32]
        );
        return Uuid::parse_str(&with_dashes).map_err(|e| anyhow::anyhow!("Invalid UUID: {}", e));
    }

    anyhow::bail!("Invalid UUID format: '{}'", s)
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
) -> OpenResult<impl Iterator<Item = podfive_core::Result<podfive_core::ReadData>> + 'a> {
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

/// Add run infos from a reader to a writer, deduplicating by acquisition_id.
///
/// Returns the mapping from acquisition_id to writer index.
pub fn add_run_infos_deduplicated(
    reader: &Reader,
    writer: &mut Writer,
    run_info_map: &mut HashMap<String, u32>,
) -> anyhow::Result<()> {
    for run_info in reader.run_infos() {
        if !run_info_map.contains_key(&run_info.acquisition_id) {
            let idx = writer.add_run_info(run_info.clone())?;
            run_info_map.insert(run_info.acquisition_id.clone(), idx);
        }
    }
    Ok(())
}

/// Get the new run_info index for a read, using the run_info_map.
pub fn map_run_info_index(
    reader: &Reader,
    read_run_info_index: u32,
    run_info_map: &HashMap<String, u32>,
) -> u32 {
    if let Some(original_run_info) = reader.get_run_info(read_run_info_index as usize) {
        *run_info_map
            .get(&original_run_info.acquisition_id)
            .unwrap_or(&0)
    } else {
        0
    }
}

/// Collected dictionary values from scanning POD5 files.
#[derive(Debug, Default)]
pub struct ScannedDictionaries {
    pub pore_types: BTreeSet<String>,
    pub end_reasons: BTreeSet<String>,
    pub total_read_count: u64,
}

/// Scan POD5 files to collect dictionary values for reads matching a filter.
///
/// If `filter_ids` is Some, only collect values for reads whose IDs are in the set.
/// If `filter_ids` is None, collect values for all reads.
pub fn scan_dictionary_values(
    files: &[PathBuf],
    filter_ids: Option<&HashSet<Uuid>>,
) -> ScannedDictionaries {
    let mut result = ScannedDictionaries::default();

    for file_path in files {
        if let Ok(reader) = Reader::open(file_path) {
            if let Ok(reads_iter) = reader.reads() {
                for read in reads_iter.flatten() {
                    result.total_read_count += 1;
                    // Only collect values for matching reads (or all if no filter)
                    let should_collect = filter_ids
                        .map(|ids| ids.contains(&read.read_id))
                        .unwrap_or(true);
                    if should_collect {
                        result.pore_types.insert(read.pore_type.clone());
                        result.end_reasons.insert(read.end_reason.to_string());
                    }
                }
            }
        }
    }

    result
}

/// A reporter that limits the number of warnings emitted.
#[derive(Debug)]
pub struct LimitedWarningReporter {
    limit: u64,
    count: u64,
}

impl LimitedWarningReporter {
    /// Create a new reporter with the given limit.
    pub fn new(limit: u64) -> Self {
        Self { limit, count: 0 }
    }

    /// Report a warning, returning true if it was emitted.
    pub fn warn(&mut self, message: &str) -> bool {
        self.count += 1;
        if self.count <= self.limit {
            eprintln!("Warning: {}", message);
            true
        } else {
            false
        }
    }

    /// Get the total count of warnings (including suppressed).
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Check if any warnings were suppressed.
    #[allow(dead_code)]
    pub fn has_suppressed(&self) -> bool {
        self.count > self.limit
    }
}

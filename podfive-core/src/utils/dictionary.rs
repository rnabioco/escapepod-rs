//! Dictionary value scanning utilities.

use crate::Reader;
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use uuid::Uuid;

/// Collected dictionary values from scanning POD5 files.
#[derive(Debug, Default, Clone)]
pub struct ScannedDictionaries {
    /// Unique pore types found across all reads.
    pub pore_types: BTreeSet<String>,
    /// Unique end reasons found across all reads.
    pub end_reasons: BTreeSet<String>,
    /// Total number of reads scanned.
    pub total_read_count: u64,
}

/// Scan POD5 files to collect dictionary values for reads matching a filter.
///
/// This function scans multiple POD5 files and collects unique pore types and
/// end reasons. This is useful for pre-batching operations where you need to
/// know all dictionary values before writing.
///
/// # Arguments
///
/// * `files` - Slice of POD5 file paths to scan
/// * `filter_ids` - If `Some`, only collect values for reads whose IDs are in the set.
///   If `None`, collect values for all reads.
///
/// # Example
///
/// ```no_run
/// use podfive_core::utils::scan_dictionary_values;
/// use std::path::PathBuf;
///
/// let files = vec![PathBuf::from("input.pod5")];
/// let dictionaries = scan_dictionary_values(&files, None);
///
/// println!("Found {} pore types", dictionaries.pore_types.len());
/// println!("Found {} end reasons", dictionaries.end_reasons.len());
/// ```
pub fn scan_dictionary_values<P: AsRef<Path>>(
    files: &[P],
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

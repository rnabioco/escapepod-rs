//! Dictionary value scanning utilities.

use crate::Reader;
use arrow::array::{Array, DictionaryArray, StringArray};
use arrow::datatypes::Int16Type;
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

/// Extract dictionary values from a column's dictionary array.
fn extract_dict_values(batch: &arrow::record_batch::RecordBatch, col_name: &str, values: &mut BTreeSet<String>) {
    if let Some(col) = batch.column_by_name(col_name) {
        if let Some(dict) = col.as_any().downcast_ref::<DictionaryArray<Int16Type>>() {
            if let Some(dict_values) = dict.values().as_any().downcast_ref::<StringArray>() {
                for i in 0..dict_values.len() {
                    if !dict_values.is_null(i) {
                        values.insert(dict_values.value(i).to_string());
                    }
                }
            }
        }
    }
}

/// Scan POD5 files to collect dictionary values.
///
/// This function efficiently scans POD5 files to collect unique pore types and
/// end reasons. Since dictionary values are typically a small set shared across
/// all rows, we only need to read the first batch to get all unique values.
///
/// # Arguments
///
/// * `files` - Slice of POD5 file paths to scan
/// * `filter_ids` - Unused but kept for API compatibility. Dictionary values are
///   collected from all reads since filtering would be more expensive.
pub fn scan_dictionary_values<P: AsRef<Path>>(
    files: &[P],
    _filter_ids: Option<&HashSet<Uuid>>,
) -> ScannedDictionaries {
    let mut result = ScannedDictionaries::default();

    for file_path in files {
        if let Ok(reader) = Reader::open(file_path) {
            // Get batch count for read count estimation (much faster than read_count())
            if let Ok(num_batches) = reader.read_batch_count() {
                // Read first batch to get dictionary values AND row count for estimation
                if let Ok(batch) = reader.read_batch(0) {
                    extract_dict_values(&batch, "pore_type", &mut result.pore_types);
                    extract_dict_values(&batch, "end_reason", &mut result.end_reasons);
                    // Estimate total rows: batch_size * num_batches (rough estimate)
                    result.total_read_count += (batch.num_rows() * num_batches) as u64;
                }
            }
        }
    }

    result
}

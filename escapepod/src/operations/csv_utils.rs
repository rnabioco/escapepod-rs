//! Shared CSV parsing utilities for UUID-keyed mappings.

use crate::utils::parse_uuid_flexible;
use crate::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use uuid::Uuid;

/// Parse a CSV file mapping read IDs to a string value column.
///
/// The CSV must have headers including `read_id` and a column named `value_column`.
/// UUIDs can be in standard format (with dashes) or compact format (32 hex chars).
///
/// When `skip_empty_values` is true, rows where both the read_id and value are empty
/// are silently skipped. When false, only rows with empty read_id are skipped (empty
/// values are kept).
pub fn parse_csv_uuid_mapping(
    csv_path: impl AsRef<Path>,
    value_column: &str,
    skip_empty_values: bool,
) -> Result<HashMap<Uuid, String>> {
    let file = File::open(csv_path.as_ref())?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(reader);

    let mut mapping = HashMap::new();

    // Check headers
    let headers = csv_reader
        .headers()
        .map_err(|e| Error::Parse(format!("CSV header error: {}", e)))?
        .clone();

    let read_id_col = headers
        .iter()
        .position(|h| h == "read_id")
        .ok_or_else(|| Error::Parse("CSV must have a 'read_id' column".to_string()))?;

    let value_col = headers
        .iter()
        .position(|h| h == value_column)
        .ok_or_else(|| Error::Parse(format!("CSV must have a '{}' column", value_column)))?;

    for (line_num, result) in csv_reader.records().enumerate() {
        let record = result
            .map_err(|e| Error::Parse(format!("CSV error on line {}: {}", line_num + 2, e)))?;

        let read_id_str = record
            .get(read_id_col)
            .ok_or_else(|| Error::Parse(format!("Missing read_id on line {}", line_num + 2)))?;

        let value = record.get(value_col).ok_or_else(|| {
            Error::Parse(format!("Missing {} on line {}", value_column, line_num + 2))
        })?;

        if read_id_str.is_empty() {
            continue;
        }
        if skip_empty_values && value.is_empty() {
            continue;
        }

        // Parse UUID (handle both with and without dashes)
        let uuid = parse_uuid_flexible(read_id_str).map_err(|_| {
            Error::Parse(format!(
                "Invalid UUID '{}' on line {}",
                read_id_str,
                line_num + 2
            ))
        })?;

        mapping.insert(uuid, value.to_string());
    }

    Ok(mapping)
}

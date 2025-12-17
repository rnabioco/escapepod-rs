//! Field definitions and extraction for ReadData.
//!
//! This module provides constants and utilities for working with read fields
//! in a configurable way, useful for tabular output and field selection.

use crate::ReadData;

/// All available fields for output.
pub const ALL_FIELDS: &[&str] = &[
    "read_id",
    "channel",
    "well",
    "pore_type",
    "read_number",
    "start_sample",
    "median_before",
    "end_reason",
    "end_reason_forced",
    "num_samples",
    "num_minknow_events",
    "calibration_offset",
    "calibration_scale",
    "run_info",
    "open_pore_level",
];

/// Default fields when no include/exclude specified.
pub const DEFAULT_FIELDS: &[&str] = &[
    "read_id",
    "channel",
    "well",
    "read_number",
    "start_sample",
    "num_samples",
    "end_reason",
];

/// Error type for field operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FieldError {
    /// An unknown field was specified.
    #[error("Unknown field '{0}'. Available fields: read_id, channel, well, pore_type, read_number, start_sample, median_before, end_reason, end_reason_forced, num_samples, num_minknow_events, calibration_offset, calibration_scale, run_info, open_pore_level")]
    UnknownField(String),

    /// No fields were selected for output.
    #[error("No fields selected for output")]
    NoFieldsSelected,
}

/// Determine which fields to use based on include/exclude options.
///
/// # Arguments
///
/// * `include` - Comma-separated list of fields to include (uses only these fields)
/// * `exclude` - Comma-separated list of fields to exclude from the base set
/// * `ids_only` - If true, return only the read_id field
///
/// # Returns
///
/// A vector of field names to use, or an error if validation fails.
///
/// # Example
///
/// ```
/// use escapepod::determine_fields;
///
/// // Use defaults
/// let fields = determine_fields(None, None, false).unwrap();
///
/// // Include specific fields
/// let fields = determine_fields(Some("read_id,channel,well"), None, false).unwrap();
/// assert_eq!(fields, vec!["read_id", "channel", "well"]);
///
/// // Exclude fields from defaults
/// let fields = determine_fields(None, Some("channel"), false).unwrap();
/// assert!(!fields.contains(&"channel".to_string()));
/// ```
pub fn determine_fields(
    include: Option<&str>,
    exclude: Option<&str>,
    ids_only: bool,
) -> Result<Vec<String>, FieldError> {
    use std::collections::HashSet;

    if ids_only {
        return Ok(vec!["read_id".to_string()]);
    }

    let all_fields_set: HashSet<&str> = ALL_FIELDS.iter().copied().collect();

    let base_fields: Vec<&str> = if let Some(include_str) = include {
        // Use only specified fields
        let requested: Vec<&str> = include_str.split(',').map(|s| s.trim()).collect();
        for f in &requested {
            if !all_fields_set.contains(*f) {
                return Err(FieldError::UnknownField((*f).to_string()));
            }
        }
        requested
    } else {
        // Start with defaults
        DEFAULT_FIELDS.to_vec()
    };

    let final_fields: Vec<String> = if let Some(exclude_str) = exclude {
        let excluded: HashSet<&str> = exclude_str.split(',').map(|s| s.trim()).collect();
        for f in &excluded {
            if !all_fields_set.contains(*f) {
                return Err(FieldError::UnknownField((*f).to_string()));
            }
        }
        base_fields
            .into_iter()
            .filter(|f| !excluded.contains(*f))
            .map(String::from)
            .collect()
    } else {
        base_fields.into_iter().map(String::from).collect()
    };

    if final_fields.is_empty() {
        return Err(FieldError::NoFieldsSelected);
    }

    Ok(final_fields)
}

/// Get the value of a field from a ReadData struct.
///
/// # Arguments
///
/// * `read` - The ReadData to extract the field from
/// * `field` - The name of the field to extract
///
/// # Returns
///
/// The field value as a formatted string. Unknown fields return an empty string.
///
/// # Example
///
/// ```no_run
/// use escapepod::{Reader, get_field_value};
///
/// let reader = Reader::open("example.pod5")?;
/// for read in reader.reads()?.flatten() {
///     let id = get_field_value(&read, "read_id");
///     let channel = get_field_value(&read, "channel");
///     println!("{}\t{}", id, channel);
/// }
/// # Ok::<(), escapepod::Error>(())
/// ```
pub fn get_field_value(read: &ReadData, field: &str) -> String {
    match field {
        "read_id" => read.read_id.to_string(),
        "channel" => read.channel.to_string(),
        "well" => read.well.to_string(),
        "pore_type" => read.pore_type.clone(),
        "read_number" => read.read_number.to_string(),
        "start_sample" => read.start_sample.to_string(),
        "median_before" => format!("{:.2}", read.median_before),
        "end_reason" => read.end_reason.to_string(),
        "end_reason_forced" => read.end_reason_forced.to_string(),
        "num_samples" => read.num_samples.to_string(),
        "num_minknow_events" => read.num_minknow_events.to_string(),
        "calibration_offset" => format!("{:.4}", read.calibration_offset),
        "calibration_scale" => format!("{:.6}", read.calibration_scale),
        "run_info" => read.run_info_index.to_string(),
        "open_pore_level" => format!("{:.2}", read.open_pore_level),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_fields_default() {
        let fields = determine_fields(None, None, false).unwrap();
        assert_eq!(
            fields,
            DEFAULT_FIELDS
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_determine_fields_ids_only() {
        let fields = determine_fields(None, None, true).unwrap();
        assert_eq!(fields, vec!["read_id".to_string()]);
    }

    #[test]
    fn test_determine_fields_include() {
        let fields = determine_fields(Some("read_id,channel,well"), None, false).unwrap();
        assert_eq!(fields, vec!["read_id", "channel", "well"]);
    }

    #[test]
    fn test_determine_fields_include_with_spaces() {
        let fields = determine_fields(Some("read_id, channel, well"), None, false).unwrap();
        assert_eq!(fields, vec!["read_id", "channel", "well"]);
    }

    #[test]
    fn test_determine_fields_exclude() {
        let fields = determine_fields(None, Some("read_id,channel"), false).unwrap();
        assert!(!fields.contains(&"read_id".to_string()));
        assert!(!fields.contains(&"channel".to_string()));
        assert!(fields.contains(&"well".to_string()));
    }

    #[test]
    fn test_determine_fields_include_and_exclude() {
        let fields = determine_fields(
            Some("read_id,channel,well,num_samples"),
            Some("channel"),
            false,
        )
        .unwrap();
        assert_eq!(fields, vec!["read_id", "well", "num_samples"]);
    }

    #[test]
    fn test_determine_fields_unknown_include() {
        let result = determine_fields(Some("read_id,unknown_field"), None, false);
        assert!(matches!(result, Err(FieldError::UnknownField(_))));
    }

    #[test]
    fn test_determine_fields_unknown_exclude() {
        let result = determine_fields(None, Some("unknown_field"), false);
        assert!(matches!(result, Err(FieldError::UnknownField(_))));
    }

    #[test]
    fn test_determine_fields_all_excluded() {
        let exclude = DEFAULT_FIELDS.join(",");
        let result = determine_fields(None, Some(&exclude), false);
        assert!(matches!(result, Err(FieldError::NoFieldsSelected)));
    }

    #[test]
    fn test_all_fields_recognized() {
        let fields = determine_fields(Some(&ALL_FIELDS.join(",")), None, false).unwrap();
        assert_eq!(fields.len(), ALL_FIELDS.len());
    }
}

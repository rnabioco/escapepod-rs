//! View command implementation.
//!
//! Produces a tabular summary of reads from POD5 files.

use crate::util::resolve_pod5_inputs;
use podfive_core::{ReadData, Reader};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Available fields for output
const ALL_FIELDS: &[&str] = &[
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

/// Default fields when no include/exclude specified
const DEFAULT_FIELDS: &[&str] = &[
    "read_id",
    "channel",
    "well",
    "read_number",
    "start_sample",
    "num_samples",
    "end_reason",
];

pub fn run(
    input: PathBuf,
    include: Option<String>,
    exclude: Option<String>,
    ids_only: bool,
    output: Option<PathBuf>,
    separator: String,
    no_header: bool,
) -> anyhow::Result<()> {
    // Resolve input to list of POD5 files (supports directories)
    let files = resolve_pod5_inputs(&input)?;

    // Determine which fields to output
    let fields = determine_fields(include.as_deref(), exclude.as_deref(), ids_only)?;

    // Set up output writer
    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };

    // Write header (only once for all files)
    if !no_header && !ids_only {
        writeln!(writer, "{}", fields.join(&separator))?;
    }

    // Process all files
    for file_path in &files {
        let reader = Reader::open(file_path)?;

        // Write reads
        for read_result in reader.reads()? {
            let read = read_result?;

            if ids_only {
                writeln!(writer, "{}", read.read_id)?;
            } else {
                let values: Vec<String> = fields.iter().map(|f| get_field_value(&read, f)).collect();
                writeln!(writer, "{}", values.join(&separator))?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn determine_fields(
    include: Option<&str>,
    exclude: Option<&str>,
    ids_only: bool,
) -> anyhow::Result<Vec<String>> {
    if ids_only {
        return Ok(vec!["read_id".to_string()]);
    }

    let all_fields_set: HashSet<&str> = ALL_FIELDS.iter().copied().collect();

    let base_fields: Vec<&str> = if let Some(include_str) = include {
        // Use only specified fields
        let requested: Vec<&str> = include_str.split(',').map(|s| s.trim()).collect();
        for f in &requested {
            if !all_fields_set.contains(*f) {
                anyhow::bail!(
                    "Unknown field '{}'. Available fields: {}",
                    f,
                    ALL_FIELDS.join(", ")
                );
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
                anyhow::bail!(
                    "Unknown field '{}'. Available fields: {}",
                    f,
                    ALL_FIELDS.join(", ")
                );
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
        anyhow::bail!("No fields selected for output");
    }

    Ok(final_fields)
}

fn get_field_value(read: &ReadData, field: &str) -> String {
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
        assert_eq!(fields, DEFAULT_FIELDS.iter().map(|s| s.to_string()).collect::<Vec<_>>());
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
        // Default fields minus read_id and channel
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
        ).unwrap();
        assert_eq!(fields, vec!["read_id", "well", "num_samples"]);
    }

    #[test]
    fn test_determine_fields_unknown_include() {
        let result = determine_fields(Some("read_id,unknown_field"), None, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown field"));
    }

    #[test]
    fn test_determine_fields_unknown_exclude() {
        let result = determine_fields(None, Some("unknown_field"), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown field"));
    }

    #[test]
    fn test_determine_fields_all_excluded() {
        // Exclude all default fields
        let exclude = DEFAULT_FIELDS.join(",");
        let result = determine_fields(None, Some(&exclude), false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No fields selected"));
    }

    #[test]
    fn test_all_fields_recognized() {
        // Test that all declared fields are valid
        let fields = determine_fields(Some(&ALL_FIELDS.join(",")), None, false).unwrap();
        assert_eq!(fields.len(), ALL_FIELDS.len());
    }
}

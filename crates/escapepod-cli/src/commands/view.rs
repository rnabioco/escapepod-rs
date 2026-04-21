//! View command implementation.
//!
//! Produces a tabular summary of reads from POD5 files.

use crate::util::{
    OpenResult, get_reads_iter_with_warning, open_reader_with_warning, resolve_pod5_inputs,
};
use escapepod_signal::{determine_fields, get_field_value};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

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

    // Determine which fields to output (using core library)
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
    let is_directory = files.len() > 1;
    for file_path in &files {
        let reader = match open_reader_with_warning(file_path, is_directory) {
            OpenResult::Ok(r) => r,
            OpenResult::Skip => continue,
            OpenResult::Err(e) => return Err(e),
        };

        // Write reads
        let reads_iter = match get_reads_iter_with_warning(&reader, file_path, is_directory) {
            OpenResult::Ok(iter) => iter,
            OpenResult::Skip => continue,
            OpenResult::Err(e) => return Err(e),
        };

        for read_result in reads_iter {
            let read = match read_result {
                Ok(r) => r,
                Err(_) => continue, // Skip individual read errors silently
            };

            if ids_only {
                writeln!(writer, "{}", read.read_id)?;
            } else {
                let values: Vec<String> =
                    fields.iter().map(|f| get_field_value(&read, f)).collect();
                writeln!(writer, "{}", values.join(&separator))?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use escapepod_signal::{ALL_FIELDS, DEFAULT_FIELDS, determine_fields};

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
        )
        .unwrap();
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
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No fields selected")
        );
    }

    #[test]
    fn test_all_fields_recognized() {
        // Test that all declared fields are valid
        let fields = determine_fields(Some(&ALL_FIELDS.join(",")), None, false).unwrap();
        assert_eq!(fields.len(), ALL_FIELDS.len());
    }
}

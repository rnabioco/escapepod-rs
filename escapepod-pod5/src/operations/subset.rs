//! Subset operation utilities for POD5 files.

use crate::Result;
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

use super::csv_utils::parse_csv_uuid_mapping;

/// Parse a CSV file mapping read IDs to output files.
///
/// The CSV must have headers including `read_id` and `output` columns.
/// UUIDs can be in standard format (with dashes) or compact format (32 hex chars).
///
/// # Arguments
///
/// * `csv_path` - Path to the CSV file
///
/// # Returns
///
/// A HashMap mapping each read UUID to its target output filename.
///
/// # Example
///
/// ```no_run
/// use escapepod_pod5::operations::parse_csv_mapping;
///
/// let mapping = parse_csv_mapping("mapping.csv")?;
/// for (uuid, output) in &mapping {
///     println!("{} -> {}", uuid, output);
/// }
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
///
/// # CSV Format
///
/// ```csv
/// read_id,output
/// a1b2c3d4-e5f6-7890-abcd-ef1234567890,sample1.pod5
/// b2c3d4e5f6a78901bcdef12345678901,sample2.pod5
/// ```
pub fn parse_csv_mapping(csv_path: impl AsRef<Path>) -> Result<HashMap<Uuid, String>> {
    parse_csv_uuid_mapping(csv_path, "output", true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_csv_mapping_valid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,output").unwrap();
        writeln!(
            temp_file,
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890,sample1.pod5"
        )
        .unwrap();
        writeln!(
            temp_file,
            "b2c3d4e5-f6a7-8901-bcde-f12345678901,sample2.pod5"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let mapping = parse_csv_mapping(temp_file.path()).unwrap();
        assert_eq!(mapping.len(), 2);

        let uuid1 = Uuid::parse_str("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        let uuid2 = Uuid::parse_str("b2c3d4e5-f6a7-8901-bcde-f12345678901").unwrap();

        assert_eq!(mapping.get(&uuid1), Some(&"sample1.pod5".to_string()));
        assert_eq!(mapping.get(&uuid2), Some(&"sample2.pod5".to_string()));
    }

    #[test]
    fn test_parse_csv_mapping_no_dashes() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,output").unwrap();
        writeln!(temp_file, "a1b2c3d4e5f67890abcdef1234567890,sample1.pod5").unwrap();
        temp_file.flush().unwrap();

        let mapping = parse_csv_mapping(temp_file.path()).unwrap();
        assert_eq!(mapping.len(), 1);
    }

    #[test]
    fn test_parse_csv_mapping_empty_lines() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,output").unwrap();
        writeln!(
            temp_file,
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890,sample1.pod5"
        )
        .unwrap();
        writeln!(temp_file).unwrap(); // Empty line
        writeln!(temp_file, ",").unwrap(); // Empty fields
        writeln!(
            temp_file,
            "b2c3d4e5-f6a7-8901-bcde-f12345678901,sample2.pod5"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let mapping = parse_csv_mapping(temp_file.path()).unwrap();
        assert_eq!(mapping.len(), 2);
    }

    #[test]
    fn test_parse_csv_mapping_missing_header() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "uuid,file").unwrap(); // Wrong headers
        writeln!(
            temp_file,
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890,sample1.pod5"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let result = parse_csv_mapping(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_csv_mapping_whitespace_trimmed() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id , output").unwrap();
        writeln!(
            temp_file,
            " a1b2c3d4-e5f6-7890-abcd-ef1234567890 , sample1.pod5 "
        )
        .unwrap();
        temp_file.flush().unwrap();

        let mapping = parse_csv_mapping(temp_file.path()).unwrap();
        assert_eq!(mapping.len(), 1);

        let uuid = Uuid::parse_str("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        assert_eq!(mapping.get(&uuid), Some(&"sample1.pod5".to_string()));
    }

    #[test]
    fn test_parse_csv_mapping_invalid_uuid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,output").unwrap();
        writeln!(temp_file, "not-a-valid-uuid,sample1.pod5").unwrap();
        temp_file.flush().unwrap();

        let result = parse_csv_mapping(temp_file.path());
        assert!(result.is_err());
    }
}

//! Split operation for POD5 files by barcode classification.
//!
//! Reads a CSV file with barcode classifications and splits reads into separate POD5 files.

use crate::Result;
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

use super::csv_utils::parse_csv_uuid_mapping;

/// Parse a CSV file mapping read IDs to barcodes.
///
/// The CSV must have headers including `read_id` and `barcode` columns.
/// UUIDs can be in standard format (with dashes) or compact format (32 hex chars).
///
/// # Arguments
///
/// * `csv_path` - Path to the CSV file (output from classify command)
///
/// # Returns
///
/// A HashMap mapping each read UUID to its barcode assignment.
///
/// # Example
///
/// ```no_run
/// use escapepod_pod5::operations::parse_barcode_mapping;
///
/// let mapping = parse_barcode_mapping("classifications.csv")?;
/// for (uuid, barcode) in &mapping {
///     println!("{} -> {}", uuid, barcode);
/// }
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
///
/// # CSV Format
///
/// ```csv
/// read_id,barcode,distance,second_best_distance,ratio,confident
/// a1b2c3d4-e5f6-7890-abcd-ef1234567890,barcode01,0.1234,0.5678,0.2173,true
/// b2c3d4e5f6a78901bcdef12345678901,barcode02,0.2345,0.6789,0.3456,false
/// ```
pub fn parse_barcode_mapping(csv_path: impl AsRef<Path>) -> Result<HashMap<Uuid, String>> {
    // Try "barcode" first (DTW classify output), then "predicted_barcode" (SVM classify output)
    match parse_csv_uuid_mapping(csv_path.as_ref(), "barcode", false) {
        Ok(mapping) => Ok(mapping),
        Err(_) => parse_csv_uuid_mapping(csv_path, "predicted_barcode", false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_barcode_mapping_valid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            "read_id,barcode,distance,second_best_distance,ratio,confident"
        )
        .unwrap();
        writeln!(
            temp_file,
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890,barcode01,0.1234,0.5678,0.2173,true"
        )
        .unwrap();
        writeln!(
            temp_file,
            "b2c3d4e5-f6a7-8901-bcde-f12345678901,barcode02,0.2345,0.6789,0.3456,false"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let mapping = parse_barcode_mapping(temp_file.path()).unwrap();
        assert_eq!(mapping.len(), 2);

        let uuid1 = Uuid::parse_str("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        let uuid2 = Uuid::parse_str("b2c3d4e5-f6a7-8901-bcde-f12345678901").unwrap();

        assert_eq!(mapping.get(&uuid1), Some(&"barcode01".to_string()));
        assert_eq!(mapping.get(&uuid2), Some(&"barcode02".to_string()));
    }

    #[test]
    fn test_parse_barcode_mapping_empty_barcode() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,barcode").unwrap();
        writeln!(temp_file, "a1b2c3d4-e5f6-7890-abcd-ef1234567890,barcode01").unwrap();
        writeln!(temp_file, "b2c3d4e5-f6a7-8901-bcde-f12345678901,").unwrap(); // Empty barcode
        temp_file.flush().unwrap();

        let mapping = parse_barcode_mapping(temp_file.path()).unwrap();
        assert_eq!(mapping.len(), 2);

        let uuid2 = Uuid::parse_str("b2c3d4e5-f6a7-8901-bcde-f12345678901").unwrap();
        assert_eq!(mapping.get(&uuid2), Some(&"".to_string()));
    }

    #[test]
    fn test_parse_barcode_mapping_missing_header() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "uuid,bc").unwrap(); // Wrong headers
        writeln!(temp_file, "a1b2c3d4-e5f6-7890-abcd-ef1234567890,barcode01").unwrap();
        temp_file.flush().unwrap();

        let result = parse_barcode_mapping(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_barcode_mapping_invalid_uuid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "read_id,barcode").unwrap();
        writeln!(temp_file, "not-a-valid-uuid,barcode01").unwrap();
        temp_file.flush().unwrap();

        let result = parse_barcode_mapping(temp_file.path());
        assert!(result.is_err());
    }
}

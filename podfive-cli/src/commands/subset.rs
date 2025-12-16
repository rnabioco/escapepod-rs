//! Subset command implementation.
//!
//! Splits reads into multiple output files based on a CSV mapping.

use crate::progress::create_progress_bar;
use crate::style;
use podfive_core::operations::parse_csv_mapping;
use podfive_core::{Reader, RunInfoData, Writer, WriterOptions};
use std::collections::HashMap;
use std::path::PathBuf;

pub fn run(
    input: PathBuf,
    csv_file: PathBuf,
    output_dir: PathBuf,
    force: bool,
) -> anyhow::Result<()> {
    // Parse the CSV mapping file
    let mapping = parse_csv_mapping(&csv_file)?;

    if mapping.is_empty() {
        anyhow::bail!("No valid mappings found in CSV file");
    }

    // Get unique output files
    let output_files: Vec<&String> = mapping
        .values()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    println!(
        "{} {} reads into {} output file(s)",
        style::action("Subsetting"),
        style::count(mapping.len()),
        style::value(output_files.len())
    );

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    // Check for existing files if not forcing
    if !force {
        for output_name in &output_files {
            let output_path = output_dir.join(output_name);
            if output_path.exists() {
                anyhow::bail!(
                    "Output file {} already exists. Use --force to overwrite.",
                    output_path.display()
                );
            }
        }
    }

    // Open reader
    let reader = Reader::open(&input)?;
    let run_infos: Vec<RunInfoData> = reader.run_infos().to_vec();

    // Create writers for each output file
    let mut writers: HashMap<String, Writer> = HashMap::new();
    let mut write_counts: HashMap<String, u64> = HashMap::new();
    let options = WriterOptions::default();

    for output_name in output_files {
        let output_path = output_dir.join(output_name);
        let mut writer = Writer::create(&output_path, options.clone())?;

        // Add all run infos to each writer
        for run_info in &run_infos {
            writer.add_run_info(run_info.clone())?;
        }

        writers.insert(output_name.clone(), writer);
        write_counts.insert(output_name.clone(), 0);
    }

    // Set up progress
    let read_count = reader.read_count()?;
    let progress = create_progress_bar(read_count as u64, "Subsetting")?;
    progress.set_message("reads");

    let mut matched = 0u64;
    let mut unmatched = 0u64;

    // Process reads
    for read_result in reader.reads()? {
        let read = read_result?;

        if let Some(output_name) = mapping.get(&read.read_id) {
            // Use compressed signal to avoid decompress/recompress overhead
            let compressed_signal = reader.get_compressed_signal_for_rows(&read.signal_rows)?;

            let new_read = read.for_writing_same_run();

            if let Some(writer) = writers.get_mut(output_name) {
                writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;
                *write_counts.get_mut(output_name).unwrap() += 1;
            }

            matched += 1;
        } else {
            unmatched += 1;
        }

        progress.inc(1);
    }

    progress.finish_with_message("done");

    // Finalize all writers
    for (_name, writer) in writers {
        writer.finish()?;
    }

    // Print summary
    println!("\n{}", style::header("Subset summary:"));
    println!("  Matched reads: {}", style::count(matched));
    println!(
        "  Unmatched reads: {}",
        if unmatched > 0 {
            style::warning(unmatched)
        } else {
            unmatched.to_string()
        }
    );
    println!("\n{}", style::label("Output files:"));
    for (name, count) in &write_counts {
        let path = output_dir.join(name);
        println!(
            "  {} ({} reads)",
            style::path(path.display()),
            style::count(count)
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use podfive_core::operations::parse_csv_mapping;
    use podfive_core::utils::parse_uuid_flexible;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    #[test]
    fn test_parse_uuid_standard_format() {
        let uuid = parse_uuid_flexible("a1b2c3d4-e5f6-7890-abcd-ef1234567890").unwrap();
        assert_eq!(uuid.to_string(), "a1b2c3d4-e5f6-7890-abcd-ef1234567890");
    }

    #[test]
    fn test_parse_uuid_no_dashes() {
        let uuid = parse_uuid_flexible("a1b2c3d4e5f67890abcdef1234567890").unwrap();
        assert_eq!(uuid.to_string(), "a1b2c3d4-e5f6-7890-abcd-ef1234567890");
    }

    #[test]
    fn test_parse_uuid_uppercase() {
        let uuid = parse_uuid_flexible("A1B2C3D4-E5F6-7890-ABCD-EF1234567890").unwrap();
        assert_eq!(uuid.to_string(), "a1b2c3d4-e5f6-7890-abcd-ef1234567890");
    }

    #[test]
    fn test_parse_uuid_invalid() {
        assert!(parse_uuid_flexible("not-a-uuid").is_err());
        assert!(parse_uuid_flexible("").is_err());
        assert!(parse_uuid_flexible("a1b2c3d4").is_err());
    }

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
        assert!(result.unwrap_err().to_string().contains("read_id"));
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
        assert!(result.unwrap_err().to_string().contains("Invalid UUID"));
    }
}

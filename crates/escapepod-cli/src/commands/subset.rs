//! Subset command implementation.
//!
//! Splits reads into multiple output files based on a CSV mapping.
//! Uses the optimized filter_files() path for each output group.

use crate::commands::profile::PhaseTimer;
use crate::style;
use escapepod_signal::operations::{FilterOptions, filter_files, parse_csv_mapping};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tracing::info;
use uuid::Uuid;

pub fn run(
    input: PathBuf,
    csv_file: PathBuf,
    output_dir: PathBuf,
    threads: Option<usize>,
    force: bool,
    profile: bool,
) -> anyhow::Result<()> {
    // Bound the combined width of the per-group and per-batch parallelism.
    // Like `filter`, subset does NOT default to all CPUs (see DEFAULT_THREADS);
    // raise it with `-t` on a machine you own.
    let num_threads = threads.unwrap_or(crate::commands::DEFAULT_THREADS);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok(); // Ignore error if pool already initialized

    let mut timer = PhaseTimer::new();
    timer.phase("Parse CSV mapping");

    // Parse the CSV mapping file
    let mapping = parse_csv_mapping(&csv_file)?;

    if mapping.is_empty() {
        anyhow::bail!("No valid mappings found in CSV file");
    }

    // Group read IDs by output file
    let mut groups: HashMap<String, HashSet<Uuid>> = HashMap::new();
    for (read_id, output_name) in &mapping {
        groups
            .entry(output_name.clone())
            .or_default()
            .insert(*read_id);
    }

    let num_groups = groups.len();
    let total_reads = mapping.len();

    info!(
        "{} {} reads into {} output file(s)",
        style::action("Subsetting"),
        style::count(total_reads),
        style::value(num_groups)
    );

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    // Check for existing files if not forcing
    if !force {
        for output_name in groups.keys() {
            let output_path = output_dir.join(output_name);
            if output_path.exists() {
                anyhow::bail!(
                    "Output file {} already exists. Use --force to overwrite.",
                    output_path.display()
                );
            }
        }
    }

    let options = FilterOptions {
        signal_batch_size: 1_000,
        read_batch_size: 10_000,
    };

    timer.phase("Filter & write groups");
    // Process all output groups in parallel using the optimized filter path
    let group_list: Vec<_> = groups.into_iter().collect();
    let results: Vec<anyhow::Result<(String, u64)>> = group_list
        .par_iter()
        .map(|(output_name, group_ids)| {
            let output_path = output_dir.join(output_name);
            let input_files = [&input];
            let result =
                filter_files(&input_files, &output_path, group_ids, options.clone(), None)?;
            Ok((output_name.clone(), result.matched_reads))
        })
        .collect();

    let mut total_matched = 0u64;
    let mut group_rows: Vec<(PathBuf, u64)> = Vec::new();
    for result in results {
        let (name, matched) = result?;
        let output_path = output_dir.join(&name);
        group_rows.push((output_path, matched));
        total_matched += matched;
    }

    let unmatched = (total_reads as u64).saturating_sub(total_matched);

    // Styled multi-line report; gate on verbosity instead of per-line tracing events.
    if tracing::enabled!(tracing::Level::INFO) {
        for (output_path, matched) in &group_rows {
            eprintln!(
                "  {} ({} reads)",
                style::path(output_path.display()),
                style::count(*matched)
            );
        }
        eprintln!("\n{}", style::header("Subset summary:"));
        eprintln!("  Matched reads: {}", style::count(total_matched));
        eprintln!(
            "  Unmatched reads: {}",
            if unmatched > 0 {
                style::warning(unmatched)
            } else {
                unmatched.to_string()
            }
        );
    }

    timer.report(profile);

    Ok(())
}

#[cfg(test)]
mod tests {
    use escapepod_signal::operations::parse_csv_mapping;
    use escapepod_signal::parse_uuid_flexible;
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

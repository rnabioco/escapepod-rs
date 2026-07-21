//! Subset command implementation.
//!
//! Splits reads into multiple output files based on a CSV mapping, in a single
//! pass over the input (see `subset_file`): the input is scanned once and each
//! read routed to its group's writer, rather than re-scanning the whole input
//! once per output group.

use crate::commands::profile::PhaseTimer;
use crate::style;
use crate::util::check_output_not_input;
use escapepod_signal::Durability;
use escapepod_signal::operations::{FilterOptions, parse_csv_mapping, subset_file};
use std::collections::HashSet;
use std::path::PathBuf;
use tracing::info;

pub fn run(
    input: PathBuf,
    csv_file: PathBuf,
    output_dir: PathBuf,
    threads: Option<usize>,
    force: bool,
    profile: bool,
    durability: Durability,
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

    // Unique output files (a group) — one per distinct value in the mapping.
    let unique_outputs: HashSet<&String> = mapping.values().collect();
    let num_groups = unique_outputs.len();
    let total_reads = mapping.len();

    info!(
        "{} {} reads into {} output file(s)",
        style::action("Subsetting"),
        style::count(total_reads),
        style::value(num_groups)
    );

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    // A group name can collide with the input file when the output directory
    // is the input's own directory, which would replace the source mid-run.
    let inputs = vec![input.clone()];
    for output_name in &unique_outputs {
        let output_path = output_dir.join(output_name);
        check_output_not_input(&output_path, &inputs)?;
        if output_path.exists() && !force {
            anyhow::bail!(
                "Output file {} already exists. Use --force to overwrite.",
                output_path.display()
            );
        }
    }

    let options = FilterOptions {
        signal_batch_size: 1_000,
        read_batch_size: 10_000,
        durability,
    };

    timer.phase("Split (single pass)");
    // One pass over the input: scan the reads table once, partition by group,
    // then write every group's file in parallel against the shared mmap.
    // `SubsetOutcome` already sorts both lists by group name, so the report
    // order is deterministic even though groups are written in parallel.
    let results = subset_file(&input, &mapping, &output_dir, options)?;

    // Each failed group produced no file at all; name them rather than
    // reporting a partial subset as if it were complete.
    if !results.failures.is_empty() {
        for (group, err) in &results.failures {
            tracing::error!("{}: {}", style::path(group), err);
        }
        anyhow::bail!("{} output file(s) failed to write", results.failures.len());
    }

    let mut total_matched = 0u64;
    let mut group_rows: Vec<(PathBuf, u64)> = Vec::new();
    for (name, matched) in &results.groups {
        group_rows.push((output_dir.join(name), *matched));
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

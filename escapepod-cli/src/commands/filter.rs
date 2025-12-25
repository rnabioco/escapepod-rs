//! Filter command implementation.
//!
//! Filters reads from POD5 files based on various criteria including read IDs,
//! sample count (read length), and end reasons.
//! Uses batch-level parallelism with rayon and block-level copying for maximum performance.

use crate::progress::create_progress_bar;
use crate::style;
use crate::util::resolve_pod5_inputs;
use escapepod::operations::{
    filter_files_with_criteria, read_ids_from_file, FilterCriteria, FilterOptions,
};
use escapepod::types::EndReason;
use std::collections::HashSet;
use std::path::PathBuf;

pub fn run(
    input: PathBuf,
    ids_file: Option<PathBuf>,
    min_samples: Option<u64>,
    max_samples: Option<u64>,
    end_reason: Option<Vec<String>>,
    exclude_end_reason: Option<Vec<String>>,
    output: PathBuf,
) -> anyhow::Result<()> {
    // Resolve input to list of POD5 files (supports directories)
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    // Build filter criteria
    let mut criteria = FilterCriteria::default();

    // Load read IDs if specified
    if let Some(ref ids_path) = ids_file {
        let ids = read_ids_from_file(ids_path)?;
        if ids.is_empty() {
            anyhow::bail!("No read IDs found in {}", ids_path.display());
        }
        criteria.read_ids = Some(ids);
    }

    // Set sample count filters
    criteria.min_samples = min_samples;
    criteria.max_samples = max_samples;

    // Parse end reason filters
    if let Some(reasons) = end_reason {
        let parsed: HashSet<EndReason> = reasons
            .iter()
            .map(|s| EndReason::from(s.as_str()))
            .collect();
        criteria.include_end_reasons = Some(parsed);
    }

    if let Some(reasons) = exclude_end_reason {
        let parsed: HashSet<EndReason> = reasons
            .iter()
            .map(|s| EndReason::from(s.as_str()))
            .collect();
        criteria.exclude_end_reasons = Some(parsed);
    }

    // Validate that at least one criterion is set
    if criteria.is_empty() {
        anyhow::bail!(
            "No filter criteria specified. Use --ids, --min-samples, --max-samples, \
             --end-reason, or --exclude-end-reason"
        );
    }

    // Print filtering info
    println!(
        "{} {}",
        style::action("Filtering"),
        if is_directory {
            format!(
                "{} ({} files)",
                style::path(input.display()),
                style::value(files.len())
            )
        } else {
            style::path(input.display()).to_string()
        }
    );

    // Print active criteria
    if let Some(ref ids) = criteria.read_ids {
        println!(
            "  {} {} read IDs from {}",
            style::label("IDs:"),
            style::count(ids.len()),
            style::path(ids_file.as_ref().unwrap().display())
        );
    }
    if let Some(min) = criteria.min_samples {
        println!("  {} >= {}", style::label("Samples:"), style::value(min));
    }
    if let Some(max) = criteria.max_samples {
        println!("  {} <= {}", style::label("Samples:"), style::value(max));
    }
    if let Some(ref reasons) = criteria.include_end_reasons {
        let reason_strs: Vec<_> = reasons.iter().map(|r| r.as_str()).collect();
        println!(
            "  {} {}",
            style::label("End reasons:"),
            reason_strs.join(", ")
        );
    }
    if let Some(ref reasons) = criteria.exclude_end_reasons {
        let reason_strs: Vec<_> = reasons.iter().map(|r| r.as_str()).collect();
        println!(
            "  {} {}",
            style::label("Exclude end reasons:"),
            reason_strs.join(", ")
        );
    }

    println!(
        "{} {}",
        style::label("Output:"),
        style::path(output.display())
    );

    // Estimate total reads for progress bar (we'll update as we go)
    let filter_bar = create_progress_bar(0, "Filtering")?;
    filter_bar.set_length(0); // Will be set by first progress callback

    let bar_for_callback = filter_bar.clone();

    // Create progress callback
    let progress: Box<dyn Fn(u64, u64) + Send + Sync> =
        Box::new(move |current: u64, total: u64| {
            bar_for_callback.set_length(total);
            bar_for_callback.set_position(current);
        });

    // Use the core library's parallel filter
    let options = FilterOptions {
        signal_batch_size: 1_000,
        read_batch_size: 10_000,
    };

    let result = filter_files_with_criteria(&files, &output, &criteria, options, Some(progress))?;

    filter_bar.finish_with_message(format!("{} matched", result.matched_reads));

    let percentage = result.match_percentage();
    println!(
        "{} {} reads from {} total ({})",
        style::action("Filtered"),
        style::count(result.matched_reads),
        result.total_reads,
        style::percentage(format!("{:.1}%", percentage))
    );

    // ID-specific warnings only if filtering by IDs
    if let Some(ref ids) = criteria.read_ids {
        let not_found = (ids.len() as u64).saturating_sub(result.matched_reads);
        if not_found > 0 {
            println!(
                "{} {} requested IDs were not found in the input",
                style::warning_label("Warning:"),
                style::warning(not_found)
            );
        }
        if result.matched_reads > ids.len() as u64 {
            println!(
                "{} {} duplicate reads matched across multiple files",
                style::note_label("Note:"),
                style::warning(result.matched_reads - ids.len() as u64)
            );
        }
    }

    // Report any errors encountered
    if result.read_errors > 0 || result.signal_errors > 0 {
        eprintln!(
            "{} encountered {} read error(s) and {} signal error(s)",
            style::error_label("Warning:"),
            style::error(result.read_errors),
            style::error(result.signal_errors)
        );
    }

    Ok(())
}

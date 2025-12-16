//! Filter command implementation.
//!
//! Filters reads from POD5 files based on a list of read IDs.
//! Uses batch-level parallelism with rayon and block-level copying for maximum performance.

use crate::progress::create_progress_bar;
use crate::style;
use crate::util::resolve_pod5_inputs;
use podfive_core::operations::{filter_files, read_ids_from_file, FilterOptions};
use std::path::PathBuf;

pub fn run(input: PathBuf, ids_file: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    // Resolve input to list of POD5 files (supports directories)
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!(
        "{} {} using IDs from {}",
        style::action("Filtering"),
        if is_directory {
            format!(
                "{} ({} files)",
                style::path(input.display()),
                style::value(files.len())
            )
        } else {
            style::path(input.display())
        },
        style::path(ids_file.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(output.display())
    );

    // Read IDs from file (using core library)
    let ids = read_ids_from_file(&ids_file)?;
    println!("Loaded {} read IDs to filter", style::count(ids.len()));

    if ids.is_empty() {
        anyhow::bail!("No read IDs found in {}", ids_file.display());
    }

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

    let result = filter_files(&files, &output, &ids, options, Some(progress))?;

    filter_bar.finish_with_message(format!("{} matched", result.matched_reads));

    let percentage = result.match_percentage();
    println!(
        "{} {} reads from {} total ({})",
        style::action("Filtered"),
        style::count(result.matched_reads),
        result.total_reads,
        style::percentage(format!("{:.1}%", percentage))
    );

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

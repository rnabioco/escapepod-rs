//! Merge command implementation.
//!
//! Thin wrapper around escapepod::merge_files.

use crate::progress::create_spinner;
use crate::style;
use crate::util::resolve_pod5_inputs;
use escapepod::{merge_files, MergeOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub fn run(inputs: Vec<PathBuf>, output: PathBuf, duplicate_ok: bool) -> anyhow::Result<()> {
    if inputs.is_empty() {
        anyhow::bail!("No input files specified");
    }

    // Expand any directories to individual POD5 files
    let mut all_files = Vec::new();
    for input in &inputs {
        let files = resolve_pod5_inputs(input)?;
        all_files.extend(files);
    }

    if all_files.is_empty() {
        anyhow::bail!("No POD5 files found in specified inputs");
    }

    let num_files = all_files.len();
    eprintln!(
        "{} {} files into {}",
        style::action("Merging"),
        style::count(num_files),
        style::path(output.display()),
    );

    let options = MergeOptions {
        duplicate_ok,
        read_batch_size: 100_000,
    };

    // Create progress indicator
    let spinner = create_spinner("Processing")?;
    spinner.set_message("files...");

    // Track progress
    let progress = Arc::new(AtomicUsize::new(0));
    let progress_clone = progress.clone();
    let spinner_clone = spinner.clone();

    let callback = move |current: usize, total: usize| {
        progress_clone.store(current, Ordering::Relaxed);
        spinner_clone.set_message(format!("files... ({}/{})", current, total));
    };

    // Run merge
    let result = merge_files(&all_files, &output, &options, Some(&callback))?;

    spinner.finish_with_message(format!(
        "{} signal rows processed",
        style::count(result.signal_rows)
    ));

    println!(
        "{} {} reads into {}",
        style::action("Merged"),
        style::count(result.reads_written),
        style::path(output.display())
    );

    if result.duplicates_skipped > 0 {
        println!(
            "{} {} duplicate reads",
            style::note_label("Skipped"),
            style::warning(result.duplicates_skipped)
        );
    }

    Ok(())
}

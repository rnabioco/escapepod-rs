//! Repack command implementation.
//!
//! Repacks POD5 files using block-level signal copying (no decompression/recompression).
//! Files are processed in parallel using rayon. Supports directories as input.

use crate::commands::profile::PhaseTimer;
use crate::progress::create_progress_bar;
use crate::style;
use crate::util::collect_pod5_inputs;
use escapepod::{RepackOptions, repack_files};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn run(
    inputs: Vec<PathBuf>,
    output_dir: PathBuf,
    force: bool,
    profile: bool,
) -> anyhow::Result<()> {
    let mut timer = PhaseTimer::new();
    timer.phase("Resolve inputs");
    let all_files = collect_pod5_inputs(&inputs)?;

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    // Check for existing files if not forcing
    if !force {
        for input_path in &all_files {
            if let Some(file_name) = input_path.file_name() {
                let output_path = output_dir.join(file_name);
                if output_path.exists() {
                    anyhow::bail!(
                        "Output file {} already exists. Use --force to overwrite.",
                        output_path.display()
                    );
                }
            }
        }
    }

    println!(
        "{} {} file(s) to {}",
        style::action("Repacking"),
        style::count(all_files.len()),
        style::path(output_dir.display())
    );

    let overall_bar = create_progress_bar(all_files.len() as u64, "Repacking")?;
    let bar_progress = Arc::new(AtomicU64::new(0));

    // Build file pairs (input, output)
    let file_pairs: Vec<(PathBuf, PathBuf)> = all_files
        .iter()
        .filter_map(|input_path| {
            input_path.file_name().map(|file_name| {
                let output_path = output_dir.join(file_name);
                (input_path.clone(), output_path)
            })
        })
        .collect();

    let options = RepackOptions {
        force,
        ..RepackOptions::default()
    };

    // Create progress callback
    let bar = overall_bar.clone();
    let progress_counter = bar_progress.clone();
    let progress_callback: escapepod::ProgressCallback = Box::new(move |p: escapepod::Progress| {
        let prev = progress_counter.swap(p.current, Ordering::Relaxed);
        if p.current > prev {
            bar.inc(p.current - prev);
        }
    });

    timer.phase("Repack (parallel)");
    // Process files in parallel using the new operation
    let result = repack_files(&file_pairs, options, Some(progress_callback));

    overall_bar.finish_with_message("done");

    println!(
        "{} {} reads across {} file(s)",
        style::action("Repacked"),
        style::count(result.total_reads),
        style::value(result.files_processed)
    );

    if result.files_skipped > 0 {
        println!(
            "  {} {} file(s) skipped",
            style::action("Warning:"),
            style::count(result.files_skipped as u64)
        );
    }

    timer.report(profile);

    Ok(())
}

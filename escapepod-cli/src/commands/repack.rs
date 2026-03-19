//! Repack command implementation.
//!
//! Repacks POD5 files using block-level signal copying (no decompression/recompression).
//! Files are processed in parallel using rayon. Supports directories as input.

use crate::progress::create_progress_bar;
use crate::style;
use crate::util::resolve_pod5_inputs;
use escapepod::{RepackOptions, repack_files};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn run(inputs: Vec<PathBuf>, output_dir: PathBuf, force: bool) -> anyhow::Result<()> {
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
    let progress_callback: Box<dyn Fn(usize, usize) + Send + Sync> =
        Box::new(move |done: usize, _total: usize| {
            let prev = progress_counter.swap(done as u64, Ordering::Relaxed);
            if done as u64 > prev {
                bar.inc((done as u64) - prev);
            }
        });

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

    Ok(())
}

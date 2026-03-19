//! Merge command implementation.
//!
//! Thin wrapper around escapepod::merge_files.

use crate::progress::create_progress_bar;
use crate::style;
use crate::util::resolve_pod5_inputs;
use escapepod::{MergeOptions, MergePhase, MergeProgress, merge_files};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub fn run(
    inputs: Vec<PathBuf>,
    output: PathBuf,
    duplicate_ok: bool,
    threads: Option<usize>,
    profile: bool,
) -> anyhow::Result<()> {
    // Configure rayon thread pool if threads specified
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // Ignore error if pool already initialized
    }

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

    // Create progress bar
    let progress_bar = create_progress_bar(num_files as u64, "Loading")?;
    progress_bar.set_message("metadata");

    // Track current phase and timing
    let current_phase = Mutex::new(None::<MergePhase>);
    let phase_start = Mutex::new(Instant::now());
    let phase_times: Mutex<Vec<(MergePhase, Duration)>> = Mutex::new(Vec::new());
    let total_start = Instant::now();

    let callback = |progress: MergeProgress| {
        let mut phase_guard = current_phase.lock().unwrap();

        // Detect phase transitions
        if *phase_guard != Some(progress.phase) {
            // Record previous phase duration
            if let Some(prev_phase) = *phase_guard {
                let elapsed = phase_start.lock().unwrap().elapsed();
                phase_times.lock().unwrap().push((prev_phase, elapsed));
            }
            *phase_start.lock().unwrap() = Instant::now();
            *phase_guard = Some(progress.phase);

            match progress.phase {
                MergePhase::LoadingMetadata => {
                    progress_bar.set_prefix("Loading");
                    progress_bar.set_message("metadata");
                    progress_bar.set_length(progress.total as u64);
                    progress_bar.set_position(0);
                }
                MergePhase::WritingSignal => {
                    progress_bar.set_prefix("Writing");
                    progress_bar.set_message("signal");
                    progress_bar.set_length(progress.total as u64);
                    progress_bar.set_position(0);
                }
                MergePhase::WritingReads => {
                    progress_bar.set_prefix("Writing");
                    progress_bar.set_message("reads");
                    progress_bar.set_length(progress.total as u64);
                    progress_bar.set_position(0);
                }
            }
        }

        progress_bar.set_position(progress.current as u64);
    };

    // Run merge
    let result = merge_files(&all_files, &output, &options, Some(&callback))?;

    // Record final phase duration
    if let Some(last_phase) = *current_phase.lock().unwrap() {
        let elapsed = phase_start.lock().unwrap().elapsed();
        phase_times.lock().unwrap().push((last_phase, elapsed));
    }
    let total_elapsed = total_start.elapsed();

    progress_bar.finish_and_clear();

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

    // Print profiling info if requested
    if profile {
        eprintln!();
        eprintln!("{}", style::action("Profile"));
        let times = phase_times.lock().unwrap();
        for (phase, duration) in times.iter() {
            let phase_name = match phase {
                MergePhase::LoadingMetadata => "Loading metadata (parallel)",
                MergePhase::WritingSignal => "Writing signal data",
                MergePhase::WritingReads => "Writing reads table",
            };
            let pct = (duration.as_secs_f64() / total_elapsed.as_secs_f64()) * 100.0;
            eprintln!(
                "  {:<30} {:>8.2}s ({:>5.1}%)",
                phase_name,
                duration.as_secs_f64(),
                pct
            );
        }
        eprintln!("  {:<30} {:>8.2}s", "Total", total_elapsed.as_secs_f64());

        // Additional stats
        let output_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        let throughput = output_size as f64 / total_elapsed.as_secs_f64() / 1_000_000.0;
        eprintln!();
        eprintln!("  Output size: {:.2} MB", output_size as f64 / 1_000_000.0);
        eprintln!("  Throughput:  {:.2} MB/s", throughput);
        eprintln!("  Files:       {}", num_files);
        eprintln!("  Reads:       {}", result.reads_written);
    }

    Ok(())
}

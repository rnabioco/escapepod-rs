//! Merge command implementation.
//!
//! Thin wrapper around escapepod_signal::merge_files.

use crate::progress::create_progress_bar;
use crate::style;
use crate::util::{check_output_writable, collect_pod5_inputs};
use escapepod_signal::{MergeOptions, MergePhase, MergeProgress, merge_files};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::info;

pub fn run(
    inputs: Vec<PathBuf>,
    output: PathBuf,
    duplicate_ok: bool,
    threads: Option<usize>,
    force: bool,
    profile: bool,
) -> anyhow::Result<()> {
    // Bound the parallelism. Merge scales across input files, but like the
    // other block-copy commands it does NOT default to all CPUs (see
    // DEFAULT_THREADS) — that is antisocial on a shared node for a largely
    // I/O-bound copy. Raise it with `-t` on a machine you own.
    let num_threads = threads.unwrap_or(crate::commands::DEFAULT_THREADS);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok(); // Ignore error if pool already initialized

    check_output_writable(&output, force)?;

    let all_files = collect_pod5_inputs(&inputs)?;

    let num_files = all_files.len();
    info!(
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

    // Track current phase and timing. One mutex holds the entire tracker so
    // each callback takes a single lock instead of five. LoadingMetadata
    // fires the callback from a parallel par_iter, so the old multi-mutex
    // dance caused real lock traffic on large input sets.
    struct PhaseTracker {
        current: Option<MergePhase>,
        start: Instant,
        times: Vec<(MergePhase, Duration)>,
    }
    let tracker = Mutex::new(PhaseTracker {
        current: None,
        start: Instant::now(),
        times: Vec::new(),
    });
    let total_start = Instant::now();

    let callback = |progress: MergeProgress| {
        let mut t = tracker.lock().unwrap();

        if t.current != Some(progress.phase) {
            if let Some(prev) = t.current {
                let elapsed = t.start.elapsed();
                t.times.push((prev, elapsed));
            }
            t.start = Instant::now();
            t.current = Some(progress.phase);

            match progress.phase {
                MergePhase::LoadingMetadata => {
                    progress_bar.set_prefix("Loading");
                    progress_bar.set_message("metadata");
                }
                MergePhase::WritingSignal => {
                    progress_bar.set_prefix("Writing");
                    progress_bar.set_message("signal");
                }
                MergePhase::WritingReads => {
                    progress_bar.set_prefix("Writing");
                    progress_bar.set_message("reads");
                }
            }
            progress_bar.set_length(progress.total as u64);
            progress_bar.set_position(0);
        }

        progress_bar.set_position(progress.current as u64);
    };

    // Run merge
    let result = merge_files(&all_files, &output, &options, Some(&callback))?;

    // Record final phase duration
    let phase_times = {
        let mut t = tracker.lock().unwrap();
        if let Some(last) = t.current {
            let elapsed = t.start.elapsed();
            t.times.push((last, elapsed));
        }
        std::mem::take(&mut t.times)
    };
    let total_elapsed = total_start.elapsed();

    progress_bar.finish_and_clear();

    info!(
        "{} {} reads into {}",
        style::action("Merged"),
        style::count(result.reads_written),
        style::path(output.display())
    );

    if result.duplicates_skipped > 0 {
        info!(
            "{} {} duplicate reads",
            style::note_label("Skipped"),
            style::warning(result.duplicates_skipped)
        );
    }

    // Print profiling info if requested
    if profile {
        eprintln!();
        eprintln!("{}", style::action("Profile"));
        for (phase, duration) in phase_times.iter() {
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

//! Detect subcommand - LLR-based adapter boundary detection.

use super::types::ReadBoundaries;
use super::utils::{collect_reads_with_signals, configure_thread_pool, normalize_signal};
use crate::progress::create_progress_bar;
use crate::style;
use escapepod::segmentation::detect_adapter;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Arguments for the detect subcommand.
#[derive(Debug, clap::Args)]
pub struct DetectArgs {
    /// Input POD5 file(s)
    #[arg(required = true, value_name = "FILES")]
    pub input: Vec<PathBuf>,

    /// Output CSV file for detected boundaries
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// Minimum observations for adapter segment
    #[arg(long, default_value = "200", value_name = "N")]
    pub min_adapter: usize,

    /// Border trim size
    #[arg(long, default_value = "50", value_name = "N")]
    pub border_trim: usize,

    /// Number of threads for parallel processing
    #[arg(short = 'j', long, default_value = "4", value_name = "N")]
    pub threads: usize,
}

/// Run the detect subcommand using LLR boundary detection.
pub fn run(args: DetectArgs) -> anyhow::Result<()> {
    println!(
        "{} adapter boundaries using LLR algorithm",
        style::action("Detecting"),
    );
    println!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );
    println!(
        "{} min_adapter={}, border_trim={}",
        style::label("Parameters:"),
        style::value(args.min_adapter),
        style::value(args.border_trim)
    );

    // Set thread pool size
    configure_thread_pool(args.threads);

    // Collect all reads with their signals
    let all_reads = collect_reads_with_signals(&args.input)?;

    println!(
        "{} {} reads to process",
        style::label("Found:"),
        style::count(all_reads.len())
    );

    let progress_bar = create_progress_bar(all_reads.len() as u64, "Detecting")?;

    // Process reads in parallel
    let results: Vec<ReadBoundaries> = all_reads
        .par_iter()
        .map(|(read_id, num_samples, signal)| {
            // Normalize signal
            let normalized = normalize_signal(signal);

            // Detect adapter using LLR
            let (adapter_start, adapter_end) =
                detect_adapter(&normalized, args.min_adapter, args.border_trim);

            progress_bar.inc(1);

            ReadBoundaries {
                read_id: *read_id,
                num_samples: *num_samples,
                adapter_start,
                adapter_end,
            }
        })
        .collect();

    progress_bar.finish_with_message("complete");

    // Write results
    let output_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(output_file);

    writeln!(writer, "read_id,num_samples,adapter_start,adapter_end")?;

    let mut detected_count = 0;
    for boundaries in &results {
        writeln!(
            writer,
            "{},{},{},{}",
            boundaries.read_id,
            boundaries.num_samples,
            boundaries.adapter_start,
            boundaries.adapter_end
        )?;
        if boundaries.has_valid_adapter() {
            detected_count += 1;
        }
    }

    writer.flush()?;

    println!(
        "{} boundaries written to {}",
        style::action("Detected"),
        style::path(args.output.display())
    );
    println!(
        "{} {} reads with detected adapters",
        style::label("Result:"),
        style::count(detected_count)
    );

    Ok(())
}

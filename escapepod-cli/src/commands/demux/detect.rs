//! Detect subcommand - LLR-based adapter boundary detection.

use super::types::ReadBoundaries;
use super::utils::{
    collect_read_metadata, configure_thread_pool, downscale_signal, normalize_signal, open_readers,
    ReadMetadata,
};
use crate::progress::create_progress_bar;
use crate::style;
use escapepod::segmentation::detect_adapter;
use escapepod::Reader;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

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

    /// Downscale factor for signal processing (1 = no downscaling, 10 = WarpDemuX-compatible)
    #[arg(long, default_value = "1", value_name = "N")]
    pub downscale: usize,

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
        "{} min_adapter={}, border_trim={}, downscale={}",
        style::label("Parameters:"),
        style::value(args.min_adapter),
        style::value(args.border_trim),
        style::value(args.downscale)
    );

    // Set thread pool size
    configure_thread_pool(args.threads);

    // Open all POD5 files - Arc-wrapped for parallel access
    let readers = open_readers(&args.input)?;

    // Collect read metadata (fast - no signal decompression)
    let read_metadata = collect_read_metadata(&readers)?;

    println!(
        "{} {} reads to process",
        style::label("Found:"),
        style::count(read_metadata.len())
    );

    let progress_bar = create_progress_bar(read_metadata.len() as u64, "Detecting")?;

    // Process reads in parallel - signal decompression is now parallelized
    let downscale = args.downscale.max(1); // Ensure at least 1
    let results: Vec<ReadBoundaries> = read_metadata
        .par_iter()
        .filter_map(|meta| {
            process_read(&readers, meta, downscale, &args, &progress_bar)
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

/// Process a single read: decompress signal, normalize, detect adapter.
fn process_read(
    readers: &[Arc<Reader>],
    meta: &ReadMetadata,
    downscale: usize,
    args: &DetectArgs,
    progress_bar: &indicatif::ProgressBar,
) -> Option<ReadBoundaries> {
    // Get the reader for this file
    let reader = &readers[meta.file_index];

    // Decompress signal (this is the expensive operation, now parallelized)
    let signal = match reader.get_signal(&meta.signal_rows) {
        Ok(s) => s,
        Err(_) => {
            progress_bar.inc(1);
            return None;
        }
    };

    // Normalize signal
    let normalized = normalize_signal(&signal);

    // Optionally downscale signal
    let (processed_signal, scale_factor) = if downscale > 1 {
        (downscale_signal(&normalized, downscale), downscale)
    } else {
        (normalized, 1)
    };

    // Scale parameters for downscaled signal
    let scaled_min_adapter = args.min_adapter / scale_factor;
    let scaled_border_trim = args.border_trim / scale_factor;

    // Detect adapter using LLR
    let (adapter_start, adapter_end) = detect_adapter(
        &processed_signal,
        scaled_min_adapter.max(1),
        scaled_border_trim.max(1),
    );

    // Scale results back to original resolution
    let adapter_start = adapter_start * scale_factor;
    let adapter_end = adapter_end * scale_factor;

    progress_bar.inc(1);

    Some(ReadBoundaries {
        read_id: meta.read_id,
        num_samples: meta.num_samples,
        adapter_start,
        adapter_end,
    })
}

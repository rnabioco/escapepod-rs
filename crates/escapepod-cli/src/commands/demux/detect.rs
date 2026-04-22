//! Detect subcommand - LLR-based adapter boundary detection.

use super::types::ReadBoundaries;
use super::utils::{
    configure_thread_pool, normalize_signal, process_reads_par, total_read_count,
};
use crate::progress::create_progress_bar;
use crate::style;
use escapepod_signal::segmentation::{detect_adapter, downscale};
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
    #[arg(
        long,
        default_value = "200",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub min_adapter: usize,

    /// Border trim size
    #[arg(
        long,
        default_value = "50",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub border_trim: usize,

    /// Downscale factor for signal processing (1 = no downscaling, 10 = WarpDemuX-compatible)
    #[arg(
        long,
        default_value = "1",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub downscale: usize,

    /// Adapter detection method.
    ///
    /// `llr` (default) uses the built-in log-likelihood ratio detector.
    /// `cnn` uses a port of ADAPTed's BoundariesCNN (opt-in via `--features
    /// cnn-detect`; requires a `.onnx` model produced by
    /// `scripts/export_adapter_cnn_to_onnx.py` from a local ADAPTed install
    /// — those weights are CC BY-NC 4.0 and not bundled with this crate).
    #[arg(
        long,
        default_value = "llr",
        value_name = "{llr,cnn}",
        help_heading = "Advanced Options"
    )]
    pub method: String,

    /// Path to the ADAPTed CNN ONNX model (only used with `--method cnn`).
    #[cfg(feature = "cnn-detect")]
    #[arg(long, value_name = "FILE", help_heading = "Advanced Options")]
    pub cnn_model: Option<PathBuf>,

    /// Number of threads for parallel processing (default: all CPUs)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// Run the detect subcommand.
pub fn run(args: DetectArgs) -> anyhow::Result<()> {
    match args.method.as_str() {
        "llr" => run_llr(args),
        "cnn" => {
            #[cfg(feature = "cnn-detect")]
            {
                run_cnn(args)
            }
            #[cfg(not(feature = "cnn-detect"))]
            {
                let _ = args;
                anyhow::bail!(
                    "--method cnn requires a build with `--features cnn-detect`. \
                     Rebuild with: cargo build --release -p escapepod-cli \
                     --features \"demux cnn-detect\"."
                );
            }
        }
        other => anyhow::bail!("unknown --method `{other}`; expected `llr` or `cnn`"),
    }
}

/// Run the detect subcommand using LLR boundary detection.
fn run_llr(args: DetectArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Detect adapters");
    let profile = args.profile;
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

    let total = total_read_count(&args.input);
    println!(
        "{} {} reads to process",
        style::label("Found:"),
        style::count(total)
    );

    let progress_bar = create_progress_bar(total as u64, "Detecting")?;

    let downscale_factor = args.downscale.max(1);
    let min_adapter = args.min_adapter;
    let border_trim = args.border_trim;

    let results: Vec<ReadBoundaries> = process_reads_par(
        &args.input,
        Some(&progress_bar),
        |read_id, num_samples, signal| {
            let normalized = normalize_signal(signal);

            let (processed_signal, scale_factor) = if downscale_factor > 1 {
                // Truncate to a whole multiple of downscale_factor so the
                // last (partial) chunk is dropped — matches the historical
                // cli behavior and WarpDemuX's numpy-style downsampling.
                let truncated = (normalized.len() / downscale_factor) * downscale_factor;
                (
                    downscale(&normalized[..truncated], downscale_factor),
                    downscale_factor,
                )
            } else {
                (normalized, 1)
            };

            let scaled_min_adapter = min_adapter / scale_factor;
            let scaled_border_trim = border_trim / scale_factor;

            let (adapter_start, adapter_end) = detect_adapter(
                &processed_signal,
                scaled_min_adapter.max(1),
                scaled_border_trim.max(1),
            );

            ReadBoundaries {
                read_id,
                num_samples,
                adapter_start: adapter_start * scale_factor,
                adapter_end: adapter_end * scale_factor,
            }
        },
    )?;

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

    eprintln!(
        "{} boundaries written to {}",
        style::action("Detected"),
        style::path(args.output.display())
    );
    eprintln!(
        "{} {} reads with detected adapters",
        style::label("Result:"),
        style::count(detected_count)
    );

    timer.report(profile);

    Ok(())
}

/// Run the detect subcommand using the ADAPTed CNN (opt-in).
///
/// Requires a user-supplied ONNX model (ADAPTed weights are CC BY-NC
/// and must not be bundled). See `scripts/export_adapter_cnn_to_onnx.py`.
#[cfg(feature = "cnn-detect")]
fn run_cnn(args: DetectArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    use escapepod_demux::AdapterCnn;

    let mut timer = PhaseTimer::new();
    timer.phase("Detect adapters (CNN)");
    let profile = args.profile;

    let cnn_model_path = args
        .cnn_model
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--method cnn requires --cnn-model <FILE>"))?;

    eprintln!(
        "{} the ADAPTed CNN path uses CC BY-NC 4.0 weights you supply via --cnn-model; \
         downstream non-commercial-use obligations apply.",
        style::label("Note:")
    );

    println!(
        "{} adapter boundaries using ADAPTed CNN",
        style::action("Detecting"),
    );
    println!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    println!(
        "{} {}",
        style::label("Model:"),
        style::path(cnn_model_path.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );

    configure_thread_pool(args.threads);

    let cnn =
        AdapterCnn::load(cnn_model_path).map_err(|e| anyhow::anyhow!("loading CNN model: {e}"))?;

    let total = total_read_count(&args.input);
    println!(
        "{} {} reads to process",
        style::label("Found:"),
        style::count(total)
    );

    let progress_bar = create_progress_bar(total as u64, "Detecting (CNN)")?;

    let results: Vec<ReadBoundaries> = process_reads_par(
        &args.input,
        Some(&progress_bar),
        |read_id, num_samples, signal| {
            // MAD normalization inside the CNN's `prepare_data` is
            // scale-invariant, so raw i16 → f32 matches the pA-calibrated
            // path WarpDemuX uses bit-for-bit post-normalization.
            let signal_f32: Vec<f32> = signal.iter().map(|&s| s as f32).collect();
            let adapter_end = cnn.detect_adapter_end(&signal_f32).unwrap_or(0);
            ReadBoundaries {
                read_id,
                num_samples,
                // ADAPTed's CNN sets adapter_start=0 always — boundary
                // detection on this path is single-ended.
                adapter_start: 0,
                adapter_end,
            }
        },
    )?;

    progress_bar.finish_with_message("complete");

    let output_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(output_file);
    writeln!(writer, "read_id,num_samples,adapter_start,adapter_end")?;

    let mut detected = 0;
    for b in &results {
        writeln!(
            writer,
            "{},{},{},{}",
            b.read_id, b.num_samples, b.adapter_start, b.adapter_end
        )?;
        if b.has_valid_adapter() {
            detected += 1;
        }
    }
    writer.flush()?;

    eprintln!(
        "{} boundaries written to {}",
        style::action("Detected"),
        style::path(args.output.display())
    );
    eprintln!(
        "{} {} reads with detected adapters",
        style::label("Result:"),
        style::count(detected)
    );

    timer.report(profile);
    Ok(())
}

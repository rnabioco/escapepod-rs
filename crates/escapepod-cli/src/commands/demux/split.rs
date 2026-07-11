//! Split subcommand - split reads into separate POD5 files by barcode.

use super::utils::configure_thread_pool;
use crate::style;
use escapepod_signal::operations::{FilterOptions, parse_barcode_mapping, subset_files};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use tabled::{builder::Builder, settings::Style};
use tracing::{info, warn};
use uuid::Uuid;

/// Arguments for the split subcommand.
#[derive(Debug, clap::Args)]
pub struct SplitArgs {
    /// Input POD5 file(s)
    #[arg(required = true, value_name = "FILES")]
    pub input: Vec<PathBuf>,

    /// Classifications CSV file (from classify command)
    #[arg(long, required = true, value_name = "FILE")]
    pub classifications: PathBuf,

    /// Output directory for demultiplexed files
    #[arg(short = 'd', long, required = true, value_name = "DIR")]
    pub output_dir: PathBuf,

    /// Prefix for output filenames
    #[arg(long, default_value = "barcode", value_name = "PREFIX")]
    pub prefix: String,

    /// Only write classified reads; drop unclassified instead of writing them
    /// to a separate file
    #[arg(long)]
    pub classified_only: bool,

    /// Number of threads for parallel processing (default: all CPUs)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// Statistics for a single barcode output.
struct BarcodeOutput {
    barcode: String,
    read_count: u64,
    file_size: u64,
}

/// Run the split subcommand.
pub fn run(args: SplitArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Split");
    let profile = args.profile;
    info!(
        "{} reads into separate POD5 files by barcode",
        style::action("Splitting"),
    );
    info!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    info!(
        "{} {}",
        style::label("Classifications:"),
        style::path(args.classifications.display())
    );
    info!(
        "{} {}",
        style::label("Output dir:"),
        style::path(args.output_dir.display())
    );

    // Set thread pool size
    configure_thread_pool(args.threads);

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output_dir)?;

    // Parse the classifications CSV
    info!("{} classifications...", style::action("Loading"));
    let barcode_mapping = parse_barcode_mapping(&args.classifications)?;
    info!(
        "{} {} classified reads",
        style::label("Loaded:"),
        style::count(barcode_mapping.len())
    );

    // Build the read_id -> output-filename map the single-pass splitter wants,
    // plus a filename -> barcode label lookup for the summary. Empty barcodes
    // fold into "unclassified"; those reads are dropped (never added to the
    // map) when --unclassified=false.
    let mut read_to_group: HashMap<Uuid, String> = HashMap::with_capacity(barcode_mapping.len());
    let mut file_to_barcode: HashMap<String, String> = HashMap::new();
    let mut unique_barcodes: HashSet<&str> = HashSet::new();
    let mut skipped_unclassified: u64 = 0;
    for (read_id, barcode) in &barcode_mapping {
        let barcode_key = if barcode.is_empty() {
            "unclassified"
        } else {
            barcode.as_str()
        };
        unique_barcodes.insert(barcode_key);

        if barcode_key == "unclassified" && args.classified_only {
            skipped_unclassified += 1;
            continue;
        }

        let filename = format!("{}_{}.pod5", args.prefix, barcode_key);
        file_to_barcode
            .entry(filename.clone())
            .or_insert_with(|| barcode_key.to_string());
        read_to_group.insert(*read_id, filename);
    }

    info!(
        "{} {} unique barcodes",
        style::label("Found:"),
        style::count(unique_barcodes.len())
    );
    if skipped_unclassified > 0 {
        warn!(
            "{} {} unclassified reads (--classified-only)",
            style::warning_label("Skipping"),
            style::count(skipped_unclassified)
        );
    }

    // Single pass over the inputs: scan each once, route every read to its
    // barcode's writer, rather than re-scanning every input once per barcode.
    let options = FilterOptions {
        signal_batch_size: 1_000,
        read_batch_size: 10_000,
    };
    let mut results = subset_files(&args.input, &read_to_group, &args.output_dir, options)?;
    // Deterministic report order (group write order is nondeterministic).
    results.sort_by(|a, b| a.0.cmp(&b.0));

    let mut barcode_stats = Vec::with_capacity(results.len());
    for (filename, read_count) in &results {
        let output_path = args.output_dir.join(filename);
        let file_size = fs::metadata(&output_path)?.len();
        let barcode = file_to_barcode
            .get(filename)
            .cloned()
            .unwrap_or_else(|| filename.clone());
        barcode_stats.push(BarcodeOutput {
            barcode,
            read_count: *read_count,
            file_size,
        });
    }

    // Print summary table
    print_summary(&barcode_stats);

    timer.report(profile);

    Ok(())
}

/// Print a summary table of all processed barcodes.
fn print_summary(stats: &[BarcodeOutput]) {
    // Styled multi-line report; gate on verbosity instead of per-line tracing events.
    if tracing::enabled!(tracing::Level::INFO) {
        println!("\n{}", style::action("Summary:"));
        println!("{}", style::label("─".repeat(60)));

        let mut builder = Builder::default();
        builder.push_record(vec!["Barcode", "Reads", "File Size"]);

        for output in stats {
            let size_mb = (output.file_size as f64) / (1024.0 * 1024.0);
            builder.push_record(vec![
                output.barcode.clone(),
                format!("{}", output.read_count),
                format!("{:.2} MB", size_mb),
            ]);
        }

        let mut table = builder.build();
        table.with(Style::rounded());
        println!("{}", table);

        let total_reads: u64 = stats.iter().map(|s| s.read_count).sum();
        let total_size: u64 = stats.iter().map(|s| s.file_size).sum();
        let total_size_mb = (total_size as f64) / (1024.0 * 1024.0);

        println!("\n{}", style::label("─".repeat(60)));
        println!(
            "{} {} reads across {} files ({:.2} MB total)",
            style::action("Split"),
            style::count(total_reads),
            style::count(stats.len()),
            total_size_mb
        );
    }
}

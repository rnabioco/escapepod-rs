//! Split subcommand - split reads into separate POD5 files by barcode.

use super::utils::configure_thread_pool;
use crate::style;
use escapepod_signal::operations::{FilterOptions, filter_files, parse_barcode_mapping};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use tabled::{builder::Builder, settings::Style};
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

    /// Include unclassified reads in a separate file
    #[arg(long, default_value = "true", value_name = "BOOL")]
    pub unclassified: bool,

    /// Number of threads for parallel processing
    #[arg(short = 'j', long, default_value = "4", value_name = "N")]
    pub threads: usize,
}

/// Statistics for a single barcode output.
struct BarcodeOutput {
    barcode: String,
    read_count: u64,
    file_size: u64,
}

/// Run the split subcommand.
pub fn run(args: SplitArgs) -> anyhow::Result<()> {
    println!(
        "{} reads into separate POD5 files by barcode",
        style::action("Splitting"),
    );
    println!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    println!(
        "{} {}",
        style::label("Classifications:"),
        style::path(args.classifications.display())
    );
    println!(
        "{} {}",
        style::label("Output dir:"),
        style::path(args.output_dir.display())
    );

    // Set thread pool size
    configure_thread_pool(args.threads);

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output_dir)?;

    // Parse the classifications CSV
    println!("{} classifications...", style::action("Loading"));
    let barcode_mapping = parse_barcode_mapping(&args.classifications)?;
    println!(
        "{} {} classified reads",
        style::label("Loaded:"),
        style::count(barcode_mapping.len())
    );

    // Group read IDs by barcode
    let barcode_groups = group_by_barcode(&barcode_mapping);

    // Get sorted list of barcodes for consistent processing order
    let mut barcodes: Vec<String> = barcode_groups.keys().cloned().collect();
    barcodes.sort();

    println!(
        "{} {} unique barcodes",
        style::label("Found:"),
        style::count(barcodes.len())
    );

    // Process each barcode
    let mut barcode_stats = Vec::new();

    for barcode in &barcodes {
        // Skip unclassified if requested
        if barcode == "unclassified" && !args.unclassified {
            let count = barcode_groups.get(barcode).map(|s| s.len()).unwrap_or(0);
            println!(
                "{} {} unclassified reads (--unclassified=false)",
                style::warning_label("Skipping"),
                style::count(count)
            );
            continue;
        }

        let read_ids = barcode_groups.get(barcode).unwrap();
        let output = process_barcode(&args, barcode, read_ids)?;
        barcode_stats.push(output);
    }

    // Print summary table
    print_summary(&barcode_stats);

    Ok(())
}

/// Group read IDs by their barcode assignment.
fn group_by_barcode(mapping: &HashMap<Uuid, String>) -> HashMap<String, HashSet<Uuid>> {
    let mut groups: HashMap<String, HashSet<Uuid>> = HashMap::new();

    for (read_id, barcode) in mapping {
        let barcode_key = if barcode.is_empty() {
            "unclassified".to_string()
        } else {
            barcode.clone()
        };
        groups.entry(barcode_key).or_default().insert(*read_id);
    }

    groups
}

/// Process a single barcode, writing matching reads to a new POD5 file.
fn process_barcode(
    args: &SplitArgs,
    barcode: &str,
    read_ids: &HashSet<Uuid>,
) -> anyhow::Result<BarcodeOutput> {
    let output_filename = format!("{}_{}.pod5", args.prefix, barcode);
    let output_path = args.output_dir.join(&output_filename);

    println!(
        "{} {} ({} reads)...",
        style::action("Processing"),
        style::value(barcode),
        style::count(read_ids.len())
    );

    // Use the filter operation to extract reads for this barcode
    let options = FilterOptions {
        signal_batch_size: 1_000,
        read_batch_size: 10_000,
    };

    let result = filter_files(&args.input, &output_path, read_ids, options, None)?;

    // Get file size
    let file_size = fs::metadata(&output_path)?.len();

    println!(
        "  {} {} reads to {}",
        style::action("Wrote"),
        style::count(result.matched_reads),
        style::path(&output_filename)
    );

    Ok(BarcodeOutput {
        barcode: barcode.to_string(),
        read_count: result.matched_reads,
        file_size,
    })
}

/// Print a summary table of all processed barcodes.
fn print_summary(stats: &[BarcodeOutput]) {
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

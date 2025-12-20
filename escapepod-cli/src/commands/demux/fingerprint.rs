//! Fingerprint subcommand - extract signal features from adapter regions.

use super::types::ReadFingerprint;
use super::utils::{
    configure_thread_pool, extract_fingerprint_from_signal, parse_boundaries_csv, parse_norm_method,
};
use crate::progress::create_progress_bar;
use crate::style;
use escapepod::Reader;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use uuid::Uuid;

/// Arguments for the fingerprint subcommand.
#[derive(Debug, clap::Args)]
pub struct FingerprintArgs {
    /// Input POD5 file(s)
    #[arg(required = true, value_name = "FILES")]
    pub input: Vec<PathBuf>,

    /// Detected boundaries CSV (from detect command)
    #[arg(long, required = true, value_name = "FILE")]
    pub boundaries: PathBuf,

    /// Output fingerprints file
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// Number of segments for fingerprint
    #[arg(long, default_value = "10", value_name = "N")]
    pub num_segments: usize,

    /// Window width for t-test segmentation
    #[arg(long, default_value = "5", value_name = "N")]
    pub window_width: usize,

    /// Normalization method (zscore, minmax, median, none)
    #[arg(long, default_value = "zscore", value_name = "METHOD")]
    pub normalize: String,

    /// Number of threads for parallel processing
    #[arg(short = 'j', long, default_value = "4", value_name = "N")]
    pub threads: usize,
}

/// Run the fingerprint subcommand.
pub fn run(args: FingerprintArgs) -> anyhow::Result<()> {
    println!("{} barcode fingerprints", style::action("Extracting"));
    println!(
        "{} {} POD5 file(s)",
        style::label("Input:"),
        style::count(args.input.len())
    );
    println!(
        "{} {}",
        style::label("Boundaries:"),
        style::path(args.boundaries.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );

    // Parse normalization method
    let norm_method = parse_norm_method(&args.normalize)?;

    // Set thread pool size
    configure_thread_pool(args.threads);

    // Read boundaries CSV
    let boundaries_map = parse_boundaries_csv(&args.boundaries)?;

    println!(
        "{} {} boundary records with valid adapters",
        style::label("Loaded:"),
        style::count(boundaries_map.len())
    );

    // Collect reads that have boundaries
    let mut reads_to_process: Vec<(Uuid, usize, usize, Vec<i16>)> = Vec::new();

    for path in &args.input {
        let reader = Reader::open(path)?;
        if let Ok(reads) = reader.reads() {
            for read_result in reads {
                let read = read_result?;
                if let Some(boundaries) = boundaries_map.get(&read.read_id) {
                    if !read.signal_rows.is_empty() {
                        if let Ok(signal) = reader.get_signal(&read.signal_rows) {
                            reads_to_process.push((
                                read.read_id,
                                boundaries.adapter_start,
                                boundaries.adapter_end,
                                signal,
                            ));
                        }
                    }
                }
            }
        }
    }

    println!(
        "{} {} reads to fingerprint",
        style::label("Processing:"),
        style::count(reads_to_process.len())
    );

    let progress_bar = create_progress_bar(reads_to_process.len() as u64, "Fingerprinting")?;

    // Process reads in parallel
    let fingerprints: Vec<ReadFingerprint> = reads_to_process
        .par_iter()
        .filter_map(|(read_id, adapter_start, adapter_end, signal)| {
            let result = extract_fingerprint_from_signal(
                signal,
                *adapter_start,
                *adapter_end,
                args.num_segments,
                args.window_width,
                norm_method,
                *read_id,
            );

            progress_bar.inc(1);
            result
        })
        .collect();

    progress_bar.finish_with_message("complete");

    // Write fingerprints
    write_fingerprints_csv(&args.output, &fingerprints)?;

    println!(
        "{} {} fingerprints written to {}",
        style::action("Extracted"),
        style::count(fingerprints.len()),
        style::path(args.output.display())
    );

    Ok(())
}

/// Write fingerprints to a CSV file.
fn write_fingerprints_csv(path: &PathBuf, fingerprints: &[ReadFingerprint]) -> anyhow::Result<()> {
    let output_file = File::create(path)?;
    let mut writer = BufWriter::new(output_file);

    // Header: read_id,fp_0,fp_1,...,fp_n
    write!(writer, "read_id")?;
    if let Some(first_fp) = fingerprints.first() {
        for i in 0..first_fp.values.len() {
            write!(writer, ",fp_{}", i)?;
        }
    }
    writeln!(writer)?;

    // Data rows
    for fp in fingerprints {
        write!(writer, "{}", fp.read_id)?;
        for val in &fp.values {
            write!(writer, ",{:.6}", val)?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

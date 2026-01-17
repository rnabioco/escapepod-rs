//! Fingerprint subcommand - extract signal features from adapter regions.

use super::types::{ReadBoundaries, ReadFingerprint};
use super::utils::{
    collect_read_metadata, configure_thread_pool, extract_fingerprint_from_signal, open_readers,
    parse_boundaries_csv, parse_norm_method, ReadMetadata,
};
use crate::progress::create_progress_bar;
use crate::style;
use escapepod::dtw::NormMethod;
use escapepod::segmentation::{
    mad_normalize_with_clipping, segment_with_consensus, ConsensusConfig,
    CONSENSUS_RNA004_130BPS_V1_0,
};
use escapepod::Reader;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
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

    /// Start sample offset within adapter region for fingerprinting
    #[arg(long, default_value = "1000", value_name = "N")]
    pub segment_start: usize,

    /// End sample offset within adapter region for fingerprinting
    #[arg(long, default_value = "2000", value_name = "N")]
    pub segment_end: usize,

    /// Number of segments for fingerprint
    #[arg(long, default_value = "10", value_name = "N")]
    pub num_segments: usize,

    /// Window width for t-test segmentation
    #[arg(long, default_value = "5", value_name = "N")]
    pub window_width: usize,

    /// Normalization method (zscore, minmax, median, none)
    #[arg(long, default_value = "zscore", value_name = "METHOD")]
    pub normalize: String,

    /// Use consensus-guided segmentation (WarpDemuX-compatible)
    ///
    /// This uses DTW alignment to find the barcode region within the adapter,
    /// matching WarpDemuX's fingerprint extraction algorithm.
    #[arg(long)]
    pub consensus: bool,

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
    if args.consensus {
        println!(
            "{} consensus-guided (WarpDemuX-compatible)",
            style::label("Mode:")
        );
    }

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

    // Open all POD5 files - Arc-wrapped for parallel access
    let readers = open_readers(&args.input)?;

    // Collect read metadata (fast - no signal decompression)
    let all_metadata = collect_read_metadata(&readers)?;

    // Filter to only reads with boundaries
    let reads_to_process: Vec<_> = all_metadata
        .into_iter()
        .filter(|meta| boundaries_map.contains_key(&meta.read_id))
        .collect();

    println!(
        "{} {} reads to fingerprint",
        style::label("Processing:"),
        style::count(reads_to_process.len())
    );

    let progress_bar = create_progress_bar(reads_to_process.len() as u64, "Fingerprinting")?;

    // Process reads in parallel - signal decompression is now parallelized
    let fingerprints: Vec<ReadFingerprint> = if args.consensus {
        // Consensus-guided segmentation (WarpDemuX-compatible)
        let consensus_config = ConsensusConfig::default();
        reads_to_process
            .par_iter()
            .filter_map(|meta| {
                process_fingerprint_consensus(
                    &readers,
                    meta,
                    &boundaries_map,
                    &consensus_config,
                    &progress_bar,
                )
            })
            .collect()
    } else {
        // Original t-test based segmentation
        reads_to_process
            .par_iter()
            .filter_map(|meta| {
                process_fingerprint(
                    &readers,
                    meta,
                    &boundaries_map,
                    &args,
                    norm_method,
                    &progress_bar,
                )
            })
            .collect()
    };

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

/// Process a single read: decompress signal, extract fingerprint.
fn process_fingerprint(
    readers: &[Arc<Reader>],
    meta: &ReadMetadata,
    boundaries_map: &HashMap<Uuid, ReadBoundaries>,
    args: &FingerprintArgs,
    norm_method: NormMethod,
    progress_bar: &indicatif::ProgressBar,
) -> Option<ReadFingerprint> {
    // Get boundaries for this read
    let boundaries = boundaries_map.get(&meta.read_id)?;

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

    // Compute the region within the adapter for fingerprinting
    let region_start = boundaries.adapter_start + args.segment_start;
    let region_end = (boundaries.adapter_start + args.segment_end).min(boundaries.adapter_end);

    if region_end <= region_start {
        progress_bar.inc(1);
        return None;
    }

    let result = extract_fingerprint_from_signal(
        &signal,
        region_start,
        region_end,
        args.num_segments,
        args.window_width,
        norm_method,
        meta.read_id,
    );

    progress_bar.inc(1);
    result
}

/// Process a single read using consensus-guided segmentation.
fn process_fingerprint_consensus(
    readers: &[Arc<Reader>],
    meta: &ReadMetadata,
    boundaries_map: &HashMap<Uuid, ReadBoundaries>,
    config: &ConsensusConfig,
    progress_bar: &indicatif::ProgressBar,
) -> Option<ReadFingerprint> {
    // Get boundaries for this read
    let boundaries = boundaries_map.get(&meta.read_id)?;

    // Get the reader for this file
    let reader = &readers[meta.file_index];

    // Decompress signal
    let signal = match reader.get_signal(&meta.signal_rows) {
        Ok(s) => s,
        Err(_) => {
            progress_bar.inc(1);
            return None;
        }
    };

    // Extract adapter region
    if boundaries.adapter_end <= boundaries.adapter_start {
        progress_bar.inc(1);
        return None;
    }

    let adapter_signal: Vec<f32> = signal[boundaries.adapter_start..boundaries.adapter_end]
        .iter()
        .map(|&x| x as f32)
        .collect();

    // Normalize signal (MAD normalization with clipping, matching WarpDemuX)
    let normalized = mad_normalize_with_clipping(&adapter_signal, 5.0);

    // Apply consensus-guided segmentation
    let result = segment_with_consensus(&normalized, &CONSENSUS_RNA004_130BPS_V1_0, config);

    progress_bar.inc(1);

    result.map(|seg| ReadFingerprint {
        read_id: meta.read_id,
        values: seg.fingerprint.into_iter().map(|x| x as f64).collect(),
    })
}

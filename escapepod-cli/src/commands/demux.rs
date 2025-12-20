//! Demux command implementation.
//!
//! Barcode demultiplexing workflow for Oxford Nanopore reads.
//! Includes adapter detection, barcode fingerprinting, and classification.

use crate::progress::create_progress_bar;
use crate::style;
use escapepod::dtw::{dtw_distance_matrix, normalize_fingerprint, Fingerprint, NormMethod};
use escapepod::segmentation::{detect_adapter, mad_normalize, segment_signal};
use escapepod::Reader;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use uuid::Uuid;

/// Main demux command arguments.
#[derive(Debug, clap::Args)]
pub struct DemuxArgs {
    #[command(subcommand)]
    pub command: DemuxCommand,
}

/// Demux subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum DemuxCommand {
    /// Detect adapter boundaries in reads using LLR algorithm
    #[command(after_help = "\
Examples:
  escapepod demux detect input.pod5 -o boundaries.csv
  escapepod demux detect *.pod5 -o boundaries.csv --min-adapter 200 -j 8
")]
    Detect(DetectArgs),

    /// Extract barcode fingerprints from adapter regions
    #[command(after_help = "\
Examples:
  escapepod demux fingerprint input.pod5 --boundaries boundaries.csv -o fingerprints.csv
  escapepod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv
")]
    Fingerprint(FingerprintArgs),

    /// Classify reads by barcode using DTW distance
    #[command(after_help = "\
Examples:
  escapepod demux classify fingerprints.csv --reference barcodes.csv -o classifications.csv
  escapepod demux classify fingerprints.csv --reference barcodes.csv -o out.csv --window 10
")]
    Classify(ClassifyArgs),
}

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

/// Arguments for the classify subcommand.
#[derive(Debug, clap::Args)]
pub struct ClassifyArgs {
    /// Input fingerprints file
    #[arg(value_name = "FILE")]
    pub fingerprints: PathBuf,

    /// Reference barcode fingerprints (training data)
    #[arg(long, required = true, value_name = "FILE")]
    pub reference: PathBuf,

    /// Output classifications file
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// DTW window constraint (Sakoe-Chiba band width)
    #[arg(long, value_name = "N")]
    pub window: Option<usize>,

    /// Minimum distance ratio for confident classification
    #[arg(long, default_value = "0.8", value_name = "RATIO")]
    pub min_ratio: f32,
}

/// Run the demux command.
pub fn run(args: DemuxArgs) -> anyhow::Result<()> {
    match args.command {
        DemuxCommand::Detect(detect_args) => run_detect(detect_args),
        DemuxCommand::Fingerprint(fingerprint_args) => run_fingerprint(fingerprint_args),
        DemuxCommand::Classify(classify_args) => run_classify(classify_args),
    }
}

/// Run the detect subcommand using LLR boundary detection.
fn run_detect(args: DetectArgs) -> anyhow::Result<()> {
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
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // Collect all reads with their signals
    let mut all_reads: Vec<(Uuid, u64, Vec<i16>)> = Vec::new();

    for path in &args.input {
        let reader = Reader::open(path)?;
        if let Ok(reads) = reader.reads() {
            for read_result in reads {
                let read = read_result?;
                // Get the signal data using signal_rows
                if !read.signal_rows.is_empty() {
                    if let Ok(signal) = reader.get_signal(&read.signal_rows) {
                        all_reads.push((read.read_id, read.num_samples, signal));
                    }
                }
            }
        }
    }

    println!(
        "{} {} reads to process",
        style::label("Found:"),
        style::count(all_reads.len())
    );

    let progress_bar = create_progress_bar(all_reads.len() as u64, "Detecting")?;

    // Process reads in parallel
    let results: Vec<(Uuid, u64, usize, usize)> = all_reads
        .par_iter()
        .map(|(read_id, num_samples, signal)| {
            // Convert i16 signal to f32
            let signal_f32: Vec<f32> = signal.iter().map(|&s| s as f32).collect();

            // Apply MAD normalization
            let normalized = if signal_f32.len() > 10 {
                mad_normalize(&signal_f32)
            } else {
                signal_f32
            };

            // Detect adapter using LLR
            let (adapter_start, adapter_end) =
                detect_adapter(&normalized, args.min_adapter, args.border_trim);

            progress_bar.inc(1);

            (*read_id, *num_samples, adapter_start, adapter_end)
        })
        .collect();

    progress_bar.finish_with_message("complete");

    // Write results
    let output_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(output_file);

    writeln!(writer, "read_id,num_samples,adapter_start,adapter_end")?;

    let mut detected_count = 0;
    for (read_id, num_samples, adapter_start, adapter_end) in results {
        writeln!(
            writer,
            "{},{},{},{}",
            read_id, num_samples, adapter_start, adapter_end
        )?;
        if adapter_end > adapter_start {
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

/// Run the fingerprint subcommand - extract signal features from adapter regions.
fn run_fingerprint(args: FingerprintArgs) -> anyhow::Result<()> {
    println!("{} barcode fingerprints", style::action("Extracting"),);
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
    let norm_method = match args.normalize.to_lowercase().as_str() {
        "zscore" => NormMethod::ZScore,
        "minmax" => NormMethod::MinMax,
        "median" => NormMethod::Median,
        "none" => NormMethod::None,
        _ => {
            anyhow::bail!(
                "Invalid normalization method: {}. Use zscore, minmax, median, or none",
                args.normalize
            );
        }
    };

    // Set thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // Read boundaries CSV
    let boundaries_file = File::open(&args.boundaries)?;
    let boundaries_reader = BufReader::new(boundaries_file);

    let mut boundaries_map: HashMap<Uuid, (u64, usize, usize)> = HashMap::new();
    let mut line_count = 0;

    for line in boundaries_reader.lines() {
        let line = line?;
        line_count += 1;

        // Skip header
        if line_count == 1 {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            if let Ok(read_id) = Uuid::parse_str(parts[0]) {
                let num_samples = parts[1].parse::<u64>().unwrap_or(0);
                let adapter_start = parts[2].parse::<usize>().unwrap_or(0);
                let adapter_end = parts[3].parse::<usize>().unwrap_or(0);
                if adapter_end > adapter_start {
                    boundaries_map.insert(read_id, (num_samples, adapter_start, adapter_end));
                }
            }
        }
    }

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
                if let Some(&(_, adapter_start, adapter_end)) = boundaries_map.get(&read.read_id) {
                    if !read.signal_rows.is_empty() {
                        if let Ok(signal) = reader.get_signal(&read.signal_rows) {
                            reads_to_process.push((read.read_id, adapter_start, adapter_end, signal));
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
    let fingerprints: Vec<(Uuid, Vec<f64>)> = reads_to_process
        .par_iter()
        .filter_map(|(read_id, adapter_start, adapter_end, signal)| {
            // Extract adapter region
            let start = *adapter_start;
            let end = (*adapter_end).min(signal.len());

            if end <= start || end - start < args.window_width * 2 {
                progress_bar.inc(1);
                return None;
            }

            // Convert to f32
            let adapter_signal: Vec<f32> = signal[start..end].iter().map(|&s| s as f32).collect();

            // Segment the adapter region
            let segments = segment_signal(
                &adapter_signal,
                args.window_width,
                args.num_segments.saturating_sub(1),
                args.window_width,
            );

            if segments.is_empty() {
                progress_bar.inc(1);
                return None;
            }

            // Extract segment means as fingerprint
            let fingerprint_values: Vec<f32> = segments.iter().map(|(_, _, mean)| *mean as f32).collect();

            // Normalize the fingerprint
            let mut fp = Fingerprint::new(fingerprint_values, *read_id);
            normalize_fingerprint(&mut fp, norm_method);

            progress_bar.inc(1);

            Some((*read_id, fp.values.iter().map(|&v| v as f64).collect()))
        })
        .collect();

    progress_bar.finish_with_message("complete");

    // Write fingerprints
    let output_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(output_file);

    // Header: read_id,fp_0,fp_1,...,fp_n
    write!(writer, "read_id")?;
    if let Some((_, first_fp)) = fingerprints.first() {
        for i in 0..first_fp.len() {
            write!(writer, ",fp_{}", i)?;
        }
    }
    writeln!(writer)?;

    // Data rows
    for (read_id, fp_values) in &fingerprints {
        write!(writer, "{}", read_id)?;
        for val in fp_values {
            write!(writer, ",{:.6}", val)?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;

    println!(
        "{} {} fingerprints written to {}",
        style::action("Extracted"),
        style::count(fingerprints.len()),
        style::path(args.output.display())
    );

    Ok(())
}

/// Run the classify subcommand using DTW distance.
fn run_classify(args: ClassifyArgs) -> anyhow::Result<()> {
    println!("{} reads by barcode using DTW", style::action("Classifying"),);
    println!(
        "{} {}",
        style::label("Fingerprints:"),
        style::path(args.fingerprints.display())
    );
    println!(
        "{} {}",
        style::label("Reference:"),
        style::path(args.reference.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );
    if let Some(w) = args.window {
        println!("{} {}", style::label("DTW window:"), style::value(w));
    }

    // Read reference fingerprints
    let ref_file = File::open(&args.reference)?;
    let ref_reader = BufReader::new(ref_file);

    let mut reference_fps: Vec<(String, Vec<f32>)> = Vec::new();
    let mut header_seen = false;

    for line in ref_reader.lines() {
        let line = line?;
        if !header_seen {
            header_seen = true;
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let barcode_name = parts[0].to_string();
            let values: Vec<f32> = parts[1..]
                .iter()
                .filter_map(|s| s.parse::<f32>().ok())
                .collect();
            if !values.is_empty() {
                reference_fps.push((barcode_name, values));
            }
        }
    }

    println!(
        "{} {} reference barcodes",
        style::label("Loaded:"),
        style::count(reference_fps.len())
    );

    if reference_fps.is_empty() {
        anyhow::bail!("No valid reference fingerprints found");
    }

    // Read query fingerprints
    let query_file = File::open(&args.fingerprints)?;
    let query_reader = BufReader::new(query_file);

    let mut query_fps: Vec<(Uuid, Vec<f32>)> = Vec::new();
    header_seen = false;

    for line in query_reader.lines() {
        let line = line?;
        if !header_seen {
            header_seen = true;
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(read_id) = Uuid::parse_str(parts[0]) {
                let values: Vec<f32> = parts[1..]
                    .iter()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                if !values.is_empty() {
                    query_fps.push((read_id, values));
                }
            }
        }
    }

    println!(
        "{} {} query fingerprints",
        style::label("Loaded:"),
        style::count(query_fps.len())
    );

    if query_fps.is_empty() {
        anyhow::bail!("No valid query fingerprints found");
    }

    // Extract values for DTW computation
    let query_values: Vec<Vec<f32>> = query_fps.iter().map(|(_, v)| v.clone()).collect();
    let ref_values: Vec<Vec<f32>> = reference_fps.iter().map(|(_, v)| v.clone()).collect();

    println!("{} DTW distances...", style::action("Computing"));

    // Compute distance matrix
    let distances = dtw_distance_matrix(&query_values, &ref_values, args.window);

    // Classify each query
    let output_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(output_file);

    writeln!(
        writer,
        "read_id,barcode,distance,second_best_distance,ratio,confident"
    )?;

    let mut confident_count = 0;
    let mut unclassified_count = 0;

    for (i, (read_id, _)) in query_fps.iter().enumerate() {
        let row = distances.row(i);

        // Find best and second-best matches
        let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (best_idx, best_dist) = indexed[0];
        let second_best_dist = if indexed.len() > 1 {
            indexed[1].1
        } else {
            f32::INFINITY
        };

        let ratio = if second_best_dist > 0.0 {
            best_dist / second_best_dist
        } else {
            0.0
        };

        let confident = ratio <= args.min_ratio;
        let barcode_name = &reference_fps[best_idx].0;

        if confident {
            confident_count += 1;
        } else {
            unclassified_count += 1;
        }

        writeln!(
            writer,
            "{},{},{:.4},{:.4},{:.4},{}",
            read_id, barcode_name, best_dist, second_best_dist, ratio, confident
        )?;
    }

    writer.flush()?;

    println!(
        "{} classifications written to {}",
        style::action("Wrote"),
        style::path(args.output.display())
    );
    println!(
        "{} {} confident, {} unclassified",
        style::label("Result:"),
        style::count(confident_count),
        style::warning(unclassified_count)
    );

    Ok(())
}

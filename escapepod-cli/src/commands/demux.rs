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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use uuid::Uuid;
use walkdir::WalkDir;

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

    /// Train reference barcode fingerprints from known samples
    #[command(after_help = "\
Examples:
  escapepod demux train --input-dir barcodes/ -o reference.json
  escapepod demux train --assignments assignments.csv -o reference.json
")]
    Train(TrainArgs),
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

/// Arguments for the train subcommand.
#[derive(Debug, clap::Args)]
pub struct TrainArgs {
    /// Input directory with barcode subdirectories (mutually exclusive with --assignments)
    #[arg(long, value_name = "DIR", conflicts_with = "assignments")]
    pub input_dir: Option<PathBuf>,

    /// CSV file with read_id,barcode,pod5_file columns (mutually exclusive with --input-dir)
    #[arg(long, value_name = "FILE", conflicts_with = "input_dir")]
    pub assignments: Option<PathBuf>,

    /// Output JSON file for reference fingerprints
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// Start sample for fingerprint region
    #[arg(long, default_value = "1000", value_name = "N")]
    pub segment_start: usize,

    /// End sample for fingerprint region
    #[arg(long, default_value = "2000", value_name = "N")]
    pub segment_end: usize,

    /// Number of segments for fingerprinting
    #[arg(long, default_value = "10", value_name = "N")]
    pub num_segments: usize,

    /// Window width for t-test segmentation
    #[arg(long, default_value = "5", value_name = "N")]
    pub window_width: usize,

    /// Normalization method (zscore, minmax, median, none)
    #[arg(long, default_value = "zscore", value_name = "METHOD")]
    pub normalize: String,

    /// Minimum observations for adapter segment
    #[arg(long, default_value = "200", value_name = "N")]
    pub min_adapter: usize,

    /// Border trim size for adapter detection
    #[arg(long, default_value = "50", value_name = "N")]
    pub border_trim: usize,

    /// Number of threads for parallel processing
    #[arg(short = 'j', long, default_value = "4", value_name = "N")]
    pub threads: usize,
}

/// Run the demux command.
pub fn run(args: DemuxArgs) -> anyhow::Result<()> {
    match args.command {
        DemuxCommand::Detect(detect_args) => run_detect(detect_args),
        DemuxCommand::Fingerprint(fingerprint_args) => run_fingerprint(fingerprint_args),
        DemuxCommand::Classify(classify_args) => run_classify(classify_args),
        DemuxCommand::Train(train_args) => run_train(train_args),
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

/// Barcode statistics for training output.
#[derive(Debug, Serialize, Deserialize)]
struct BarcodeStats {
    fingerprint: Vec<f64>,
    read_count: usize,
    std_dev: Vec<f64>,
}

/// Training parameters.
#[derive(Debug, Serialize, Deserialize)]
struct TrainParams {
    segment_start: usize,
    segment_end: usize,
    num_segments: usize,
}

/// Training output JSON structure.
#[derive(Debug, Serialize, Deserialize)]
struct TrainingOutput {
    barcodes: HashMap<String, BarcodeStats>,
    params: TrainParams,
}

/// Run the train subcommand - generate reference fingerprints from known samples.
fn run_train(args: TrainArgs) -> anyhow::Result<()> {
    println!("{} reference barcode fingerprints", style::action("Training"));

    // Validate that either input_dir or assignments is provided
    if args.input_dir.is_none() && args.assignments.is_none() {
        anyhow::bail!("Either --input-dir or --assignments must be provided");
    }

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

    // Collect barcode assignments: read_id -> (barcode, pod5_path)
    let assignments = if let Some(input_dir) = &args.input_dir {
        collect_assignments_from_directory(input_dir)?
    } else if let Some(assignments_file) = &args.assignments {
        collect_assignments_from_csv(assignments_file)?
    } else {
        unreachable!()
    };

    println!(
        "{} {} read assignments across {} barcodes",
        style::label("Loaded:"),
        style::count(assignments.len()),
        style::count(
            assignments
                .values()
                .map(|(bc, _)| bc.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        )
    );

    // Group reads by POD5 file for efficient reading
    let mut reads_by_file: HashMap<PathBuf, Vec<(Uuid, String)>> = HashMap::new();
    for (read_id, (barcode, pod5_path)) in &assignments {
        reads_by_file
            .entry(pod5_path.clone())
            .or_insert_with(Vec::new)
            .push((*read_id, barcode.clone()));
    }

    println!(
        "{} {} POD5 files to process",
        style::label("Files:"),
        style::count(reads_by_file.len())
    );

    // Extract fingerprints for all reads
    let all_fingerprints = extract_fingerprints_from_assignments(
        &reads_by_file,
        &args,
        norm_method,
    )?;

    // Group fingerprints by barcode
    let mut barcode_fingerprints: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for (read_id, fingerprint) in &all_fingerprints {
        if let Some((barcode, _)) = assignments.get(read_id) {
            barcode_fingerprints
                .entry(barcode.clone())
                .or_insert_with(Vec::new)
                .push(fingerprint.clone());
        }
    }

    println!(
        "{} {} total fingerprints extracted",
        style::label("Extracted:"),
        style::count(all_fingerprints.len())
    );

    // Compute consensus fingerprints
    let mut training_output = TrainingOutput {
        barcodes: HashMap::new(),
        params: TrainParams {
            segment_start: args.segment_start,
            segment_end: args.segment_end,
            num_segments: args.num_segments,
        },
    };

    for (barcode, fingerprints) in barcode_fingerprints {
        let consensus = compute_consensus_fingerprint(&fingerprints);
        let std_dev = compute_std_dev_fingerprint(&fingerprints, &consensus);

        training_output.barcodes.insert(
            barcode.clone(),
            BarcodeStats {
                fingerprint: consensus.iter().map(|&v| v as f64).collect(),
                read_count: fingerprints.len(),
                std_dev: std_dev.iter().map(|&v| v as f64).collect(),
            },
        );

        println!(
            "{} {} fingerprints from {} reads",
            style::label(&format!("{}:", barcode)),
            style::action("Computed consensus"),
            style::count(fingerprints.len())
        );
    }

    // Write JSON output
    let output_file = File::create(&args.output)?;
    let writer = BufWriter::new(output_file);
    serde_json::to_writer_pretty(writer, &training_output)?;

    println!(
        "{} reference fingerprints written to {}",
        style::action("Trained"),
        style::path(args.output.display())
    );
    println!(
        "{} {} barcodes",
        style::label("Total:"),
        style::count(training_output.barcodes.len())
    );

    Ok(())
}

/// Collect read assignments from directory structure.
/// Each subdirectory represents a barcode, containing POD5 files.
fn collect_assignments_from_directory(
    input_dir: &PathBuf,
) -> anyhow::Result<HashMap<Uuid, (String, PathBuf)>> {
    let mut assignments = HashMap::new();

    // Iterate through subdirectories
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let barcode = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| anyhow::anyhow!("Invalid directory name"))?
                .to_string();

            // Find all POD5 files in this barcode directory
            for pod5_entry in WalkDir::new(&path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s == "pod5")
                        .unwrap_or(false)
                })
            {
                let pod5_path = pod5_entry.path().to_path_buf();
                let reader = Reader::open(&pod5_path)?;

                if let Ok(reads) = reader.reads() {
                    for read_result in reads {
                        let read = read_result?;
                        assignments.insert(read.read_id, (barcode.clone(), pod5_path.clone()));
                    }
                }
            }
        }
    }

    Ok(assignments)
}

/// Collect read assignments from CSV file.
/// Expected columns: read_id, barcode, pod5_file
fn collect_assignments_from_csv(
    csv_path: &PathBuf,
) -> anyhow::Result<HashMap<Uuid, (String, PathBuf)>> {
    let mut assignments = HashMap::new();
    let file = File::open(csv_path)?;
    let reader = BufReader::new(file);

    let mut line_count = 0;
    for line in reader.lines() {
        let line = line?;
        line_count += 1;

        // Skip header
        if line_count == 1 {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            if let Ok(read_id) = Uuid::parse_str(parts[0]) {
                let barcode = parts[1].to_string();
                let pod5_file = PathBuf::from(parts[2]);
                assignments.insert(read_id, (barcode, pod5_file));
            }
        }
    }

    Ok(assignments)
}

/// Extract fingerprints from reads based on assignments.
fn extract_fingerprints_from_assignments(
    reads_by_file: &HashMap<PathBuf, Vec<(Uuid, String)>>,
    args: &TrainArgs,
    norm_method: NormMethod,
) -> anyhow::Result<HashMap<Uuid, Vec<f32>>> {
    let total_reads: usize = reads_by_file.values().map(|v| v.len()).sum();
    let progress_bar = create_progress_bar(total_reads as u64, "Processing")?;

    let all_fingerprints: HashMap<Uuid, Vec<f32>> = reads_by_file
        .par_iter()
        .flat_map(|(pod5_path, read_list)| {
            let mut fingerprints = Vec::new();
            let read_ids: std::collections::HashSet<Uuid> =
                read_list.iter().map(|(id, _)| *id).collect();

            if let Ok(reader) = Reader::open(pod5_path) {
                if let Ok(reads) = reader.reads() {
                    for read_result in reads {
                        if let Ok(read) = read_result {
                            if read_ids.contains(&read.read_id) && !read.signal_rows.is_empty() {
                                if let Ok(signal) = reader.get_signal(&read.signal_rows) {
                                    // Convert i16 signal to f32
                                    let signal_f32: Vec<f32> =
                                        signal.iter().map(|&s| s as f32).collect();

                                    // Apply MAD normalization
                                    let normalized = if signal_f32.len() > 10 {
                                        mad_normalize(&signal_f32)
                                    } else {
                                        signal_f32
                                    };

                                    // Detect adapter using LLR
                                    let (adapter_start, adapter_end) = detect_adapter(
                                        &normalized,
                                        args.min_adapter,
                                        args.border_trim,
                                    );

                                    if adapter_end > adapter_start {
                                        // Extract the specified region
                                        let region_start = adapter_start + args.segment_start;
                                        let region_end =
                                            (adapter_start + args.segment_end).min(adapter_end);

                                        if region_end > region_start
                                            && region_end <= normalized.len()
                                        {
                                            let region_signal =
                                                &normalized[region_start..region_end];

                                            // Segment the region
                                            let segments = segment_signal(
                                                region_signal,
                                                args.window_width,
                                                args.num_segments.saturating_sub(1),
                                                args.window_width,
                                            );

                                            if !segments.is_empty() {
                                                // Extract segment means as fingerprint
                                                let fingerprint_values: Vec<f32> = segments
                                                    .iter()
                                                    .map(|(_, _, mean)| *mean as f32)
                                                    .collect();

                                                // Normalize the fingerprint
                                                let mut fp = Fingerprint::new(
                                                    fingerprint_values,
                                                    read.read_id,
                                                );
                                                normalize_fingerprint(&mut fp, norm_method);

                                                fingerprints
                                                    .push((read.read_id, fp.values.clone()));
                                            }
                                        }
                                    }
                                }
                            }
                            progress_bar.inc(1);
                        }
                    }
                }
            }

            fingerprints
        })
        .collect();

    progress_bar.finish_with_message("complete");

    Ok(all_fingerprints)
}

/// Compute consensus fingerprint as element-wise median.
fn compute_consensus_fingerprint(fingerprints: &[Vec<f32>]) -> Vec<f32> {
    if fingerprints.is_empty() {
        return Vec::new();
    }

    let length = fingerprints[0].len();
    let mut consensus = Vec::with_capacity(length);

    for i in 0..length {
        let mut values: Vec<f32> = fingerprints.iter().map(|fp| fp[i]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };

        consensus.push(median);
    }

    consensus
}

/// Compute element-wise standard deviation.
fn compute_std_dev_fingerprint(fingerprints: &[Vec<f32>], consensus: &[f32]) -> Vec<f32> {
    if fingerprints.is_empty() {
        return Vec::new();
    }

    let length = consensus.len();
    let mut std_dev = Vec::with_capacity(length);

    for i in 0..length {
        let mean = consensus[i];
        let variance = fingerprints
            .iter()
            .map(|fp| (fp[i] - mean).powi(2))
            .sum::<f32>()
            / fingerprints.len() as f32;
        std_dev.push(variance.sqrt());
    }

    std_dev
}

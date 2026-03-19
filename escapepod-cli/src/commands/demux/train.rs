//! Train subcommand - generate reference barcode fingerprints from known samples.

use super::types::{BarcodeStats, TrainParams, TrainingOutput};
use super::utils::{
    compute_consensus_fingerprint, compute_std_dev_fingerprint, configure_thread_pool,
    extract_fingerprint_from_signal, normalize_signal, parse_norm_method,
};
use crate::progress::create_progress_bar;
use crate::style;
use escapepod::Reader;
use escapepod::segmentation::detect_adapter;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;
use uuid::Uuid;
use walkdir::WalkDir;

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
    #[arg(
        long,
        default_value = "1000",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub segment_start: usize,

    /// End sample for fingerprint region
    #[arg(
        long,
        default_value = "2000",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub segment_end: usize,

    /// Number of segments for fingerprinting
    #[arg(
        long,
        default_value = "10",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub num_segments: usize,

    /// Window width for t-test segmentation
    #[arg(
        long,
        default_value = "5",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub window_width: usize,

    /// Normalization method (zscore, minmax, median, none)
    #[arg(
        long,
        default_value = "zscore",
        value_name = "METHOD",
        help_heading = "Advanced Options"
    )]
    pub normalize: String,

    /// Minimum observations for adapter segment
    #[arg(
        long,
        default_value = "200",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub min_adapter: usize,

    /// Border trim size for adapter detection
    #[arg(
        long,
        default_value = "50",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub border_trim: usize,

    /// Number of threads for parallel processing
    #[arg(short = 'j', long, default_value = "4", value_name = "N")]
    pub threads: usize,
}

/// Run the train subcommand.
pub fn run(args: TrainArgs) -> anyhow::Result<()> {
    println!(
        "{} reference barcode fingerprints",
        style::action("Training")
    );

    // Validate that either input_dir or assignments is provided
    if args.input_dir.is_none() && args.assignments.is_none() {
        anyhow::bail!("Either --input-dir or --assignments must be provided");
    }

    // Parse normalization method
    let norm_method = parse_norm_method(&args.normalize)?;

    // Set thread pool size
    configure_thread_pool(args.threads);

    // Collect barcode assignments: read_id -> (barcode, pod5_path)
    let assignments = if let Some(ref input_dir) = args.input_dir {
        collect_assignments_from_directory(input_dir)?
    } else if let Some(ref assignments_file) = args.assignments {
        collect_assignments_from_csv(assignments_file)?
    } else {
        unreachable!()
    };

    let unique_barcodes: HashSet<_> = assignments.values().map(|(bc, _)| bc.as_str()).collect();

    println!(
        "{} {} read assignments across {} barcodes",
        style::label("Loaded:"),
        style::count(assignments.len()),
        style::count(unique_barcodes.len())
    );

    // Group reads by POD5 file for efficient reading
    let reads_by_file = group_reads_by_file(&assignments);

    println!(
        "{} {} POD5 files to process",
        style::label("Files:"),
        style::count(reads_by_file.len())
    );

    // Extract fingerprints for all reads
    let all_fingerprints = extract_fingerprints(&reads_by_file, &args, norm_method)?;

    // Group fingerprints by barcode
    let barcode_fingerprints = group_fingerprints_by_barcode(&all_fingerprints, &assignments);

    println!(
        "{} {} total fingerprints extracted",
        style::label("Extracted:"),
        style::count(all_fingerprints.len())
    );

    // Compute consensus fingerprints and build output
    let output_file = File::create(&args.output)?;
    let writer = BufWriter::new(output_file);

    let training_output = build_training_output(&args, &barcode_fingerprints);
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

/// Group reads by POD5 file for efficient reading.
fn group_reads_by_file(
    assignments: &HashMap<Uuid, (String, PathBuf)>,
) -> HashMap<PathBuf, Vec<(Uuid, String)>> {
    let mut reads_by_file: HashMap<PathBuf, Vec<(Uuid, String)>> = HashMap::new();

    for (read_id, (barcode, pod5_path)) in assignments {
        reads_by_file
            .entry(pod5_path.clone())
            .or_default()
            .push((*read_id, barcode.clone()));
    }

    reads_by_file
}

/// Extract fingerprints from reads based on assignments.
fn extract_fingerprints(
    reads_by_file: &HashMap<PathBuf, Vec<(Uuid, String)>>,
    args: &TrainArgs,
    norm_method: escapepod::dtw::NormMethod,
) -> anyhow::Result<HashMap<Uuid, Vec<f32>>> {
    let total_reads: usize = reads_by_file.values().map(|v| v.len()).sum();
    let progress_bar = create_progress_bar(total_reads as u64, "Processing")?;

    let all_fingerprints: HashMap<Uuid, Vec<f32>> = reads_by_file
        .par_iter()
        .flat_map(|(pod5_path, read_list)| {
            let mut fingerprints = Vec::new();
            let read_ids: HashSet<Uuid> = read_list.iter().map(|(id, _)| *id).collect();

            if let Ok(reader) = Reader::open(pod5_path) {
                if let Ok(reads) = reader.reads() {
                    for read in reads.flatten() {
                        if read_ids.contains(&read.read_id) && !read.signal_rows.is_empty() {
                            if let Ok(signal) = reader.get_signal(&read.signal_rows) {
                                if let Some(fp) = extract_training_fingerprint(
                                    &signal,
                                    args,
                                    norm_method,
                                    read.read_id,
                                ) {
                                    fingerprints.push((read.read_id, fp));
                                }
                            }
                        }
                        progress_bar.inc(1);
                    }
                }
            }

            fingerprints
        })
        .collect();

    progress_bar.finish_with_message("complete");

    Ok(all_fingerprints)
}

/// Extract a fingerprint from a training read.
fn extract_training_fingerprint(
    signal: &[i16],
    args: &TrainArgs,
    norm_method: escapepod::dtw::NormMethod,
    read_id: Uuid,
) -> Option<Vec<f32>> {
    // Normalize signal
    let normalized = normalize_signal(signal);

    // Detect adapter using LLR
    let (adapter_start, adapter_end) =
        detect_adapter(&normalized, args.min_adapter, args.border_trim);

    if adapter_end <= adapter_start {
        return None;
    }

    // Extract the specified region
    let region_start = adapter_start + args.segment_start;
    let region_end = (adapter_start + args.segment_end).min(adapter_end);

    if region_end <= region_start || region_end > normalized.len() {
        return None;
    }

    // Use the utility function to extract fingerprint
    let fp = extract_fingerprint_from_signal(
        signal,
        region_start,
        region_end,
        args.num_segments,
        args.window_width,
        norm_method,
        read_id,
        None,
        None,
    )?;

    Some(fp.values.iter().map(|&v| v as f32).collect())
}

/// Group fingerprints by barcode.
fn group_fingerprints_by_barcode(
    fingerprints: &HashMap<Uuid, Vec<f32>>,
    assignments: &HashMap<Uuid, (String, PathBuf)>,
) -> HashMap<String, Vec<Vec<f32>>> {
    let mut barcode_fingerprints: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    for (read_id, fingerprint) in fingerprints {
        if let Some((barcode, _)) = assignments.get(read_id) {
            barcode_fingerprints
                .entry(barcode.clone())
                .or_default()
                .push(fingerprint.clone());
        }
    }

    barcode_fingerprints
}

/// Build the training output JSON structure.
fn build_training_output(
    args: &TrainArgs,
    barcode_fingerprints: &HashMap<String, Vec<Vec<f32>>>,
) -> TrainingOutput {
    let mut training_output = TrainingOutput {
        barcodes: HashMap::new(),
        params: TrainParams {
            segment_start: args.segment_start,
            segment_end: args.segment_end,
            num_segments: args.num_segments,
        },
    };

    for (barcode, fingerprints) in barcode_fingerprints {
        let consensus = compute_consensus_fingerprint(fingerprints);
        let std_dev = compute_std_dev_fingerprint(fingerprints, &consensus);

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
            style::label(format!("{}:", barcode)),
            style::action("Computed consensus"),
            style::count(fingerprints.len())
        );
    }

    training_output
}

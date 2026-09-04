//! Train subcommand - generate reference barcode fingerprints from known samples.

use super::types::{BarcodeStats, TrainParams, TrainingOutput};
use super::utils::parse_norm_method;
use crate::progress::create_progress_bar;
use crate::style;
use escapepod_demux::{
    compute_consensus_fingerprint, compute_std_dev_fingerprint, extract_fingerprint_from_signal,
};
use escapepod_signal::segmentation::{detect_adapter, normalize_signal};
use escapepod_signal::{Reader, ReadsBatchView};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;
use tracing::info;
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

    /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// Run the train subcommand.
pub fn run(args: TrainArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Train");
    let profile = args.profile;
    info!(
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

    // Collect barcode assignments: read_id -> (barcode, pod5_path)
    let assignments = if let Some(ref input_dir) = args.input_dir {
        collect_assignments_from_directory(input_dir)?
    } else if let Some(ref assignments_file) = args.assignments {
        collect_assignments_from_csv(assignments_file)?
    } else {
        unreachable!()
    };

    let unique_barcodes: HashSet<_> = assignments.values().map(|(bc, _)| bc.as_str()).collect();

    info!(
        "{} {} read assignments across {} barcodes",
        style::label("Loaded:"),
        style::count(assignments.len()),
        style::count(unique_barcodes.len())
    );

    // Group reads by POD5 file for efficient reading
    let reads_by_file = group_reads_by_file(&assignments);

    info!(
        "{} {} POD5 files to process",
        style::label("Files:"),
        style::count(reads_by_file.len())
    );

    // Extract fingerprints, bucketed by barcode as they are produced.
    let barcode_fingerprints = extract_fingerprints_by_barcode(&reads_by_file, &args, norm_method)?;

    let total_extracted: usize = barcode_fingerprints.values().map(|v| v.len()).sum();
    info!(
        "{} {} total fingerprints extracted",
        style::label("Extracted:"),
        style::count(total_extracted)
    );

    // Compute consensus fingerprints and build output
    let output_file = File::create(&args.output)?;
    let writer = BufWriter::new(output_file);

    let training_output = build_training_output(&args, &barcode_fingerprints);
    serde_json::to_writer_pretty(writer, &training_output)?;

    info!(
        "{} reference fingerprints written to {}",
        style::action("Trained"),
        style::path(args.output.display())
    );
    info!(
        "{} {} barcodes",
        style::label("Total:"),
        style::count(training_output.barcodes.len())
    );

    timer.report(profile);

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

                // Only the read_id is needed here, so use the lighter
                // view.read_id() rather than decoding all 22 fields.
                if let Ok(batches) = reader.read_batches() {
                    for batch_result in batches {
                        let batch = batch_result?;
                        let view = ReadsBatchView::new(&batch, false)?;
                        for row in 0..view.num_rows() {
                            let read_id = view.read_id(row)?;
                            assignments.insert(read_id, (barcode.clone(), pod5_path.clone()));
                        }
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
        if parts.len() >= 3
            && let Ok(read_id) = Uuid::parse_str(parts[0])
        {
            let barcode = parts[1].to_string();
            let pod5_file = PathBuf::from(parts[2]);
            assignments.insert(read_id, (barcode, pod5_file));
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

/// Extract fingerprints from the assigned reads, grouped by barcode.
///
/// Structured like the `fingerprint` and `detect` subcommands rather than the
/// old per-file `par_iter`: files are walked *sequentially* (one ascending mmap
/// sweep of the signal table per Arrow batch, #72 — fanning I/O across reads or
/// files instead turns a 288 MB/s sequential read into N concurrent
/// demand-paging streams), and the parallelism lives *inside* each batch, where
/// the expensive work is. The old shape parallelized only across files, so the
/// common single-POD5 training run used one core no matter the `-t` setting.
///
/// Two further savings on the way in:
/// * rows are gated on the cheap `view.read_id(row)` before `view.read(row)`
///   decodes all 22 fields, so reads outside the assignment set cost one UUID
///   read instead of a full record;
/// * fingerprints are accumulated straight into per-barcode buckets, which
///   removes the intermediate `read_id -> fingerprint` map and the full clone
///   of every fingerprint that the separate grouping pass used to make.
fn extract_fingerprints_by_barcode(
    reads_by_file: &HashMap<PathBuf, Vec<(Uuid, String)>>,
    args: &TrainArgs,
    norm_method: escapepod_signal::dtw::NormMethod,
) -> anyhow::Result<HashMap<String, Vec<Vec<f32>>>> {
    let total_reads: usize = reads_by_file.values().map(|v| v.len()).sum();
    let progress_bar = create_progress_bar(total_reads as u64, "Processing")?;

    let mut by_barcode: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    for (pod5_path, read_list) in reads_by_file {
        // Barcode lookup for this file's assigned reads. Borrowed, so the
        // per-read barcode `String` is never cloned during extraction.
        let wanted: HashMap<Uuid, &str> = read_list
            .iter()
            .map(|(id, bc)| (*id, bc.as_str()))
            .collect();

        let Ok(reader) = Reader::open(pod5_path) else {
            continue;
        };
        let Ok(batches) = reader.read_batches() else {
            continue;
        };

        for batch_result in batches {
            let Ok(batch) = batch_result else { continue };
            let Ok(view) = ReadsBatchView::new(&batch, false) else {
                continue;
            };

            // Metadata-only pre-filter. `read_id` is one fixed-width column;
            // `read()` decodes the whole record, so only assigned rows pay it.
            let reads: Vec<_> = (0..view.num_rows())
                .filter(|&row| {
                    view.read_id(row)
                        .map(|id| wanted.contains_key(&id))
                        .unwrap_or(false)
                })
                .filter_map(|row| view.read(row).ok())
                .filter(|r| !r.signal_rows.is_empty())
                .collect();
            if reads.is_empty() {
                continue;
            }

            let keyed: Vec<(usize, Vec<u64>)> = reads
                .iter()
                .enumerate()
                .map(|(i, r)| (i, r.signal_rows.clone()))
                .collect();
            let Ok(bulk) = reader.get_compressed_signal_bulk(&keyed) else {
                continue;
            };

            // Decompress + LLR-detect + fingerprint in parallel, decoding the
            // same leading window `demux detect --method llr` does
            // (`LLR_DECODE_BOUND`), so the reference bank is built from the
            // boundaries the queries will get.
            // Tag with the read index and re-sort: `get_compressed_signal_bulk`
            // does not promise to return rows in request order, and the
            // per-barcode push order decides the summation order inside
            // `compute_std_dev_fingerprint` (an f32 sum, so order changes the
            // low bits). Sorting makes a training run reproducible — the old
            // grouping pass iterated a `HashMap<Uuid, _>`, so it was not.
            let mut batch_fps: Vec<(usize, &str, Vec<f32>)> = bulk
                .par_iter()
                .filter_map(|(i, chunks)| {
                    let r = &reads[*i];
                    let barcode = *wanted.get(&r.read_id)?;
                    let signal = super::utils::decode_chunks_to(
                        chunks,
                        Some(super::utils::LLR_DECODE_BOUND),
                    )?;
                    let fp = extract_training_fingerprint(&signal, args, norm_method, r.read_id)?;
                    Some((*i, barcode, fp))
                })
                .collect();
            batch_fps.sort_by_key(|(i, _, _)| *i);

            for (_, barcode, fp) in batch_fps {
                by_barcode.entry(barcode.to_string()).or_default().push(fp);
            }

            // One update per Arrow batch instead of a contended atomic add per
            // read. Counts reads *attempted*, so the bar reflects work done
            // even when a signal fails to decompress.
            progress_bar.inc(reads.len() as u64);
        }
    }

    progress_bar.finish_with_message("complete");

    Ok(by_barcode)
}

/// Extract a fingerprint from a training read.
fn extract_training_fingerprint(
    signal: &[i16],
    args: &TrainArgs,
    norm_method: escapepod_signal::dtw::NormMethod,
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

    // Use the utility function to extract fingerprint.
    // `demux train` (the WarpDemux-style reference-fingerprint path) never
    // needs dwell features — those are an SVM-training-only augmentation.
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
        false,
    )?;

    Some(fp.values.iter().map(|&v| v as f32).collect())
}

/// Build the training output JSON structure.
fn build_training_output(
    args: &TrainArgs,
    barcode_fingerprints: &HashMap<String, Vec<Vec<f32>>>,
) -> TrainingOutput {
    let mut training_output = TrainingOutput {
        barcodes: std::collections::BTreeMap::new(),
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

        info!(
            "{} {} fingerprints from {} reads",
            style::label(format!("{}:", barcode)),
            style::action("Computed consensus"),
            style::count(fingerprints.len())
        );
    }

    training_output
}

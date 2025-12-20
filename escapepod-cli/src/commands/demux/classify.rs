//! Classify subcommand - barcode classification using DTW distance.

use super::utils::parse_reference_csv;
use crate::style;
use escapepod::dtw::dtw_distance_matrix;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use uuid::Uuid;

/// Arguments for the classify subcommand.
#[derive(Debug, clap::Args)]
pub struct ClassifyArgs {
    /// Input fingerprints file
    #[arg(value_name = "FILE")]
    pub fingerprints: PathBuf,

    /// Reference barcode fingerprints (training data)
    #[arg(long, value_name = "FILE")]
    pub reference: Option<PathBuf>,

    /// Trained WarpDemuX model (JSON format)
    #[arg(long, value_name = "FILE")]
    pub model: Option<PathBuf>,

    /// Output classifications file
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// DTW window constraint (Sakoe-Chiba band width)
    #[arg(long, value_name = "N")]
    pub window: Option<usize>,

    /// Minimum distance ratio for confident classification (CSV mode only)
    #[arg(long, default_value = "0.8", value_name = "RATIO")]
    pub min_ratio: f32,
}

/// Classification result for output.
struct ClassifyResult {
    read_id: Uuid,
    barcode: String,
    confidence: f64,
    best_distance: f64,
    second_best_distance: f64,
    is_confident: bool,
}

/// Run the classify subcommand.
pub fn run(mut args: ClassifyArgs) -> anyhow::Result<()> {
    // Check that either reference or model is provided
    if args.reference.is_none() && args.model.is_none() {
        anyhow::bail!("Either --reference or --model must be provided");
    }

    if args.reference.is_some() && args.model.is_some() {
        anyhow::bail!("Cannot specify both --reference and --model");
    }

    // Dispatch to appropriate classification method
    if let Some(model_path) = args.model.take() {
        run_with_model(args, model_path)
    } else if let Some(reference_path) = args.reference.take() {
        run_with_csv(args, reference_path)
    } else {
        unreachable!()
    }
}

/// Run classification using a trained WarpDemuX model.
fn run_with_model(args: ClassifyArgs, model_path: PathBuf) -> anyhow::Result<()> {
    use escapepod::demux::{classify_read, load_model};

    println!(
        "{} reads using WarpDemuX model",
        style::action("Classifying")
    );
    println!(
        "{} {}",
        style::label("Fingerprints:"),
        style::path(args.fingerprints.display())
    );
    println!(
        "{} {}",
        style::label("Model:"),
        style::path(model_path.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(args.output.display())
    );

    // Load the model
    println!("{} model...", style::action("Loading"));
    let model = load_model(&model_path)?;

    println!(
        "{} {} training samples, {} features, threshold={:.3} ({})",
        style::label("Model:"),
        style::count(model.num_samples()),
        style::value(model.feature_dim()),
        style::value(model.threshold),
        model.threshold_type
    );

    // Read query fingerprints
    let query_fps = parse_query_fingerprints_f64(&args.fingerprints)?;

    println!(
        "{} {} query fingerprints",
        style::label("Loaded:"),
        style::count(query_fps.len())
    );

    if query_fps.is_empty() {
        anyhow::bail!("No valid query fingerprints found");
    }

    // Classify each query
    println!("{} reads...", style::action("Classifying"));

    let results: Vec<ClassifyResult> = query_fps
        .iter()
        .map(|(read_id, fingerprint)| {
            let result = classify_read(&model, fingerprint);
            ClassifyResult {
                read_id: *read_id,
                barcode: result.barcode,
                confidence: result.confidence,
                best_distance: result.best_distance,
                second_best_distance: result.second_best_distance,
                is_confident: result.is_confident,
            }
        })
        .collect();

    // Write output
    write_model_classifications(&args.output, &results)?;

    let confident_count = results.iter().filter(|r| r.is_confident).count();
    let unclassified_count = results.len() - confident_count;

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

/// Run classification using CSV reference fingerprints.
fn run_with_csv(args: ClassifyArgs, reference_path: PathBuf) -> anyhow::Result<()> {
    println!(
        "{} reads by barcode using DTW",
        style::action("Classifying")
    );
    println!(
        "{} {}",
        style::label("Fingerprints:"),
        style::path(args.fingerprints.display())
    );
    println!(
        "{} {}",
        style::label("Reference:"),
        style::path(reference_path.display())
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
    let reference_fps = parse_reference_csv(&reference_path)?;

    println!(
        "{} {} reference barcodes",
        style::label("Loaded:"),
        style::count(reference_fps.len())
    );

    if reference_fps.is_empty() {
        anyhow::bail!("No valid reference fingerprints found");
    }

    // Read query fingerprints
    let query_fps = parse_query_fingerprints_f32(&args.fingerprints)?;

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
    let ref_values: Vec<Vec<f32>> = reference_fps.iter().map(|fp| fp.values.clone()).collect();

    println!("{} DTW distances...", style::action("Computing"));

    // Compute distance matrix
    let distances = dtw_distance_matrix(&query_values, &ref_values, args.window);

    // Classify each query
    let results: Vec<ClassifyResult> = query_fps
        .iter()
        .enumerate()
        .map(|(i, (read_id, _))| {
            let row = distances.row(i);

            // Find best and second-best matches
            let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

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
            let barcode_name = reference_fps[best_idx].barcode.clone();

            ClassifyResult {
                read_id: *read_id,
                barcode: barcode_name,
                confidence: (1.0 - ratio) as f64,
                best_distance: best_dist as f64,
                second_best_distance: second_best_dist as f64,
                is_confident: confident,
            }
        })
        .collect();

    // Write output
    write_csv_classifications(&args.output, &results)?;

    let confident_count = results.iter().filter(|r| r.is_confident).count();
    let unclassified_count = results.len() - confident_count;

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

/// Parse query fingerprints as f64 (for model classification).
fn parse_query_fingerprints_f64(path: &PathBuf) -> anyhow::Result<Vec<(Uuid, Vec<f64>)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut fingerprints = Vec::new();
    let mut header_seen = false;

    for line in reader.lines() {
        let line = line?;
        if !header_seen {
            header_seen = true;
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(read_id) = Uuid::parse_str(parts[0]) {
                let values: Vec<f64> = parts[1..]
                    .iter()
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if !values.is_empty() {
                    fingerprints.push((read_id, values));
                }
            }
        }
    }

    Ok(fingerprints)
}

/// Parse query fingerprints as f32 (for CSV classification).
fn parse_query_fingerprints_f32(path: &PathBuf) -> anyhow::Result<Vec<(Uuid, Vec<f32>)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut fingerprints = Vec::new();
    let mut header_seen = false;

    for line in reader.lines() {
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
                    fingerprints.push((read_id, values));
                }
            }
        }
    }

    Ok(fingerprints)
}

/// Write model classification results to CSV.
fn write_model_classifications(path: &PathBuf, results: &[ClassifyResult]) -> anyhow::Result<()> {
    let output_file = File::create(path)?;
    let mut writer = BufWriter::new(output_file);

    writeln!(
        writer,
        "read_id,barcode,confidence,best_distance,second_best_distance,is_confident"
    )?;

    for result in results {
        writeln!(
            writer,
            "{},{},{:.6},{:.4},{:.4},{}",
            result.read_id,
            result.barcode,
            result.confidence,
            result.best_distance,
            result.second_best_distance,
            result.is_confident
        )?;
    }

    writer.flush()?;
    Ok(())
}

/// Write CSV classification results to CSV.
fn write_csv_classifications(path: &PathBuf, results: &[ClassifyResult]) -> anyhow::Result<()> {
    let output_file = File::create(path)?;
    let mut writer = BufWriter::new(output_file);

    writeln!(
        writer,
        "read_id,barcode,distance,second_best_distance,ratio,confident"
    )?;

    for result in results {
        let ratio = if result.second_best_distance > 0.0 {
            result.best_distance / result.second_best_distance
        } else {
            0.0
        };

        writeln!(
            writer,
            "{},{},{:.4},{:.4},{:.4},{}",
            result.read_id,
            result.barcode,
            result.best_distance,
            result.second_best_distance,
            ratio,
            result.is_confident
        )?;
    }

    writer.flush()?;
    Ok(())
}

//! Train SVM subcommand - train an SVM model from fingerprints.
//!
//! This command trains a DTW-SVM model that can classify reads by barcode
//! with probability output. Only available with the `train` feature.

use crate::style;
use escapepod::demux::{train_svm, TrainConfig};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Arguments for the train-svm subcommand.
#[derive(Debug, clap::Args)]
pub struct TrainSvmArgs {
    /// CSV file with fingerprints (read_id, barcode, feat1, feat2, ...)
    #[arg(short, long, required = true, value_name = "FILE")]
    pub fingerprints: PathBuf,

    /// Output JSON file for the trained SVM model
    #[arg(short, long, required = true, value_name = "FILE")]
    pub output: PathBuf,

    /// RBF kernel gamma parameter
    #[arg(long, default_value = "1.0", value_name = "VALUE")]
    pub gamma: f64,

    /// Power to raise distances before exponential
    #[arg(long, default_value = "1.0", value_name = "VALUE")]
    pub power: f64,

    /// SVM regularization parameter C
    #[arg(long, default_value = "1.0", value_name = "VALUE")]
    pub c: f64,

    /// DTW window constraint (Sakoe-Chiba band)
    #[arg(long, value_name = "N")]
    pub window: Option<usize>,

    /// Per-class confidence thresholds (comma-separated)
    #[arg(long, value_name = "VALUES")]
    pub thresholds: Option<String>,
}

/// Run the train-svm subcommand.
pub fn run(args: TrainSvmArgs) -> anyhow::Result<()> {
    println!("{} SVM model from fingerprints", style::action("Training"));

    // Load fingerprints from CSV
    let (fingerprints, labels, barcode_names) = load_fingerprints(&args.fingerprints)?;

    println!(
        "{} {} fingerprints across {} barcodes",
        style::label("Loaded:"),
        style::count(fingerprints.len()),
        style::count(barcode_names.len())
    );

    // Parse thresholds if provided
    let thresholds = args.thresholds.as_ref().map(|t| {
        t.split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect()
    });

    // Build training config
    let config = TrainConfig {
        gamma: args.gamma,
        power: args.power,
        c: args.c,
        window: args.window,
        thresholds,
    };

    println!(
        "{} gamma={}, power={}, C={}",
        style::label("Config:"),
        config.gamma,
        config.power,
        config.c
    );

    // Train the SVM
    println!("{} DTW distance matrix...", style::action("Computing"));
    let model = train_svm(fingerprints, labels, &config)?;

    // Save the model
    model.save(&args.output)?;

    println!(
        "{} SVM model written to {}",
        style::action("Trained"),
        style::path(args.output.display())
    );
    println!(
        "{} {} classes, {} support vectors",
        style::label("Model:"),
        style::count(model.n_classes),
        style::count(model.support_indices.len())
    );

    Ok(())
}

/// Load fingerprints from CSV file.
///
/// Expected format: read_id,barcode,feat1,feat2,...,featN
/// Returns: (fingerprints, labels, barcode_names)
fn load_fingerprints(
    path: &PathBuf,
) -> anyhow::Result<(Vec<Vec<f64>>, Vec<i32>, Vec<String>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut fingerprints = Vec::new();
    let mut labels = Vec::new();
    let mut barcode_to_id: HashMap<String, i32> = HashMap::new();
    let mut barcode_names = Vec::new();
    let mut next_id = 0i32;

    let mut line_count = 0;
    for line in reader.lines() {
        let line = line?;
        line_count += 1;

        // Skip header
        if line_count == 1 {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue; // Need at least read_id, barcode, and one feature
        }

        // parts[0] is read_id (ignored for training)
        let barcode = parts[1].to_string();

        // Get or assign barcode ID
        let label = *barcode_to_id.entry(barcode.clone()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            barcode_names.push(barcode);
            id
        });

        // Parse features
        let features: Vec<f64> = parts[2..]
            .iter()
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();

        if !features.is_empty() {
            fingerprints.push(features);
            labels.push(label);
        }
    }

    if fingerprints.is_empty() {
        anyhow::bail!("No valid fingerprints found in {}", path.display());
    }

    Ok((fingerprints, labels, barcode_names))
}

//! Train SVM subcommand - train an SVM model from fingerprints.
//!
//! This command trains a DTW-SVM model that can classify reads by barcode
//! with probability output. Only available with the `train` feature.

use crate::style;
use escapepod_demux::{TrainConfig, train_svm};
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
    #[arg(
        long,
        default_value = "1.0",
        value_name = "VALUE",
        help_heading = "Advanced Options"
    )]
    pub gamma: f64,

    /// Power to raise distances before exponential
    #[arg(
        long,
        default_value = "1.0",
        value_name = "VALUE",
        help_heading = "Advanced Options"
    )]
    pub power: f64,

    /// SVM regularization parameter C
    #[arg(
        long,
        default_value = "1.0",
        value_name = "VALUE",
        help_heading = "Advanced Options"
    )]
    pub c: f64,

    /// DTW window constraint (Sakoe-Chiba band)
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub window: Option<usize>,

    /// Per-class confidence thresholds (comma-separated)
    #[arg(long, value_name = "VALUES", help_heading = "Advanced Options")]
    pub thresholds: Option<String>,

    /// Run the DTW distance matrix on the GPU (requires `--features gpu`).
    #[cfg(feature = "gpu")]
    #[arg(long, help_heading = "Advanced Options")]
    pub gpu: bool,

    /// Print per-phase timing breakdown after completion
    #[arg(long)]
    pub profile: bool,
}

/// Whether the user requested the GPU path. Expands to `false` in builds
/// compiled without the `gpu` feature.
#[inline]
fn gpu_requested(args: &TrainSvmArgs) -> bool {
    #[cfg(feature = "gpu")]
    {
        args.gpu
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = args;
        false
    }
}

/// Run the train-svm subcommand.
pub fn run(args: TrainSvmArgs) -> anyhow::Result<()> {
    use crate::commands::profile::PhaseTimer;
    let mut timer = PhaseTimer::new();
    timer.phase("Train SVM");
    let profile = args.profile;

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

    // Train the SVM — GPU path for the all-pairs DTW when requested, CPU
    // otherwise.
    let model = if gpu_requested(&args) {
        #[cfg(feature = "gpu")]
        {
            use escapepod_demux::train_svm_gpu;
            println!(
                "{} DTW distance matrix on GPU...",
                style::action("Computing")
            );
            train_svm_gpu(fingerprints, labels, &config)?
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("--gpu flag is only defined when the `gpu` feature is enabled")
        }
    } else {
        println!("{} DTW distance matrix...", style::action("Computing"));
        train_svm(fingerprints, labels, &config)?
    };

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

    timer.report(profile);

    Ok(())
}

/// Load fingerprints from CSV file.
///
/// Expected format: read_id,barcode,feat1,feat2,...,featN
/// Returns: (fingerprints, labels, barcode_names)
type FingerprintData = (Vec<Vec<f64>>, Vec<i32>, Vec<String>);

fn load_fingerprints(path: &PathBuf) -> anyhow::Result<FingerprintData> {
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

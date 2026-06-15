//! Train SVM subcommand - train an SVM model from fingerprints.
//!
//! This command trains a DTW-SVM model that can classify reads by barcode
//! with probability output. Only available with the `train` feature.

use super::fp_io::read_labeled_fingerprints;
use crate::style;
use escapepod_demux::{TrainConfig, train_svm};
use std::path::PathBuf;
use tracing::info;

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

    /// Randomly subsample each barcode class down to at most N fingerprints
    /// before training. Required for large datasets since the all-pairs DTW
    /// distance matrix is O(N^2) memory (easily overflows GPU VRAM). Sampling
    /// is balanced (same cap per class) and deterministic under `--seed`.
    #[arg(long, value_name = "N", help_heading = "Advanced Options")]
    pub max_per_class: Option<usize>,

    /// RNG seed for `--max-per-class` subsampling
    #[arg(
        long,
        default_value = "42",
        value_name = "N",
        help_heading = "Advanced Options"
    )]
    pub seed: u64,

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

    info!("{} SVM model from fingerprints", style::action("Training"));

    // Load fingerprints from CSV. When `--max-per-class` is set, sampling
    // happens during the scan (per-class reservoir) so we never materialize
    // the full CSV in memory — critical for multi-GB training sets.
    let subsample = args.max_per_class.map(|cap| (cap, args.seed));
    let (fingerprints, labels, barcode_names, total_seen) =
        read_labeled_fingerprints(&args.fingerprints, subsample)?;

    if let Some(cap) = args.max_per_class {
        info!(
            "{} {} -> {} kept (cap {} per class, seed {})",
            style::action("Streamed+subsampled:"),
            style::count(total_seen),
            style::count(fingerprints.len()),
            style::count(cap),
            style::value(args.seed),
        );
    } else {
        info!(
            "{} {} fingerprints across {} barcodes",
            style::label("Loaded:"),
            style::count(fingerprints.len()),
            style::count(barcode_names.len())
        );
    }

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

    info!(
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
            info!(
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
        info!("{} DTW distance matrix...", style::action("Computing"));
        train_svm(fingerprints, labels, &config)?
    };

    // Save the model
    model.save(&args.output)?;

    info!(
        "{} SVM model written to {}",
        style::action("Trained"),
        style::path(args.output.display())
    );
    info!(
        "{} {} classes, {} support vectors",
        style::label("Model:"),
        style::count(model.n_classes),
        style::count(model.support_indices.len())
    );

    timer.report(profile);

    Ok(())
}

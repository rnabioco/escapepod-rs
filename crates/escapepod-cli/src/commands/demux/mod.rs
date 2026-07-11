//! Demux command implementation.
//!
//! Barcode demultiplexing workflow for Oxford Nanopore reads.
//! Includes adapter detection, barcode fingerprinting, and classification.
//!
//! ## Workflow
//!
//! 1. **detect**: LLR-based adapter boundary detection
//! 2. **fingerprint**: Extract signal features from adapter regions
//! 3. **classify**: Assign reads to barcodes using DTW distance
//! 4. **split**: Write demultiplexed reads to separate POD5 files
//! 5. **train**: Generate reference barcodes from known samples
//! 6. **train-svm**: Train SVM model from fingerprints (requires `train` feature)

mod classify;
mod detect;
mod fingerprint;
mod fp_io;
mod run;
mod split;
mod train;
mod types;
mod utils;

#[cfg(feature = "train")]
mod train_svm;

pub use classify::ClassifyArgs;
pub use detect::DetectArgs;
pub use fingerprint::FingerprintArgs;
pub use run::RunArgs;
pub use split::SplitArgs;
pub use train::TrainArgs;

#[cfg(feature = "train")]
pub use train_svm::TrainSvmArgs;

/// Main demux command arguments.
///
/// With no subcommand, `demux` runs the fused streaming pipeline (detect +
/// fingerprint + classify in one pass → demultiplexed POD5). The granular
/// `detect`/`fingerprint`/`classify`/`split`/`train*` subcommands remain for
/// advanced/debugging use.
#[derive(Debug, clap::Args)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
pub struct DemuxArgs {
    /// Advanced per-stage subcommands. Omit to run the fused pipeline.
    #[command(subcommand)]
    pub command: Option<DemuxCommand>,

    /// Fused-pipeline arguments (used when no subcommand is given).
    #[command(flatten)]
    pub run: RunArgs,
}

/// Demux subcommands (advanced / per-stage).
#[derive(Debug, clap::Subcommand)]
pub enum DemuxCommand {
    /// [advanced] Detect adapter boundaries in reads using LLR algorithm
    #[command(after_help = "\
Examples:
  escpod demux detect input.pod5 -o boundaries.csv
  escpod demux detect *.pod5 -o boundaries.csv --min-adapter 200 -j 8
")]
    Detect(DetectArgs),

    /// Extract barcode fingerprints from adapter regions
    #[command(after_help = "\
Examples:
  escpod demux fingerprint input.pod5 --boundaries boundaries.csv -o fingerprints.csv
  escpod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv
")]
    Fingerprint(FingerprintArgs),

    /// Classify reads by barcode using DTW distance
    #[command(after_help = "\
Examples:
  escpod demux classify fingerprints.csv --reference barcodes.csv -o classifications.csv
  escpod demux classify fingerprints.csv --reference barcodes.csv -o out.csv --window 10
")]
    Classify(ClassifyArgs),

    /// Split reads into separate POD5 files by barcode
    #[command(after_help = "\
Examples:
  escpod demux split input.pod5 --classifications classifications.csv --output-dir demuxed/
  escpod demux split *.pod5 --classifications classifications.csv -d out/ --prefix bc

Reads each input once and routes every read to its barcode's writer (rather than
re-scanning per barcode). This keeps the full input resident while writing, so
peak memory tracks input size — size --mem accordingly for large runs.
")]
    Split(SplitArgs),

    /// Train reference barcode fingerprints from known samples
    #[command(after_help = "\
Examples:
  escpod demux train --input-dir barcodes/ -o reference.json
  escpod demux train --assignments assignments.csv -o reference.json
")]
    Train(TrainArgs),

    /// Train SVM model from fingerprints (requires --features train)
    #[cfg(feature = "train")]
    #[command(
        name = "train-svm",
        after_help = "\
Examples:
  escpod demux train-svm -f fingerprints.csv -o model.json
  escpod demux train-svm -f fingerprints.csv -o model.json --gamma 0.5 --window 10
"
    )]
    TrainSvm(TrainSvmArgs),
}

/// Run the demux command.
pub fn run(args: DemuxArgs) -> anyhow::Result<()> {
    let Some(command) = args.command else {
        // No subcommand → the fused streaming pipeline.
        return run::run(args.run);
    };
    match command {
        DemuxCommand::Detect(detect_args) => detect::run(detect_args),
        DemuxCommand::Fingerprint(fingerprint_args) => fingerprint::run(fingerprint_args),
        DemuxCommand::Classify(classify_args) => classify::run(classify_args),
        DemuxCommand::Split(split_args) => split::run(split_args),
        DemuxCommand::Train(train_args) => train::run(train_args),
        #[cfg(feature = "train")]
        DemuxCommand::TrainSvm(train_svm_args) => train_svm::run(train_svm_args),
    }
}

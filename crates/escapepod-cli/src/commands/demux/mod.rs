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
//! 4. **basecall**: CTC-CRF barcode basecalling from detected boundaries
//! 5. **split**: Write demultiplexed reads to separate POD5 files
//! 6. **train**: Generate reference barcodes from known samples
//! 7. **train-svm**: Train SVM model from fingerprints (requires `train` feature)

#[cfg(feature = "crf-decode")]
mod basecall;
mod classify;
mod detect;
mod fingerprint;
mod fp_io;
mod info;
#[cfg(feature = "demux-models")]
pub mod models;
mod run;
mod split;
mod train;
mod types;
mod utils;

#[cfg(feature = "train")]
mod train_svm;

#[cfg(feature = "crf-decode")]
pub use basecall::BasecallArgs;
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

    /// Manage boundary/barcode model downloads and the local cache
    #[cfg(feature = "demux-models")]
    #[command(after_help = "\
Examples:
  escpod demux models list
  escpod demux models fetch wdx4_rna004
  escpod demux models path

Models are gitignored upstream and distributed via GitHub Releases. Fetch on a
networked node before submitting compute jobs — resolution at run time never
touches the network, so a missing model fails fast instead of hanging.
")]
    Models {
        #[command(subcommand)]
        command: models::ModelsCommand,
    },

    /// Basecall barcode regions with a CTC-CRF model
    #[cfg(feature = "crf-decode")]
    #[command(after_help = "\
Examples:
  escpod demux basecall in.pod5 --boundaries bnd.csv --model crf_export/ \
      --barcodes refs.csv -o classifications.csv
  escpod demux basecall *.pod5 --boundaries bnd.csv --model crf_export/ -o seqs.csv

With --barcodes (a `name,sequence` CSV) each read is assigned to its closest
reference by edit distance and the output feeds `escpod demux split` directly.
Without it, only decoded sequences are emitted. Confidence is the edit-distance
margin to the second-best reference.

The encoder runs on the GPU by default (--device auto) when built with
`gpu` (onnxruntime + CUDA) and a device is visible; --device gpu demands it
and --device cpu forbids it. The lattice decode is the encoder's own business.
")]
    Basecall(BasecallArgs),

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

/// The `-t/-j` value for whichever demux stage was invoked.
///
/// Lives here rather than in `main` because it destructures [`DemuxCommand`].
/// See `crate::requested_threads` for why the value is read back out of the
/// parsed args instead of being a global clap flag.
pub fn requested_threads(args: &DemuxArgs) -> Option<usize> {
    match &args.command {
        // `args_conflicts_with_subcommands`: no subcommand means the fused
        // pipeline, whose flags live in `run`.
        None => args.run.threads,
        Some(DemuxCommand::Detect(a)) => a.threads,
        Some(DemuxCommand::Fingerprint(a)) => a.threads,
        #[cfg(feature = "demux-models")]
        Some(DemuxCommand::Models { .. }) => None,
        #[cfg(feature = "crf-decode")]
        Some(DemuxCommand::Basecall(a)) => a.threads,
        Some(DemuxCommand::Classify(a)) => a.threads,
        Some(DemuxCommand::Split(a)) => a.threads,
        Some(DemuxCommand::Train(a)) => a.threads,
        // train-svm has no --threads flag.
        #[cfg(feature = "train")]
        Some(DemuxCommand::TrainSvm(_)) => None,
    }
}

/// Run the demux command.
///
/// POD5 inputs are resolved here, once, for every subcommand that takes them:
/// a directory expands to the `*.pod5` files under it and a path that does not
/// exist is an error, exactly as `merge`/`view`/`index` have always behaved
/// (`crate::util::collect_pod5_inputs`). Doing it at the dispatch point rather
/// than in each `run` keeps the four stages from drifting apart again — before
/// this, `detect` and `split` died on a directory with a bare `ENODEV` from the
/// mmap while `fingerprint` and `basecall` skipped it and wrote a header-only
/// CSV with a zero exit status (escapepod-rs#293).
pub fn run(args: DemuxArgs) -> anyhow::Result<()> {
    use crate::util::collect_pod5_inputs;

    let Some(command) = args.command else {
        // No subcommand → the fused streaming pipeline. Its inputs are resolved
        // inside `run::run`, not here: `--info` describes a model and takes no
        // POD5 at all, so resolution has to happen after that early return.
        return run::run(args.run);
    };
    match command {
        DemuxCommand::Detect(mut detect_args) => {
            detect_args.input = collect_pod5_inputs(&detect_args.input)?;
            detect::run(detect_args)
        }
        DemuxCommand::Fingerprint(mut fingerprint_args) => {
            fingerprint_args.input = collect_pod5_inputs(&fingerprint_args.input)?;
            fingerprint::run(fingerprint_args)
        }
        #[cfg(feature = "demux-models")]
        DemuxCommand::Models { command } => models::run(command),
        #[cfg(feature = "crf-decode")]
        DemuxCommand::Basecall(mut basecall_args) => {
            basecall_args.input = collect_pod5_inputs(&basecall_args.input)?;
            basecall::run(basecall_args)
        }
        DemuxCommand::Classify(classify_args) => classify::run(classify_args),
        DemuxCommand::Split(mut split_args) => {
            split_args.input = collect_pod5_inputs(&split_args.input)?;
            split::run(split_args)
        }
        DemuxCommand::Train(train_args) => train::run(train_args),
        #[cfg(feature = "train")]
        DemuxCommand::TrainSvm(train_svm_args) => train_svm::run(train_svm_args),
    }
}

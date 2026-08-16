//! `escpod signal` — read-level models over the raw signal.
//!
//! The group exists to keep the *word* `classify` unambiguous. `escpod demux
//! classify` assigns a barcode from a DTW/GBM fingerprint; the tRNA charging
//! model asks an entirely different question of an entirely different input
//! (POD5 + aligned BAM, anchored on the CCA–aa junction in reference
//! coordinates). Those two lived one keystroke apart as `escpod demux
//! classify` and a bare top-level `escpod classify`, which is a trap for
//! anyone reading a shell history or a pipeline script. Naming the group for
//! what it operates on — the signal itself, rather than a barcode — separates
//! them.
//!
//! Unlike [`demux`](crate::commands::demux) and
//! [`resquiggle`](crate::commands::resquiggle), which have a default action
//! and so need `args_conflicts_with_subcommands`/`subcommand_negates_reqs`,
//! `signal` is a pure namespace: the subcommand is required and there is
//! nothing to negate.

pub mod classify;

pub use classify::ClassifyArgs;

/// Signal-level read classification subcommands.
#[derive(clap::Subcommand)]
pub enum SignalCommand {
    /// Classify reads against a model bundle (tRNA charging) from POD5 +
    /// aligned BAM
    #[command(after_help = "\
Examples:
  escpod signal classify reads.pod5 -b aln.bam -r ref.fa -m bundle/ -o out.bam
  escpod signal classify reads.pod5 -b aln.bam -r ref.fa -m bundle/ -o out.bam \\
      --tsv calls.tsv

The model bundle carries the whole feature recipe (offsets, stat layout, the
k-mer table pinned by sha256, the recommended operating point) — it is not
configurable by flag, because a caller computing the features differently gets
a wrong answer rather than an error.
")]
    Classify(ClassifyArgs),
}

/// The `-t/-j` value for whichever signal subcommand was invoked.
///
/// Lives here rather than in `main` because it destructures [`SignalCommand`].
/// See `crate::requested_threads` for why the value is read back out of the
/// parsed args instead of being a global clap flag.
pub fn requested_threads(command: &SignalCommand) -> Option<usize> {
    match command {
        SignalCommand::Classify(args) => args.threads,
    }
}

/// Run the signal command.
pub fn run(command: SignalCommand) -> anyhow::Result<()> {
    match command {
        SignalCommand::Classify(args) => classify::run(args),
    }
}

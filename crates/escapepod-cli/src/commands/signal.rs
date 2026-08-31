//! `escpod signal` — a hidden, deprecated namespace alias for `escpod
//! classify`.
//!
//! The charging classifier lived under this group for one release cycle
//! (0.11.0), moved there so the *word* `classify` could not be confused with
//! `escpod demux classify`. That fixed a smaller problem than it created:
//! every other tool in this binary is one word, and burying the only
//! read-level model runner under a namespace made it the exception — harder
//! to find in `escpod --help`, and longer to type in every pipeline that runs
//! it. The two commands were never actually ambiguous in use: one is a
//! `demux` stage that consumes a fingerprint CSV, the other is a top-level
//! command that takes a POD5 and an aligned BAM.
//!
//! So the group is gone from the help output and survives only as a
//! compatibility shim: `escpod signal classify` parses the same
//! [`ClassifyArgs`] and forwards to the same runner, with a warning naming its
//! replacement. See [`crate::commands::classify`] for the command itself.

pub use crate::commands::classify::ClassifyArgs;

/// The one subcommand the deprecated group ever had.
#[derive(clap::Subcommand)]
pub enum SignalCommand {
    /// Deprecated alias for `escpod classify`.
    #[command(hide = true)]
    Classify(ClassifyArgs),
}

/// The `-t/-j` value for whichever signal subcommand was invoked.
///
/// Lives here rather than in `main` because it destructures [`SignalCommand`].
/// See `crate::requested_threads` for why the value is read back out of the
/// parsed args instead of being a global clap flag — a deprecated path that
/// silently loses `-j` is the #155 bug wearing a different hat.
pub fn requested_threads(command: &SignalCommand) -> Option<usize> {
    match command {
        SignalCommand::Classify(args) => args.threads,
    }
}

/// Run the deprecated alias: warn, then forward to the real command.
pub fn run(command: SignalCommand) -> anyhow::Result<()> {
    match command {
        SignalCommand::Classify(args) => {
            tracing::warn!("`escpod signal classify` is deprecated; use `escpod classify`.");
            crate::commands::classify::run(args)
        }
    }
}

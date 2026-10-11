//! `escpod sidecar` — the group for everything that writes the `.p5s` sidecar.
//!
//! `index` and `annotate` write the same file through the same writers, with
//! the same footer identity binding and collection-sidecar handling, and
//! `annotate` already builds the read index when none exists — so `index` is
//! the labels-free case of the same operation. Two unrelated-looking top-level
//! commands hid that; the group is the noun they share, and it leaves room for
//! further sidecar tooling without another top-level name.
//!
//! This module only nests: [`IndexArgs`] and [`AnnotateArgs`] are the
//! definitions the commands themselves own, and [`run`] hands them to the
//! existing handlers. Nothing about either command is restated here.
//!
//! The top-level `escpod index` and `escpod annotate` survive as hidden,
//! deprecated forwarders — the pattern the retired `escpod signal classify`
//! followed from 0.18 to 0.26 (#375). They parse the same args structs, so a
//! flag added to the real command reaches both spellings, and warn once on
//! stderr naming the replacement. [`run_legacy_index`] and
//! [`run_legacy_annotate`] are those forwarders.

use crate::commands::annotate::{self, AnnotateArgs};
use crate::commands::index::{self, IndexArgs};

/// The subcommands of `escpod sidecar`.
#[derive(clap::Subcommand)]
pub enum SidecarCommand {
    /// Build the .p5s caches — read index and signal batch geometry — for fast
    /// read lookup and fast signal access
    Index(IndexArgs),

    /// Record per-read annotations (e.g. demux barcodes) in the .p5s sidecar
    Annotate(Box<AnnotateArgs>),
}

/// The `-t/-j` value for whichever sidecar subcommand was invoked.
///
/// Lives here rather than in `main` because it destructures
/// [`SidecarCommand`]. See `crate::requested_threads` for why the value is
/// read back out of the parsed args instead of being a global clap flag — a
/// path that silently loses `-j` is the #155 bug wearing a different hat.
/// Exhaustive on purpose, for the same reason that one is.
pub fn requested_threads(command: &SidecarCommand) -> Option<usize> {
    match command {
        SidecarCommand::Index(args) => args.threads,
        SidecarCommand::Annotate(args) => args.threads,
    }
}

/// Run `escpod sidecar <subcommand>`.
pub fn run(command: SidecarCommand) -> anyhow::Result<()> {
    match command {
        SidecarCommand::Index(args) => index::run(args),
        SidecarCommand::Annotate(args) => annotate::run(*args),
    }
}

/// Run the deprecated `escpod index`: warn, then forward to the real command.
pub fn run_legacy_index(args: IndexArgs) -> anyhow::Result<()> {
    tracing::warn!("`escpod index` is deprecated; use `escpod sidecar index`.");
    index::run(args)
}

/// Run the deprecated `escpod annotate`: warn, then forward to the real command.
pub fn run_legacy_annotate(args: AnnotateArgs) -> anyhow::Result<()> {
    tracing::warn!("`escpod annotate` is deprecated; use `escpod sidecar annotate`.");
    annotate::run(args)
}

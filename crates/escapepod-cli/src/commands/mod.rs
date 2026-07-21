//! CLI command implementations.

/// Default rayon pool size for the block-copy / bulk-file commands (`filter`,
/// `subset`, `merge`, `index`) when `-t` is not given. Deliberately small:
/// these commands are largely I/O-bound and fan out across files, so defaulting
/// to all CPUs would monopolize a shared node for little gain. Users raise it
/// with `-t` on a machine they own.
pub const DEFAULT_THREADS: usize = 8;

pub mod bam_filter;
#[cfg(feature = "demux")]
pub mod demux;
pub mod filter;
#[cfg(feature = "experimental")]
pub mod index;
pub mod inspect;
pub mod merge;
pub mod profile;
#[cfg(feature = "experimental")]
pub mod repack;
#[cfg(feature = "experimental")]
pub mod resquiggle;
#[cfg(feature = "experimental")]
pub mod resquiggle_models;
pub mod subset;
pub mod summary;
pub mod view;

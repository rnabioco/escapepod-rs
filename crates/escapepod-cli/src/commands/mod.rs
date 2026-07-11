//! CLI command implementations.

/// Default rayon pool size for the block-copy commands (`filter`, `subset`,
/// `merge`) when `-t` is not given. Deliberately small: these commands are
/// largely I/O-bound copies, so defaulting to all CPUs would monopolize a
/// shared node for little gain. Users raise it with `-t` on a machine they own.
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
pub mod subset;
pub mod summary;
pub mod view;

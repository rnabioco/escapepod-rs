//! CLI command implementations.
//!
//! Commands do not size the rayon pool themselves — `main` does that once,
//! before dispatch. See [`crate::threads`] for why.

#[cfg(feature = "experimental")]
pub mod annotate;
pub mod bam_filter;
#[cfg(feature = "demux")]
pub mod demux;
pub mod filter;
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
#[cfg(feature = "classify")]
pub mod signal;
pub mod subset;
pub mod summary;
pub mod view;

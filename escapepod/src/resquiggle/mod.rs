//! Signal-to-sequence alignment refinement (resquiggle).
//!
//! This module ports the core resquiggle algorithms from fishnet, providing
//! banded dynamic programming for refining basecaller signal-to-base mappings
//! against a kmer level model.

pub mod bands;
pub mod dp;
pub mod kmer_table;
pub mod refine;
pub mod rescale;
pub mod types;

pub use kmer_table::KmerTable;
pub use refine::{calculate_initial_scaling, refine_signal_map, RefinementResult};
pub use types::{RefineAlgo, RefineSettings, RescaleAlgo, RoughRescaleAlgo};

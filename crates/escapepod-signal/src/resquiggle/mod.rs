// SPDX-License-Identifier: MIT
// Algorithm inspired by fishnet (Brickner et al.); independent implementation.

//! Signal-to-sequence alignment refinement (resquiggle).
//!
//! This module is inspired by the resquiggle algorithms from fishnet, providing
//! banded dynamic programming for refining basecaller signal-to-base mappings
//! against a kmer level model.

pub mod adaptive_dp;
pub mod bands;
pub mod dp;
pub mod dp_raw_penalty;
pub mod kmer_table;
pub mod refine;
pub mod rescale;
pub mod types;

pub use dp_raw_penalty::banded_dp_with_penalty_table;
pub use kmer_table::KmerTable;
pub use refine::{
    RefinementResult, calculate_initial_scaling, refine_signal_map, reverse_query_to_signal_map,
};
pub use types::{
    BandingAlgo, RefineAlgo, RefineSettings, RescaleAlgo, RescaleFilterParams, RoughRescaleAlgo,
};

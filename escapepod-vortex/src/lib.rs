//! Experimental POD5 → Vortex converter and benchmarks.
//!
//! Scope (Phase 1, the benchmark deliverable): convert the POD5 signal column
//! to a Vortex file using BtrBlocks default encoding (cascade includes pco where
//! beneficial), and compare on-disk size + decode-to-i16 throughput vs VBZ.
//!
//! If the benchmark shows Vortex wins, we expand to full round-trip (reads +
//! run_info tables) per the approved plan.

pub mod delta_scheme;
pub mod error;
pub mod signal;

pub use error::{Error, Result};

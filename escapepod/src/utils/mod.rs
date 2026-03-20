//! Utility functions for POD5 operations.
//!
//! This module provides common utilities used across the library:
//! - UUID parsing and handling
//! - Statistics computation
//! - Table building for POD5 output

pub mod dictionary;
mod statistics;
pub(crate) mod table_builders;
mod uuid;

// Public re-exports (exposed through lib.rs)
pub use statistics::{Statistics, compute_n50, compute_statistics};
pub use uuid::parse_uuid_flexible;

//! Utility functions for POD5 operations.
//!
//! This module provides common utilities used across the library:
//! - UUID parsing and handling
//! - Run info deduplication
//! - Dictionary value scanning
//! - Statistics computation

mod dictionary;
mod run_info;
mod statistics;
mod uuid;

pub use dictionary::{scan_dictionary_values, ScannedDictionaries};
pub use run_info::{add_run_infos_deduplicated, map_run_info_index};
pub use statistics::{compute_n50, compute_statistics, Statistics};
pub use uuid::parse_uuid_flexible;

//! High-level operations on POD5 files.
//!
//! This module provides functions for common POD5 operations like
//! filtering reads by ID and parsing CSV mappings for subsetting.

mod filter;
mod subset;

pub use filter::{filter_files, read_ids_from_file, FilterOptions, FilterResult};
pub use subset::parse_csv_mapping;

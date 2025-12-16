//! High-level operations on POD5 files.
//!
//! This module provides functions for common POD5 operations like
//! filtering reads by ID, repacking files, and parsing CSV mappings for subsetting.

mod filter;
mod repack;
mod subset;

pub use filter::{filter_files, read_ids_from_file, FilterOptions, FilterResult};
pub use repack::{repack_files, RepackOptions, RepackResult};
pub use subset::parse_csv_mapping;

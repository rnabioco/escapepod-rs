//! High-level operations on POD5 files.
//!
//! This module provides functions for common POD5 operations like
//! filtering reads by ID, repacking files, and parsing CSV mappings for subsetting.

pub(crate) mod csv_utils;
mod filter;
mod repack;
mod split;
mod subset;

pub use filter::{
    FilterCriteria, FilterOptions, FilterResult, filter_files, filter_files_with_criteria,
    read_ids_from_file,
};
pub use repack::{RepackOptions, RepackResult, repack_files};
pub use split::parse_barcode_mapping;
pub use subset::parse_csv_mapping;

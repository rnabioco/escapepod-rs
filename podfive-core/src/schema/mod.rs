//! Arrow schema definitions for POD5 tables.

pub mod reads;
pub mod run_info;
pub mod signal;

pub use reads::reads_schema;
pub use run_info::run_info_schema;
pub use signal::signal_schema;

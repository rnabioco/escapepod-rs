//! CLI-only types used by the demux subcommands. Domain types
//! (fingerprints, boundaries, etc.) live in [`escapepod_demux`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Barcode statistics for training output.
#[derive(Debug, Serialize, Deserialize)]
pub struct BarcodeStats {
    /// Consensus fingerprint values
    pub fingerprint: Vec<f64>,
    /// Number of reads used to compute the consensus
    pub read_count: usize,
    /// Standard deviation at each fingerprint position
    pub std_dev: Vec<f64>,
}

/// Training parameters stored in the output.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainParams {
    /// Start sample for fingerprint region
    pub segment_start: usize,
    /// End sample for fingerprint region
    pub segment_end: usize,
    /// Number of segments for fingerprinting
    pub num_segments: usize,
}

/// Training output JSON structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingOutput {
    /// Map of barcode name to statistics
    pub barcodes: HashMap<String, BarcodeStats>,
    /// Training parameters used
    pub params: TrainParams,
}

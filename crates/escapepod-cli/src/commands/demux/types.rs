//! Common types for demux subcommands.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use uuid::Uuid;

/// Read boundary information from adapter detection.
#[derive(Debug, Clone)]
pub struct ReadBoundaries {
    /// The read identifier
    pub read_id: Uuid,
    /// Total number of samples in the read
    pub num_samples: u64,
    /// Start position of the adapter region
    pub adapter_start: usize,
    /// End position of the adapter region
    pub adapter_end: usize,
}

impl ReadBoundaries {
    /// Check if the adapter region is valid (end > start).
    pub fn has_valid_adapter(&self) -> bool {
        self.adapter_end > self.adapter_start
    }
}

/// A fingerprint extracted from a read's adapter region.
#[derive(Debug, Clone)]
pub struct ReadFingerprint {
    /// The read identifier
    pub read_id: Uuid,
    /// The fingerprint feature values
    pub values: Vec<f64>,
}

impl ReadFingerprint {
    /// Create a new read fingerprint.
    pub fn new(read_id: Uuid, values: Vec<f64>) -> Self {
        Self { read_id, values }
    }
}

/// A reference barcode fingerprint for classification.
#[derive(Debug, Clone)]
pub struct BarcodeFingerprint {
    /// The barcode name (e.g., "BC01")
    pub barcode: String,
    /// The fingerprint feature values
    pub values: Vec<f32>,
}

impl BarcodeFingerprint {
    /// Create a new barcode fingerprint.
    pub fn new(barcode: String, values: Vec<f32>) -> Self {
        Self { barcode, values }
    }
}

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

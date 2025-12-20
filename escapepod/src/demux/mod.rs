//! Barcode demultiplexing using WarpDemuX-style models.
//!
//! This module provides support for loading and using trained WarpDemuX models
//! to classify nanopore reads by barcode. The workflow includes:
//!
//! 1. Load a trained model from JSON (exported using `export_warpdemux_model.py`)
//! 2. Extract fingerprints from reads (using DTW fingerprinting)
//! 3. Classify reads using DTW distance + RBF kernel similarity
//!
//! # Example
//!
//! ```no_run
//! use escapepod::demux::{load_model, classify_read};
//! use std::path::Path;
//!
//! // Load the model
//! let model = load_model(Path::new("model.json"))?;
//!
//! // Classify a fingerprint
//! let fingerprint = vec![0.1, 0.2, 0.3, 0.4, 0.5];
//! let result = classify_read(&model, &fingerprint);
//!
//! println!("Barcode: {}", result.barcode);
//! println!("Confidence: {:.3}", result.confidence);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod classify;
mod model;

pub use classify::{classify_read, ClassificationResult};
pub use model::{load_model, KernelParams, WarpDemuxModel};

# Demux Feature Priorities

This document tracks the remaining features needed to make escapepod's demux command production-ready.

## Current Status

The demux command has the core algorithms implemented:
- Signal segmentation (LLR boundary detection, windowed t-test, MAD normalization)
- DTW distance calculation with Sakoe-Chiba band constraint
- RBF kernel conversion for SVM compatibility
- Fingerprint extraction from adapter regions
- CLI scaffolding with `boundaries`, `fingerprints`, and `classify` subcommands

## High Priority

### 1. WarpDemuX Model Support
**Goal**: Enable use of pre-trained WarpDemuX models for classification

WarpDemuX models (`.joblib` files) contain:
- `_X`: Training fingerprints (numpy array)
- `_y`: Training labels
- `svm_params`: SVM hyperparameters (C, class_weight)
- `kernel_params`: RBF kernel parameters (gamma, power)
- `label_mapper`: Barcode name ↔ numeric ID mapping
- `thresh`, `thresh_type`, `thresh_mode`: Classification thresholds

**Implementation approach**:
1. Python export script to convert `.joblib` → portable format (JSON/MessagePack)
2. Rust model loader to deserialize exported models
3. Classification using DTW + RBF kernel with exported training fingerprints
4. Apply thresholds for unclassified read filtering

**Why high priority**: This enables immediate use without requiring users to train their own models.

### 2. Training Mode
**Goal**: Generate reference barcodes from known samples

Both WarpDemuX and ADAPTed support training from:
- Single-plex runs (one barcode per file)
- Pre-classified reads with known assignments

**Implementation approach**:
1. Accept ground-truth barcode assignments (CSV or directory structure)
2. Extract fingerprints from known reads
3. Compute consensus fingerprints per barcode
4. Export reference fingerprints for classification

### 3. Split Command
**Goal**: Output demultiplexed reads to separate POD5 files

Current `classify` only outputs CSV assignments. Production workflows need:
- Separate POD5 file per barcode
- Unclassified reads to separate file
- Summary statistics per output file

**Implementation approach**:
1. Use existing filter infrastructure
2. Group reads by classification
3. Write separate POD5 files using `FileWriter`
4. Report per-barcode statistics

## Medium Priority

### 4. Poly(A) Tail Detection
**Goal**: Detect and measure poly(A) tails for RNA samples

ADAPTed includes poly(A) detection as part of adapter finding. Key for:
- RNA-seq applications
- Direct RNA sequencing
- Tail length analysis

**Implementation approach**:
1. Detect poly(A) signal pattern after adapter
2. Estimate tail length from signal duration
3. Include in output metadata

### 5. Batch Processing with Resume
**Goal**: Handle large datasets with checkpointing

For processing thousands of POD5 files:
- Progress tracking across files
- Resume from interruption
- Parallel file processing

**Implementation approach**:
1. Track processed files in state file
2. Support `--resume` flag
3. Use rayon for parallel file processing (already have this pattern in merge)

## Lower Priority

### 6. Enhanced QC Metrics
**Goal**: Report quality metrics for demux decisions

Additional metrics for troubleshooting:
- Distance distributions per barcode
- Confidence scores
- Boundary detection quality
- Signal quality indicators

---

## Implementation Order

Recommended order based on user impact:

1. **WarpDemuX model support** - Immediate value, enables use without training
2. **Split command** - Required for most workflows
3. **Training mode** - Enables custom barcode sets
4. **Batch processing** - Production scalability
5. **Poly(A) detection** - RNA-specific enhancement
6. **Enhanced QC** - Nice-to-have diagnostics

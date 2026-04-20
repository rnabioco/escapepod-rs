# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

escapepod-rs is a pure Rust implementation for reading and writing POD5 files, the native file format for Oxford Nanopore sequencing data. The project provides both a library crate (`escapepod`) and a CLI tool (`escapepod-cli`).

## Requirements

- Rust 1.88 or later (matches `workspace.package.rust-version`)

## Build Commands

```bash
# Build
cargo build --release

# Build with training support (enables SVM model training)
cargo build --release --features train

# Run tests
cargo test

# Run a specific test
cargo test test_round_trip_single_read

# Run clippy lints
cargo clippy

# Build optimized for the current CPU (enables AVX2, etc.)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run the CLI (after building)
./target/release/escpod <command>
```

## Benchmarking & Profiling

### Build profiles

- `release` — ship build: fat LTO, `codegen-units=1`, stripped, `panic=abort`.
- `release-with-debug` — release speed with symbols retained (for `samply`/`perf`).
- `bench` — inherits release; used by `cargo bench` so microbenchmarks match release codegen.
- `profiling` — inherits release but turns LTO off and splits `codegen-units=16` so profilers see real call graphs instead of inlined soup.
- `dist` — ship build for release artifacts.
- `dev.package."*"` — dev dependencies compile at `opt-level = 2` so test iteration is fast.

### Microbenchmarks (criterion)

`escapepod/benches/hot_paths.rs` covers the audit hot paths: DTW, resquiggle DP, fingerprint MAD, VBZ encode/decode, DTW matrix.

```bash
# Full run
cargo bench --bench hot_paths

# Subset
cargo bench --bench hot_paths -- vbz

# A/B against a saved baseline
cargo bench --bench hot_paths -- --save-baseline <name>
cargo bench --bench hot_paths -- --baseline <name>     # compare future runs
```

Env overrides: `ESCAPEPOD_BENCH_THREADS=N` (rayon pool size for the matrix bench), `ESCAPEPOD_BENCH_SAMPLES=N` (criterion sample size for slow groups).

### End-to-end (hyperfine vs. Python pod5)

```bash
cargo build --release
./benchmarks/benchmark.sh /path/to/pod5/dir
```

Runs `inspect summary`, `view`, `merge`, `filter`, `subset` via hyperfine against Python `pod5` (installed in the pixi env). Results persist as JSON under `/tmp/escapepod_benchmark/`. Historical numbers are in `benchmarks/README.md`.

### Profiling workflow

```bash
# 1. build with symbols kept
cargo build --profile release-with-debug -p escapepod-cli
# 2. record with samply (pixi-provided binary recommended)
samply record target/release-with-debug/escpod <args>
# 3. flamegraph-style view in browser
```

For perf/valgrind where inlining hides frames, swap `release-with-debug` for `profiling` (LTO off).

### Runtime verbosity

The CLI is wired to `tracing` with stderr output. Control the level via CLI flags or `RUST_LOG`:

```bash
escpod -v inspect summary file.pod5      # info
escpod -vv merge *.pod5 -o out.pod5      # debug
RUST_LOG=escapepod::reader=trace escpod view file.pod5   # module-scoped
escpod -q merge …                         # errors only
```

## Architecture

### Workspace Structure

- **escapepod**: Core library for reading/writing POD5 files
- **escapepod-cli**: CLI binary that uses escapepod

### POD5 File Format

POD5 is a container format wrapping Apache Arrow IPC (Feather v2) tables:

```
<POD5 signature>
<section marker>
<Signal table (Arrow IPC)><section marker>
<Run Info table (Arrow IPC)><section marker>
<Reads table (Arrow IPC)><section marker>
<FOOTER magic>
<FlatBuffer footer>
<footer length>
<section marker>
<POD5 signature>
```

### Core Library (escapepod)

- **reader/file_reader.rs**: Memory-mapped file reader using `memmap2`. Opens POD5 files, parses the FlatBuffer footer, and provides iterators over reads and signal data.
- **writer/file_writer.rs**: Buffered writer that constructs POD5 files. Handles signal chunking, VBZ compression, and batching of Arrow record batches.
- **compression/**: VBZ signal compression (SVB16 + ZSTD pipeline)
  - `svb16/mod.rs`: Scalar SVB16 encode/decode + runtime dispatch to SIMD.
  - `svb16/simd_ssse3.rs`: SSSE3 encode/decode (~2× vs scalar on x86_64 w/ SSSE3).
  - `svb16/tables.rs`: `pshufb` shuffle + length tables for both directions.
  - `vbz.rs`: Full VBZ pipeline combining SVB16 with ZSTD compression
- **footer.rs**: Manual FlatBuffer parsing for the POD5 footer (locates embedded Arrow tables)
- **schema/**: Arrow schema definitions for reads, signal, and run_info tables
- **types.rs**: Core data types (`ReadData`, `RunInfoData`, `EndReason`, etc.)
- **merge.rs**: File merging operations with run info deduplication
- **operations/**: High-level file operations
  - `filter.rs`: Filter reads by criteria (ID list, sample count, end reason)
  - `repack.rs`: Repack files with new compression settings
  - `subset.rs`: Split reads into multiple files by barcode or CSV mapping
- **demux/**: Barcode demultiplexing module
  - `mod.rs`: Main demux API with WarpDemuX model loading
  - `svm.rs`: SVM-based classification with probability output
  - `train.rs`: Model training from labeled fingerprints
- **dtw/**: Dynamic Time Warping distance computation
  - `distance.rs`: DTW algorithm with Sakoe-Chiba band constraint
  - `fingerprint.rs`: Signal fingerprint representation
  - `kernel.rs`: DTW distance to kernel conversion for SVM
- **segmentation/**: Signal segmentation algorithms
  - `llr.rs`: Log-Likelihood Ratio boundary detection
  - `ttest.rs`: T-test based changepoint segmentation
  - `normalize.rs`: MAD, z-score, and min-max normalization

### CLI Commands

- `view`: Display reads as TSV with configurable columns
- `inspect`: Show file metadata (summary, reads list, specific read)
- `summary`: Comprehensive summary with statistics
- `merge`: Combine multiple POD5 files (parallel reading with rayon)
- `filter`: Extract reads by ID list or criteria (sample count, end reason)
- `bam-filter`: Filter reads based on paired BAM file (mapped status, region, quality)
- `repack`: Repack files for optimized storage
- `subset`: Split reads into multiple files based on CSV mapping
- `resquiggle`: Refine signal-to-base mapping using banded DP with POD5 signal and BAM move tables
- `demux`: Barcode demultiplexing workflow with subcommands:
  - `detect`: LLR-based adapter boundary detection
  - `fingerprint`: T-test segmentation for barcode fingerprints
  - `classify`: DTW-based barcode classification
  - `split`: Split reads by barcode into separate files
  - `train`: Train reference fingerprints from known samples
  - `train-svm`: Train SVM model (requires `train` feature)

### Key Patterns

**Block-level copying**: For merge/filter operations, signal data is kept compressed (`CompressedSignalChunk` with `Arc<[u8]>`) to avoid decompression/recompression overhead. Use `add_read_with_compressed_signal()` instead of `add_read()` when copying between files.

**Dictionary tracking**: The writer maintains O(1) lookup for pore types and end reasons using HashMap indexes alongside Vec storage for Arrow dictionary encoding.

**Run info deduplication**: When merging files, run infos are deduplicated by `acquisition_id` to avoid redundant entries.

## Dependencies

### Core Library (escapepod)
- **arrow**: Arrow IPC file reading/writing
- **memmap2**: Memory-mapped file I/O
- **zstd**: ZSTD compression for VBZ
- **flatbuffers**: FlatBuffer footer parsing
- **uuid**: Read ID handling
- **ndarray**: Array operations for signal processing
- **csv**: CSV parsing for filter IDs and barcode mappings
- **serde/serde_json**: JSON model serialization
- **linfa/linfa-svm**: SVM training (optional, requires `train` feature)

### CLI (escapepod-cli)
- **clap**: CLI argument parsing
- **rayon**: Parallel iteration for merge operations
- **tabled**: Table formatting for CLI output
- **noodles-bam/sam**: BAM file support for bam-filter command
- **walkdir**: Directory traversal

## Test Data

Test POD5 files from Oxford Nanopore are in `ext/nanopore-dna-data/pod5/`. The `ext/pod5-file-format/` directory contains the official POD5 specification.

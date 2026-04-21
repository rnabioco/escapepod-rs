# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

escapepod-rs is a pure Rust implementation for reading and writing POD5 files, the native file format for Oxford Nanopore sequencing data. The workspace splits the library into two layers — `escapepod-pod5` for format I/O and `escapepod-signal` for signal-processing algorithms — plus a CLI tool (`escapepod-cli`) and Python bindings (`escapepod-python`).

## Requirements

- Rust 1.88 or later (matches `workspace.package.rust-version`)

## Build Commands

```bash
# Build
cargo build --release

# Build with training support (enables SVM model training)
cargo build --release --features train

# Build with GPU-accelerated DTW (experimental, opt-in; needs the CUDA
# driver + libnvrtc at runtime — no nvcc required at build time).
# The pixi `gpu` env provides libnvrtc via conda-forge's cuda-nvrtc package
# and sets LD_LIBRARY_PATH automatically:
pixi run -e gpu cargo build --release --features gpu -p escapepod-cli

# Test / bench on a GPU node (SLURM account `gpu_rbi`, partition `gpu`):
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e gpu cargo test --features gpu -p escapepod-signal --test gpu_dtw
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e gpu cargo bench --features gpu --bench hot_paths_gpu

# Use on a node with a visible GPU:
escpod demux classify --model model.json reads.fp.csv --gpu -o out.tsv

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

`crates/escapepod-signal/benches/hot_paths.rs` covers the audit hot paths: DTW, resquiggle DP, fingerprint MAD, VBZ encode/decode, DTW matrix.

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
RUST_LOG=escapepod_signal::reader=trace escpod view file.pod5   # module-scoped
escpod -q merge …                         # errors only
```

## Architecture

### Workspace Structure

- **escapepod-pod5**: POD5 format I/O layer — reader, writer, VBZ compression, footer parsing, block-level merge/filter/repack/subset operations.
- **escapepod-signal**: Signal-processing algorithms (DTW, resquiggle, segmentation) layered on top of `escapepod-pod5`. Re-exports the pod5 surface so downstream consumers can depend on a single crate.
- **escapepod-demux**: WarpDemuX-compatible barcode demultiplexing — SVM model loaders, DTW+SVM classifier, Platt scaling, optional SVM training (`train`), GPU DTW batch classify (`gpu`), ADAPTed CNN adapter detection (`cnn-detect`). Depends on `escapepod-signal` for DTW and fingerprint primitives.
- **escapepod-cli**: CLI binary (`escpod`). Demux commands require building with `--features demux` (pulls in `escapepod-demux`); forward features `train`, `gpu`, `cnn-detect` propagate to the demux crate.
- **escapepod-python**: pyo3 bindings.

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

### Format layer (escapepod-pod5)

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

### Signal layer (escapepod-signal)

- **dtw/**: Dynamic Time Warping distance computation
  - `distance.rs`: DTW algorithm with Sakoe-Chiba band constraint
  - `fingerprint.rs`: Signal fingerprint representation
  - `kernel.rs`: DTW distance to kernel conversion for SVM
  - `cuda/`: GPU-accelerated DTW distance matrix (opt-in `gpu` feature)
- **segmentation/**: Signal segmentation algorithms
  - `llr.rs`: Log-Likelihood Ratio boundary detection
  - `ttest.rs`: T-test based changepoint segmentation (scipy-compatible peak detection)
  - `normalize.rs`: MAD, z-score, and min-max normalization
- **resquiggle/**: Signal-to-base alignment refinement (banded DP, rescaling, drift correction)

### Demux layer (escapepod-demux)

- `model.rs`: `WarpDemuxModel` and `DtwSvmModel` JSON loaders.
- `classify.rs`: Per-read DTW classifier (`classify_read`), shared top-2 threshold logic, optional batched GPU classifier.
- `svm.rs`: Full SVM predictor — RBF kernel, OvO dual coefficients, Platt scaling, multiclass probability coupling, batched GPU classify.
- `probability.rs`: softmax / coupling utilities.
- `train.rs` (feature: `train`): SVM training via linfa-svm, optional GPU all-pairs DTW matrix.
- `adapter_cnn.rs` (feature: `cnn-detect`): port of ADAPTed's `BoundariesCNN` through tract-onnx. Loads ADAPTed-exported ONNX model at runtime; weights are CC BY-NC 4.0 and are NOT bundled.

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

### Format crate (escapepod-pod5)
- **arrow**: Arrow IPC file reading/writing
- **memmap2**: Memory-mapped file I/O
- **zstd**: ZSTD compression for VBZ
- **flatbuffers**: FlatBuffer footer parsing
- **uuid**: Read ID handling
- **csv**: CSV parsing for filter IDs and barcode mappings
- **byteorder**, **thiserror**, **tempfile**, **rayon**

### Signal crate (escapepod-signal)
- **escapepod-pod5**: re-exported as `pod5` plus the full type surface
- **ndarray**: Array operations for signal processing
- **rand**, **flate2**: resquiggle internals
- **serde/serde_json**: JSON model serialization (demux)
- **linfa/linfa-svm**: SVM training (optional, requires `train` feature)

### CLI (escapepod-cli)
- **clap**: CLI argument parsing
- **rayon**: Parallel iteration for merge operations
- **tabled**: Table formatting for CLI output
- **noodles-bam/sam**: BAM file support for bam-filter command
- **walkdir**: Directory traversal

## Test Data

Test POD5 files from Oxford Nanopore are in `ext/nanopore-dna-data/pod5/`. The `ext/pod5-file-format/` directory contains the official POD5 specification.

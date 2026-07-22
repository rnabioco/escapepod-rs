# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

escapepod-rs is a pure Rust implementation for reading and writing POD5 files, the native file format for Oxford Nanopore sequencing data. The workspace splits the library into two layers — `escapepod-pod5` for format I/O and `escapepod-signal` for signal-processing algorithms — plus the `escapepod-cli` crate (the `escpod` CLI binary, with an optional umbrella library re-exporting the layers) and Python bindings (`escapepod-python`).

## Requirements

- Rust 1.95 or later (matches `workspace.package.rust-version`)

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
    pixi run -e gpu cargo nextest run --features gpu -p escapepod-signal --test gpu_dtw
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e gpu cargo bench --features gpu --bench hot_paths_gpu

# Use on a node with a visible GPU:
escpod demux classify --model model.json reads.fp.csv --gpu -o out.tsv

# Run tests (cargo-nextest)
cargo nextest run

# Doctests — nextest does not run doctests, keep them on a separate invocation
cargo test --doc --workspace

# Run a specific test
cargo nextest run test_round_trip_single_read

# Run clippy lints
cargo clippy

# Run the CLI (after building)
./target/release/escpod <command>

# mold is the linker full time: the base [activation] in pixi.toml exports
# LD_PRELOAD=$CONDA_PREFIX/lib/mold/mold-wrapper.so for EVERY environment, so
# any `pixi run [-e <env>] cargo …` (and maturin) links with mold — no
# `mold -run` and no `-e dev` needed for that. Works with system gcc 11.5 (no
# `-fuse-ld=mold` support required) and needs no glibc-static. Release
# artifacts shipped via GitHub Releases build in CI against musl (static),
# outside pixi, and are unaffected; local builds remain dynamic gnu by design.
#
# The `dev` env additionally provides cargo-nextest + convenience task wrappers
# (which are now just bare `cargo …`, since mold is already on):
pixi run -e dev build        # cargo build
pixi run -e dev build-rel    # cargo build --release (dynamic gnu)
pixi run -e dev test         # cargo nextest run
pixi run -e dev test-doc     # cargo test --doc (nextest skips doctests)
pixi run -e dev check        # cargo check
pixi run -e dev clippy       # cargo clippy --workspace --all-targets

# GPU builds also link with mold (dev-gpu, or any gpu env):
pixi run -e dev-gpu cargo build --features gpu -p escapepod-cli
```

### Build baseline and SLURM builds

Local Linux/x86_64 builds pin `-C target-cpu=x86-64-v3` via `.cargo/config.toml`
(AVX2 + FMA + BMI2 + POPCNT + F16C). This is portable across every node in
the cluster (Broadwell login, Cascade Lake rna, Ice Lake gpu). Do **not**
use `target-cpu=native`: a binary built on a gpu node uses Ice Lake-only
instructions (VBMI, VPCLMULQDQ, …) that SIGILL on rna. Hot kernels that
want AVX-512 do so via `#[target_feature]` + runtime `is_x86_feature_detected!`
dispatch, not a global baseline bump.

The login node has only 2 cores — wrap any heavy build or bench in `srun -p rna`:

```bash
# build / test — 32 logical CPUs (16 physical cores + HT) is enough.
# These already link with mold (base activation, see above); no -e dev needed.
srun -p rna -c 32 --mem=32G pixi run cargo build --release
srun -p rna -c 32 --mem=32G pixi run cargo nextest run --workspace

# The dev env adds cargo-nextest + task wrappers. mold is multithreaded on the
# link step, so the 32-core allocation helps both compile and link phases.
srun -p rna -c 32 --mem=32G pixi run -e dev build-rel
srun -p rna -c 32 --mem=32G pixi run -e dev test

# throughput-sensitive demux runs — ask for a full socket (48 logical = 24 physical + HT).
# SLURM's `-c 32` only allocates 16 physical cores on rna's Gold 6240R, not 32;
# `-c 48` fills the socket and gives ~20% more wall-clock speedup on fingerprint
# without crossing NUMA boundaries. Crossing sockets (`-c 64`+) is a crapshoot on
# a shared node — other jobs on the second socket regress wall time.
srun -p rna -c 48 --mem=64G pixi run escpod demux fingerprint …
```

### Running on Alpine (CU Boulder RMACC/ACCESS)

The `rna`/`gpu` partitions above are the **Beevol** (CU Anschutz) cluster. On
**Alpine** the SLURM model differs: every partition needs an **explicit `--qos=`
matched to the partition** plus `-A amc-general` (the CU Anschutz allocation) — the
partition name alone is not enough. Translation:

| Purpose | Beevol | Alpine |
|---|---|---|
| CPU build / test / bench | `srun -p rna -c 32 --mem=32G` | `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G` |
| CPU full node (throughput) | `srun -p rna -c 48` | `srun -p amilan --qos=normal -A amc-general -c 64 --mem=120G` |
| GPU test / bench | `srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1` | `srun -p aa100 --qos=gpu-normal -A amc-general -c 16 --gres=gpu:1` |
| Light build (login is 2 cores) | login node | `srun -p acompile --qos=compile -A amc-general -c 16 --mem=32G` |

QOS↔partition: amilan→`normal`|`long`; aa100/al40→`gpu-normal`|`gpu-long`;
acompile→`compile`; atesting→`testing` (1 h cap). Default mem is 3840 MB/core.

Notes specific to Alpine:
- `amilan` nodes are AMD EPYC Milan, 2×32 = **64 physical cores, no hyperthreading**,
  245 GB. So the Beevol "`-c 48` fills a socket / `-c 64` crosses NUMA" tuning does
  **not** apply — `-c N` is N physical cores; one full node is 64 across 2 sockets.
- GPU CUDA work must use `aa100` (A100) or `al40` (L40). `ami100` is AMD/ROCm and
  **cannot** run the CUDA `gpu`/`cnn-gpu` features.
- All of amilan/aa100/acompile are Zen3 x86-64, so the pinned `target-cpu=x86-64-v3`
  baseline is portable across them — build once, run anywhere, no SIGILL risk.
- Toolchain lives in pixi, not on the bare PATH: mold links every env (base
  activation); `-e dev` adds cargo-nextest + task wrappers, `-e warpdemux-bench`
  (hyperfine + pod5, for `benchmarks/benchmark.sh`), `-e gpu` (CUDA runtime /
  libnvrtc). Wrap invocations accordingly, e.g.
  `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G pixi run -e dev test`.

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

All CLI status/progress/warning output flows through `tracing` to stderr (the
custom `EscpodFormatter` in `main.rs` renders `timestamp LEVEL [target] message`).
Command *data* — TSV/CSV rows, `inspect`/`summary` reports, ID lists — stays on
**stdout** via `println!`, so it can be piped/redirected independently of logs.

Default level is **info** (status visible out of the box). Control via CLI flags
or `RUST_LOG` (which always wins when set):

```bash
escpod inspect summary file.pod5         # info (default): status + warnings
escpod -v merge *.pod5 -o out.pod5       # debug
escpod -vv merge *.pod5 -o out.pod5      # trace
escpod -q merge …                         # errors only (status + progress bars suppressed)
RUST_LOG=escapepod_signal::reader=trace escpod view file.pod5   # module-scoped
```

Progress bars/spinners (`progress.rs`) are status output too: they auto-hide
when the level drops below info (i.e. under `-q`). Multi-line styled report
blocks (e.g. `merge --profile` timings, demux summaries) are gated on
`tracing::enabled!(Level::INFO)` rather than emitted as per-line events.

When adding output: use `tracing::info!`/`warn!`/`error!` for diagnostics
(don't hand-prefix messages with `Warning:`/`Note:` — the formatter prints the
level); use `println!`→stdout only for the command's actual data product.

## Architecture

### Workspace Structure

- **escapepod-pod5**: POD5 format I/O layer — reader, writer, VBZ compression, footer parsing, block-level merge/filter/repack/subset operations.
- **escapepod-signal**: Signal-processing algorithms (DTW, resquiggle, segmentation) layered on top of `escapepod-pod5`. Re-exports the pod5 surface so downstream consumers can depend on a single crate.
- **escapepod-demux**: WarpDemuX-compatible barcode demultiplexing — SVM model loaders, DTW+SVM classifier, Platt scaling, optional SVM training (`train`), GPU DTW batch classify (`gpu`), boundary-CNN adapter detection (`cnn-detect`, CPU tract; optional onnxruntime CUDA via `cnn-gpu`). Depends on `escapepod-signal` for DTW and fingerprint primitives.
- **escapepod-cli**: the `escpod` CLI binary (built by the default `cli` feature) plus an optional umbrella library (imported as `escapepod_cli`) that re-exports `pod5`/`signal`/`demux` behind feature flags for `default-features = false` consumers. Demux commands require building with `--features demux` (pulls in `escapepod-demux`); forward features `train`, `gpu`, `cnn-detect`, `cnn-gpu` propagate to the demux crate.
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
- `adapter_cnn.rs` (feature: `cnn-detect`): runs a user-supplied boundary-CNN ONNX graph (`[B,1,L] -> [B,2,L]`) through tract-onnx at runtime, one read at a time (CPU). Architecture-agnostic — works with escapepod-models' `adapter_rna004` TCN (CC-BY) or an ADAPTed `BoundariesCNN` export (CC BY-NC, NOT bundled). Shared `prep_adapter_signal`/`decode_adapter_end`/`group_by_len` helpers + a batched `detect_adapter_end_batch` (exact-length grouping, no cross-read padding) back both the CPU and GPU paths bit-identically. Prep truncates the model input to `search_window + receptive-field margin` (`ESCAPEPOD_CNN_MARGIN`, default 256 ⇒ cap 806; output-preserving since the local graph can't reach past it) — this also collapses every read longer than the cap onto one length so the GPU can batch them. A load-time shape probe rejects a wrong-output-shape model up front instead of silently writing `adapter_end=0`.
- `adapter_cnn_gpu.rs` (feature: `cnn-gpu`, implies `cnn-detect`): the architecture-agnostic **GPU** path — same ONNX graph + same prep/decode, run batched through onnxruntime's CUDA execution provider via the `ort` crate (`load-dynamic`; needs a CUDA-enabled `libonnxruntime` on `ORT_DYLIB_PATH` + a visible GPU at run time, nothing at build time). `escpod demux detect --method cnn --gpu`. The TCN is **inference-bound, not I/O-bound** (LLR detect = 0.25 s/20k vs CNN = 77 s/20k; ~99.6% is inference), so GPU pays off — ~7.6× end-to-end on an A30 at 20k reads (grows at scale; isolated inference is ~99× tract). `detect --gpu` runs a dedicated GPU consumer thread (the ort/CUDA session builds while CPU producers decode+prep in parallel, overlapping init) fed prepped, length-bucketed blocks through a bounded channel (`AdapterCnnGpu::detect_prepped`). The **fused pipeline** (`escpod demux --method cnn --gpu`) also runs GPU detection: all producers (`produce_cpu`/`produce_cpu_gbm`/`produce_gpu`) detect via `Detector::detect_batch` over windowed decode-once blocks (`DETECT_WINDOW`), so the GPU does one batched call per window while preserving the single-stream I/O (#72) — measured ~7.2× end-to-end (116 s→16.2 s, A30, 20k GBM) with 99.99% classification parity vs CPU detect. NOT the old arch-locked CUDA kernel (removed in #80, hardwired to BoundariesCNN); this runs any `[B,1,L]->[B,2,L]` graph. tract has no efficient batched conv, so CPU detection stays per-read.

### CLI Commands

- `view`: Display reads as TSV with configurable columns
- `inspect`: Show file metadata (summary, reads list, specific read)
- `summary`: Comprehensive summary with statistics
- `merge`: Combine multiple POD5 files (parallel reading with rayon)
- `filter`: Extract reads by ID list or criteria (sample count, end reason)
- `bam-filter`: Filter reads based on paired BAM file (mapped status, region, quality)
- `repack`: Repack files for optimized storage
- `subset`: Split reads into multiple files based on CSV mapping
- `resquiggle`: Refine signal-to-base mapping using banded DP with POD5 signal and BAM move tables. Takes a k-mer level table via `--kmer-table <path>` or a named model via `--kmer-model <name>` (DNA + RNA; `dna_r10.4.1_e8.2_400bps`, `rna004`, …). Named models resolve from a local cache (`$ESCAPEPOD_KMER_CACHE` → `$XDG_CACHE_HOME/escapepod/kmer_models` → `~/.cache/…`) that is **never** populated at runtime — build with `--features models-download` and prefetch on a networked login node (`escpod resquiggle models fetch --all`) before submitting compute jobs (Alpine/Beevol compute nodes can't reach GitHub). Tables come from nanoporetech/kmer_models (MPL-2.0), pinned to a commit + sha256; the code path uses `ureq`/rustls so the static-musl release stays OpenSSL-free.
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

### CLI (escapepod, `cli` feature)
- **clap**: CLI argument parsing
- **rayon**: Parallel iteration for merge operations
- **tabled**: Table formatting for CLI output
- **noodles-bam/sam**: BAM file support for bam-filter command
- **walkdir**: Directory traversal

## Test Data

Test POD5 files from Oxford Nanopore are in `ext/nanopore-dna-data/pod5/`. The `ext/pod5-file-format/` directory contains the official POD5 specification.

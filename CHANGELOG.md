# Changelog

## Unreleased

### Added

- **POD5 reads-table schema V5**: the `expected_open_pore_level` and
  `selected_read_level` fields (both `float32`), introduced upstream in pod5
  0.3.44, are now read, written, merged/filtered, surfaced in `view`/`inspect`
  (and as selectable output fields), and exposed on the Python `ReadData` /
  `Writer.add_read` API. Files escapepod writes are now stamped `pod5_version`
  0.3.44 and verified readable by the reference ONT `pod5` reader; existing
  V0–V4 files still read, with the new fields defaulting to 0.0.
- Defensive pre-mmap probe when opening a POD5 file: the header and footer are
  read through ordinary I/O before the file is memory-mapped, so a truncated
  file or an archive "stub" (unresident data on HSM/tape-backed filesystems)
  surfaces a recoverable error instead of an uncatchable SIGBUS on first page
  fault. Mirrors upstream pod5 0.3.37; set `POD5_DISABLE_MMAP_OPEN=1` to skip.
- `escpod demux <file> --model M -d OUT` now runs a **fused, streaming
  pipeline** by default: each read's signal is decoded once, then detect +
  fingerprint + classify run in a single pass and reads are routed
  (block-level compressed copy) into per-barcode POD5s — no intermediate
  boundaries/fingerprints/classifications files. The granular
  `detect`/`fingerprint`/`classify`/`split`/`train` subcommands remain for
  advanced use. `--classifications` writes the per-read CSV only when asked.
- Experimental GPU primitives (behind `--features gpu`) for the demux signal
  chain — SVB16 decode, t-test fingerprint, LLR detect — kept as validated,
  reusable kernels. They are **not** used by `escpod demux`: measurement shows
  the prep stages run faster on a multi-core CPU than on the GPU.

### Removed

- The **GPU CNN adapter-detection path** (`demux detect --method cnn --gpu`,
  plus the `--cnn-weights` flag and `scripts/dump_adapter_cnn_weights.py`). Its
  hand-written CUDA kernels were hardcoded to ADAPTed's `BoundariesCNN`
  topology (3× Conv1d + ConvTranspose1d, fixed K=7/C=64) and could not run any
  other architecture — including escapepod-models' replacement TCN. CNN
  detection (`--method cnn`) now runs **only** through the architecture-agnostic
  tract-onnx CPU path (`adapter_cnn.rs`), which accepts any `[B,1,L] -> [B,2,L]`
  ONNX graph. This is not a regression at typical scales: `detect` is dominated
  by POD5 read + signal prep, not CNN compute, so the CPU path is as fast or
  faster (the GPU flag's own help already said as much). Removing it also drops
  the CC-BY-NC `.weights` dumper. If a GPU CNN path is ever needed again, add an
  ONNX-generic backend (e.g. ORT CUDA EP) rather than a per-architecture kernel.

### Changed

- The LLR detect `--downscale` default is now **10** (the WarpDemuX-native
  mode) for `demux` and `demux detect`, up from 1. This makes detect — the
  dominant prep stage — ~5× faster, with ~98% barcode agreement versus
  full-resolution (ds=1). Pass `--downscale 1` to restore full-resolution
  detect.
- Dependency bumps (no behavior change): Arrow ecosystem `arrow` + `parquet`
  58 → 59, `tabled` 0.20 → 0.21, and the `noodles-*` BAM stack (`bam` 0.90,
  `sam` 0.85, `bgzf` 0.47, `core` 0.20, `csi` 0.56). `ndarray` is held at 0.16
  — the `linfa` SVM stack still pins it, so 0.17 is blocked upstream.

### Performance

- Codebase-wide optimization/refactor sweep (#86), all bit-identical output:
  - **Resquiggle adaptive banded DP ~31% faster** — the per-base traceback no
    longer heap-allocates a `Vec` per base; the whole read shares one flat
    buffer.
  - **O(1) POD5 read-batch access** — `read_batch(i)` / `read_ids_from_batch(i)`
    now seek via the Arrow IPC footer instead of decoding every preceding batch.
    Iterating a many-read-batch file (e.g. the Python `Reader` read iterator)
    drops from O(B²) to O(B) batch decodes: ~10× faster random batch access and
    ~2.6× faster full-file iteration on a 1.65M-read / 166-batch file.
  - **Signal median computations are O(n) instead of O(n log n)** — the SVM
    kernel γ-heuristic, Theil–Sen rescale, and resquiggle dwell median now use
    `select_nth_unstable` instead of a full sort.
  - Smaller per-read allocations on the demux/classify and fingerprint hot
    paths (MAD-normalization scratch reuse, Platt coupling workspace sized once).
- Internal consolidation with no behavior change: six duplicated median impls
  unified into `escapepod-signal::stats`; the SVM RBF-kernel mapping and the
  CPU/GPU CNN batch packing/scatter each live in one shared helper.

### Fixed

- Resolved a PyPI name collision: both the `escapepod` CLI crate and the
  `escapepod-python` bindings crate declared `name = "escapepod"`. The PyPI
  `escapepod` distribution is the **Python `Reader` bindings**
  (`escapepod-python`); the `escpod` CLI now ships via `cargo install
  escapepod` and GitHub release binaries only, so its maturin `pyproject.toml`
  (a `bindings = "bin"` wheel) has been removed. This reverses the 0.5.1 note
  about `pip install escapepod` installing the CLI.

## 0.5.1 (2026-06-14)

### Changed

- The CLI now ships from the `escapepod` crate (renamed from
  `escapepod-cli`), so `cargo install escapepod` installs the `escpod`
  binary. The same crate doubles as an umbrella library: with
  `default-features = false` plus `pod5` / `signal` / `demux`, it
  re-exports each layer (e.g. `escapepod::signal`) without pulling in the
  CLI's dependency tree. `demux` stays opt-in until it stabilizes.
- The maturin binary wheel is published as `escapepod` (was
  `escapepod-cli`) so `pip install escapepod` matches `cargo install`.

### Fixed

- Packaging `readme` pointed at a nonexistent path, which made
  `cargo package` fail; the workspace now points every publishable crate
  at the root `README.md`. `escapepod-python` is marked `publish = false`.
- `demux fingerprint` (test fixture): labeled-Parquet temp files lacked a
  `.parquet` suffix, so format detection read them as CSV and the parquet
  loaders failed with an "invalid UTF-8" error.

### Build / Tooling

- Gated the `train`-only labeled-fingerprint loaders behind
  `#[cfg(feature = "train")]`, removing dead-code warnings from
  `--features demux` builds.
- Bumped GitHub Actions to current majors (checkout v6, upload-artifact
  v7, download-artifact v8, setup-python v6, setup-pixi v0.9.6,
  action-gh-release v3, actions-netlify v4), clearing the Node 20
  deprecation warnings.

## 0.5.0 (2026-04-27)

### Added

- `demux fingerprint`: Parquet output when `-o` ends in `.parquet`, plus
  an `--emit-dwell` flag that adds per-segment dwell-time features.
- `demux classify` (CLI): `fp_io` module reads fingerprint inputs from
  both Parquet and CSV (gzipped CSV included); new flags
  `--gpu-chunk-cells` and `--threads`, with model auto-detection so
  `--model` accepts any supported format.
- `escapepod-demux`: `AnyModel` enum and `load_any_model()` for
  format-agnostic SVM/DTW model loading.
- `escapepod-signal`: SVM helper CUDA kernels exposed via function
  handles for downstream GPU pipelines.

### Performance

- `demux classify` (GPU): on-GPU RBF + OvO decision pipeline
  (`GpuSvmContext`) keeps SVM evaluation on the device; producer/
  consumer pipeline parallelizes the consumer side and bumps the
  default chunk to 4G cells with channel depth 2 for better GPU
  utilization on long runs. Per-chunk indicatif progress bar surfaces
  throughput.
- `escapepod-demux`: RBF kernel fast paths for `power == 1.0` and
  `power == 2.0` skip the generic `powf` call.
- `escapepod-pod5`: filter and merge hot paths reworked; remaining
  `reader.reads()` callers now batch-amortize the schema/footer parse,
  and a `PoreType` newtype removes per-read string churn.

### Fixed

- `train` (multiclass OvO): dropped an unused SMO solve path that ran
  during training without contributing to the final model.

### Build / Tooling

- Pixi `dev` env wires `mold -run` for fast local links (system gcc
  11.5, no glibc-static needed); release artifacts in CI continue to
  build against musl.
- Docs: benchmark page leads with bulk operations (`merge`, `filter`,
  `subset`, `repack`); `inspect` and `view` demoted to a secondary
  section.

## 0.4.0 (2026-04-22)

### Performance

- `demux fingerprint`: nested `par_iter` streams signals across files and
  reads so fingerprinting a 48-file run drops from ~32 min to ~9.8 s on
  the rna partition.
- `demux classify` / `train-svm`: reusable per-thread `SvmWorkspace` and
  a streaming (rayon fan-out + single writer) output path cut RSS by
  ~37% and remove a serialize-then-write stall.
- `svb16`: AVX2 decode path (16 samples/iter), preferred at runtime over
  SSSE3 when available.
- `dtw`: split the inner band loop so the trailing segment auto-
  vectorizes under AVX2; the x86-64-v3 baseline (AVX2 + FMA + BMI2 +
  POPCNT + F16C) is now pinned in `.cargo/config.toml` for portability
  across Broadwell/Cascade Lake/Ice Lake cluster nodes.
- `segmentation::llr`: allocation-free `best_split` and an opt-in
  early-stop variant.
- CLI: progress-bar updates throttled out of hot paths.

### Changed

- Moved from CLI into libraries (additive for library consumers):
  - `ReadBoundaries` and fingerprint types/helpers now live in
    `escapepod-demux`.
  - `normalize_signal(&[i16])` and the CLI's `downscale_signal` now
    live in `escapepod-signal` (the CLI's duplicate was removed).
- Docs: recommend `srun -c 48` for throughput-sensitive demux runs on
  the rna partition (fills one socket without crossing NUMA).

## 0.3.1 (2026-04-21)

### Added

- `resquiggle::banded_dp_with_penalty_table` — banded Viterbi DP variant
  that accepts a caller-supplied short-dwell penalty table and uses its
  length as the check horizon. Tie-break is strict (`<`), matching the
  Remora-compatible refinement pipeline. Complements the existing
  `banded_dp` which builds the asymmetric penalty internally.
- `segmentation::mad_normalize_robust` — median-MAD normalization with
  the 1.4826 Gaussian scale factor and graceful fallback (returns
  `signal - median` without dividing) when MAD is essentially zero.
  Complements `mad_normalize`, which panics on constant signals.

### Performance

- Hot-path audit across fingerprint MAD, SVM prediction, `view`, and
  `merge`. Fingerprint MAD uses a single-pass median/MAD with an
  in-place partition; SVM prediction reuses per-thread scratch buffers
  and avoids redundant kernel evaluations on the OvO dual path; CLI
  `view` streams reads with reused formatting buffers; `merge` switches
  to mmap-backed readers where possible to cut per-file overhead.

### Fixed

- `escapepod-python` cdylib now links cleanly under a plain
  `cargo build` on macOS. A `build.rs` emits the pyo3
  extension-module link args (equivalent to
  `pyo3_build_config::add_extension_module_link_args()`), scoped to the
  cdylib, so the build no longer fails with undefined `_Py*` symbols
  when maturin is not driving the build. macOS is now in the CI matrix
  for `check`, `test`, and `clippy` to catch regressions.

### Changed

- Workspace crates moved under `crates/` (no public-API change).
- Docs reorganised with an "experimental tools" section; `resquiggle`
  and `index` CLI subcommands are gated behind their respective Cargo
  features.

## 0.3.0 (2026-04-20)

### Breaking

- Barcode demultiplexing moved out of `escapepod-signal` into a new
  `escapepod-demux` crate. The `escapepod_signal::demux` module is gone;
  downstream code should depend on `escapepod-demux` directly and
  import from `escapepod_demux::...` (model loaders, `classify_read`,
  SVM predictor/trainer, Platt scaling, GPU batch classify, ADAPTed
  CNN adapter detection). The `escpod demux` CLI surface is unchanged,
  but `escapepod-cli`'s `demux` Cargo feature now pulls in the new
  crate; the `train`, `gpu`, and `cnn-detect` features forward to it.

### Added

- GPU-accelerated DTW for demux, opt-in via `--features gpu` on
  `escapepod-signal` and `escapepod-cli`. Wires up `escpod demux classify
  --gpu` (WarpDemuX model, CSV reference, and SVM model paths) and
  `escpod demux train-svm --gpu`. CUDA kernel is NVRTC-compiled at
  runtime, so no `nvcc` is required at build time — only the CUDA driver
  and `libnvrtc` at run time. On the lab cluster, `pixi run -e gpu …`
  provisions `cuda-nvrtc` via conda-forge and sets `LD_LIBRARY_PATH`.
  Anti-diagonal kernel with shared-memory-cached fingerprints; single-
  warp blocks with `__syncwarp()` and `__launch_bounds__(32, 64)`.
  Measured ~7.7× speedup over CPU rayon on A30 at 1024×40×110 and
  8192×40×110 workloads (criterion, band w=10).
- `GpuDtwContext`, `dtw_distance_matrix_gpu`, `classify_reads_gpu`,
  `classify_with_svm_batch_gpu`, `compute_distance_matrix_gpu`,
  `train_svm_gpu` public API on `escapepod-signal` (all `gpu`-gated).
- CPU `classify_read` now uses `dtw_distance_bounded` with the running
  second-best squared distance as an upper bound, skipping DTW work for
  training fingerprints that cannot change the top-2. Safe for both
  ratio and kernel threshold paths.

### Fixed

- **Behavior change for windowed DTW.** The 2-row banded DP in
  `dtw_distance` / `dtw_distance_bounded` was leaving stale
  `curr[j_start - 1]` values from earlier rows, letting the recurrence
  read an out-of-band predecessor and occasionally find a shorter-than-
  valid path. The classical Sakoe-Chiba band treats those cells as
  unreachable; we now re-seed the boundary to `INF` at the top of each
  row and also short-circuit to `INF` when `|n − m| > w` (the endpoint
  itself is outside the band and the DP would otherwise propagate a
  stale in-band value through the trailing empty rows). Only affects
  callers that pass `Some(window)`; unwindowed DTW is unchanged. In
  practice the difference is small but non-zero on real data — any
  downstream classify output produced with a band constraint may shift
  slightly, with GPU and CPU now agreeing bit-for-bit up to f32
  summation order.

## 0.2.0 (2026-04-20)

### Breaking

- Workspace split into two library crates. POD5 format I/O (reader, writer,
  VBZ compression, merge/filter/repack/subset, schema, footer, types, errors)
  now lives in the new `escapepod-pod5` crate. The crate formerly called
  `escapepod` has been renamed to `escapepod-signal` and contains the
  signal-processing algorithms (DTW, resquiggle, segmentation) layered on
  top of `escapepod-pod5`. Downstream consumers depending on `escapepod`
  by name must rename to `escapepod-signal`; the pod5 surface is
  re-exported from `escapepod-signal` so most `use escapepod::...` paths
  translate to `use escapepod_signal::...` with no other changes.
- Barcode demultiplexing is now opt-in. The `escapepod-signal::demux`
  module and the `escpod demux` CLI subcommand require building with
  `--features demux`; the `train` feature now implies `demux`.

### Added

- `escapepod-pod5` crate for POD5 format I/O.
- `demux` Cargo feature on both `escapepod-signal` and `escapepod-cli`.

### Changed

- README no longer advertises barcode demultiplexing as a shipped feature;
  `docs/cli/demux.md` carries an experimental admonition.
- CLI now declares demux and resquiggle commands as experimental in the
  commands index.

### Removed

- Empty `escapepod-vortex/` directory (content preserved on the
  `escapepod-vortex` branch).
- Stale `PROGRESS.md` and the top-level `examples/test_ipc.rs` scratch
  file; `examples/dtw_example.rs` moved under
  `escapepod-signal/examples/`.

## 0.1.3 (2026-04-20)

### Added

- Tracing-based runtime verbosity (`-v`/`-vv`/`-q`, `RUST_LOG`)
- `release-with-debug` and `profiling` build profiles; phase timer
- Criterion microbenches covering audit hot paths (`escapepod/benches/hot_paths.rs`)

### Changed

- SSSE3 SIMD encode/decode for SVB16 (~2× vs scalar on x86_64)
- Audit-driven hot-path optimizations across reader, DTW, demux, DP
- Dropped `escapepod-vortex` workspace member
- `[profile.bench]` pinned to inherit from release

### Fixed

- Clippy lints: `unnecessary_sort_by`, `needless_range_loop`

## 0.1.0 (2026-03-19)

First stable release of escapepod-rs.

### Added

- **index**: `.p5i` sidecar read index for fast UUID lookup (`escpod index`), with zstd-compressed entry blocks, sorted-vec binary search, and file size checksum validation
- **filter**: Sample count and end reason filters, stdin support for read IDs, fast `reads_by_ids()` path for UUID-only filtering
- **subset**: Accelerated subsetting via indexed batch lookup
- **merge**: Parallel I/O optimization

### Fixed

- Include ZSTD content size in VBZ frames for Dorado/pod5 compatibility
- POD5 forward compatibility with Python pod5 library
- Correct pore count in summary table

# Changelog

## Unreleased

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

# Benchmark Results

Comparison of `podfive-rs` vs the official Python `pod5` tool.

## Test Environment

- **CPU**: Apple M3 Pro
- **Memory**: 18 GB
- **OS**: macOS (Darwin 25.1.0)
- **Date**: 2025-12-15

## Test Data

| File | Size | Reads |
|------|------|-------|
| FBC74904_90b682e0_e09f0700_0.pod5 | 897 MB | 23,123 |
| FBC74904_90b682e0_e09f0700_1.pod5 | 2.5 GB | 783 |
| FBC74904_90b682e0_e09f0700_10.pod5 | 3.1 GB | 1,020 |
| **Total** | **6.5 GB** | **24,926** |

## Results Summary

| Command | podfive-rs | pod5 (Python) | Speedup |
|---------|------------|---------------|---------|
| inspect summary | 3 ms | 237 ms | **79x faster** |
| view | 19 ms | 460 ms | **24x faster** |
| merge (3 files, 6.5 GB) | 11.1 s | 4.3 s | 2.6x slower |
| repack | 7.5 s | 869 ms | 8.7x slower |
| filter (10% of reads) | 2.8 s | 546 ms | 5.1x slower |

## Analysis

### Where podfive-rs excels

- **Read-only operations**: `inspect` and `view` commands are dramatically faster (26-64x) due to:
  - No Python interpreter startup overhead
  - Memory-mapped file I/O
  - Efficient Arrow table iteration

### Where pod5 (Python) is faster

- **Write operations**: `merge`, `repack`, and `filter` are currently slower in podfive-rs:
  - The Python `pod5` tool uses optimized C++ libraries (lib_pod5) under the hood
  - `filter`: Despite using LRU-cached signal batch lookups and block-level copying (no decompression), podfive-rs iterates through all reads sequentially. The Python tool may have indexed access.
  - `merge`/`repack`: The C++ library has highly optimized batch-level operations

## Running Benchmarks

```bash
# Build release binary first
cargo build --release

# Run full benchmark suite
./benchmarks/benchmark.sh data/pod5/

# Run merge-only benchmark
./benchmarks/merge_benchmark.sh data/pod5/
```

### Requirements

- `hyperfine`: `brew install hyperfine`
- `pod5`: `pip install pod5` (in `~/.venv/bin/pod5`)

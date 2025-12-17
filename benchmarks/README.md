# Benchmark Results

Comparison of `escapepod-rs` vs the official Python `pod5` tool.

## Test Environment

- **CPU**: Apple M3 Pro
- **Memory**: 18 GB
- **OS**: macOS (Darwin 25.1.0)
- **Date**: 2025-12-16

## Test Data

| File | Size | Reads |
|------|------|-------|
| FBC74904_90b682e0_e09f0700_0.pod5 | 897 MB | 23,123 |
| FBC74904_90b682e0_e09f0700_1.pod5 | 2.5 GB | 783 |
| FBC74904_90b682e0_e09f0700_10.pod5 | 3.1 GB | 1,020 |
| **Total** | **6.5 GB** | **24,926** |

## Results Summary

| Command | escapepod-rs | pod5 (Python) | Speedup |
|---------|------------|---------------|---------|
| inspect summary | 5 ms | 253 ms | **56x faster** |
| view | 19 ms | 586 ms | **30x faster** |
| merge (3 files, 6.5 GB) | 1.6 s | 4.9 s | **3x faster** |
| repack | 7.9 s | 917 ms | 8.6x slower |
| filter (10% of reads) | 3.1 s | 593 ms | 5.2x slower |

## Analysis

### Where escapepod-rs excels

- **Read-only operations**: `inspect` and `view` commands are dramatically faster (30-56x) due to:
  - No Python interpreter startup overhead
  - Memory-mapped file I/O
  - Efficient Arrow table iteration

- **Merge operations**: `merge` is now **3x faster** than pod5 thanks to:
  - Parallel file reading with rayon
  - Raw Arrow IPC batch copying without deserialization
  - Zero-copy async I/O via scoped threads
  - Memory-mapped input files
  - 16 MB buffered sequential writes

### Where pod5 (Python) is faster

- **Write operations**: `repack` and `filter` are slower in escapepod-rs:
  - The Python `pod5` tool uses optimized C++ libraries (lib_pod5) under the hood
  - `filter`: Despite using LRU-cached signal batch lookups and block-level copying (no decompression), escapepod-rs iterates through all reads sequentially. The Python tool may have indexed access.
  - `repack`: Requires full decompression/recompression in escapepod-rs. The C++ library has optimized batch-level operations.

## Running Benchmarks

```bash
# Build release binary first
cargo build --release

# Run full benchmark suite
./benchmarks/benchmark.sh data/pod5/
```

### Requirements

- `hyperfine`: `brew install hyperfine`
- `pod5`: `pip install pod5` (in `~/.venv/bin/pod5`)

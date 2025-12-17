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

| Command | escapepod | pod5 (Python) | Speedup |
|---------|-----------|---------------|---------|
| inspect summary | 5 ms | 225 ms | **43x faster** |
| view | 18 ms | 458 ms | **25x faster** |
| merge (3 files, 6.5 GB) | 1.3 s | 4.6 s | **3.6x faster** |
| filter (10% of reads) | 66 ms | 539 ms | **8x faster** |

## Analysis

### Where escapepod excels

- **Read-only operations**: `inspect` and `view` commands are dramatically faster (25-43x) due to:
  - No Python interpreter startup overhead
  - Memory-mapped file I/O
  - Efficient Arrow table iteration

- **Merge operations**: `merge` is **3.6x faster** than pod5 thanks to:
  - Parallel file reading with rayon
  - Raw Arrow IPC batch copying without deserialization
  - Zero-copy async I/O via scoped threads
  - Memory-mapped input files
  - 16 MB buffered sequential writes

- **Filter operations**: `filter` is **8x faster** than pod5 due to:
  - Batch-level parallelism with rayon
  - Block-level signal copying (preserves compression)
  - Efficient read ID lookup with HashSet
  - Streaming writes without intermediate buffering

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

# Benchmark Results

Comparison of `escapepod-rs` vs the official Python `pod5` tool (v0.3.36).

## Test Environment

- **CPU**: Intel Xeon Gold 6240R @ 2.40GHz (96 cores)
- **Memory**: 753 GB
- **OS**: Linux (RHEL 9)
- **Storage**: NFS
- **Date**: 2026-03-20

## Test Data

| File | Size | Reads |
|------|------|-------|
| PAY38817_82d9df02_82c8ff31_0.pod5 | 1.5 GB | 159,673 |
| PAY38817_82d9df02_82c8ff31_1.pod5 | 1.5 GB | 153,075 |
| **Total** | **3.0 GB** | **312,748** |

## Results Summary

| Command | escapepod | pod5 (Python/C++) | Speedup |
|---------|-----------|---------------|---------|
| inspect summary | 36 ms | 1.7 s | **47x faster** |
| view | 238 ms | 4.5 s | **19x faster** |
| merge (1 thread, 2 files, 3 GB) | 4.1 s | 4.1 s | ~1x |
| merge (4 threads) | 3.0 s | 4.1 s | **1.4x faster** |
| filter (10% of reads) | 513 ms | 4.7 s | **9x faster** |
| subset (2 groups) | 2.8 s | 8.3 s | **3x faster** |

## Analysis

### Where escapepod excels

- **Read-only operations**: `inspect` and `view` commands are dramatically faster (19-47x) due to:
  - No Python interpreter startup overhead
  - Memory-mapped file I/O
  - Efficient Arrow table iteration

- **Filter and subset operations**: `filter` is **9x faster** and `subset` is **3x faster** than pod5 due to:
  - Parallel group processing with rayon
  - Block-level signal copying (preserves compression)
  - Indexed batch lookup via `.p5i` or `reads_by_ids()` fast path
  - Single-pass signal extraction per output group

- **Merge operations**: At 1 thread, both tools are ~equal (I/O-bound on NFS). With 4 threads, escapepod is **1.4x faster** thanks to parallel metadata loading and zero-copy signal forwarding.

## Running Benchmarks

```bash
# Build release binary first
cargo build --release

# Run full benchmark suite
./benchmarks/benchmark.sh data/pod5/
```

### Requirements

- `hyperfine`: `cargo install hyperfine` or system package manager
- `pod5`: `pip install pod5` or `pixi add pod5`

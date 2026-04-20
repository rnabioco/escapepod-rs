# Benchmark Results

Comparison of `escapepod-rs` vs the official Python `pod5` tool (v0.3.36).

## 2026-04-19 run (post-SIMD, post-audit)

Run on the 2026-04 perf branch with SSSE3 SIMD SVB16 + release LTO profile.
Note: none of the benchmarked commands decompress signal (inspect/view are
metadata-only; filter/subset use compressed-passthrough), so the SVB16
SIMD wins are invisible to this suite — see `escapepod/benches/hot_paths.rs`
for microbenchmarks that exercise decode/encode directly.

### Test Data

| File | Size | Reads |
|------|------|-------|
| no_aaRS_caps_deacyl_b5.pod5 | 4.4 GB | 520,851 |

### Results Summary

| Command | escapepod-rs | pod5 (Python) | Speedup |
|---------|-------------:|--------------:|--------:|
| inspect summary | 47.9 ms ± 2.6 | 1.854 s ± 0.009 | **38.7×** |
| view (→/dev/null) | 594 ms ± 7 | 5.873 s ± 0.009 | **9.9×** |
| filter (10 % of reads) | 1.43 s ± 0.05 | 9.82 s ± 0.11 | **6.9×** |
| subset (2 groups) | 19.1 s ± 0.9 | 26.8 s ± 0.4 | **1.4×** |
| merge | skipped (single-file input) | | |

### Microbenchmarks (criterion) — SVB16 SIMD vs scalar

SSSE3 `_mm_shuffle_epi8` + prefix-sum delta decode. Measured with
`cargo bench --bench hot_paths`, release profile with fat LTO.

| Microbench | scalar | SSSE3 | Δ |
|---|---:|---:|---:|
| vbz/encode/1000 | 8.75 µs | 6.84 µs | −21.9 % |
| vbz/encode/10000 | 44.9 µs | 25.3 µs | −43.4 % (~1.77×) |
| vbz/encode/100000 | 365 µs | 170 µs | −53.3 % (~2.15×) |
| vbz/decode/1000 | 4.97 µs | 3.15 µs | −36.5 % |
| vbz/decode/10000 | 33.0 µs | 14.7 µs | −55.4 % (~2.24×) |
| vbz/decode/100000 | 306 µs | 120 µs | −60.6 % (~2.54×) |

## 2026-03-20 run (pre-audit)

### Test Data

| File | Size | Reads |
|------|------|-------|
| PAY38817_82d9df02_82c8ff31_0.pod5 | 1.5 GB | 159,673 |
| PAY38817_82d9df02_82c8ff31_1.pod5 | 1.5 GB | 153,075 |
| **Total** | **3.0 GB** | **312,748** |

### Results Summary

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

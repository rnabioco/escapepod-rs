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

| Command | escapepod | pod5 (Python/C++) | Speedup |
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

---

## Basecall Quality Benchmark

Measures the impact of signal downsampling (via `podfive archive`) on basecalling accuracy.

### Purpose

The `podfive archive` command reduces POD5 file sizes by downsampling signal data. This benchmark quantifies how downsampling affects basecalling quality across different model tiers.

### Running the Benchmark

```bash
# Build release binary first
cargo build --release

# Run basecall quality benchmark
./benchmarks/basecall_benchmark.sh <pod5_file> <reference.fa> <output_dir> [factors]
```

**Arguments:**
- `pod5_file` - Input POD5 file to benchmark
- `reference.fa` - Reference genome for alignment (FASTA)
- `output_dir` - Directory for results
- `factors` - Downsample factors (default: "2 4")

**Example:**
```bash
./benchmarks/basecall_benchmark.sh data/test.pod5 ref/genome.fa results/
./benchmarks/basecall_benchmark.sh data/test.pod5 ref/genome.fa results/ "2 4 8"
```

### Requirements

- `dorado` - ONT basecaller (in PATH)
- `samtools` - BAM file handling
- Python 3 with `pysam` and `pandas`

### Metrics Collected

| Metric | Description |
|--------|-------------|
| **mean_qscore** | Mean Q-score per read |
| **identity** | Alignment identity (matches / aligned bases) |
| **sub_rate** | Substitution (mismatch) rate |
| **ins_rate** | Insertion rate |
| **del_rate** | Deletion rate |
| **mapped_pct** | Percentage of reads mapped |
| **mean_mapq** | Mean mapping quality |

### Expected Output

The benchmark tests all model tiers (fast, hac, sup) and generates:

- `quality_summary.tsv` - Summary table of metrics
- `quality_metrics.json` - Detailed results in JSON format

**Example output:**

| model | condition | reads | mapped_pct | mean_qscore | identity | sub_rate | ins_rate | del_rate |
|-------|-----------|-------|------------|-------------|----------|----------|----------|----------|
| fast  | original  | 1000  | 98.5       | 15.2        | 92.1     | 3.2      | 2.1      | 2.6      |
| fast  | 2x DS     | 1000  | 97.8       | 14.5        | 90.8     | 3.8      | 2.5      | 2.9      |
| fast  | 4x DS     | 1000  | 95.2       | 13.1        | 88.2     | 4.5      | 3.2      | 4.1      |
| hac   | original  | 1000  | 99.2       | 18.5        | 95.2     | 2.1      | 1.2      | 1.5      |
| ...   | ...       | ...   | ...        | ...         | ...      | ...      | ...      | ...      |

### Interpretation

- **2x downsampling**: Minimal impact on HAC models (~1-2% accuracy loss)
- **4x downsampling**: Noticeable impact on all models (~3-5% accuracy loss)
- **8x+ downsampling**: Significant degradation, suitable only for basic QC

# Benchmark Results

Comparison of `escapepod-rs` vs the official Python `pod5` tool (v0.3.36)
and the reference barcode-demultiplexer (WarpDemuX / ADAPTed,
`KleistLab/WarpDemuX`).

## Demux vs WarpDemuX (2026-04-20)

Harness: `benchmarks/benchmark_demux.sh`. Auto-dispatches onto SLURM
(default: `-p rna -A rbi -c 16`; `--gpu` → `-p gpu -A gpu_rbi -c 16
--gres=gpu:1`) and reports single-node wall-clock. Compute node: 16
cores allocated; GPU = NVIDIA A30.

Input: `ext/WarpDemuX/test_data/demux/4000_rna004.pod5` (78 MB, 4000
reads). Both tools use WarpDemuX's bundled `WDX4_rna004_v1_0` SVM model
— escpod reads it after a one-shot conversion via
`scripts/convert_warpdemux_model.py`.

### Adapter detection (hyperfine, 3 runs, 1 warmup)

| Command | Time | Speedup |
|---|---:|---:|
| `escpod demux detect` | **1.591 s** ± 0.003 | — |
| `adapted detect` (LLR) | 15.272 s ± 0.055 | — |

`escpod detect` is **~9.6× faster** than ADAPTed's LLR detector at the
same `-j 16`.

### End-to-end pipeline (wall-clock, single run)

| Tool | Stages | Time | Speedup |
|---|---|---:|---:|
| `escpod` (CPU) | detect + fingerprint `--warpdemux-compat` + classify `--svm-model` | **3.43 s** | **5.5×** |
| `escpod` (GPU, `--gpu`) | same + batched GPU DTW | 3.33 s | 5.7× |
| `warpdemux demux -m WDX4_rna004_v1_0` | full pipeline | 19.02 s | 1× |

GPU is within noise of CPU at this input size — with 4000 reads × 851
training fingerprints the DTW step is short enough that NVRTC compile
(~100 ms) + H2D transfer eat the kernel speedup. The GPU path is useful
on much larger inputs where DTW dominates; the
`hot_paths_gpu` microbench at 8192 × 40 fingerprints measures a 7.7×
speedup on the kernel in isolation.

### Classification agreement (`compare_demux_results.py`)

Using the **same** SVM model (`WDX4_rna004_v1_0`). Layered tests while
closing the gap:

| Experiment | Input | Agreement |
|---|---|---:|
| Layer A: WDX boundaries + WDX fingerprints → `escpod classify` | `barcode_fpts_0.npz` → CSV | **97.14 %** (100 % on conf ≥ 0.5, r = 0.9958) |
| Layer B, LLR detect (escpod default) | — | **93.37 %** |
| Layer B, CNN detect (`--method cnn`) | — | **96.95 %** |

Layer A showed early that the SVM / DTW / kernel / Platt pipeline is
faithful. Closing the original Layer-B gap (23 % → ~97 %) required
three fixes, all in fingerprint extraction:

1. **Extract padding.** WDX's `sig_extract.padding = 100` — the
   adapter region fed to clipping + segmentation is
   `signal[adapter_start-100 : adapter_end+100]`, not
   `signal[adapter_start : adapter_end]`. Missing this shifted every
   changepoint and the retained last-25 segment means. **Biggest
   single contributor.**
2. **scipy-matching `find_changepoints`.** Local-max detection switched
   to strict `>` with scipy-style plateau-midpoint collapse; removed
   the `t_score > 0.0` filter. Verified by direct Rust-vs-scipy
   comparison: the two produce bit-identical peak sets on real
   adapter signals.
3. **CNN adapter detection (`--method cnn`).** Ports ADAPTed's
   `BoundariesCNN` through tract-onnx. Closes most of the LLR vs CNN
   boundary gap (CNN reaches Layer-A ceiling; LLR stops a few points
   short because the LLR detector occasionally disagrees with the CNN
   on adapter_end by ≥ 20 samples).

The remaining ~3 % on the LLR path and ~0.1 % on the CNN path trace to
boundary-detection differences on hard reads — not a compat bug.

### Reproducing

```bash
# One-time setup
git clone https://github.com/KleistLab/WarpDemuX ext/WarpDemuX
git clone https://github.com/KleistLab/ADAPTed    ext/ADAPTed
pixi install -e warpdemux-bench
pixi run -e warpdemux-bench install-warpdemux

# CPU build (default)
srun -p rna -A rbi -c 32 cargo build --release \
    -p escapepod --features "demux train"

# GPU build (adds --gpu variant)
pixi install -e gpu
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e gpu cargo build --release \
    -p escapepod --features "demux train gpu"

# Run — auto-dispatches to the right SLURM partition
./benchmarks/benchmark_demux.sh          # CPU only
./benchmarks/benchmark_demux.sh --gpu    # adds the GPU variant
```

---

Comparison of `escapepod-rs` vs the official Python `pod5` tool (v0.3.36).

## 2026-04-19 run (post-SIMD, post-audit)

Run on the 2026-04 perf branch with SSSE3 SIMD SVB16 + release LTO profile.
The commands that move bulk data — `filter`, `subset`, `bam-filter`, `merge`
— are the ones that matter on real workflows; `inspect`/`view` are
metadata-only and included below only for completeness.

None of the benchmarked commands decompress signal (inspect/view hit
metadata; filter/subset/merge use compressed-passthrough), so the SVB16
SIMD wins are invisible to this suite — see `escapepod/benches/hot_paths.rs`
for microbenchmarks that exercise decode/encode directly.

### Test Data

| File | Size | Reads |
|------|------|-------|
| no_aaRS_caps_deacyl_b5.pod5 | 4.4 GB | 520,851 |

### Bulk data operations

| Command | escapepod-rs | pod5 (Python) | Speedup |
|---------|-------------:|--------------:|--------:|
| filter (10 % of reads, 4.4 GB → ~440 MB) | **1.43 s** ± 0.05 | 9.82 s ± 0.11 | **6.9×** |
| subset (split into 2 groups, 4.4 GB) | **19.1 s** ± 0.9 | 26.8 s ± 0.4 | **1.4×** |
| bam-filter (mapped-only, region, MAPQ) | escpod-only | — | — |
| merge | skipped (single-file input, see 2026-03-20 run) | | |

`bam-filter` has no Python counterpart in `pod5`; it reuses the same
block-level compressed-signal passthrough as `filter`, so the 4.4 GB
filter numbers are a reasonable proxy for its I/O path.

### Metadata operations (small absolute times)

| Command | escapepod-rs | pod5 (Python) | Speedup |
|---------|-------------:|--------------:|--------:|
| inspect summary | 47.9 ms ± 2.6 | 1.854 s ± 0.009 | 38.7× |
| view (→/dev/null) | 594 ms ± 7 | 5.873 s ± 0.009 | 9.9× |

These commands finish in well under a second either way — the speedup
ratio looks dramatic but the wall-clock difference is negligible in a
pipeline.

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

### Bulk data operations

| Command | escapepod-rs | pod5 (Python/C++) | Speedup |
|---------|-------------:|------------------:|--------:|
| filter (10 % of reads, 3 GB) | **513 ms** | 4.7 s | **9×** |
| subset (2 groups, 3 GB) | **2.8 s** | 8.3 s | **3×** |
| merge (4 threads, 2 files, 3 GB) | **3.0 s** | 4.1 s | **1.4×** |
| merge (1 thread) | 4.1 s | 4.1 s | ~1× (I/O-bound on NFS) |

### Metadata operations

| Command | escapepod-rs | pod5 (Python/C++) | Speedup |
|---------|-------------:|------------------:|--------:|
| inspect summary | 36 ms | 1.7 s | 47× |
| view | 238 ms | 4.5 s | 19× |

## Analysis

### Where escapepod moves the needle

- **Filter / subset / bam-filter** share one code path: block-level
  compressed-signal passthrough with parallel group writes via rayon,
  plus the `reads_by_ids()` fast path for indexed batch lookup. That
  gives **~9×** on filter and **~3×** on subset in absolute seconds
  saved on multi-GB files — the wins scale with input size, unlike
  the metadata commands.

- **Merge** is I/O-bound at 1 thread (both tools sit at ~4 s on NFS).
  With 4 threads, parallel metadata loading + zero-copy signal
  forwarding give a **1.4×** win, and the `Arc<[u8]>` compressed
  chunks avoid any decompress/recompress round-trip.

- **bam-filter** has no Python counterpart. It reuses the `filter`
  passthrough path, so its steady-state throughput is bounded by the
  same block-level copy cost as `filter`.

### Metadata commands (inspect, view)

Dramatically faster on paper (19–47×) thanks to no Python interpreter
startup, memory-mapped I/O, and tight Arrow iteration — but the
absolute times are tens to hundreds of milliseconds either way. This
matters for interactive use; it doesn't change pipeline wall-clock.

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

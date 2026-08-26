# Benchmark Results

## Fused demux pipeline rework (2026-07-26)

Input: `Ma_20aa.pod5` — 1,220,602 reads, 10.4 GB, RNA004. Node: `rna`
(Cascade Lake), 48 logical cores, output to node-local disk (406 MB/s
sequential write). `escpod filter` copying the same reads block-level takes
**50.9 s** — that is the I/O floor any demux run is measured against.

| Run | Before | After | |
|---|---:|---:|---:|
| GBM model, 48 threads | 151.0 s | **65.1 s** | **2.32×** |
| DTW-SVM model, 48 threads | 147.5 s | **115.8 s** | 1.27× |
| Peak RSS (GBM) | 13.5 GB | 12.7 GB | — |
| CPU utilization (GBM) | 511% | 1273% | — |

Barcode assignments are unchanged (identical per-barcode counts), and the
staged `detect` / `fingerprint` subcommands produce byte-identical output.

The dominant fix was structural, not arithmetic. Before, `drive_blocks` filled a
block and processed it serially, so no I/O was in flight during detect+classify;
combined with per-barcode writer threads (86% of these reads route to one
barcode) the pipeline left ~43 of 48 cores idle and ran **faster at 8 threads
(108 s) than at 48 (151 s)**. Overlapping the reader with the processing loop
and budgeting the router channels removed that inversion.

Note the diagnostic value of the two models: swapping GBM for DTW-SVM used to
cost *nothing* in wall time (151.0 → 147.5 s) despite ~6× the CPU, proving
classification was entirely hidden. After the fix the same swap costs ~50 s —
classify is now on the critical path, which is what makes the classifier
optimizations and GPU DTW worth anything.

### Reader tuning

Block size and reader-thread count were swept on the same input; both are now
fixed in the code at the measured optimum. Only the filler count remains
tunable (`ESCAPEPOD_DEMUX_FILLERS`), as the escape hatch for cold network
filesystems — see below.

Block size (GBM model). 128 MB is the pick; larger is worse on both axes:

| MB | wall | peak RSS | CPU |
|---:|---:|---:|---:|
| 32 | 65.5 s | 12.53 GB | 1327% |
| 64 | 66.2 s | 12.55 GB | 1285% |
| **128** | **63.8 s** | 12.68 GB | 1230% |
| 256 | 69.9 s | 12.86 GB | 1072% |
| 512 | 74.4 s | 13.07 GB | 970% |

Reader threads and queue depth, after the classifier and prep work below:

| fillers / queue | wall | CPU | peak RSS |
|---|---:|---:|---:|
| 1 / 1 | 56.9 s | 771% | 12.2 GB |
| **2 / 2** | **48.4 s** | 918% | 13.0 GB |
| 4 / 4 | 48.1 s | 931% | 14.5 GB |
| 8 / 4 | 48.5 s | 923% | 17.3 GB |

A deeper queue alone is worth ~3%; the second reader thread is the real lever
(-15%). Past two it is flat and costs real memory.

**Caveat:** these runs used BeeGFS input that had been read repeatedly on a
256 GB node, so it may have been partly page-cached. The cold case is *not*
independently validated. Two threads doing coarse ascending sweeps of
alternating Arrow batches is qualitatively unlike the per-read demand paging
that #72 fixed (48 threads faulting per read measured 0.3 MB/s against
288 MB/s for one sweep), but if a cold mount regresses, set
`ESCAPEPOD_DEMUX_FILLERS=1`.

### What the pipeline is *not* bound by

Measured while tuning, so these are not re-derived:

- **Not write-bound.** Sending output to `/dev/shm` instead of disk changes
  nothing (60.3 s vs 59.8 s), and no Arrow/writer symbols appear in the profile
  above 1% — the per-barcode writers do block-level compressed copies and
  almost no CPU. Adding writer threads would not help.
- **Input read is the cost.** Staging the input to node-local disk instead of
  BeeGFS moves it from 59.8 s to 47.0 s.

### GPU DTW classify (A30) — does not pay off on a full node

Same input and binary, run on a `gpu` node (64 cores, A30) so CPU and GPU share
hardware. **Allocate the whole node**: the CPU core count dominates this
comparison, and starving it produces a misleading GPU win.

| DTW-SVM, A30 node | `--device cpu` | `--device gpu` | |
|---|---:|---:|---:|
| `-c 64` (full node) | **113.0 s** | 132.4 s | GPU **0.85×** — slower |
| `-c 16` | 206.4 s | 123.5 s | GPU 1.67× — an artifact of CPU starvation |

At 64 cores the CPU DTW (lane-parallel AVX2, see `DTW_LANES`) beats the A30, and
the GPU also costs ~2.2 GB extra RSS. The GPU path is worth reaching for only
when cores are scarce relative to the DTW workload — e.g. a shared node where
you can get a GPU but not a full socket.

**This is the measurement behind the DTW carve-out in `--device auto`**: `auto`
leaves DTW on the CPU even with an idle GPU in the node, and `--device gpu` is
how you override that. See `crate::device::Stage::auto_prefers_gpu`.

### Would a GPU path help GBM models? No.

Measured with `examples/profile_gbm` on the shipped 5-class model (222
iterations, 25 features), single thread on `rna`:

| GBM classify | per read | throughput |
|---|---:|---:|
| scalar | 82.5 µs | 12.1 k reads/s |
| **batch-8 (shipping)** | **28.6 µs** | 34.9 k reads/s |

At 1.22 M reads that is ~35 core-seconds, i.e. **~4% of the 65 s pipeline**.
Even an infinitely fast GPU GBM would return ~4% by Amdahl. And DTW — a far
more GPU-friendly, arithmetic-dense kernel with ~4× the per-read work — already
*loses* to the CPU on a full node (above), so a branch-heavy tree walk over a
~1.1 MB arena that fits in L2 is a strictly worse candidate.

If GPU effort is wanted, it belongs in `--method cnn` adapter detection, which
is genuinely inference-bound.

### Classifier microbenchmarks

`cargo bench -p escapepod-demux --bench classify`, `rna`, per-read
`predict_with_workspace` on the shipped WDX4 shape (851 refs × 25 features):

| Change | 851 refs | 20-class decision fn |
|---|---:|---:|
| baseline | 150.1 µs | 366.9 µs |
| `DTW_LANES` 16 → 32 | 121.4 µs (−18.9%) | — |
| + class-ranged OvO decision | **110.4 µs (−26.4%)** | **26.7 µs (−92.7%)** |

Both are bit-identical to the previous results (covered by parity tests).


## vs. official `pod5` (2026-07-26)

Harness: `hyperfine`, `rna` node (Gold 6240R, 2x24 cores + HT = 96 logical;
`-c 48` = one full socket, per the CLAUDE.md guidance on not crossing NUMA).
escapepod-rs `--release`; `pod5` 0.3.44 from the `warpdemux-bench` pixi env.

Input: 1.06 GB / 122,061 reads (a 10% `filter` of `Ma_20aa.pod5`), on BeeGFS,
outputs to node-local disk.

### Bulk data operations

| Command | escapepod-rs | pod5 (Python) | Speedup |
|---|---:|---:|---:|
| `filter`, all reads (copy-all) | **1.760 s** ± 0.020 | 4.989 s ± 0.220 | **2.83x** |
| `filter`, 10% of reads | **0.800 s** ± 0.026 | 3.759 s ± 0.074 | **4.70x** |
| `subset`, 2 groups | **1.342 s** ± 0.117 | 4.867 s ± 0.011 | **3.63x** |

### Metadata operations

| Command | escapepod-rs | pod5 (Python) | Speedup |
|---|---:|---:|---:|
| `inspect summary` | **36.1 ms** ± 1.3 | 1.912 s ± 0.011 | **53x** |
| `view` (-> /dev/null) | **226 ms** ± 1.5 | 4.945 s ± 0.035 | **21.9x** |

### Notes on making this comparison fair

- **Pick the input size deliberately.** An earlier run of the same `filter`
  10% comparison against the full 10.4 GB `Ma_20aa.pod5` measured only 1.23x.
  That is not a different result, it is a different workload: the output is
  ~1 GB either way, so both tools spend most of their time reading the same
  10.4 GB input and the ratio collapses toward 1. Size the input so the
  operation, not the shared read, dominates.
- **`escpod filter -i <ids>` and `--min-samples 0` are different code paths.**
  A pure ID list takes the `is_uuid_only` fast path (`reads_by_ids`, skipping
  non-matching batches); a criteria filter scans the reads table. Do not
  compare timings across the two.
- **`pod5 subset --csv` is not the path to benchmark.** Its CSV format is one
  row per target with every read ID inline, and parsing that was OOM-killed at
  48 GB on this 1 GB input. The `--summary` + `--columns` table path (used
  above) is the documented and workable one.
- pod5's subset output totalled 2.1 GB against a 1.06 GB input, so it is not
  writing byte-comparable output here; the wall-clock ratio above should be
  read with that in mind.

The historical numbers below used `no_aaRS_caps_deacyl_b5.pod5` (4.4 GB), which
no longer exists on this system, so they are kept for reference rather than
being directly comparable to the table above.

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
| `escpod` (GPU, `--device gpu`) | same + batched GPU DTW | 3.33 s | 5.7× |
| `warpdemux demux -m WDX4_rna004_v1_0` | full pipeline | 19.02 s | 1× |

GPU is within noise of CPU at this input size — with 4000 reads × 851
training fingerprints the DTW step is short enough that NVRTC compile
(~100 ms) + H2D transfer eat the kernel speedup. The GPU path is useful
on much larger inputs where DTW dominates; the
`hot_paths_gpu` microbench at 8192 × 40 fingerprints measures a 7.7×
speedup on the kernel in isolation.

### Classification agreement — parity ladder (2026-06-14 update)

The stage-isolation harness `benchmarks/benchmark_demux_parity.sh` runs
four layers that swap escpod stages in one at a time, so the agreement
drop between adjacent layers attributes any gap to a specific stage. All
layers classify with the **same** converted `WDX4_rna004_v1_0` model and
are compared against WarpDemuX's own predictions.

| Layer | boundaries / fingerprints | overall | conf ≥ 0.5 |
|---|---|---:|---:|
| A — WDX bounds + WDX fpts → `escpod classify` | WDX / WDX | **99.63 %** | **100.00 %** |
| B-bounds — WDX bounds + escpod fpt | WDX / escpod | 99.61 % | 100.00 % |
| B-cnn — escpod CNN detect (`--method cnn`) | escpod / escpod | **99.26 %** | 100.00 % |
| B-llr — escpod LLR detect (default) | escpod / escpod | 94.14 % | 96.34 % |

**Confident reads (conf ≥ 0.5) are at 100 % parity through Layer B-cnn.**

#### The DTW warping-penalty fix (97.1 % → 99.6 % ceiling)

Earlier the Layer-A ceiling sat at **97.14 %** — even with identical WDX
boundaries *and* fingerprints, `escpod classify` disagreed ~2.9 %. The
root cause was the DTW distance: WarpDemuX models carry a
`dtaidistance` warping **`penalty`** (`WDX4` = 0.1) added to the two
non-diagonal (expansion / compression) DP transitions, and escpod
applied none. Plumbing it through (`DtwSvmModel.penalty`, extracted by
`convert_warpdemux_model.py`; applied in `dtw_distance_penalty`) lifts
Layer A to **99.63 %** and confident reads to **100 %**.

Subtlety that bit once: `dtaidistance`'s penalty is expressed in
*non-squared* distance space while escpod's DP accumulates squared local
costs, so each warp step adds **`penalty²`** (verified directly:
`dtaidistance.dtw.distance([0,0,0],[0], penalty=0.1) == sqrt(2·0.1²)`).
Adding the raw `penalty` over-penalizes 10× and *regresses* parity to
~80 %. The GPU kernel applies the identical `penalty²` so GPU
classify matches CPU (test: `gpu_svm_batch::parity_svm_classify_batch_penalty`).

The remaining Layer-A 0.37 % are all low-confidence near-ties
(`wdx_conf < 0.5`) that flip on f32-vs-f64 DTW rounding — escpod keeps
f32 DTW for throughput; the residual is below the confident-call gate.

Earlier work (still in place) closed the original 23 % → 97 % Layer-B
gap with three fingerprint-extraction fixes: WDX's `sig_extract.padding
= 100`; scipy-matching `find_changepoints` (strict `>` + plateau
midpoint); and the ADAPTed `BoundariesCNN` port (`--method cnn`). The
residual ~5 % on the **default LLR** path (94.14 % vs B-cnn 99.26 %) is
boundary detection — **use `--method cnn` for parity**; LLR occasionally
disagrees with the CNN on `adapter_end` by ≥ 20 samples on hard reads.

### Reproducing

```bash
# One-time setup
git clone https://github.com/KleistLab/WarpDemuX ext/WarpDemuX
git clone https://github.com/KleistLab/ADAPTed    ext/ADAPTed
pixi install -e warpdemux-bench
pixi run -e warpdemux-bench install-warpdemux

# CPU build (default)
srun -p rna -A rbi -c 32 cargo build --release \
    -p escapepod-cli --features "demux train"

# GPU build (adds the `--device gpu` variant)
pixi install -e gpu
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e gpu cargo build --release \
    -p escapepod-cli --features "demux train gpu"

# Run — auto-dispatches to the right SLURM partition
./benchmarks/benchmark_demux.sh                       # CPU only, WDX4
./benchmarks/benchmark_demux.sh --gpu                 # adds the GPU variant
./benchmarks/benchmark_demux.sh --model WDX10_rna004_v1_0   # larger DTW workload
```

#### Example sweep (2026-06-14, parallel fan-out; AlaRS_all20_b4 real run)

`benchmark_demux_matrix.sh` across WDX4/6/10 × {4k bundled, 25k, 100k real}
× {cpu, gpu}, 18 cells fanned out concurrently (cpu `-c 24`, gpu A30). The
GPU classify (DTW) only earns its keep at scale; at 4k reads NVRTC compile +
H2D transfer dominate.

| model | n_reads | escpod CPU s | escpod GPU s | speedup CPU | speedup GPU |
|---|---:|---:|---:|---:|---:|
| WDX4  | 3,786  | 1.81  | 1.57  | 17.6× | 20.3× |
| WDX4  | 55,864 | 30.23 | **10.00** | 3.8× | **11.5×** |
| WDX6  | 55,864 | 43.14 | **10.71** | 2.8× | **11.1×** |
| WDX10 | 3,786  | 2.85  | 3.53  | 12.3× | 9.9× |
| WDX10 | 55,864 | 60.44 | **12.06** | 1.6× | **7.9×** |

At 100k reads the GPU is **~5× faster than escpod-CPU** for the heaviest model
(WDX10: 60.4 → 12.1 s) — DTW dominates and the A30 pays off. At 4k it's within
noise or slower. Agreement (default LLR path) is 93–96% across the sweep and is
model/boundary-bound, not affected by the device.

#### Harness scripts (2026-06-14)

| Script | Purpose |
|---|---|
| `benchmark_demux.sh` | Single cell: detect+fingerprint+classify vs WarpDemuX. Flags: `--model NAME`, `--gpu` (the script's own flag; it passes `--device gpu` to escpod, and `--device cpu` for the CPU arm), `--out-dir DIR`, `--emit-tsv FILE`. |
| `make_demux_inputs.sh` | Builds reproducible size tiers (4k/25k/100k reads) from a real run via `escpod filter`; the bundled 4000-read file is always the smallest tier. |
| `benchmark_demux_matrix.sh` | Sweeps {models} × {tiers} × {cpu,gpu}, one srun per device, → `matrix.tsv` + `matrix.md` (speed + agreement per cell). |
| `benchmark_demux_parity.sh` | The stage-isolation ladder above. `--dump-mismatches` writes a per-read CSV for root-causing. Needs the CNN ONNX for the B-cnn layer. |

```bash
# Full speed+agreement matrix across models and dataset sizes:
./benchmarks/benchmark_demux_matrix.sh \
    --models "WDX4_rna004_v1_0 WDX6_rna004_v1_0 WDX10_rna004_v1_0" \
    --tiers "4000 25000 100000" --devices "cpu gpu" --src /path/to/real_run

# Parity ladder + per-read mismatch dump:
#   (B-cnn needs scripts/export_adapter_cnn_to_onnx.py -> benchmarks/adapter_cnn_rna004.onnx,
#    built from a local ADAPTed install; CC BY-NC weights are not redistributed.)
./benchmarks/benchmark_demux_parity.sh --dump-mismatches
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

## GPU vs CPU TCN adapter detection (2026-06-16)

Decision benchmark for running the escapepod-models `adapter_rna004@v1.0.1` TCN
(`demux detect --method cnn`) on GPU. Input: `bench_wdx/sub20k/sub.pod5` (20k
RNA004 tRNA reads, mean 10,206 samples → downscaled TCN input L≈920). CPU =
rna 16c; GPU = A30 via onnxruntime CUDA EP (probe `bench_tcn_gpu/ort_probe.py`).

**`detect --method cnn` is inference-bound, not I/O-bound.** LLR detect on the
same read+decode path is **0.25 s / 20k**; the CNN/TCN path is **77.49 s / 20k**
→ inference is **~99.6 %** of wall-clock. (The #80 "detect is POD5-read + prep
bound, not CNN-compute bound" reasoning holds for LLR, *not* the TCN.)

| Inference path | reads/s | vs current |
|---|---:|---:|
| CPU tract per-read (current `escpod`) | 258 | 1× |
| CPU onnxruntime, batched B=1024 | 738 | 2.9× |
| **GPU onnxruntime, batched B=1024 (A30)** | **25,656** | **~99×** |
| I/O+decode floor (LLR, same path) | ~80,000 | — |

GPU is **~99×** over the current tract path and **~35×** over the best
CPU-batched path; batching alone (no CUDA) is a **2.9×** partial win. Verdict:
**GO** — clears the ≥1.5× gate by a wide margin. Right mechanism is the `ort`
crate + CUDA execution provider (architecture-agnostic, any `[B,1,L]->[B,2,L]`
ONNX) — **not** reverting #80's BoundariesCNN-locked CUDA kernel, which cannot
run the TCN. Numbers are synthetic fixed-L throughput; a production path needs
length-bucketing and GPU-vs-tract `adapter_end` parity on real signals.

## Python library: escapepod vs pod5 (2026-07-09)

Harness: `benchmarks/benchmark_python.py` (wrapper `benchmark_python.sh`).
Unlike `benchmark.sh` — which times the `escpod` vs `pod5` **CLIs** with
hyperfine — this exercises the two **Python libraries** in-process, so the
numbers reflect library work, not interpreter startup + import. Each case runs
the same logical operation through both libraries, warms up, times several
repetitions, and reports the median; a checksum asserts the two libraries agree
(int16 signal is bit-identical, pA sums match within last-ULP float32 drift).

Input: `ext/WarpDemuX/test_data/demux/4000_rna004.pod5` (81 MB, 4000 reads).
Node: `rna` partition, `escapepod` 0.5.1 vs `pod5` 0.3.39, 7 runs / 2 warmup.

| Benchmark | escapepod | pod5 | speedup |
|---|---:|---:|---:|
| open + metadata | 4.21 ms | 16.50 ms | **3.92×** |
| read_ids | 476 µs | 541 µs | 1.14× |
| iterate read metadata | 4.20 ms | 3.52 ms | 0.84× |
| read all signal (int16) | 209.7 ms | 264.1 ms | **1.26×** |
| read all signal (pA) | 217.4 ms | 355.0 ms | **1.63×** |
| metadata → pandas | 15.3 ms | 2.5 ms | 0.16× |
| random-access selection | 333 µs | 162 µs | 0.49× |
| read all signal (batched, `get_signals`) | 284.1 ms | — | esc-only |

speedup = pod5 median / escapepod median (>1 ⇒ escapepod faster). escapepod wins
on file open and signal decode (VBZ + calibration); pod5 wins on the pandas
export (its metadata is already an Arrow table, so `to_pandas` is near-free,
whereas escapepod builds the frame) and on tiny random-access selections. The
escapepod-only batched `get_signals` case has no pod5 API equivalent.

### Larger file (20k reads)

A 20 000-read subset of a 2 GB real run
(`ThrRS_ser_b6.pod5`, 220k reads total; subset written via `--limit 20000`).
Node: `rna`, escapepod 0.5.1 vs pod5 0.3.39, 5 runs / 1 warmup.

| Benchmark | escapepod | pod5 | speedup |
|---|---:|---:|---:|
| open + metadata | 133 µs | 612 µs | **4.59×** |
| read_ids | 2.54 ms | 2.64 ms | 1.04× |
| iterate read metadata | 21.4 ms | 17.4 ms | 0.81× |
| read all signal (int16) | 541.2 ms | 758.8 ms | **1.40×** |
| read all signal (pA) | 551.0 ms | 1.047 s | **1.90×** |
| metadata → pandas | 75.7 ms | 8.2 ms | 0.11× |
| random-access selection (500) | 1.18 ms | 835 µs | 0.71× |
| read all signal (batched, `get_signals`) | 602.3 ms | — | esc-only |

The picture holds and sharpens at scale: escapepod's signal-decode lead widens
(pA 1.63× → **1.90×**). One gap stands out as a real optimization target:

- **`metadata → pandas` gets *worse* with size** (0.16× → 0.11×). `to_dict`/
  `to_pandas` built Python **lists of boxed scalars** per column, which pandas
  then re-parsed into numpy arrays; pod5 goes Arrow→pandas columnar with no
  per-element boxing. **Fixed in #99** by emitting numpy columns directly
  (`to_pandas` 4k: 15.3 → 8.0 ms, 0.16× → 0.36×). The residual is upstream
  `ReadData` materialization in `collect_inner` (~63 % of wall at 220k reads:
  heap `Uuid`/`pore_type`/`end_reason` strings + an unused `signal_rows` `Vec`
  per read) — building columns straight from the Arrow batch is tracked in #98.

The `get_signals` (batched) row looked slower here (602 ms vs the per-read
541 ms), but a controlled A/B rules that out as a measurement artifact:
`get_signal_bulk` is batch-grouped + rayon-parallel VBZ decode, and on the same
reads at 16 threads it runs **427 ms vs 526 ms per-read (1.23×)**, scaling 2.8×
from 1→16 threads. Per-read decode is single-threaded and flat across thread
count, so `get_signals` is the right API for bulk signal — the batched number
above just reflects run-to-run variance on that particular subset, not a
regression. (A streaming/iterator variant that avoids materializing every signal
at once would cut its peak memory, but is not needed for throughput.)

**Why random-access selection is slower:** the test file has no `.p5i` index
sidecar, so escapepod's `reads(selection=…)` falls back to a linear scan of the
whole reads table — decoding every read_id and testing set membership, O(total
reads) — while pod5 resolves the selection against an in-memory read-id index,
O(selection). Building the index first (`reader.build_index()`, or shipping the
`.p5i`) drops escapepod from 0.47 ms → 0.21 ms, on par with pod5's 0.16–0.23 ms.
The table reports the default (no-index) path; for repeated random access on one
reader, call `build_index()` once up front.

### Reproducing

```bash
# Rebuild the extension first if the Rust bindings changed — a stale wheel would
# benchmark an old API (the wrapper warns if installed version != workspace).
pixi run -e python-test maturin develop --release \
    --manifest-path crates/escapepod-python/Cargo.toml

# Small file (runs anywhere)
./benchmarks/benchmark_python.sh ext/WarpDemuX/test_data/demux/4000_rna004.pod5

# Large file — cap the signal loops to N reads (writes a subset first) and run
# under SLURM so it's off the 2-core login node
srun -p rna -c 32 --mem=32G ./benchmarks/benchmark_python.sh \
    big.pod5 --limit 20000 --runs 5 --json /tmp/pybench.json
```

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
- Python-library benchmark (`benchmark_python.sh`): both `escapepod` and `pod5`
  in the `python-test` pixi env — `pixi run -e python-test maturin develop
  --release --manifest-path crates/escapepod-python/Cargo.toml`

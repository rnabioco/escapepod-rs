# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

escapepod-rs is a pure Rust implementation for reading and writing POD5 files, the native file format for Oxford Nanopore sequencing data. The workspace splits the library into two layers — `escapepod-pod5` for format I/O and `escapepod-signal` for signal-processing algorithms — plus the `escapepod-cli` crate (the `escpod` CLI binary, with an optional umbrella library re-exporting the layers) and Python bindings (`escapepod-python`).

## Requirements

- Rust 1.95 or later (matches `workspace.package.rust-version`)

## Build Commands

```bash
# Build
cargo build --release

# Build with training support (enables SVM model training)
cargo build --release --features train

# Build with GPU support (opt-in). NOTHING CUDA is needed at build time —
# cudarc and ort both dlopen at run time — so this is a plain cargo build and
# needs no GPU env:
cargo build --release --features gpu -p escapepod-cli

# Create the `gpu` runtime environment ONCE, on a networked node. Everything it
# needs (CUDA 12 libs, cuDNN, and libonnxruntime itself) is an ordinary
# conda-forge package described by pixi.lock; there is no fetch step and no
# out-of-band directory. Use the task rather than `pixi install -e gpu`: the
# environment targets a `__cuda` platform that a login node cannot satisfy, and
# the task supplies the install-time probe (see pixi.toml for why that is sound):
pixi run install-gpu
# Full runtime story + verification: docs/experimental/demux.md ("GPU acceleration")

# Test / bench on a GPU node (SLURM account `gpu_rbi`, partition `gpu`).
# Note `-e dev-gpu`, not `-e gpu`: cargo-nextest lives in the `dev` feature.
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e dev-gpu cargo nextest run --features gpu -p escapepod-signal --test gpu_dtw
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e gpu cargo bench --features gpu --bench hot_paths_gpu

# Use on a node with a visible GPU. `--device auto` (the default) places each
# stage where it wins — GPU for CNN detect + CRF encoder, CPU for DTW — so a
# GPU build usually needs no flag at all. `--device gpu` demands the device and
# errors instead of falling back; `--device cpu` forces CPU. The old `--gpu` is
# a hidden deprecated alias for `--device gpu`.
escpod demux reads.pod5 --model <bundle> --annotate
escpod demux classify --model model.json reads.fp.csv --device gpu -o out.tsv

# Run tests (cargo-nextest)
cargo nextest run

# Doctests — nextest does not run doctests, keep them on a separate invocation
cargo test --doc --workspace

# Run a specific test
cargo nextest run test_round_trip_single_read

# Run clippy lints
cargo clippy

# Run the CLI (after building)
./target/release/escpod <command>

# mold is the linker full time: the base [activation] in pixi.toml exports
# LD_PRELOAD=$CONDA_PREFIX/lib/mold/mold-wrapper.so for EVERY environment, so
# any `pixi run [-e <env>] cargo …` (and maturin) links with mold — no
# `mold -run` and no `-e dev` needed for that. Works with system gcc 11.5 (no
# `-fuse-ld=mold` support required) and needs no glibc-static. Release
# artifacts shipped via GitHub Releases build in CI against musl (static),
# outside pixi, and are unaffected; local builds remain dynamic gnu by design.
#
# The `dev` env additionally provides cargo-nextest + convenience task wrappers
# (which are now just bare `cargo …`, since mold is already on):
pixi run -e dev build        # cargo build
pixi run -e dev build-rel    # cargo build --release (dynamic gnu)
pixi run -e dev test         # cargo nextest run
pixi run -e dev test-doc     # cargo test --doc (nextest skips doctests)
pixi run -e dev check        # cargo check
pixi run -e dev clippy       # cargo clippy --workspace --all-targets

# GPU builds also link with mold (dev-gpu, or any gpu env):
pixi run -e dev-gpu cargo build --features gpu -p escapepod-cli
```

### Allocator

The `escpod` binary uses mimalloc (`escapepod-cli`'s `mimalloc` feature, on by
default). The reason is the release artifact: on `escpod classify` with 8
threads the 0.20.0 static-musl tarball made 9.2 M voluntary context switches
against 79 k for a glibc build of the same code, and ran 2.35× slower in wall
and 3.9× in CPU — with `--profile dist` and `-C target-cpu=haswell` on both.
It is not musl as such: a static-musl build of the same source with zig's musl
(1.2.x, mallocng) runs like glibc. The tarball's strings carry
`GCC: (GNU) 9.4.0` — `cross`'s old musl image, whose musl predates mallocng
and takes one global lock per allocation. mimalloc keeps every Rust allocation
off that path whichever libc the artifact links, and is worth a further ~10%
of CPU on glibc. `cargo build --no-default-features --features cli` is the
same binary on the system allocator, kept buildable so the two can be measured
against each other (`benchmarks/README.md`). A local static-musl build for
such a measurement: `rustup target add x86_64-unknown-linux-musl`, zig from
conda-forge (`pixi global install zig`), `cargo install cargo-zigbuild`, then
`cargo zigbuild --profile dist --target x86_64-unknown-linux-musl -p escapepod-cli`
with `RUSTFLAGS="-C target-cpu=haswell"` — the release workflow's flags. The
release workflow itself still builds with `cross`; whether its image's musl is
the cause or a symptom is the next CI-built tarball's measurement to make.

### Build baseline and SLURM builds

Local Linux/x86_64 builds pin `-C target-cpu=x86-64-v3` via `.cargo/config.toml`
(AVX2 + FMA + BMI2 + POPCNT + F16C). This is portable across every node in
the cluster (Broadwell login, Cascade Lake rna, Ice Lake gpu). Do **not**
use `target-cpu=native`: a binary built on a gpu node uses Ice Lake-only
instructions (VBMI, VPCLMULQDQ, …) that SIGILL on rna. Hot kernels that
want AVX-512 do so via `#[target_feature]` + runtime `is_x86_feature_detected!`
dispatch, not a global baseline bump.

The login node has only 2 cores — wrap any heavy build or bench in `srun -p rna`:

```bash
# build / test — 32 logical CPUs (16 physical cores + HT) is enough.
# These already link with mold (base activation, see above); no -e dev needed.
srun -p rna -c 32 --mem=32G pixi run cargo build --release
srun -p rna -c 32 --mem=32G pixi run cargo nextest run --workspace

# The dev env adds cargo-nextest + task wrappers. mold is multithreaded on the
# link step, so the 32-core allocation helps both compile and link phases.
srun -p rna -c 32 --mem=32G pixi run -e dev build-rel
srun -p rna -c 32 --mem=32G pixi run -e dev test

# throughput-sensitive demux runs — ask for a full socket (48 logical = 24 physical + HT).
# SLURM's `-c 32` only allocates 16 physical cores on rna's Gold 6240R, not 32;
# `-c 48` fills the socket and gives ~20% more wall-clock speedup on fingerprint
# without crossing NUMA boundaries. Crossing sockets (`-c 64`+) is a crapshoot on
# a shared node — other jobs on the second socket regress wall time.
srun -p rna -c 48 --mem=64G pixi run escpod demux fingerprint …
```

### Running on Alpine (CU Boulder RMACC/ACCESS)

The `rna`/`gpu` partitions above are the **Beevol** (CU Anschutz) cluster. On
**Alpine** the SLURM model differs: every partition needs an **explicit `--qos=`
matched to the partition** plus `-A amc-general` (the CU Anschutz allocation) — the
partition name alone is not enough. Translation:

| Purpose | Beevol | Alpine |
|---|---|---|
| CPU build / test / bench | `srun -p rna -c 32 --mem=32G` | `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G` |
| CPU full node (throughput) | `srun -p rna -c 48` | `srun -p amilan --qos=normal -A amc-general -c 64 --mem=120G` |
| GPU test / bench | `srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1` | `srun -p aa100 --qos=gpu-normal -A amc-general -c 16 --gres=gpu:1` |
| Light build (login is 2 cores) | login node | `srun -p acompile --qos=compile -A amc-general -c 16 --mem=32G` |

QOS↔partition: amilan→`normal`|`long`; aa100/al40→`gpu-normal`|`gpu-long`;
acompile→`compile`; atesting→`testing` (1 h cap). Default mem is 3840 MB/core.

Notes specific to Alpine:
- `amilan` nodes are AMD EPYC Milan, 2×32 = **64 physical cores, no hyperthreading**,
  245 GB. So the Beevol "`-c 48` fills a socket / `-c 64` crosses NUMA" tuning does
  **not** apply — `-c N` is N physical cores; one full node is 64 across 2 sockets.
- GPU CUDA work must use `aa100` (A100) or `al40` (L40). `ami100` is AMD/ROCm and
  **cannot** run the CUDA `gpu` feature.
- All of amilan/aa100/acompile are Zen3 x86-64, so the pinned `target-cpu=x86-64-v3`
  baseline is portable across them — build once, run anywhere, no SIGILL risk.
- Toolchain lives in pixi, not on the bare PATH: mold links every env (base
  activation); `-e dev` adds cargo-nextest + task wrappers, `-e warpdemux-bench`
  (hyperfine + pod5, for `benchmarks/benchmark.sh`), `-e gpu` (CUDA runtime /
  libnvrtc). Wrap invocations accordingly, e.g.
  `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G pixi run -e dev test`.

## Benchmarking & Profiling

### Build profiles

- `release` — ship build: fat LTO, `codegen-units=1`, stripped, `panic=abort`.
- `release-with-debug` — release speed with symbols retained (for `samply`/`perf`).
- `bench` — inherits release; used by `cargo bench` so microbenchmarks match release codegen.
- `profiling` — inherits release but turns LTO off and splits `codegen-units=16` so profilers see real call graphs instead of inlined soup.
- `dist` — ship build for release artifacts.
- `dev.package."*"` — dev dependencies compile at `opt-level = 2` so test iteration is fast.

### Microbenchmarks (criterion)

`crates/escapepod-signal/benches/hot_paths.rs` covers the audit hot paths: DTW, resquiggle DP, fingerprint MAD, VBZ encode/decode, DTW matrix.

```bash
# Full run
cargo bench --bench hot_paths

# Subset
cargo bench --bench hot_paths -- vbz

# A/B against a saved baseline
cargo bench --bench hot_paths -- --save-baseline <name>
cargo bench --bench hot_paths -- --baseline <name>     # compare future runs
```

Env overrides: `ESCAPEPOD_BENCH_THREADS=N` (rayon pool size for the matrix bench), `ESCAPEPOD_BENCH_SAMPLES=N` (criterion sample size for slow groups).

`crates/escapepod-classify/benches/charging.rs` covers the per-read charging path — `finalize`, `expected_levels_z`, `junction_features`, `feature_grid` — over both span modes. Self-contained by design: `ext/` fixtures do not exist in CI, so the read, the signal and the (full 4^9) k-mer table are synthesised. `ESCAPEPOD_CHARGING_BUNDLE=<dir>` adds a `scorer` group against a real bundle (needs `--features fnn-onnx`); without it that group skips itself and says so, since the weights are not redistributable.

```bash
cargo bench -p escapepod-classify --bench charging
ESCAPEPOD_CHARGING_BUNDLE=/path/to/bundle \
  cargo bench -p escapepod-classify --bench charging --features fnn-onnx
```

### End-to-end (hyperfine vs. Python pod5)

```bash
cargo build --release
./benchmarks/benchmark.sh /path/to/pod5/dir
```

Runs `inspect summary`, `view`, `merge`, `filter`, `subset` via hyperfine against Python `pod5` (installed in the pixi env). Results persist as JSON under `/tmp/escapepod_benchmark/`. Historical numbers are in `benchmarks/README.md`.

`benchmarks/benchmark_charging.sh` does the same job for `escpod classify` and additionally A/Bs escpod *builds* — one `--bin` per arm, so a released tarball can be measured beside a local build on one input. Two properties of it are load-bearing and were learned by getting them wrong: **arms are interleaved, not grouped**, because the first arm to touch a POD5 set pays to page it in (212 s against 55–68 s warm on a 12 GB set), and **CPU is reported beside wall**, because a cold run's CPU time reads *lower* than the same run warm (107 s against 264) — blocked page faults are not CPU, so either half read alone invents a 2× that is not there. Rep 1 is a warm-up; compare arms within rep 2+.

```bash
srun -p rna -c 8 --mem 32G -- benchmarks/benchmark_charging.sh \
    --pod5 run/pod5 --bam sample.bam --reference ref.fa --model bundle/ \
    --bin /path/to/escpod-0.20.0 --bin ./target/release/escpod
```

`benchmarks/benchmark_demux_crf.sh` is the same harness for the fused CRF demux path — the command the aa-tRNA-seq pipeline actually runs (`escpod demux --model ldx=… --model fdx=… --annotate --classifications … --ref-scores`), which `benchmark_demux.sh` (SVM/DTW against WarpDemuX, hyperfine, one binary) never measured. Same interleaving and CPU-beside-wall rules, plus a third: **the input is never touched** — `--annotate` writes its `.p5s` beside the path *as given*, so each arm runs against a symlink in node-local scratch and the script asserts no sidecar appeared beside the canonical file. `--pod5`/`--truth` default to the labelled 20k dual-axis set at `data/bench/dual_axis_20k/` (gitignored; provenance in `benchmarks/README.md`); `--model` is required. After the reps, `benchmarks/evaluate_demux_crf.py` scores the last rep's CSVs against the truth and every arm against arm 0 read for read, attributing disagreements by `crf_margin` — the contract an encoder change is held to (#331). The per-read counterpart is `crates/escapepod-demux/benches/crf_encoder.rs`, gated on `ESCAPEPOD_CRF_BUNDLE` like the charging `scorer` group.

```bash
srun -p rna -c 32 --mem 32G -- benchmarks/benchmark_demux_crf.sh \
    --model ldx=bundles/barcode_crf_ldx32_rna004@v0.2.1 --model fdx=bundles/barcode_crf_fdx4_rna004@v0.1.1 \
    --bin /path/to/escpod-0.20.0 --bin ./target/release/escpod --threads 32
ESCAPEPOD_CRF_BUNDLE=bundles/barcode_crf_ldx32_rna004@v0.2.1 \
  cargo bench -p escapepod-demux --features crf-decode --bench crf_encoder
```

### Profiling workflow

```bash
# 1. build with symbols kept
cargo build --profile release-with-debug -p escapepod-cli
# 2. record with samply (pixi-provided binary recommended)
samply record target/release-with-debug/escpod <args>
# 3. flamegraph-style view in browser
```

For perf/valgrind where inlining hides frames, swap `release-with-debug` for `profiling` (LTO off).

### Runtime verbosity

All CLI status/progress/warning output flows through `tracing` to stderr (the
custom `EscpodFormatter` in `main.rs` renders `timestamp LEVEL [target] message`).
Command *data* — TSV/CSV rows, `inspect`/`summary` reports, ID lists — stays on
**stdout** via `println!`, so it can be piped/redirected independently of logs.

Default level is **info** for escpod's own crates; dependencies are pinned at
**warn** so third-party chatter (tract logs every SIMD kernel it probes on each
`demux detect --method cnn` run) stays out of normal output. `-v`/`-vv` raise
escpod's level only — `RUST_LOG` is the escape hatch for dependency logs, and
always wins when set:

```bash
escpod inspect summary file.pod5         # info (default): status + warnings
escpod -v merge *.pod5 -o out.pod5       # debug
escpod -vv merge *.pod5 -o out.pod5      # trace
escpod -q merge …                         # errors only (status + progress bars suppressed)
RUST_LOG=escapepod_signal::reader=trace escpod view file.pod5   # module-scoped
```

Progress bars/spinners (`progress.rs`) are status output too: they auto-hide
when the level drops below info (i.e. under `-q`). Multi-line styled report
blocks (e.g. `merge --profile` timings, demux summaries) are gated on
`tracing::enabled!(Level::INFO)` rather than emitted as per-line events.

When adding output: use `tracing::info!`/`warn!`/`error!` for diagnostics
(don't hand-prefix messages with `Warning:`/`Note:` — the formatter prints the
level); use `println!`→stdout only for the command's actual data product.

## Architecture

### Workspace Structure

- **escapepod-pod5**: POD5 format I/O layer — reader, writer, VBZ compression, footer parsing, block-level merge/filter/repack/subset operations.
- **escapepod-signal**: Signal-processing algorithms (DTW, resquiggle, segmentation) layered on top of `escapepod-pod5`. Re-exports the pod5 surface so downstream consumers can depend on a single crate.
- **escapepod-demux**: barcode demultiplexing — two independent approaches. (1) WarpDemuX-compatible SVM model loaders, DTW+SVM classifier, Platt scaling, optional SVM training (`train`), GPU DTW batch classify (`gpu`). (2) CTC-CRF barcode basecalling (`crf-decode`, CPU tract; onnxruntime CUDA via `gpu`). Plus boundary-CNN adapter detection (`cnn-detect`, CPU tract; optional onnxruntime CUDA via `gpu`), shared by both. `gpu` is deliberately **atomic** — one flag covering CUDA DTW, GPU CNN detect and the GPU CRF encoder, implying `cnn-detect` + `crf-decode` for the shared prep/decode. It replaced granular `gpu`/`gpu` flags whose stated purpose was library consumers and CI isolation builds; neither held up (the `fqxv-align` git dependency makes this crate unpublishable, so there are no external consumers), and their only real cost was runtime prose in the CLI explaining which half of `--gpu` was a silent no-op on a partially-GPU build. Depends on `escapepod-signal` for DTW and fingerprint primitives.
- **escapepod-classify**: read-level classification against escapepod-models bundles — today the tRNA charging (aminoacylation) classifier over POD5 + aligned BAM (#204). Anchors on the CCA–aa junction in *reference* coordinates (CCAGGC motif +3), maps ref→query via CIGAR and query→signal via the move table (Remora convention, `move_pos * stride + ts`), detects per run whether the move table indexes reversed signal (voted, ≥50 reads + 95% consensus — getting it wrong silently mirrors every window), computes per-base dwell/mean/std + the z-scored k-mer residual with everything before the common arm masked, and runs the bundle's binary classifier. The recipe travels in the bundle `metadata.json` (feature order/offsets, k-mer table pinned by sha256, operating point) — never in flags. That schema is **closed** (`deny_unknown_fields`): every key in the file is a rule the model was built with, so one this runtime does not implement is refused rather than dropped at parse time, and the builder's prose is *named* rather than allowed by omission. `provenance`/`metrics`/`caveats` are exempt and free-form. Two rules are refused outright because the runtime could nearly run them (a non-empty `refinement.opts`, a per-read-transforming `features.feature_set`); `abstain` is carried on `ChargingBundle` and warned about instead of applied (#230).

  **A third variant: the signal window.** `waveform_model` (`waveform-onnx` feature) reads a *window* rather than a column vector — three tensors assembled by `escapepod_signal::chunk` (normalised current + its k-mer residual, the sequence k-mer context scattered along the signal axis, 12 per-base dwell/level rows) — and emits a **single BCE logit**. It is a second pipeline over the same two files, not a second scorer on the first one: the reference a read is scored against is rebuilt from that read's **`MD` tag** (as the training corpus does) rather than sliced from the FASTA — they differ wherever the FASTA holds an `N`, and since levels are per-9-mer one `N` blanks nine of them, which cost 169/256 chunks bit-exactness before it was found (#306); a read with no `MD` is refused rather than scored off the FASTA. The map is walked through the CIGAR into *reference* coordinates, the anchor is motif **+2** (not the feature-grid variants' +3), the spans are banded-DP refined before any feature is taken, and the frame is declared by the bundle rather than voted from the data. `bundle.rs`'s refinement refusal is therefore narrowed rather than removed — the column variants still cannot reproduce a DP pass and are still refused for it. Nothing about the rows is hard-coded: the bundle ships `waveform_model.channels.{signal,features}.order` and the runtime resolves it, since a permuted tensor has exactly the shape it should. Two more rules are cross-checked because each fails silently — `preprocessing.motif`/`motif_offset` against the `anchor` block, and `output.positive_class` against `classes` (leech assigned class ints at merge time and gave `charged` 0, so `P(charged) = 1 - sigmoid(logit)`; read it the obvious way and every call inverts). The shipped Platt calibration is **carried, not applied** — the operating point beside it is stated on the uncalibrated probability. Runs through tract like every other ONNX graph here, and is in the default build — but only since `charging_tcn_rna004@v0.1.1`. `@v0.1.0` was the one graph tract could not load, not for want of an op but because tract 0.23's *shape inference* cannot close it — **two independent causes**, either alone sufficient: dynamo's `value_info` for all 667 intermediates carrying the batch axis as the *symbol* `batch` (so a consumer that pins the batch dies at the **first convolution**, `Sym(batch) vs Val(1)`), and `adaptive_avg_pool1d(390 → 11)` open-coded into a rank-8 `GatherND` because the output does not divide the input. It is **not** `nn.MultiheadAttention` — this file, `waveform_net.rs` and escapepod-models#96 all named that until the graph was read: the `GatherND` consumes `relu_17` (last block of `signal_tcn`) and carries that pool's `(11,37)` bin mask and `(11,)` bin widths `[36,36,37,36,37,…]`, while `cross_attn` exports as plain `Mul`/`MatMul`/`Softmax`/`MatMul`/`Gemm`. #306's original suspect was right and its retraction was not. Neither onnx-simplifier nor onnxruntime's optimiser fixes it, so it could not be papered over at load time the way `fnn::hoist_conv_padding` is. It ran through `ort` (`load-dynamic`, so a *released* static-musl escpod could not run such a bundle at all) until the re-export landed — rnabioco/escapepod-models#96 (leech 0.10.0: pool as one `MatMul` against a constant segment-mean matrix, `value_info` stripped; 479→319 nodes, same weights, no retrain), plus #97's build-time gate refusing to register a graph the shipped runtime cannot load. `escapepod-demux/examples/tract_dynamo_probe.rs` is the measurement, kept re-runnable: v0.1.0 fails analysis at `node_GatherND_329` in all three modes, v0.1.1 optimizes and runs. The generalising lesson, and the reason that gate exists: **onnxruntime loaded the broken graph perfectly**, so the export's own torch round-trip stayed green — "it exports and agrees with torch" is weaker than "a runtime can load it". The feature `waveform-onnx` remains, naming the *variant* rather than a runtime. **This graph must not be scored in a batch without `escapepod_demux::onnx_rewrite::expand_instance_norm`.** tract lowers ONNX `InstanceNormalization` by reducing over every axis but the channel (`instance_norm.rs`: `(0..rank).filter(|a| a != 1)`), so for `[N, C, W]` it reduces over the **batch** as well as the spatial axis — where the operator is defined per *instance*. At the batch 1 this runtime pins, the two are identical and nothing shipped is affected; above it, every read is normalised against its neighbours and the same read's logit goes `+1.47` alone, `+0.26` in a pair, `-0.28` in 128. A sign flip, not a tolerance, and invisible to any check that scores one read at a time. The same two axes are why a CUDA transform leaves 29 `Reduce<MeanOfSquares>` on the host (`GpuReduce` takes one axis; `split_multi_axis_reduce` covers the reducers that chain, and mean-of-squares does not), so the rewrite buys correctness under batching and full device residency with one change — 62 syncs to 4, and 11.1× a core at batch 128. It is deliberately **off** in `waveform_net.rs`: the CPU path scores batch 1, where tract's lowering is already the spec, and the explicit graph is ~5% slower there. `examples/tcn_cuda_probe --dump` is the measurement and prints the host/device split as islands plus each `CudaConv`'s kernel — cuDNN and the naive `conv1d_f32_generic` share an op name, and 24 of 27 convolutions here miss cuDNN for padding asymmetrically, worth a further 4.1× via `--hoist` at three orders of parity (3.1e-5 → 1.7e-2), which is why that one is off too.

  **Two scorers, one feature space.** A bundle carries `gbm` (gradient-boosted trees, `NaN` routed natively) or `feature_model` (a small ONNX network over the same columns, `fnn-onnx` feature — tract, already in the binary via `cnn-detect`). Everything above the last step is shared verbatim, so the model-specific part of the pipeline is the `ChargingScorer` enum and nothing else; which arm a directory holds is a property of the bundle, never a flag. The network is what escapepod-models ships — 0.727 of reads callable at 99% precision against the GBM's 0.449 on a held-out flowcell — and needs three rules reproduced exactly that the GBM does not have: the fold from offsets-outer columns to `[channel, offset]`, per-channel standardisation with shipped constants, and missingness as an explicit mask channel rather than `NaN`. Each fails *silently* if guessed (a wrong fold transposes the input and still scores), so all three are declared in the bundle, the fold is checked against `features.order` at load, and parity is pinned both bit-exactly on reference feature vectors (`tests/charging_fnn_parity.rs`) and on real weights over a real corpus (`examples/verify_feature_model.rs` + `scripts/dump_feature_model_reference.py`). `FeatureNet::load` also rewrites each padded `Conv` into `Concat(zeros) + Conv(pads=0)` on the ONNX proto: tract's im2col abandons its block-copy path when `pads != 0`, which cost 82% of the shipped CNN's runtime (6.1× — 305 → 50 µs/read, bit-identical). A `Pad` node does not work (tract fuses it back) and this belongs in the loader, not the export — see `fnn::hoist_conv_padding`. Depends on `escapepod-signal` (k-mer primitives, POD5) and `escapepod-demux` (the GBM runtime); noodles-sam for record types only (BAM file I/O stays in the CLI). Parity with the training-corpus implementation (`escapepod_models.charging`) is pinned by golden-vector tests (`tests/fixtures/gen_charging_golden.py`).

  **The shipped network's recurrence runs natively.** `charging_feature_nn_rna004` is `arch: lstm` — a bidirectional LSTM, 96 hidden, over the 33 offsets, read out as mean-and-max over time into one linear layer — and through tract it cost ~490 µs/read, ~34% of the whole command's CPU with only ~14% of that in the matmul/sigmoid/tanh kernels; the rest was per-timestep bookkeeping (`Scan` state eval, plan execution, tensor allocation, dynamic-rank `ndarray` iteration, symbolic-dim eval), and an unrolled graph measured 3× *worse*. `fnn_lstm::NativeBiLstm::from_proto` recognises exactly that graph in the ONNX proto (one bidirectional `LSTM` fed by `Transpose(perm=[2,0,1])` of the input, `ReduceMean`+`ReduceMax` over time, `Concat`, `Gemm`), refuses every variation ONNX's LSTM semantics permit (peepholes, custom activations, `clip`, `input_forget`, a non-zero initial state, per-sequence lengths, `layout`) and lifts the weights into an axpy-form recurrence with a runtime-dispatched AVX2+FMA kernel. Anything else stays on tract, so a differently-exported bundle is scored the general way rather than wrongly. Not bit-identical to tract (Cephes `exp`, `tanh = 2σ(2x)−1`); the contract is agreement on the probability inside the grid's own 1e-4 tolerance, pinned against tract on random weights of the real shape in `fnn_lstm::tests`, and per read on real data (`benchmarks/README.md`). `ESCAPEPOD_FNN_TRACT=1` forces tract — the A/B lever, and the way back. The single-read kernel streams `Rᵀ` (147 KB per direction) every timestep, so `logits_batch` scores **three reads in lockstep** — one row load per 32-gate slice feeds all three accumulator sets, 4 × 3 + 3 broadcasts + 1 load = the 16 YMM registers — over weights packed block-major (`[4H/32][H][32]`, one contiguous 12 KB block per pass): 122 µs/read against 194 single-read and 490 through tract, bit-identical to the single-read path (the per-lane FMA order does not change with the batch or the layout). `classify_reads` feeds the scorer `par_chunks` of eight groups, not one (the prep between chunks evicts the weights, and a chunk short one read scores a narrower group); GBM and tract report a batch of 1. On a machine with AVX-512F the same slices run sixteen lanes wide with eight reads per row load (`run_avx512_batch`, 26 of 32 ZMM; runtime-dispatched, never a baseline bump; a lone read still takes the AVX2 kernel): 97 µs/read against 115, bit-identical across widths, `ESCAPEPOD_LSTM_BACKEND=avx2|scalar` capping the dispatch for A/Bs. The loop is bound by L2→L1 bandwidth (~23 B/cycle on rna — `examples/axpy_probe.rs` runs every shape from L1, L2 and L3; keep it re-runnable), which is why reads per row load is the axis that pays and wider slices are not; the shape table and the five dead ends are in `benchmarks/README.md`. The three kernels are `#[inline(never)]` on evidence: inlined into the dispatch beside one another, the AVX2 width-2 kernel returned wrong logits for its second read, bisected to the *form* of the match (or-pattern vs two arms) with the kernel source untouched — LLVM miscompile suspected, unsafe audited, not proven — so the kernel pins in `fnn_lstm::tests` run every backend the machine has, not the one the dispatch prefers. In the same pass: the per-read median/MAD gauge (11.5% of CPU as `select_nth_unstable_by`) now selects on order-preserving `u32` keys computed once per element (`features::median_of_keys`, bit-identical; a 256-bin radix select was tried first and measured 1.8× *slower* at ~10k samples); the k-mer level lookup goes through `KmerLevels::packed()`, a flat table indexed by 2-bit k-mer instead of a `String`-keyed hash probe per base (`KmerTable::from_levels_map` builds it from the pinned map so the values are the map's, not re-parsed); `classify_reads` walks each POD5 forward by sorting the work on `(file, first signal row)` instead of `HashMap` order; and both BAM passes' BGZF pools are sized from the rayon pool (`MultithreadedReader::new` is ONE worker).

  **Every definition of the model's input lives in this crate, not in a caller.** `recipe::FeatureRecipe` is the feature space (offsets, span mode, k-mer levels) as a borrowed view, so `feature_grid` takes a recipe rather than a whole bundle — the corpus builder that computes the same features has no weights to hand it, and forcing it to invent one is how you get a second, divergent definition. `window::signal_window` is the raw-signal counterpart (anchoring inside the junction base, `NaN` padding, the common-arm mask); it lived only in the pyo3 binding, invisible to the Rust inference path that has to reproduce it. If a rule decides what the model sees, it belongs here, and a binding or a CLI only marshals.
- **escapepod-cli**: the `escpod` CLI binary (built by the default `cli` feature) plus an optional umbrella library (imported as `escapepod_cli`) that re-exports `pod5`/`signal`/`demux` behind feature flags for `default-features = false` consumers. Demux is part of the default `cli` feature (it adds no third-party crates — ndarray/serde_json/rayon/uuid are already in the graph), as is `cnn-detect` (it does add tract-onnx, ~+21 MB stripped, but only to the binary — library consumers must already use `default-features = false` to avoid clap/noodles, so they never pay for it; `--method cnn` is what the published barcode models were trained against). `default-features = false` consumers can still select either explicitly. Forward features `train` and `gpu` propagate to the demux crate and remain opt-in because they need extra toolchains or a CUDA runtime. `gpu` is the only GPU feature here — there is no way to build half a GPU binary. (`escapepod-signal/gpu` stays narrower on purpose: cudarc DTW only, no onnxruntime.)
- **escapepod-python**: pyo3 bindings.

### POD5 File Format

POD5 is a container format wrapping Apache Arrow IPC (Feather v2) tables:

```
<POD5 signature>
<section marker>
<Signal table (Arrow IPC)><section marker>
<Run Info table (Arrow IPC)><section marker>
<Reads table (Arrow IPC)><section marker>
<FOOTER magic>
<FlatBuffer footer>
<footer length>
<section marker>
<POD5 signature>
```

### Format layer (escapepod-pod5)

- **reader/file_reader.rs**: Memory-mapped file reader using `memmap2`. Opens POD5 files, parses the FlatBuffer footer, and provides iterators over reads and signal data.
- **writer/file_writer.rs**: Buffered writer that constructs POD5 files. Handles signal chunking, VBZ compression, and batching of Arrow record batches.
- **compression/**: VBZ signal compression (SVB16 + ZSTD pipeline)
  - `svb16/mod.rs`: Scalar SVB16 encode/decode + runtime dispatch to SIMD.
  - `svb16/simd_ssse3.rs`: SSSE3 encode/decode (~2× vs scalar on x86_64 w/ SSSE3).
  - `svb16/tables.rs`: `pshufb` shuffle + length tables for both directions.
  - `vbz.rs`: Full VBZ pipeline combining SVB16 with ZSTD compression
- **footer.rs**: Manual FlatBuffer parsing for the POD5 footer (locates embedded Arrow tables)
- **schema/**: Arrow schema definitions for reads, signal, and run_info tables
- **types.rs**: Core data types (`ReadData`, `RunInfoData`, `EndReason`, etc.)
- **merge.rs**: File merging operations with run info deduplication
- **operations/**: High-level file operations
  - `filter.rs`: Filter reads by criteria (ID list, sample count, end reason)
  - `repack.rs`: Repack files with new compression settings
  - `subset.rs`: Split reads into multiple files by barcode or CSV mapping

### Signal layer (escapepod-signal)

- **dtw/**: Dynamic Time Warping distance computation
  - `distance.rs`: DTW algorithm with Sakoe-Chiba band constraint
  - `fingerprint.rs`: Signal fingerprint representation
  - `kernel.rs`: DTW distance to kernel conversion for SVM
  - `cuda/`: GPU-accelerated DTW distance matrix (opt-in `gpu` feature)
- **segmentation/**: Signal segmentation algorithms
  - `llr.rs`: Log-Likelihood Ratio boundary detection
  - `ttest.rs`: T-test based changepoint segmentation (scipy-compatible peak detection)
  - `normalize.rs`: MAD, z-score, and min-max normalization
- **resquiggle/**: Signal-to-base alignment refinement (banded DP, rescaling, drift correction)
- **chunk.rs**: windowed model inputs — one read reduced to per-sample channels and per-base rows, then cut into fixed-size chunks around a base the caller nominates. The assembly step every signal-level network needs, wired from the primitives already in this crate (`mapping`, `features::span_stats`, `seq_encoding`, `resquiggle`) in the order a trained model was fitted on. It lives here because that wiring is the kind of rule that fails *silently* — a window on the wrong side of a base, a permuted channel list, a k-mer context split `(4, 4)` instead of `(3, 5)` all produce a correctly shaped tensor of plausible numbers — and it already had two implementations (leech's Python dataset, leech-core's Rust pipeline) before a third was nearly written in `escapepod-classify` (#306). Generic by construction: the anchor is a base index, the window is `(left, right)` samples, and **the channels are a `Vec` the caller supplies** (`FeatureChannel`, `SignalChannel`), so two models reading the same rows in a different order are two values rather than two code paths.

### Demux layer (escapepod-demux)

- `model.rs`: `WarpDemuxModel` and `DtwSvmModel` JSON loaders.
- `classify.rs`: Per-read DTW classifier (`classify_read`), shared top-2 threshold logic, optional batched GPU classifier.
- `svm.rs`: Full SVM predictor — RBF kernel, OvO dual coefficients, Platt scaling, multiclass probability coupling, batched GPU classify.
- `probability.rs`: softmax / coupling utilities.
- `train.rs` (feature: `train`): SVM training via linfa-svm, optional GPU all-pairs DTW matrix.
- `adapter_cnn.rs` (feature: `cnn-detect`): runs a user-supplied boundary-CNN ONNX graph (`[B,1,L] -> [B,2,L]`) through tract-onnx at runtime, one read at a time (CPU). Architecture-agnostic — works with escapepod-models' `adapter_rna004` TCN (CC-BY) or an ADAPTed `BoundariesCNN` export (CC BY-NC, NOT bundled). Shared `prep_adapter_signal`/`decode_adapter_end`/`group_by_len` helpers + a batched `detect_adapter_end_batch` back both the CPU and GPU paths bit-identically. Prep reproduces escapepod-models' `dataset.py::prepare_signal` exactly: window `[min_obs_adapter:max_obs_trace]`, mean-pool, median/MAD-normalise, then **right-pad to a fixed `input_len` = `(max_obs_trace - min_obs_adapter)/downscale` = 1500 with `SCORE_EXCL` = −5.0**. Matching the training convention is both a correctness fix and the pipeline's single biggest speedup (#187): it is the input distribution the model was fitted on, *and* it collapses every read onto one tensor shape. The previous truncate-to-806 gave short reads their own length each — 680 distinct shapes over one production run, so a 65 k-read block issued ~680 onnxruntime calls each paying fresh cuDNN plan selection, and GPU detect measured 94% of wall (401 s → 20.2 s once fixed). Because the length is now fixed, `group_by_len` always yields exactly one group and `pack_batch` never pads; both are kept as the guard that a future ragged config cannot silently mis-pack. Decode argmaxes only `(max_obs_adapter - min_obs_adapter)/downscale` = 550 of the 1500 outputs (ADAPTed's rna004 prior) where escapepod-models' own `predict.py` argmaxes all 1500 — measured on the gold set, 0.41% of true adapter ends fall past that cap and are unreachable. A load-time shape probe rejects a wrong-output-shape model up front instead of silently writing `adapter_end=0`.
- `adapter_cnn_gpu.rs` (feature: `gpu`, which implies `cnn-detect`): the architecture-agnostic **GPU** path — same ONNX graph + same prep/decode, run batched through onnxruntime's CUDA execution provider via the `ort` crate (`load-dynamic`; needs a CUDA-enabled `libonnxruntime` on `ORT_DYLIB_PATH` + a visible GPU at run time, nothing at build time). `escpod demux detect --method cnn` on a GPU node (`--device auto` places it there; `--device gpu` demands it). The TCN is **inference-bound, not I/O-bound** (LLR detect = 0.25 s/20k vs CNN = 77 s/20k; ~99.6% is inference), so GPU pays off — ~7.6× end-to-end on an A30 at 20k reads (grows at scale; isolated inference is ~99× tract). GPU-placed `detect` runs a dedicated GPU consumer thread (the ort/CUDA session builds while CPU producers decode+prep in parallel, overlapping init) fed prepped, length-bucketed blocks through a bounded channel (`AdapterCnnGpu::detect_prepped`). The **fused pipeline** (`escpod demux --method cnn` on a GPU node) also runs GPU detection: all producers (`produce_cpu`/`produce_cpu_gbm`/`produce_gpu`) detect via `Detector::detect_batch` over windowed decode-once blocks (`DETECT_WINDOW`), so the GPU does one batched call per window while preserving the single-stream I/O (#72) — measured ~7.2× end-to-end (116 s→16.2 s, A30, 20k GBM) with 99.99% classification parity vs CPU detect. NOT the old arch-locked CUDA kernel (removed in #80, hardwired to BoundariesCNN); this runs any `[B,1,L]->[B,2,L]` graph. tract has no efficient batched conv, so CPU detection stays per-read.
- `crf/lattice.rs`: the **CTC-CRF decode** — a port of bonito's `CTC_CRF` (`bonito/crf/model.py` plus the CUDA kernels behind `koi.ctc.{fwd,bwd}_scores_cu_sparse`). Pure `f32`, **no dependencies and no feature gate**, so CI exercises it without a model file. Sparse lattice over `n_states = n_base**state_len` (256) with `n_edges = n_base+1` (5) edges each. Two traps, both pinned by tests: edges are indexed `(destination, dropped_base)` **not** `(source, emitted_base)`, so the emitted base is fixed by the destination alone (`dest % n_base`) and the closed form is `source(c,0)=c`, `source(c,1+j)=j*(n_states/n_base)+c/n_base`; and the score width is **1280 = n_states*n_edges**, not the 1024 of the linear layer (`LinearCRFEncoder` expands blanks to one per state). The decode is **two passes** and both are load-bearing: log-semiring forward/backward → per-timestep edge posteriors (`softmax`, which is the closed form of the `autograd.grad` bonito actually uses), then max-semiring forward/backward over `log(posteriors + 1e-8)` → argmax edge. Decoding raw encoder scores in one Viterbi is a different, worse decode.
- `crf/avx2.rs`, `crf/avx512.rs` (x86_64, runtime-dispatched via `Backend::best_for`): 8- and 16-wide kernels with Cephes-style vector `exp`/`ln`. The decode is transcendental-bound (~770k `exp` + ~260k `ln` per 200-timestep read) and started at **half** the total CPU cost. Measured per read on rna (`cargo bench --bench crf_decode`): scalar 12.14 ms, AVX2 1.92 ms (6.3×), AVX-512 1.19 ms (10.2×); tract's encoder is 13.9 ms for scale. This is a **prerequisite** for the GPU CRF encoder, not a nicety: with the encoder on the device the decode *is* the runtime. Forward and backward are duals, so each gets one cheap reshape (`expand` replicates each source 4×; `deinterleave4` transposes the 4:1 fold) after which every inner loop is five unit-stride rows; the `[dest][edge]`→`[edge][dest]` transpose is a 5-way de-interleave that no shuffle network handles cleanly, so it uses strided gathers. Not bit-identical to scalar (polynomial `exp`/`ln`, reassociated softmax denominator) — the contract is same-sequence, enforced by an equivalence test over every supported backend; both SIMD argmaxes break ties by flat index, so they cannot disagree with scalar on a tie. AVX-512 is runtime-detected per the build policy, never a baseline bump.
- `crf/encoder.rs` (feature: `crf-decode`): the ONNX encoder through tract. Contract is `[batch,1,chunk] -> [chunk/stride, batch, n_score]`, **time-major** — the opposite of `adapter_cnn.rs`'s batch-major `[B,2,L]`, so it needs its own load-time shape probe rather than reusing that one. Batch is pinned to 1 (tract has no efficient batched LSTM; parallelism is across reads). Standardisation constants come from the export's `metadata.json` sidecar and are **not** in `config.toml` — that file carries SeqTagger's unrelated 80.876/17.270, which silently degrades the decode.
- `crf/barcode.rs` (feature: `crf-decode`): matches a decoded sequence to barcode references by edit distance, via `fqxv-align`'s wavefront aligner (git dep on `rnabioco/fqxv`, pinned by rev; a zero-runtime-dependency leaf crate extracted for this — rnabioco/fqxv#252). WFA rather than a DP because work scales with edit *distance*: a decode sits ~4 edits from its own reference and ~10+ from the other 95, so most comparisons abandon almost immediately. References are the **last 40 nt of each barcode strand** (the training targets) and are supplied ready-to-use — deriving them from a pool-oligo table is the barcode design's business, not the demultiplexer's. Ties resolve to the lowest index with margin 0 rather than a silent coin flip.
- `crf/encoder_gpu.rs` (feature: `gpu`, which implies `crf-decode`): same ONNX graph and same decode, encoder run batched through onnxruntime's CUDA EP. The lattice decode stays on CPU by design (sequential in time, 256-wide inner dim) and fans out across rayon while the GPU runs the next batch.

### CLI Commands

- `view`: Display reads as TSV with configurable columns
- `inspect`: Show file metadata (summary, reads list, specific read)
- `summary`: Comprehensive summary with statistics
- `merge`: Combine multiple POD5 files (parallel reading with rayon)
- `filter`: Extract reads by ID list or criteria (sample count, end reason)
- `bam-filter`: Filter reads based on paired BAM file (mapped status, region, quality)
- `repack`: Repack files for optimized storage
- `subset`: Split reads into multiple files based on CSV mapping
- `resquiggle`: Refine signal-to-base mapping using banded DP with POD5 signal and BAM move tables. Takes a k-mer level table via `--kmer-table <path>` or a named model via `--kmer-model <name>` (DNA + RNA; `dna_r10.4.1_e8.2_400bps`, `rna004`, …). Named models resolve from a local cache (`$ESCAPEPOD_KMER_CACHE` → `$XDG_CACHE_HOME/escapepod/kmer_models` → `~/.cache/…`) that is **never** populated at runtime — build with `--features models-download` and prefetch on a networked login node (`escpod resquiggle models fetch --all`) before submitting compute jobs (Alpine/Beevol compute nodes can't reach GitHub). Tables come from nanoporetech/kmer_models (MPL-2.0), pinned to a commit + sha256; the code path uses `ureq`/rustls so the static-musl release stays OpenSSL-free.
- `classify`: tRNA charging (aminoacylation) classification from POD5 + aligned BAM (`escpod classify reads.pod5 -b aln.bam -r ref.fa -m bundle/ -o out.bam`). Writes `cl = round(P(charged)·255)` (uint8) directly onto the BAM — no modbase `ML`→`cl` round-trip — plus optional `--tsv`. `--orientation time|reversed` overrides the frame vote for small batches (< 50 informative reads); it is ignored (with a warning) by a `waveform_model` bundle, whose frame is declared rather than voted.

  It was `escpod signal classify` from 0.11.0 through 0.18.1, grouped under a `signal` namespace so the *word* `classify` could not be confused with `escpod demux classify` (barcode DTW/GBM). Every other tool in the binary is one word, and the two were never ambiguous in use — `demux classify` is a demux stage over a fingerprint CSV, this takes a POD5 and a BAM — so the group is retired. `escpod signal classify` survives as a **hidden deprecated alias** (`commands::signal`) that warns and forwards to the same runner, so 0.11-era scripts keep working.
- `demux`: Barcode demultiplexing workflow with subcommands:
  - `detect`: LLR-based adapter boundary detection
  - `fingerprint`: T-test segmentation for barcode fingerprints
  - `classify`: DTW-based barcode classification
  - `split`: Split reads by barcode into separate files
  - `train`: Train reference fingerprints from known samples
  - `models`: manage demux model downloads — `list`, `path`, `fetch <bundle>`. Pinned manifest of GitHub-Release bundles with per-member sha256; cache at `$ESCAPEPOD_DEMUX_MODEL_CACHE` → `$XDG_CACHE_HOME/escapepod/demux_models` → `~/.cache/…`. **Fetch is explicit and resolution never touches the network** (compute nodes can't reach GitHub — a lazy fetch would hang a job). The fetch unit is a *bundle*, so the boundary↔barcode pair can't be split. `escapepod-models` is private today, so fetching needs `GITHUB_TOKEN`/`GH_TOKEN`; the REST endpoint used also serves public repos anonymously, so no code change is needed if it opens up. Consumed by `detect --cnn-model-name` and `classify --model-name`, alongside the existing path flags.
  - `basecall`: CTC-CRF barcode basecalling from a boundaries CSV (requires `crf-decode`, in the default build). With `--barcodes <name,sequence CSV>` it also assigns each read to its closest reference by edit distance and emits `read_id,barcode` — exactly what `demux split` reads, so `detect -> basecall -> split` runs end to end with no Python. Without it, decoded sequences only. Confidence is the edit-distance margin to the second-best reference (the definition the model's published precision-at-recovery numbers used).
  - `train-svm`: Train SVM model (requires `train` feature)

### Key Patterns

**Block-level copying**: For merge/filter operations, signal data is kept compressed (`CompressedSignalChunk` with `Arc<[u8]>`) to avoid decompression/recompression overhead. Use `add_read_with_compressed_signal()` instead of `add_read()` when copying between files.

**Dictionary tracking**: The writer maintains O(1) lookup for pore types and end reasons using HashMap indexes alongside Vec storage for Arrow dictionary encoding.

**Run info deduplication**: When merging files, run infos are deduplicated by `acquisition_id` to avoid redundant entries.

## Dependencies

### Format crate (escapepod-pod5)
- **arrow**: Arrow IPC file reading/writing
- **memmap2**: Memory-mapped file I/O
- **zstd**: ZSTD compression for VBZ
- **flatbuffers**: FlatBuffer footer parsing
- **uuid**: Read ID handling
- **csv**: CSV parsing for filter IDs and barcode mappings
- **byteorder**, **thiserror**, **tempfile**, **rayon**

### Signal crate (escapepod-signal)
- **escapepod-pod5**: re-exported as `pod5` plus the full type surface
- **ndarray**: Array operations for signal processing
- **rand**, **flate2**: resquiggle internals
- **serde/serde_json**: JSON model serialization (demux)
- **linfa/linfa-svm**: SVM training (optional, requires `train` feature)

### CLI (escapepod, `cli` feature)
- **clap**: CLI argument parsing
- **rayon**: Parallel iteration for merge operations
- **tabled**: Table formatting for CLI output
- **noodles-bam/sam**: BAM file support for bam-filter command
- **walkdir**: Directory traversal

## Test Data

Test POD5 files from Oxford Nanopore are in `ext/nanopore-dna-data/pod5/`. The `ext/pod5-file-format/` directory contains the official POD5 specification.

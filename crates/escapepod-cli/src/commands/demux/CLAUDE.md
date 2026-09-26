# demux command tree

## Shape of the pipeline

- `run.rs` is the fused single-stream pipeline: decode-once blocks → detect → fingerprint/encode → classify → route to per-barcode writers and/or `.p5s` annotate. The subcommands are its stages standalone; they share `utils.rs`, not `run.rs`.
- Producers: `produce_cpu` (DTW-SVM), `produce_cpu_gbm`, `produce_gpu` (SVM, GPU DTW), `produce_cpu_crf`, `produce_cpu_crf_multi` (ldx+fdx), `produce_gpu_crf`. All but `produce_gpu` go through `drive_blocks`.
- `drive_blocks` owns reader threads + bounded channel; `fill_shard` one strided sweep + block caps; `writer_thread` one POD5 per barcode + run-info remap.

## File map

- `run.rs` — fused pipeline, `Detector`, `ClassifyModel`/`CrfHead`, `LeakIf`, annotate, summary.
- `classify.rs` — fingerprint table → barcode (SVM, GBM, WarpDemuX bank, reference CSV).
- `detect.rs` — LLR/CNN boundaries; `--emit-llr-delta`.
- `basecall.rs` — CRF decode from boundaries (`crf-decode`).
- `fingerprint.rs` — t-test fingerprints, CSV/Parquet by extension.
- `split.rs` — reads → per-barcode POD5 from CSV or `.p5s`.
- `train.rs` / `train_svm.rs` — reference bank / SVM fit (`train`).
- `models.rs` — pinned manifest + offline cache (`demux-models`; fetch needs `model-fetch`).
- `info.rs` `--info`; `fp_io.rs` table readers; `utils.rs` CSV parsers, `decode_chunks_to`, `process_reads_par`; `mod.rs` dispatch + dir expansion; `types.rs` train JSON.

## Invariants and traps

- `DEFAULT_FILLERS=2`, `DETECT_WINDOW`, `BLOCK_TARGET_BYTES`, `CRF_ENCODE_GROUP=8` are interleaved-benchmark results, not guesses.
- Block composition perturbs GPU detect output (detect.rs:598, 7/503k). Changing `produce_gpu` chunking or detect.rs `sort_by_key(num_samples)` must pass `benchmark_demux_parity.sh`.
- `crf_encoder_devices` shares GPU 0 with detect — measured; `every_visible_device_encodes` pins it.
- `LeakIf` / `mem::forget(gpu)` are the pykeio/ort#609 abort mitigation, not leaks.
- `--annotate` writes `.p5s` beside the path *as given*; never aim it at canonical data (harnesses symlink into scratch).
- Classification CSV row order is nondeterministic (rayon streaming writers); sort before diffing.
- `Detector::None`, `Calls::One|Many` are deliberate enums.
- Fused `demux` refuses the WarpDemuX legacy bank; `demux classify` accepts it.
- `--method` has no default; LLR is opt-in (17.2-point recall cost, silent).

## Harnesses

- `benchmarks/benchmark_demux.sh` — SVM/DTW vs WarpDemuX.
- `benchmarks/benchmark_demux_crf.sh` + `evaluate_demux_crf.py` — fused CRF on the 20k dual-axis set; 0-of-20000 diffs contract.
- `benchmarks/benchmark_demux_parity.sh` — output parity across arms.
- `basecall` and `classify` have **no** integration tests; the harnesses are the tests.

## Env levers

| var | effect | default | tested / documented |
|---|---|---|---|
| `ESCAPEPOD_DEMUX_FILLERS` | reader shards (cap 32) | 2 | no / CHANGELOG |
| `ESCAPEPOD_CRF_GPU_BLOCK` | reads per encoder handoff | 4096 | no / CHANGELOG |
| `ESCAPEPOD_CRF_GPU_WORKERS` | encoder sessions | 2×devices (≤8) | test skips if set / CHANGELOG |
| `ESCAPEPOD_CRF_GPU_TRACE` | stage trace (`1/true/yes/on`) | off | no / CHANGELOG |
| `ESCAPEPOD_DEMUX_MODEL_CACHE` | cache dir (then XDG, HOME) | `~/.cache/escapepod/demux_models` | yes / root CLAUDE.md |
| `GITHUB_TOKEN`/`GH_TOKEN` | `models fetch` auth | none | no / no |

`ESCAPEPOD_CRF_GPU_BATCH_ROWS`, `ESCAPEPOD_CRF_TRACT` live in escapepod-demux.

## Known debt

- run.rs:3691 `produce_gpu` bypasses `drive_blocks`; batch prelude copied ×6 (utils:318, run:2274, fingerprint:214, basecall:395, train:335).
- run.rs:2810 `produce_cpu_crf` ≡ `_multi` at n=1 (`produce_gpu_crf:3434` shows the `Calls` match).
- run.rs:857 `CrfEncoderAny` ≡ basecall.rs:146 `Basecaller`; override block pasted run:1286/basecall:544; `Gates` inlined at run:2933.
- classify.rs:829/919 streaming writers ×2 (dead `_expected`); result struct built ×5; `run_with_csv` doesn't stream.
- fp_io.rs:165/427 reservoir sampler ×2.
- LLR ×3: run:422, detect:130, train:409 (train's skips downscale — unifying changes reference banks; explicit decision).
- Env readers ×5 (run:2070–3141); `GpuTrace` ≡ detect.rs `StageTimes`.
- `for_writing` called per read in every producer and again in `writer_thread:3845`.
- `sha256_hex` ×3 (run:4357, models:670, resquiggle_models:277).
- run.rs:1422 "Only CRF bundles" bail unreachable after :1123.
- run.rs:443 `detect_with` `CnnGpu` arm unreachable.
- `--svm-model` alias (classify.rs:66) removed next minor; update `benchmarks/benchmark_demux.sh:194`, `benchmarks/README.md:894` first.

## Do not

- Read POD5 outside `drive_blocks`/`process_reads_par` — per-read faulting is 0.3 MB/s on BeeGFS (#72).
- Judge an I/O or placement change by ascending-order arms — interleave.
- Infer `--method llr` — the failure is silent.
- Define model input here — that belongs in escapepod-demux/-signal.
- Raise `ESCAPEPOD_CRF_GPU_WORKERS` by default — each session is a BFC arena that never shrinks.

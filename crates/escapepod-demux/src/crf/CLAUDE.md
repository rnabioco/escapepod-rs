# crf/

## What this is
Port of bonito's `CTC_CRF` (koi `ctc.{fwd,bwd}_scores_cu_sparse`). Encoder (native LSTM stack, tract, or ort CUDA) → two-pass lattice decode → barcode call by WFA edit distance (`barcode`) or constrained partition function (`refchain`).

## Module map
- `mod.rs` — gates/re-exports; `BoundaryInputSpec` lives here so pin plumbing builds without `crf-decode`.
- `lattice.rs` — `CrfLayout`, `CrfScratch`, scalar kernels, `Backend` dispatch; entry `decode_with*`. Always compiled.
- `avx2.rs` / `avx512.rs` — 8/16-wide mirrors; chosen by `Backend::best_for`.
- `refchain.rs` — `RefChains::build`/`forward`: `log P(ref | signal)`. Always compiled.
- `encoder.rs` — `CrfMetadata` (sidecar, window rules), `CrfEncoder`; entry `basecall_prepped*`.
- `encoder_native.rs` — `Recognized::from_proto` recognizer + `encode_*` over `escapepod_signal::lstm`.
- `encoder_gpu.rs` — `CrfEncoderGpu`: ort CUDA, zero-copy, halve-on-OOM; entry `basecall_batch*`.
- `lattice_gpu.rs` + `_kernel.rs` — batched CUDA decode + ref scan, NVRTC at load; entry `decode_*time_major`.
- `barcode.rs` — `BarcodeRefs::match_sequence` (fqxv WFA).

## Invariants and traps
- Edges are `(destination, dropped_base)`; emitted base = `dest % n_base`. Inverted → plausible garbage. `source_state_matches_bonito_idx`.
- `n_score` = 1280, not the linear layer's 1024. `rna004_geometry`.
- Two passes (log posteriors, then max over `log(post+1e-8)`); one Viterbi on raw scores is a different decode. `crf_golden`.
- SIMD/GPU: same sequence, not bit-identical; ties by flat `dest*n_edges+edge`. `simd_backends_agree_with_scalar`, `gpu_crf_lattice`.
- Standardisation from `metadata.json`, never `config.toml` (SeqTagger's 80.876/17.270 degrades silently). `sidecar_parses_*`.
- Output is time-major `[T, B, 1280]`; `adapter_cnn` is batch-major — no shared probe. `BadShape`.
- Recognizer accepts one exported shape; `self_check` (tol 0.05) gates it. `unsupported_graph_falls_back_to_tract`.
- Four evidence-backed locality decisions — SiLU output has two consumers; name equality before "producer is a Slice"; output side from `reverse_in`; trace filters Reshape consumers. Don't "simplify". `native_matches_tract_on_random_weights`.
- CPU transposes to `[edge][dest]`; GPU reads native `[dest][edge]`. `every_backend_transposes_identically`.
- CPU decode is the reference and only always-compiled path; GPU decode falls back to it.
- `blank_score` comes from the graph's `Pad`; the sidecar value is a cross-check. `blank_pad_matches_layout`.
- `refchain::logaddexp/logsumexp` skip the max's `exp`; `Semiring::reduce` doesn't (feeds SIMD). Intentional.

## Test and bench
```
cargo nextest run -p escapepod-demux --features crf-decode -E 'test(/crf|lattice|refchain|barcode|encoder/)'
cargo bench -p escapepod-demux --bench crf_decode
ESCAPEPOD_CRF_BUNDLE=<dir> cargo bench -p escapepod-demux --features crf-decode --bench crf_encoder
ESCAPEPOD_CRF_BUNDLE=<dir> cargo nextest run -p escapepod-demux --features crf-decode --test crf_encoder_native_parity
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 pixi run -e dev-gpu cargo nextest run -p escapepod-demux --features gpu --test gpu_crf_lattice
```
Encoder changes are held to `benchmarks/benchmark_demux_crf.sh` + `evaluate_demux_crf.py` over `data/bench/dual_axis_20k/`: 0/20,000 calls differ native vs tract.

## Env levers
| var | effect | default | tested / documented |
|---|---|---|---|
| `ESCAPEPOD_CRF_TRACT` | `1/true/yes/on` (via `escapepod_pod5::env::flag`): force tract | native if recognised | parity test / CHANGELOG |
| `ESCAPEPOD_CRF_DEBUG_RECOGNIZER` | echo refusal to stderr | off | no / no — deleting, `-v` shows it |
| `ESCAPEPOD_CRF_GPU_DECODE` | `0`: CPU decode | GPU if kernels load | no / CHANGELOG |
| `ESCAPEPOD_CRF_GPU_ZEROCOPY` | `0`: copy scores to host | on with GPU decode | no / CHANGELOG |
| `ESCAPEPOD_CRF_GPU_BATCH_ROWS` | rows per ort call, per worker | 1024/workers, ≥64 | unit / CHANGELOG |
| `ESCAPEPOD_CRF_GPU_WORKERS` | workers per device (read in CLI `run.rs`) | 2 | no / CHANGELOG |

## Known debt
- `encoder_native.rs:1361–1452, 887–973` — proto toolkit + LSTM-node checks duplicate `classify/fnn_lstm.rs:140–163, 172–236, 686–751`; moving to `onnx_graph` beside `onnx_rewrite`.
- `encoder_native.rs:375–553` — `(backend, n)` match duplicates `fnn_lstm::logits_batch`; moving to `escapepod_signal::lstm::run_batch`.
- `encoder_native.rs:197–298` — `encode_into` is `encode_batch_into` at n=1; bit-identity already pinned.
- `encoder_native.rs:1190–1201` + `encoder.rs:791–800` — `self_check` builds a tract plan `load` rebuilds.
- `encoder_gpu.rs` — halve-on-OOM ×4 (521/782/856 + `lattice_gpu.rs:247`), row flatten ×2 (551/672), dims check ×2 (577/742), `basecall_batch*` loop ×2.
- No production caller: `CrfEncoder::basecall`, `CrfMetadata::prep`/`refusal`, non-bundle `load*`, `CrfLatticeGpu::decode_batch`/`ScoreOrder::BatchMajor`, `CrfEncoderGpu::encode_batch`/`split_time_major` (probe + example only).
- Env flags spelled four ways (`var_os`, `== "0"`, `!= "0"`, parse).

## Do not
- Collapse `avx2.rs`/`avx512.rs` into a width macro — intrinsics differ in arity; diffability is the maintenance story.
- Swap crf's `exp8/exp16` for `escapepod_signal::lstm`'s — different rounding/coefficients change decode bits; deferred to a 20k parity run.
- Or-pattern the lone-read dispatch arms or drop `#[inline(never)]` — recorded miscompile.
- Batch the tract encoder — no efficient batched LSTM; grouping pays only natively.

# escapepod-demux

## Purpose and layering
- Two independent barcode paths — fingerprint-and-classify (DTW/SVM, GBM) and basecall-then-match (CTC-CRF, `crf/`) — sharing boundary-CNN adapter detection.
- Depends on `escapepod-signal`. Consumed by `escapepod-cli`, and by `escapepod-classify` for `GbmModel, load_gbm_model, GbmPredictor`, `onnx_rewrite::{hoist_conv_padding, expand_instance_norm}`, `cuda::device_visible`, `crf::encoder_native` — why classify depends on this crate.

## Module map
- `model.rs` — `DtwSvmModel`, `WarpDemuxModel`, `AnyModel`/`load_any_model` (sniff order: `model_type:"gbm"`, then `label_mapper` vs `label_map`).
- `classify.rs` — legacy WarpDemuX nearest-neighbour arm (`classify_read`); live in `demux classify`, kept by product decision.
- `svm/` — `predictor` (OvO, Platt, coupling), `kernel`, `workspace` (per-worker scratch), `gpu` (on-device DTW+RBF+OvO).
- `gbm.rs` — HistGradientBoosting walk over a compact arena; `predict_many` = 8-lane lockstep.
- `probability.rs` — softmax/margin/threshold head shared by SVM and GBM.
- `fingerprint.rs` — t-test segmentation → fingerprint; `MAX_FINGERPRINT_WINDOW` clamp.
- `train.rs` (`train`) — labels-only stub; predictor ignores its `dual_coef`.
- `adapter_cnn.rs` (`cnn-detect`) / `adapter_cnn_gpu.rs` (`gpu`) — prep/decode shared; only the runtime differs.
- `onnx_rewrite.rs` — proto rewrites before tract lowers a graph.
- `ort_ep.rs` (`gpu`) — the one spelling of ort's CUDA EP; `require_cuda`.
- `cuda.rs` (`gpu`) — cheap device-visible probe for `--device auto`.
- `crf/` — see `crates/escapepod-demux/src/crf/CLAUDE.md`.
- Model download/manifest is **not here**: `escapepod-cli/src/commands/demux/models.rs`.

## Feature flags
| feature | gates |
|---|---|
| `cnn-detect` | tract boundary CNN, `onnx_rewrite` |
| `crf-decode` | tract CRF encoder + `fqxv-align` (lattice decode always compiled) |
| `gpu` | **atomic**; implies both above; adds cudarc DTW/SVM, ort CUDA CNN+CRF, `cuda`, `ort_ep` |
| `train` | `train.rs`; no extra crates |
- `cnn-gpu`/`crf-gpu` no longer exist — `gpu` is the only GPU feature.

## Build, test, bench this crate alone
Wrap in `srun -p rna -c 32 --mem=32G pixi run …` (root "Build baseline and SLURM builds"); GPU per root "Build Commands".
- `cargo nextest run -p escapepod-demux` — `pipeline`, `gbm_parity`, `crf_golden` (no ONNX, no `ext/`).
- `--features cnn-detect` — `adapter_cnn_batch_parity`; skips unless `ESCAPEPOD_TEST_ADAPTER_ONNX`.
- `--features crf-decode` — `crf_encoder_native_parity`; skips unless `ESCAPEPOD_CRF_BUNDLE`.
- `--features gpu` on a gpu node — `gpu_svm_batch`, `gpu_fingerprint`, `gpu_crf_lattice`.
- `cargo test --doc -p escapepod-demux` separately.
- Benches: `classify`, `crf_decode` (none); `adapter_prep` (`cnn-detect`); `crf_encoder` (`crf-decode` + `ESCAPEPOD_CRF_BUNDLE`).

## Invariants and traps
- Prep pads to `input_len`=1500 with `SCORE_EXCL`=−5.0, stats from the unpadded window → else out-of-distribution and one cuDNN plan per shape; `prep_reports_the_pre_padding_length`.
- `group_by_len`/`pack_batch` always one group — keep; a broken invariant mis-packs silently.
- `valid_len` clamp in `decode_adapter_end` is the #187 fix (272 boundaries past read end) — load-bearing, not inert; `decode_never_returns_a_boundary_past_the_read`.
- Decode argmaxes 550 of 1500 outputs (`max_obs_adapter`=6500): 0.41% of true ends unreachable, by design.
- Zero-input load probe rejects `[_,≠2,_]`; without it every read gets `adapter_end=0`.
- `expand_instance_norm` off on CPU batch-1 (tract's lowering is the spec there); mandatory above batch 1.
- `require_cuda_ep()` only under `--device gpu`; otherwise ort silently commits to CPU — same symptom as missing libcudnn.
- `GbmPredictor::predict` is live: `<8` tail, CLI per-read fallback, classify's batch-1 scorer.
- `demux run` refuses `AnyModel::WarpDemux`; `demux classify` serves it.

## Env levers
| var | effect | default | tested/doc |
|---|---|---|---|
| `ESCAPEPOD_CNN_GPU_BATCH_ELEMS` | rows×len per ort call (else VRAM/5500, clamped 2M–64M) | derived | no / in-code |
| `ESCAPEPOD_TEST_ADAPTER_ONNX` | enables `adapter_cnn_batch_parity` | unset → skip | test-only |
| `ESCAPEPOD_CNN_MARGIN` | **retired**; historical mention only, `adapter_cnn.rs:411` | — | — |
CRF levers: `crf/CLAUDE.md`.

## Known debt
- tract load chain ×8: `adapter_cnn.rs:207`, `crf/encoder.rs:760`, `encoder_native.rs:1192,1862`, classify `fnn.rs:107`, `fnn_lstm.rs:1001`, `waveform_net.rs:213`, `waveform_net_gpu.rs:242`.
- Shape probes re-implement the forward pass: `adapter_cnn.rs:237`/`:279`; `adapter_cnn_gpu.rs:270`/`:298`.
- `sv_class` ×2 (`svm/predictor.rs:94`, `svm/gpu.rs:298`); kernel-weighted pair loop ×2 (`predictor.rs:219`, `:313`); `ovo_votes` dead (`:345`).
- `model.rs` validator copies (`186`/`493`, `268`/`521`); `svm/gpu.rs` gates per item, not `mod gpu`.

## Do not
- Change `input_len`/pad value without a retrained model — training convention.
- Replace `Concat(zeros)` with ONNX `Pad` — tract fuses it back.
- Add `.with_tf32(false)` — 1.35× slower on Ampere.
- Revert `ArenaExtendStrategy::SameAsRequested` — long streams wedge the device.
- Thread a `strict` flag through loaders — `require_cuda` is process-wide on purpose.
- Split `gpu` into finer features — builds nobody produced.
- "Fix" `group_by_len` to one batch — it is a guard.

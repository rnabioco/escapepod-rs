# Experimental

Tools in this section live outside the default build. They work, but their
APIs, flags, and output formats are not stable yet, and you opt in per build
with Cargo features.

## Commands

| Command | Feature flag | Purpose |
|---------|-------------|---------|
| [repack](repack.md) | `--features experimental` | Re-pack POD5 files with current compression settings |
| [resquiggle](resquiggle.md) | `--features experimental` | Refine signal-to-base mapping using banded DP |
| `index` | `--features experimental` | Build `.p5i` sidecar indexes for O(1) read-ID lookup |

The `index` command is intentionally undocumented in depth — it builds a read-ID
sidecar but the speedup vs. a direct scan is marginal for typical file sizes,
and the format is subject to change.

## Building

Enable one or more features at build time:

```bash
# Repack, resquiggle, and index
cargo build --release --features experimental

# Everything
cargo build --release --features experimental
```

Demux has additional sub-features layered on top:

| Feature | Enables |
|---------|---------|
| `cnn-detect` *(no flag needed — in the default build)* | CPU CNN/TCN adapter detection through `tract-onnx` (`escpod demux detect --method cnn`); bring-your-own ONNX model, no weights bundled |
| `--features train` | SVM model training via `linfa-svm` (`escpod demux train-svm`) |
| `--features gpu` | Batched GPU DTW for classify / train-svm (CUDA driver + libnvrtc required at runtime) |
| `--features cnn-gpu` | Implies `cnn-detect`; onnxruntime CUDA inference for `detect --method cnn --gpu` |
| `--features crf-gpu` | onnxruntime CUDA inference for the CTC-CRF basecall encoder (`basecall --gpu`) |

The `--features` rows each imply `demux`, so `cargo build --features train` is
enough. `cnn-detect` is listed for reference only: it ships in the default build
and needs no flag, but `cnn-gpu` builds on it and `--method cnn` is what the
published barcode models expect.

The GPU features need runtime CUDA libraries; the repository's pixi `gpu`
environment provides all of them — see [GPU setup](gpu-setup.md).

## Stability

Treat anything in this section as pre-1.0 — output formats, JSON schemas,
command names, and flag spellings may change between releases without a
deprecation window. If you script against an experimental command, pin to
a specific `escapepod-rs` version.

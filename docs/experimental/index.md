# Experimental

Tools in this section live outside the default build. They work, but their
APIs, flags, and output formats are not stable yet, and you opt in per build
with Cargo features.

## Commands

| Command | Feature flag | Purpose |
|---------|-------------|---------|
| [repack](repack.md) | `--features experimental` | Re-pack POD5 files with current compression settings |
| [resquiggle](resquiggle.md) | `--features experimental` | Refine signal-to-base mapping using banded DP |
| [demux](demux.md) | `--features demux` | WarpDemuX-compatible barcode demultiplexing (DTW + SVM) |
| `index` | `--features experimental` | Build `.p5i` sidecar indexes for O(1) read-ID lookup |

The `index` command is intentionally undocumented in depth — it builds a read-ID
sidecar but the speedup vs. a direct scan is marginal for typical file sizes,
and the format is subject to change.

## Building

Enable one or more features at build time:

```bash
# Repack, resquiggle, and index
cargo build --release --features experimental

# Barcode demultiplexing
cargo build --release --features demux

# Everything
cargo build --release --features experimental,demux
```

Demux has additional sub-features layered on top:

| Feature | Enables |
|---------|---------|
| `--features train` | SVM model training via `linfa-svm` (`escpod demux train-svm`) |
| `--features gpu` | Batched GPU DTW for classify / train-svm (CUDA driver + libnvrtc required at runtime) |
| `--features cnn-detect` | ADAPTed-style CNN adapter detection (bring-your-own ONNX model; weights are CC BY-NC 4.0 and not bundled) |

Each implies `demux`, so `cargo build --features train` is enough.

## Stability

Treat anything in this section as pre-1.0 — output formats, JSON schemas,
command names, and flag spellings may change between releases without a
deprecation window. If you script against an experimental command, pin to
a specific `escapepod-rs` version.

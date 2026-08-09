# Experimental

Tools in this section live outside the default build. They work, but their
APIs, flags, and output formats are not stable yet, and you opt in per build
with Cargo features.

## Commands

| Command | Feature flag | Purpose |
|---------|-------------|---------|
| [repack](repack.md) | `--features experimental` | Re-pack POD5 files with current compression settings |
| [resquiggle](resquiggle.md) | `--features experimental` | Refine signal-to-base mapping using banded DP |
| `index` | `--features experimental` | Build the `.p5s` sidecar read index for O(1) read-ID lookup |
| `annotate` | `--features experimental` | Record per-read annotations (demux barcodes) in the `.p5s` sidecar |

## The `.p5s` sidecar

`reads.pod5` gets one companion file, `reads.pod5.p5s`, holding any
combination of a read index (`escpod index`) and named per-read annotations
(`escpod annotate`). The POD5 itself is **never modified** — raw sequencer
output stays byte-identical and checksummable; deleting an annotation is
editing or deleting the sidecar.

The sidecar is a plain Arrow IPC (Feather v2) table — one row per read,
`read_id | batch_idx | row_idx` plus one dictionary-encoded column per
annotation — so it is directly readable without escapepod:

```python
import pyarrow.ipc as ipc
table = ipc.open_file("reads.pod5.p5s").read_all()
```

The sidecar is bound to its POD5 by file-identifier UUID and byte size
(stored in the Arrow schema metadata, checked before any data is decoded);
a stale sidecar or one copied next to the wrong file fails loudly. Writes
are atomic and section-preserving: `index` keeps annotations, `annotate`
keeps the index and other annotations.

Typical demux flow, with no intermediate per-barcode POD5s kept around:

```bash
escpod demux detect … | escpod demux basecall --barcodes … -o demux.csv
escpod annotate -a demux.csv reads.pod5      # record assignments in reads.pod5.p5s
escpod demux split reads.pod5 --sidecar -d out/   # materialize per-barcode files on demand
```

The sidecar can also carry the **experimental design** — a samplesheet
mapping barcode labels (or combinations of annotations, e.g. `ldx,edx`) to
experimental variables:

```bash
cat samplesheet.csv
# barcode,condition,replicate
# nbc01,fresh_edx01,r1
# …
escpod annotate --design samplesheet.csv reads.pod5
escpod demux split reads.pod5 --sidecar --annotation condition -d by_condition/
```

The design table is stored as JSON in the sidecar's schema metadata
(`escapepod:design`), and each of its variables is materialized as a derived
per-read column by joining across the key annotations — so `split`, pyarrow
filtering, and `reader.annotation("condition")` all work with no join logic.
Key columns are auto-detected (CSV columns that name an existing annotation;
override with `--keys`). Rewriting a key annotation (say, re-demuxing
`barcode`) automatically re-derives the dependent columns, and writing a
derived column directly is refused — update the design instead.

This format replaces the earlier `.p5i` index sidecar; delete any `.p5i`
files and rerun `escpod index`.

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

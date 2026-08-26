# Experimental

Tools in this section are functional but not yet stable — APIs, flags, and
output formats may change between releases. Most live behind the
`experimental` Cargo feature; `demux` is the exception, shipping in the
default build with output formats that are still stabilizing.

## Commands

| Command | Feature flag | Purpose |
|---------|-------------|---------|
| [demux](demux.md) | *(default build)* | Barcode demultiplexing — fused pipeline, stepwise subcommands, sidecar output |
| [annotate](annotate.md) | `--features experimental` | Record per-read annotations (demux barcodes, designs) in the `.p5s` sidecar |
| `index` | `--features experimental` | Build the `.p5s` sidecar read index for O(log n) read-ID lookup |
| [resquiggle](resquiggle.md) | `--features experimental` | Refine signal-to-base mapping using banded DP |
| [repack](repack.md) | `--features experimental` | Re-pack POD5 files with current compression settings |

## The `.p5s` sidecar

`reads.pod5` gets one companion file, `reads.pod5.p5s`, holding any
combination of a read index (`escpod index`) and named per-read annotations
(`escpod annotate`). The POD5 itself is **never modified** — raw sequencer
output stays byte-identical and checksummable; deleting an annotation is
editing or deleting the sidecar.

The sidecar is a plain Arrow table, directly readable without escapepod:

```python
import pyarrow.ipc as ipc
table = ipc.open_file("reads.pod5.p5s").read_all()
```

It is bound to its POD5 by file identifier and size, so a stale sidecar or
one copied next to the wrong file fails loudly — and says what it *was* built
from, which is the thing you want to know at that moment. Reading it directly
with pyarrow, as above, skips that check: a direct reader owns it. `index`
preserves annotations, and `annotate` preserves the index and other
annotations — each command touches only its own columns. Format details:
[The `.p5s` Sidecar](../format/sidecar.md).

Typical demux flow, with no intermediate per-barcode POD5s (or even a CSV)
kept around:

```bash
# One step: demux straight into the sidecar (no split files, no CSV)
escpod demux reads.pod5 --model <bundle> --annotate

# Then materialize exactly what a downstream tool needs, when it needs it:
escpod demux split reads.pod5 --sidecar -d out/          # all barcodes
escpod filter reads.pod5 --annotation barcode=nbc05 -o nbc05.pod5   # one group
```

`--annotate` combines with `-d` (write split files AND the sidecar) and with
`--classifications` (also keep the CSV). The stepwise route still works:
`escpod annotate -a demux.csv reads.pod5` records an existing classifications
CSV into the sidecar.

Working with annotations:

```bash
escpod inspect summary reads.pod5        # shows the sidecar: index, annotations, design
escpod annotate --list reads.pod5        # per-annotation labels + read counts
escpod annotate --remove sample reads.pod5
escpod annotate --remove-design reads.pod5   # drops the design + derived columns
escpod view reads.pod5 --include read_id,barcode,condition   # join columns into TSV
escpod filter reads.pod5 --annotation condition=fresh --annotation replicate=r1 -o out.pod5
```

`filter --annotation` is repeatable: pairs with the same name are any-of,
different names are all-of, and `--ids` intersects on top.

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

Each design variable becomes an ordinary per-read column (`condition`,
`replicate`), so `split`, `filter`, pyarrow, and
`reader.annotation("condition")` all work with no join logic. Rewriting a
key annotation (say, re-demuxing `barcode`) automatically re-derives the
dependent columns, and writing a derived column directly is refused — the
design stays the source of truth. See [annotate](annotate.md) for the full
command reference.

Note the split in feature gates: *writing* sidecars with `escpod index` /
`escpod annotate` needs `--features experimental`, while everything that
*consumes* them — `demux --annotate`, `demux split --sidecar`,
`filter --annotation`, `view`, `inspect summary` — works in the default
build.

This format replaces the earlier `.p5i` index sidecar; delete any `.p5i`
files and rerun `escpod index`.

## Building

Enable one or more features at build time:

```bash
# repack, resquiggle, index, annotate
cargo build --release --features experimental
```

Demux has two opt-in features layered on top:

| Feature | Enables |
|---------|---------|
| `--features gpu` | Every GPU path: CNN adapter detection, the CTC-CRF encoder, and DTW classify. Selected at run time with `--device` (default `auto`); CUDA libraries at run time only — see [GPU acceleration](demux.md#gpu-acceleration) |
| `--features train` | SVM model training via `linfa-svm` (`escpod demux train-svm`) |

Both imply `demux`, so `cargo build --features gpu` is enough. CPU CNN/TCN
adapter detection (`--method cnn`, what the published barcode models expect)
needs no flag — it ships in the default build.

## Stability

Treat anything in this section as pre-1.0 — output formats, JSON schemas,
command names, and flag spellings may change between releases without a
deprecation window. If you script against an experimental command, pin to
a specific `escapepod-rs` version.

# annotate

Record per-read annotations — demux barcode assignments, experimental
conditions — in the [`.p5s` sidecar](../format/sidecar.md) next to POD5
files. The POD5 itself is never modified.

Requires `--features experimental`.

## Usage

Exactly one action per invocation:

```bash
escpod annotate -a <CSV> <FILES>...            # write an annotation
escpod annotate --design <CSV> <FILES>...      # record an experimental design
escpod annotate --list <FILES>...              # show sidecar contents
escpod annotate --remove <NAME> <FILES>...     # remove one annotation
escpod annotate --remove-design <FILES>...     # remove the design + derived columns
```

`<FILES>` are POD5 files and/or directories (searched recursively).

## Options

| Option | Description |
|--------|-------------|
| `-a, --assignments <CSV>` | CSV mapping reads to labels: a `read_id` column plus `barcode` (or `predicted_barcode`) — the output of `demux classify` / `demux basecall --barcodes` |
| `--design <CSV>` | Experimental-design CSV: one column per key annotation (e.g. `barcode`, or `ldx,edx` for combinations) plus one column per variable (e.g. `condition`) |
| `--keys <COLS>` | With `--design`: which CSV columns are the keys (default: every column naming an existing annotation) |
| `--name <NAME>` | Annotation name to write with `-a` (default: `barcode`) |
| `--list` | Print each file's sidecar contents: index size, annotations with label/read counts, design |
| `--remove <NAME>` | Remove an annotation (design key/value columns are refused — update or remove the design instead) |
| `--remove-design` | Remove the design and its derived columns |
| `--force` | Replace a sidecar bound to a *different* POD5 instead of erroring |
| `-t, --threads <N>` | Threads for parallel processing across files |

## Annotations

```bash
escpod demux … --classifications demux.csv     # or any read_id,barcode CSV
escpod annotate -a demux.csv reads.pod5
```

Assignments are intersected with the reads actually present in each file;
entries for other reads are ignored, so one CSV can annotate a whole
directory of per-flowcell files. Unassigned reads are simply absent from the
annotation (no sentinel label). Writing the same `--name` again replaces
that annotation; other annotations and the read index are preserved.

If you are running the fused demux pipeline anyway, `escpod demux
--annotate` writes the sidecar directly and skips the CSV round-trip — see
[demux](../cli/demux.md#sidecar-output).

## Experimental designs

A design maps annotation labels (or combinations) to experimental
variables:

```bash
cat samplesheet.csv
# barcode,condition,replicate
# nbc01,fresh_edx01,r1
# nbc02,fresh_edx02,r1
escpod annotate --design samplesheet.csv reads.pod5
```

Each variable becomes a derived per-read column (here `condition` and
`replicate`), assigned to every read whose key-annotation labels match a
design row. Multi-key designs (`ldx,edx,condition`) match on the
combination. Key columns are auto-detected as the CSV columns that name an
existing annotation; use `--keys` when that guess would be wrong.

The design stays the source of truth: re-annotating a key column (say,
re-demuxing `barcode`) automatically re-derives the dependent columns, and
`-a --name condition` on a derived column is refused.

## Consuming annotations

Everything downstream reads the sidecar in the default build — no
`experimental` feature needed:

```bash
escpod inspect summary reads.pod5                     # sidecar section
escpod demux split reads.pod5 --sidecar -d out/       # split by annotation
escpod filter reads.pod5 --annotation condition=fresh_edx01 -o subset.pod5
escpod view reads.pod5 --include read_id,barcode,condition
```

Python: `Reader.annotation()`, `Reader.annotation_names()`,
`Reader.design()` — see [Reading Files](../python/reading.md#sidecar-annotations).

## Notes

- "Stripping" an annotation is editing or deleting the sidecar; the POD5 is
  untouched throughout.
- A sidecar is bound to its POD5 by file identifier and size; a stale or
  misplaced one fails loudly. `--force` replaces it — but only it. A sidecar
  that belongs to this POD5 and merely could not be read (truncated, or written
  by a newer escpod) is refused with or without `--force`, because its barcode
  and score columns exist nowhere else and a rebuild would replace them with an
  empty column set.
- Format details: [The `.p5s` Sidecar](../format/sidecar.md).

# annotate

Record per-read annotations — demux barcode assignments, experimental
conditions — in the [`.p5s` sidecar](../format/sidecar.md) next to POD5
files. The POD5 itself is never modified.

## Label reads from another tool's table

A read-to-label table from another demultiplexer (WarpDemuX's
`barcode_predictions_0.csv.gz`, say) can live in the run's sidecar, so
`filter --annotation` and `demux split --sidecar` work on a run demultiplexed
elsewhere — without cutting a per-sample POD5 and duplicating the signal:

```bash
escpod annotate run1/pod5/ \
    --from-table run1/predictions/barcode_predictions_0.csv.gz \
    --column barcode=predicted_barcode

escpod filter run1/pod5/reads.pod5 --annotation barcode=7 -o sample7.pod5
escpod demux split run1/pod5/reads.pod5 --sidecar -d by_barcode/
```

`--column NAME=COL` writes sidecar column `NAME` from table column `COL`;
repeat it for a dual-index run (`--column ldx=ldx_call --column fdx=fdx_call`).
The table is a CSV or TSV with one header row (a tab in the header means TSV;
gzip is detected from the file's bytes, whatever it is called, including
several gzip members in one file). The read-id column is `read_id`, or name it
with `--id-column`; a leading `#` in the header, as WarpDemuX writes
(`#read_id,…`), is ignored. A table that comes in shards is given as
`--from-table a.csv b.csv` or by repeating the flag.

- **Labels are verbatim.** Whatever the cell says, trimmed, is the label.
  WarpDemuX writes bare integers and uses `-1` for a read it could not call:
  both land in the sidecar as the strings `7` and `-1`, not as escpod's
  `BC07` or `unclassified`. So `filter --annotation barcode=BC04` selects
  nothing on such a sidecar (it warns that the label does not occur) —
  filter on `barcode=4`. Nothing is renamed or interpreted.
- **Reads the table does not label are `unclassified`** — reads absent from the
  table and reads with an empty cell alike, the same label `demux --annotate`
  writes for a read no model could place. `--unassigned-label LABEL` picks
  another word; `--unassigned-label ''` leaves those reads out of the column
  altogether (what `-a` does).
- **A table for another run is refused, and nothing is written.** If no read in
  any input is in the table — the wrong run, or the wrong `--id-column` — the
  command stops before touching a sidecar. Without that, an in-place replace
  would overwrite a good column with all-`unclassified`. `-a` has the same
  guard.
- **Bad tables fail loudly.** A row whose id is not a UUID is an error naming
  its line; so is a read that appears twice with different labels; so is a
  repeated header row (two tables concatenated — give each as its own
  `--from-table`); so is a label column with more than 65,535 distinct values
  (`--column barcode=confidence_score` is a mistake, and fails in seconds
  rather than at write time).

The sidecar is written exactly as `demux --annotate` writes it: a directory
argument gets one [collection sidecar](#a-directory-gets-one-sidecar), the
sidecar is bound to its POD5 by the footer's file identifier and size, and
writing a column that already exists replaces it in place while the rest of
the sidecar survives.

The sidecar lands beside the path *as given*. To keep a run's canonical
directory free of `.p5s` files, point `annotate` at a directory of symlinks to
its POD5s. Put `FILES` before `--from-table`, or repeat the flag per table:
`--from-table` takes every following argument as a table.

## Usage

Exactly one action per invocation:

```bash
escpod annotate --from-table <TABLE>... --column NAME=COL <FILES>...  # label reads from a table
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
| `--from-table <TABLE>...` | Read-to-label table (CSV or TSV, gzip detected); repeat or list several for shards. Needs `--column` |
| `--column <NAME=COL>` | With `--from-table`: write sidecar column `NAME` from table column `COL`, verbatim. Repeatable; at least one |
| `--id-column <COL>` | With `--from-table`: the read-id column (default `read_id`; a leading `#` in the header is ignored) |
| `--unassigned-label <LABEL>` | With `--from-table`: label for reads the table does not label (default `unclassified`); `''` leaves them unlabelled |
| `-a, --assignments <CSV>` | CSV mapping reads to labels: a `read_id` column plus `barcode` (or `predicted_barcode`) — the output of `demux classify` / `demux basecall --barcodes` |
| `--design <CSV>` | Experimental-design CSV: one column per key annotation (e.g. `barcode`, or `ldx,edx` for combinations) plus one column per variable (e.g. `condition`) |
| `--keys <COLS>` | With `--design`: which CSV columns are the keys (default: every column naming an existing annotation) |
| `--name <NAME>` | Annotation name to write with `-a` (default: `barcode`); `--from-table` names its columns with `--column` |
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
directory of per-flowcell files. With `-a`, unassigned reads are simply absent
from the annotation (no sentinel label) — unlike `--from-table`, which labels
them `unclassified` unless told otherwise. Writing the same `--name` again replaces
that annotation; other annotations and the read index are preserved.

### A directory gets one sidecar

Naming a **directory** writes a single
[collection sidecar](../format/sidecar.md#one-sidecar-for-a-directory) beside
it, covering every POD5 beneath — and no per-file sidecars:

```bash
escpod annotate -a demux.csv run1/pod5/        # writes run1/pod5.p5s, and nothing else
```

(`--from-table` does the same for a directory.)

The run's labels are one result, so they live in one file rather than being
copied into fifty. Reading is unaffected: `view --include`, `filter
--annotation`, `demux split --sidecar` and the Python API all resolve a POD5's
columns from the collection beside its directory when the file has no sidecar
of its own.

Naming files individually keeps the per-file behaviour — there is no directory
for a collection to sit beside:

```bash
escpod annotate -a demux.csv run1/pod5/*.pod5  # writes one .p5s per file
```

If you are running the fused demux pipeline anyway, `escpod demux
--annotate` writes the sidecar directly and skips the CSV round-trip — see
[demux](demux.md#sidecar-output).

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

Everything downstream reads the sidecar:

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

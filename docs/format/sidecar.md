# The `.p5s` Sidecar

The `.p5s` sidecar is escapepod's companion file for per-read metadata:
`reads.pod5` gets one `reads.pod5.p5s` holding a read index, named
annotations (e.g. demux barcode assignments), and optionally an
experimental-design table. The POD5 file itself is never modified.

This page is the on-disk specification. For how to *use* sidecars, see the
[experimental overview](../experimental/index.md#the-p5s-sidecar).

## Layout

A sidecar is a plain Arrow IPC file (Feather v2) containing a single table
with one row per read in the POD5, sorted by read UUID:

| Column | Type | Meaning |
|--------|------|---------|
| `read_id` | `FixedSizeBinary(16)` | Read UUID (tagged `minknow.uuid`, matching POD5's own reads table) |
| `batch_idx` | `UInt32` | Reads-table batch containing this read |
| `row_idx` | `UInt32` | Row within that batch |
| *(one per annotation)* | `Dictionary<Int32, Utf8>`, nullable | Label for this read; null = unassigned |
| *(one per score)* | `Float32`, nullable | Numeric value for this read; null = absent |

`batch_idx`/`row_idx` are the **read index**: they locate a read in the POD5
reads table, enabling O(log n) binary-search lookup instead of a full-table
scan. Every additional column carries per-read data, and its **Arrow type** is
what says which kind it is: a dictionary column is an *annotation* (labels, so
a 50k-read barcode annotation costs well under 1 MB) and a `Float32` column is
a *score* (e.g. a demux run's confidence). Record batches are compressed with
zstd (Arrow IPC buffer compression).

A locator is trusted only as far as the read it names: every indexed lookup
confirms that the row it landed on actually holds the requested `read_id`
before using it, and reports an error rather than returning another read's
data. See [Identity binding](#identity-binding) for why that check is separate
from the file-level one.

Because it is ordinary Arrow, any Arrow reader can consume a sidecar with no
escapepod code:

```python
import pyarrow.ipc as ipc
table = ipc.open_file("reads.pod5.p5s").read_all()
```

Reading it this way bypasses the [identity binding](#identity-binding) below —
a direct reader gets whatever the file says, including for a POD5 it does not
belong to, and owns that check itself. Compare
`table.schema.metadata[b"escapepod:file_identifier"]` against the POD5's footer
UUID if the pairing is not already guaranteed by how the file was produced.

## Identity binding

The Arrow schema metadata binds the sidecar to exactly one POD5:

| Key | Value |
|-----|-------|
| `escapepod:p5s_version` | Format version: `1`, or `2` once a `Float32` score column is present |
| `escapepod:file_identifier` | The POD5 footer's `file_identifier` UUID |
| `escapepod:pod5_size` | The POD5's byte size at write time |

Identity is checked before *anything* in the sidecar is used, including the
[signal batch geometry](#signal-batch-geometry).

Readers validate both identity keys from the IPC footer *before decoding any
record batch*. A sidecar copied next to the wrong file, or left behind after
its POD5 was replaced, therefore fails loudly ("does not match this POD5
file") rather than silently describing the wrong reads. There is no partial
acceptance: identity either matches exactly or the sidecar is rejected.

The version bump is gated on *content*, not on every write: a sidecar with only
label columns is still written as `1`, so an escpod that predates score columns
keeps reading the sidecars it can read perfectly well.

### Why a UUID and a size, and not a checksum

Identity is the POD5's `file_identifier` plus its byte length — there is no
content hash, and the sidecar does not record the POD5's path. Every POD5
written by escapepod (or by MinKNOW) mints a fresh v4 `file_identifier`, so the
UUID is a 122-bit per-file token that already answers "is this the same file?"
better than a checksum would, and `merge`/`filter`/`subset`/`repack` outputs
cannot inherit a parent's identity. The size catches truncation or a file still
being appended to.

What this deliberately does not cover is a byte edit that preserves *both* the
UUID and the exact length — which means patching a POD5 in place, against the
premise that raw sequencer output is immutable. Hashing a multi-gigabyte POD5
on every open would cost more than the reads-table scan the sidecar exists to
avoid.

A path is not recorded because location *is* the link (`reads.pod5` →
`reads.pod5.p5s`): a stored path would go stale on any legitimate move while
adding nothing the UUID does not already give.

## Provenance

Three further schema-metadata keys describe where a sidecar came from. They are
**never compared against anything** — matching them would break every legal
rename — and every one is optional, so a sidecar written before they existed
still loads:

| Key | Value |
|-----|-------|
| `escapepod:source_name` | The POD5's base name when the sidecar was written |
| `escapepod:read_count` | Reads covered by the index |
| `escapepod:writer` | What wrote it, e.g. `escapepod-pod5 0.16.1` |

They exist for the moment identity *fails*: without them the error knows only
that two UUIDs differ, which is exactly when you want a filename. With them it
reads `… does not match this POD5 file (stale or copied from another) (from
"old_run.pod5", 50000 reads, written by escapepod-pod5 0.16.1)`. `escpod
inspect summary` shows the same line for a sidecar that loads.

There is no write timestamp: the sidecar file's own mtime already records it.

## Signal batch geometry

One further optional key caches a property of the POD5 rather than of the
reads:

| Key | Value |
|-----|-------|
| `escapepod:signal_batch_rows` | Per-batch row counts of the POD5's **signal** table, run-length encoded |

The encoding is `<runs>x<rows>` terms separated by commas. A conformant POD5
has one run and a short tail, so 8866 batches encode as `8865x100,1x37` —
about fifteen bytes however many batches the file has. A non-uniform file is
recorded as it is, one term per run, rather than rejected.

### Why this is worth caching

Row count is the one field an Arrow IPC footer does not carry: a footer `Block`
records a batch's offset, metadata length and body length and nothing else. So
recovering the counts means reading every batch's own message header — one
scattered touch per batch, which on a network filesystem measured 15–24 ms
*each* when cold. It is also paid on every process start, not once: a 33 GB
file with 8866 signal batches spent 4.95 s there, 78% of a cold 5000-read
scattered fetch. With the geometry cached that becomes 4.82 ms.

**Every** sidecar write records it when it is not already there — `escpod index`,
`escpod annotate`, `demux --annotate` alike — so a sidecar cannot end up
carrying a read index and nothing else. A write that finds a value already
present carries it through untouched rather than re-measuring: identity binding
has already proved the POD5 is the same file, so a recorded geometry cannot have
gone stale under it.

`escpod index` is the exception, and deliberately: it re-measures even when a
value is present, because rebuilding the cache is the job it exists to do. It
also refuses to skip a sidecar that has an index but no geometry — otherwise the
obvious remedy for a slow file would report "already indexed" and do nothing.

An empty measurement means "could not measure", never "no batches", and leaves
any recorded value alone. Nothing is lost by declining to record: the POD5 still
holds the answer.

### Checked, not believed

The obvious cheaper trick is to read batch 0 and assume every other batch
matches. That is what the official `pod5` library and dorado do, and it is
exactly what escapepod's `nonuniform_signal_batch` diagnostic exists to catch
them out on. Recording measured counts costs a handful of bytes and is exact
for a non-uniform file too, so escapepod does not have to make that bet.

A reader that finds the key still verifies it before using it:

1. it must have exactly one entry per batch the footer describes; and
2. the **first and last** batches are read for real and compared. The last is
   the short one, so a geometry belonging to a file with a different read count
   disagrees there; the first pins the stride.

Any mismatch logs a warning and falls back to reading every header, so a stale
or corrupt cache costs time and never correctness. A sidecar bound to a
different POD5 is rejected by the [identity binding](#identity-binding) before
the geometry is consulted at all.

The two failure modes are deliberately different. The geometry degrades quietly
to the slow path because the answer is always recoverable from the POD5 itself;
the **read index** in the same file fails loudly on a stale sidecar, because
its answer is not recoverable and stepping over it would turn a stale index
into a slow wrong answer rather than a fast error.

Like the provenance keys, this one is optional on read, so adding it is not a
version bump: a sidecar written before it existed still loads, and an older
escpod ignores it.

## The experimental design

An optional design table maps combinations of annotation labels to
experimental variables (a samplesheet: `barcode → condition`, or multi-key
`ldx,edx → condition,replicate`). It is stored as JSON under the
`escapepod:design` schema-metadata key:

```json
{
  "key_columns": ["barcode"],
  "value_columns": ["condition", "replicate"],
  "rows": [["nbc01", "fresh_edx01", "r1"], ["nbc02", "fresh_edx02", "r1"]]
}
```

Rows are aligned to `key_columns` followed by `value_columns`. Each value
column is also **materialized as a derived annotation column** by joining
across the key annotations, so consumers (`split`, `filter`, pyarrow, the
Python API) read plain columns and never implement the join. The design is
the source of truth: writing a derived column directly is refused, and
rewriting a key annotation re-derives its dependents.

## Write semantics

Sidecar writes are atomic — the new file is staged beside the destination
and renamed into place — and column-preserving in both directions:
rebuilding the read index (`escpod index`) keeps annotations, and writing an
annotation (`escpod annotate`, `demux --annotate`) keeps the index and other
annotations. A crash mid-write leaves the previous sidecar intact. This
matters because the file mixes a rebuildable cache (the index) with data
products that exist nowhere else once the CSV that produced them is deleted
(the annotations).

Column names `read_id`, `batch_idx`, and `row_idx` are reserved; everything
else in the schema is treated as an annotation.

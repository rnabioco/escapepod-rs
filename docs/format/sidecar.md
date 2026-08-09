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

`batch_idx`/`row_idx` are the **read index**: they locate a read in the POD5
reads table, enabling O(log n) binary-search lookup instead of a full-table
scan. Every additional column is an **annotation** — dictionary-encoded utf8
labels, so a 50k-read barcode annotation costs well under 1 MB. Record
batches are compressed with zstd (Arrow IPC buffer compression).

Because it is ordinary Arrow, any Arrow reader can consume a sidecar with no
escapepod code:

```python
import pyarrow.ipc as ipc
table = ipc.open_file("reads.pod5.p5s").read_all()
```

## Identity binding

The Arrow schema metadata binds the sidecar to exactly one POD5:

| Key | Value |
|-----|-------|
| `escapepod:p5s_version` | Format version (currently `1`) |
| `escapepod:file_identifier` | The POD5 footer's `file_identifier` UUID |
| `escapepod:pod5_size` | The POD5's byte size at write time |

Readers validate both identity keys from the IPC footer *before decoding any
record batch*. A sidecar copied next to the wrong file, or left behind after
its POD5 was replaced, therefore fails loudly ("does not match this POD5
file") rather than silently describing the wrong reads. There is no partial
acceptance: identity either matches exactly or the sidecar is rejected.

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

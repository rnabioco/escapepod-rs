# escpod index

Build the [`.p5s` sidecar](../format/sidecar.md) caches for a POD5 file: the
**read index** (O(log n) lookup by read ID instead of a full-table scan) and
the **signal batch geometry** (the per-batch row counts of the signal table).
The POD5 itself is never modified.

Both caches are rebuildable from the POD5 at any time, which is what separates
`index` from [`annotate`](../experimental/annotate.md) — annotations are data
products that exist nowhere else, so writing them stays behind
`--features experimental`, while `index` ships in the default build.

## Usage

```bash
escpod index [OPTIONS] <FILES>...
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<FILES>...` | Input POD5 files and/or directories (searched recursively) |

## Options

| Option | Description |
|--------|-------------|
| `-f, --force` | Rebuild existing `.p5s` sidecars (annotations and scores are preserved) |
| `-t, --threads <N>` | Threads for parallel processing across files (default: 16, capped at available CPUs) |
| `-h, --help` | Print help |

## Examples

```bash
escpod index input.pod5          # index one file
escpod index *.pod5              # index all POD5 files
escpod index data_dir/           # index a directory recursively
escpod index input.pod5 --force  # rebuild an existing sidecar
```

## What gets cached, and why

### The read index

`batch_idx` / `row_idx` per read, sorted by read UUID. Without it, resolving a
set of read IDs means scanning the reads table; with it, each lookup is a
binary search. Loading the index from a sidecar rather than building it
measured 249 ms → 59 ms on a 33 GB file.

A locator is trusted only as far as the read it names: every indexed lookup
confirms the row it landed on actually holds the requested `read_id` before
using it.

### The signal batch geometry

Row count is the one field an Arrow IPC footer does not carry — a `Block`
records a batch's offset, metadata length and body length and nothing else — so
recovering the counts means reading every batch's own message header. That is
one scattered touch per batch, measured at 15–24 ms *each* when cold on a
network filesystem, and it is paid on **every process start**, not once.

A 33 GB file with 8866 signal batches spent 4.95 s there, 78% of a cold
5000-read scattered fetch. With the geometry cached that becomes 4.82 ms.

The counts are *measured*, not assumed. Reading batch 0 and extrapolating is
what the official `pod5` library and dorado do, and what escapepod's
`nonuniform_signal_batch` diagnostic exists to catch them out on — recording
the real counts costs about fifteen bytes (they are run-length encoded, so a
conformant file is one run and a short tail) and is exact for a non-uniform
file too.

## When to run it

Once, on a **networked node**, before submitting jobs against a large file —
the same reason [`escpod resquiggle models fetch`](../experimental/resquiggle.md)
exists. Every sidecar write records the geometry when it is not already there
(`escpod index`, `escpod annotate`, `demux --annotate` alike), so a sidecar
cannot end up carrying a read index and nothing else.

A sidecar that already holds both caches is skipped. One missing either — for
example written by an older `demux --annotate`, which recorded the index but
not the geometry — is completed in place, preserving annotations and scores.
`--force` re-measures even when a value is present, because rebuilding the
cache is the job this command exists to do.

## Notes

- The sidecar is bound to its POD5 by file identifier and size, so a stale or
  misplaced one fails loudly rather than describing the wrong reads.
- A cached geometry is verified before use (one entry per batch the footer
  describes; first and last batches read for real and compared). Any mismatch
  logs a warning and falls back to the full walk — a stale cache costs time,
  never correctness.
- This format replaces the earlier `.p5i` index sidecar; delete any `.p5i`
  files and rerun `escpod index`.
- Format details: [The `.p5s` Sidecar](../format/sidecar.md).

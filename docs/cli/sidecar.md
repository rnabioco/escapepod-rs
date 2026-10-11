# escpod sidecar

Write the [`.p5s` sidecar](../format/sidecar.md): the companion file that holds
a POD5's read index, its signal batch geometry and any per-read annotations.
The POD5 itself is never written — raw sequencer output stays byte-identical
and checksummable, and deleting an annotation is editing or deleting the
sidecar.

```bash
escpod sidecar <COMMAND> [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| [sidecar index](sidecar-index.md) | Build the caches: the read index and the signal batch geometry. No labels |
| [sidecar annotate](sidecar-annotate.md) | Record per-read labels — a demux CSV, another tool's read-to-label table, an experimental design — and list or remove them |

## One operation, two entry points

Both commands write the same file through the same writers, with the same
binding to the POD5 (by file identifier and size, so a stale or misplaced
sidecar fails loudly) and the same handling of a directory's
[collection sidecar](../format/sidecar.md#one-sidecar-for-a-directory).
`sidecar annotate` builds the read index when the sidecar does not have one
yet, so `sidecar index` is the labels-free form of the same operation: reach for
it to pay the cost of the index and the geometry once, on a networked node,
before submitting jobs against a large file.

Each command touches only its own columns. `sidecar index` keeps annotations
and scores, and `sidecar annotate` keeps the index and other annotations.

```bash
escpod sidecar index run1/pod5/                        # caches only
escpod sidecar annotate -a demux.csv run1/pod5/        # labels (and the index, if missing)
escpod sidecar annotate run1/pod5/ \
    --from-table calls.csv.gz --column barcode=predicted_barcode
```

## Deprecated spellings

`escpod index` and `escpod annotate` were the top-level spellings before the
group existed. They still work, with the same arguments and behaviour: each is
hidden from `--help`, logs one warning on stderr naming the new spelling, and
forwards to the command above. They will be removed in a later release, so
switch scripts now.

| Old | New |
|-----|-----|
| `escpod index` | `escpod sidecar index` |
| `escpod annotate` | `escpod sidecar annotate` |

## See also

- [The `.p5s` Sidecar](../format/sidecar.md) — the on-disk specification.
- [demux](demux.md#sidecar-output) — `demux --annotate` writes the sidecar
  directly from the fused pipeline, and `demux split --sidecar` reads it back.

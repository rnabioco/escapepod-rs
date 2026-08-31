# escpod signal classify

Read-level classification against a model bundle, from POD5 signal plus an
aligned BAM. Today that means the **tRNA charging (aminoacylation)**
classifier: for each read it asks whether the tRNA was charged, and writes the
probability onto the BAM.

```bash
escpod signal classify reads.pod5 -b aln.bam -r ref.fa -m bundle/ -o out.bam
```

Ships in the default build — no extra Cargo feature.

!!! note "`escpod signal`, not `escpod classify`"
    `escpod demux classify` assigns a barcode from a DTW/GBM fingerprint. This
    asks an entirely different question of an entirely different input. The two
    used to sit one keystroke apart as `escpod demux classify` and a bare
    top-level `escpod classify`, which is a trap for anyone reading a shell
    history. The old spelling still works as a hidden deprecated alias that
    warns and forwards here, so 0.10.0-era scripts keep running.

## Usage

```bash
escpod signal classify [OPTIONS] -b <BAM> -r <FASTA> -m <BUNDLE> -o <BAM> <POD5>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<POD5>` | Input POD5 file or directory |

## Options

| Option | Description |
|--------|-------------|
| `-b, --bam <BAM>` | Aligned BAM with move tables (`dorado --emit-moves`, tags preserved through alignment) |
| `-r, --reference <FASTA>` | Reference FASTA the BAM was aligned to; the CCA\|adapter junction is located in every record |
| `-m, --model <DIR>` | Model bundle directory (or its `metadata.json`) |
| `-o, --output <BAM>` | Output BAM: input records with `cl` added |
| `--tsv <PATH>` | Also write per-read calls as TSV (`read_id`, `reference`, `p_<class>`, `cl`, `reason`) |
| `--min-mapq <N>` | Minimum mapping quality to classify a read (default: `1`) |
| `--orientation <MODE>` | Move-table signal frame: `auto` (default), `time`, or `reversed` |
| `-t, --threads <N>` | Threads for parallel processing |
| `-h, --help` | Print help |

## Output

The `cl` tag is written as **uint8**, `round(P(charged) · 255)`, onto every
record of each classified read — no modbase `ML` → `cl` round-trip. Reads that
could not be scored carry no tag, and the run reports why each was skipped:
unmapped/filtered, low mapq, reference without junction, missing `mv`/`ns`
tags, junction not aligned, query outside move table, non-UUID read name.

`--tsv` additionally emits one row per read as
`read_id`, `reference`, `p_<positive class>`, `cl`, `reason` — the convenient
form for plotting or thresholding outside a BAM reader, and the place the skip
`reason` is recorded per read rather than only in the summary.

## Getting a model bundle

Bundles come from
[escapepod-models](https://github.com/rnabioco/escapepod-models). A bundle is a
directory holding the scorer plus a `metadata.json` describing the feature
recipe.

**The recipe travels in the bundle, never in flags** — feature order and
offsets, the k-mer table pinned by sha256, the operating point. That is
deliberate: a caller computing the features differently gets a *wrong answer*
rather than an error, so there is nothing to configure.

The metadata schema is **closed**. Every key in the file is a rule the model
was built with, so a key this runtime does not implement is refused at load
rather than silently dropped. (`provenance`, `metrics` and `caveats` are exempt
and free-form.)

### Two scorers, one feature space

A bundle carries either a gradient-boosted tree model (`gbm`, which routes
`NaN` natively) or a small ONNX network over the same columns
(`feature_model`). Which one a directory holds is a property of the bundle,
never a flag — everything upstream of the final scoring step is shared
verbatim. Both load in the default build.

The network is what escapepod-models ships, and it is the better model:
**0.727 of reads callable at 99% precision, against the GBM's 0.449** on a
held-out flowcell.

### A third variant: the signal window

`waveform_model` reads a **signal window** rather than a column vector: the
normalised current and its k-mer residual, the sequence k-mer context scattered
along the signal axis, and 12 per-base dwell/level rows — three tensors, one
BCE logit out.

It is a second pipeline over the same two files, not a second scorer on the
first one. The base-to-signal map is walked through the CIGAR into *reference*
coordinates, the anchor is the motif **+2** rather than +3, the spans are
refined by a banded DP before any feature is taken, and the signal frame comes
from the bundle instead of a vote (so `--orientation` is ignored, with a
warning).

Two things follow from that:

- **It needs a separate build.** `--features classify-waveform`, and a
  `libonnxruntime.so` on `ORT_DYLIB_PATH` at run time — so a stock release
  binary, which is static musl and cannot dlopen anything, cannot run one.
  Unlike every other ONNX graph escpod runs, this one cannot go through the
  statically linked tract: it is a PyTorch *dynamo* export, and dynamo lowers
  `adaptive_avg_pool1d` with a non-dividing output size into an
  `Unsqueeze`/`Transpose`/`GatherND`/`Transpose`/`Where` chain that tract's
  shape inference cannot close. A default build refuses such a bundle by name,
  with that hint, rather than failing at the first read. This is expected to be
  temporary — the fix is a re-export from leech >= 0.10.0, which emits that
  pool as a plain matmul; tracked as rnabioco/escapepod-models#96.
- **Its shipped Platt calibration is carried, not applied.** The operating
  point beside it is stated on the uncalibrated probability the graph emits, so
  calibrating would move the scale out from under the very threshold that ships
  with the model. escpod says so at load; a caller who wants calibrated
  probabilities must re-derive the threshold too.

## How a read is anchored

1. Locate the CCA–aa junction in **reference** coordinates (the `CCAGGC` motif,
   +3; the windowed variant anchors at +2, one base earlier, and says so in its
   own metadata).
2. Map reference → query through the CIGAR.
3. Map query → signal through the move table, Remora convention
   (`move_pos * stride + ts`).
4. Compute per-base dwell / mean / std plus the z-scored k-mer residual, with
   everything before the common arm masked.
5. Score.

### `--orientation`

Some runs index **reversed** signal in their move tables. Getting this wrong
silently mirrors every window, so escpod detects it per run by vote rather than
assuming — but the vote needs **≥ 50 informative reads and a 95% consensus** to
commit.

Leave it at `auto` for a normal run. Use `--orientation time` or
`--orientation reversed` to force the frame on a batch too small for the vote
to resolve.

## Notes

- A bundle's `abstain` rule is carried and **warned about, not applied** — if
  your bundle declares one, the calls here are unabstained and you should apply
  the rule downstream.
- Parity with the training-corpus implementation (`escapepod_models.charging`)
  is pinned by golden-vector tests, and the ONNX arm additionally by
  bit-exact reference-vector parity plus a real-weights run over a real corpus.
- BAM file I/O lives in the CLI; the classifier itself is the
  `escapepod-classify` crate, which owns every definition of the model's input.

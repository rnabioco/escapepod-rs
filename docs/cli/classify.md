# escpod classify

Read-level classification against a model bundle, from POD5 signal plus an
aligned BAM. Today that means the **tRNA charging (aminoacylation)**
classifier: for each read it asks whether the tRNA was charged, and writes the
probability onto the BAM.

```bash
escpod classify reads.pod5 -b aln.bam -r ref.fa -m bundle/ -o out.bam
```

Ships in the default build — no extra Cargo feature.

!!! note "Not `escpod demux classify`"
    `escpod demux classify` is a demux stage: it assigns a barcode from a
    DTW/GBM adapter fingerprint CSV. This command asks an entirely different
    question of an entirely different input (a POD5 and an aligned BAM).

    Between 0.11.0 and 0.18.1 this command was spelled `escpod signal
    classify`, grouped under a `signal` namespace to keep the two apart. Every
    other tool here is one word, so the group was retired in 0.19.0; the
    deprecated `escpod signal classify` alias has since been removed
    entirely — use `escpod classify`.

## Usage

```bash
escpod classify [OPTIONS] -b <BAM> -r <FASTA> -m <BUNDLE> -o <BAM> <POD5>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<POD5>` | Input POD5 file or directory |

## Options

| Option | Description |
|--------|-------------|
| `-b, --bam <BAM>` | Aligned BAM with move tables (`dorado --emit-moves`, tags preserved through alignment) |
| `-r, --reference <FASTA>` | Reference FASTA the BAM was aligned to; the CCA\|adapter junction is located in every record. The `waveform_model` variant additionally rebuilds each read's aligned reference from its `MD` tag rather than slicing this file — see below |
| `-m, --model <DIR>` | Model bundle directory (or its `metadata.json`) |
| `-o, --output <BAM>` | Output BAM: input records with `cl` added |
| `--tsv <PATH>` | Also write per-read calls as TSV (`read_id`, `reference`, `p_<class>`, `cl`, `reason`) |
| `--min-mapq <N>` | Minimum mapping quality to classify a read (default: `1`) |
| `--orientation <MODE>` | Move-table signal frame: `auto` (default), `time`, or `reversed` |
| `-t, --threads <N>` | Threads for parallel processing |
| `--device <auto\|cpu\|gpu>` | Where the `waveform_model` variant's TCN inference runs (default `auto`) — see [GPU acceleration](#gpu-acceleration). Ignored (with a one-line note) for a GBM/feature-network bundle, which has no GPU path |
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

## Provenance (`@PG` `DS`)

The output BAM's `@PG` record for `escpod-classify` carries the real invoked
command line (`CL`), and a `DS` field holding a JSON object with everything
an after-the-fact audit of the calls needs — since none of it otherwise
survives past the run's own stdout/stderr:

```json
{
  "model_id": "charging_tcn_sup6_rna004",
  "model_version": "0.1.0",
  "scorer_sha256": "…",
  "positive_class": "charged",
  "basecaller": {"model": "rna004_sup@v6.0.0", "dorado_version": "2.1.1+d66c17c"},
  "operating_point": {"probability": 0.82, "cl": 209},
  "calibration": false,
  "abstain_rule": "…",
  "device": {
    "requested": "gpu",
    "cublas_path": "/opt/conda/envs/gpu/lib/libcublas.so.12",
    "cublas_version": "12.9.1",
    "cublaslt_path": "/opt/conda/envs/gpu/lib/libcublasLt.so.12",
    "cublaslt_version": "12.9.1",
    "cublas_repaired": false,
    "gpu_batches_scored": 434,
    "parity_checked_batches": 7,
    "parity_worst_abs_dp": 0.000013
  }
}
```

Every field the bundle or the run does not carry is omitted, never
fabricated: a bundle with no `basecaller`/`operating_point`/`abstain` block
has no such key, and `calibration` alone is always present (`false` when the
bundle ships none) because it is not optional on the bundle itself.
`positive_class` (`bundle.classes[1]`) says which class `cl` and
`operating_point.probability` are a probability *of* — the one thing the
`@PG` `CL` string used to carry before it was replaced with the real argv
below.

`device` is #425's addition, and its own fields follow the same omit-rather-
than-fabricate rule, plus one more: **`requested` is always the device that
actually scored these reads, never merely the one a GPU scorer loaded for.**
Two cases report `"cpu"` despite a GPU scorer having loaded successfully:

- A `--device auto` run that hit a GPU refusal (see the `#416`/`#420`
  warnings below) and fell back to rescoring the whole run on the CPU. Here
  the cuBLAS/cuBLASLt fields are still present — the pairing loaded and
  agreed fine; a real-batch parity *divergence* is what triggered the
  fallback, and that is worth recording, not discarding — but none of the
  `gpu_batches_scored`/`parity_*` fields are, since the check that caught the
  divergence is what ended the run rather than a summary of it.
- A run smaller than one GPU batch: every read is scored by
  `classify_reads_gpu`'s own CPU-fallback tail, `gpu.logits()` is never
  actually called, and none of `device`'s GPU fields are present at all.

The remaining fields, present only when `requested == "gpu"`:

- **`cublas_path`** / **`cublas_version`** / **`cublaslt_path`** /
  **`cublaslt_version`** / **`cublas_repaired`** — the cuBLAS/cuBLASLt
  pairing `ensure_cublas_pairing()` resolved before the graph moved onto CUDA
  (rnabioco/escapepod-rs#416): which physical library file each resolved to,
  its release, and whether escpod had to reload cuBLAS against the sibling
  cuBLASLt shipped beside it to make the two agree. The paths matter as much
  as the versions — #416's root cause was a mismatched *pair of files*, and
  two runs can report identical version strings while having resolved
  different files.
- **`gpu_batches_scored`** — how many GPU batches this run actually scored
  (`gpu.logits()` calls), independent of how many of those were cross-checked.
- **`parity_checked_batches`** / **`parity_worst_abs_dp`** — how many of
  those batches were also cross-checked against the CPU scorer on real reads
  (the first, then one in every `ESCAPEPOD_WAVEFORM_GPU_PARITY_EVERY`,
  default 64) and the worst `|ΔP(charged)|` any of them showed. This is a
  *sample*, not a census: at the default cadence, `parity_checked_batches`
  is roughly `gpu_batches_scored / 64` — the two together say how much of
  the run that coverage represents, which `parity_checked_batches` alone
  cannot.

This is exactly the evidence an incident like #416 needs and, before this,
could only be reconstructed from a caller's own log retention or Slurm
accounting — neither of which escpod controls, and neither of which is
guaranteed to still exist by the time someone asks.

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

That cuts both ways, and it is worth knowing which escpod you are running: a
bundle from a *newer* builder fails to load rather than loading with its new
rule quietly ignored. Refusing to answer is the recoverable half of the trade,
but it does mean a bundle and a runtime have to be released in that order.

### Which basecaller called the corpus

A bundle may declare the basecalling model its training corpus was called with:

```json
"basecaller": {"model": "rna004_sup@v6.0.0", "dorado_version": "2.1.1+d66c17c"}
```

escpod prints it at load and **does not check it** — it would have to read the
BAM's `@PG` to know what called the reads it was handed. Checking it is worth
the trouble: the charging feature set is `mean + z-scored k-mer residual` and
the expected level is predicted from the read's own basecall, so a charging
model substantially detects *how the basecaller fails* at the aminoacyl adduct.
Scoring the same reads called with a different model loses ~0.010 AUROC, loses
~3 pp of TPR while gaining ~0.4 pp of FPR at the shipped set point (so no
threshold recovers it), and flips **3.9% of per-read calls** — while the
aggregate charged fraction moves 0.04 pp. The one number anyone would check
reads "no change" while one read in 26 answers differently.

Bundles published before this key exists do not carry it, and load fine; escpod
says so rather than assuming comparability.

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
first one. It also reads the reference differently: the sequence a read is
scored against is rebuilt from that read's **`MD` tag**, the way the training
corpus builds it, not sliced out of the FASTA. Those disagree wherever the
FASTA carries an `N` — the alignment recorded a real base there — and since
levels are looked up per 9-mer, one `N` blanks nine of them. A read with no
`MD` tag is skipped (`no MD tag`) rather than silently scored off the FASTA.

The bundle *says* which source it was built from, in
`waveform_model.preprocessing.reference_source` — absent meaning `md`, which is
what every published bundle was built with. That is a different question from
`motif_reference`, which names where the *motif* is searched (the FASTA, in
reference coordinates, which is right); reading one as an answer to both is how
this runtime got it wrong. A bundle naming a source escpod does not assemble is
refused at load, naming both, rather than scored off `MD` anyway — the ambiguity
is permanent (it is an ordered degenerate position, so no substituted base is
right for more than 55% of reads), and both sources score every read without
erroring.

The base-to-signal map is walked through the CIGAR into *reference*
coordinates, the anchor is the motif **+2** rather than +3, the spans are
refined by a banded DP before any feature is taken, and the signal frame comes
from the bundle instead of a vote (so `--orientation` is ignored, with a
warning).

Two things follow from that:

- **The bundle version decides whether it loads at all.** It goes through the
  statically linked tract like every other ONNX graph escpod runs, so a stock
  release binary can run one — but only since `charging_tcn_rna004@v0.1.1`.
  The first export, `@v0.1.0`, defeated tract's shape inference two ways at
  once — a `value_info` entry per intermediate carrying the batch axis as a
  *symbol*, so pinning the batch failed at the first convolution; and
  `adaptive_avg_pool1d(390 → 11)` open-coded into a rank-8 `GatherND` — and no
  graph rewrite fixed either. The fix was the re-export
  (rnabioco/escapepod-models#96, and #97's build-time gate so a graph the
  shipped runtime cannot load never registers again). Loading `@v0.1.0` fails
  at `into_optimized` with tract's own analysis error, naming the file.
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

## GPU acceleration

Only the `waveform_model` variant has a GPU path — the windowed TCN, batched
through [`tract-cuda`](https://github.com/sonos/tract) rather than
onnxruntime. GBM and `feature_model` bundles have no device path at all;
`--device gpu` on one of those is a no-op with a one-line log note, never an
error. Build with `--features gpu` (see [demux's GPU
section](demux.md#gpu-acceleration) for the Cargo/pixi setup — one `gpu`
feature covers every GPU-capable stage in the binary, `classify` included).

**tract-cuda needs more than the demux GPU stages do.** `detect --method cnn`
and the CRF encoder go through onnxruntime, which only `dlopen`s a prebuilt
`libonnxruntime` — the CUDA *driver* (`libcuda.so`, part of the NVIDIA driver,
always present on a GPU node) is enough for that check. `classify`'s TCN
instead builds a CUDA context and *compiles its own kernels at run time*
through NVRTC, and needs two more things the driver alone does not supply:

- The CUDA **runtime** library, `libcudart`, discoverable by name at `dlopen`
  time — not just the driver. A node can have a perfectly good driver and
  still fail here if the `libcudart` it finds is the wrong major version (see
  rnabioco/escapepod-rs#347's post-mortem below).
- CUDA **headers** (`cuda_fp16.h`, the CCCL headers) on disk at run time, for
  NVRTC to compile against — nothing else in this binary needs headers past
  build time.

The pixi `gpu` environment (`pixi run install-gpu`, then `pixi run -e gpu …` —
see [demux's runtime-libraries
section](demux.md#runtime-libraries-the-pixi-environment)) already pins both:
`cuda-cudart`/`cuda-cudart-dev` for the runtime library and `cuda-cccl` for
the headers. Run `escpod classify --device gpu` the same way you would run any
other GPU-capable command — inside `pixi run -e gpu`, or with that
environment's `LD_LIBRARY_PATH` reproduced by hand. Running the released
`-gpu` binary bare, with no CUDA environment set up by the caller, means
`libcudart` resolution falls through to whatever the *node* happens to expose
on its default library path — see the warning below.

!!! warning "rnabioco/escapepod-rs#347: `Missing symbol cudaGetDeviceProperties_v2`"
    Reported as a raw panic —
    `thread 'main' panicked at cudarc-0.19.9/.../runtime/sys/mod.rs: Missing
    symbol cudaGetDeviceProperties_v2: dlsym failed` — on nodes where the CUDA
    *driver* worked fine (other CUDA tools on the same node ran normally) but
    `escpod classify --device gpu` was invoked **without** the pixi `gpu`
    environment's `LD_LIBRARY_PATH` active. cudarc's dynamic loader tries the
    bare, unversioned `libcudart.so` before any versioned name, so on a node
    whose default library path exposes a system CUDA install of a different
    major version — CUDA 13, say — that is the one it finds, and CUDA 13
    dropped the `_v2` symbol this build's pinned CUDA 12.x API needs.

    escpod now turns this into a named error instead of a process abort
    (`--device gpu` fails with the missing-symbol detail and a pointer to this
    section; `--device auto` logs a warning and falls back to the CPU scorer
    rather than crashing) — but the fix that lets the *GPU* actually run is
    making sure `libcudart` resolves inside the pixi `gpu` environment before
    anything from the node's own CUDA install is on the search path. A
    pipeline that shadows a GPU-enabled `escpod` onto `PATH` for one command
    (rather than invoking it via `pixi run -e gpu`) has to reproduce that
    `LD_LIBRARY_PATH` itself, the same way it already would for the CNN/CRF
    onnxruntime stages.

!!! warning "rnabioco/escapepod-rs#416: wrong probabilities, no error, on some nodes"
    The same unversioned-name-first lookup, applied to cuBLAS. A CUDA runtime
    environment (the pixi `gpu` env, or a pipeline's) ships `libcublas.so.12`
    but not the unversioned `libcublas.so`, which is a `-dev` file — so
    cudarc's first candidate is answered by the node's `ldconfig` cache. On a
    node with a host toolkit registered there (compgpu03: CUDA 12.8), that
    host cuBLAS is loaded, and *its* `NEEDED libcublasLt.so.12` is then
    answered by `LD_LIBRARY_PATH` — the environment's 12.9. cuBLAS 12.8 over
    cuBLASLt 12.9 runs without complaint and returns wrong GEMMs: the TCN's
    GPU output correlated 0.009 with the CPU's on 20k reads (13.1% called
    charged against 1.5%). Either consistent pair, 12.8/12.8 or 12.9/12.9,
    is exact. A node with no second toolkit in the cache never mixes them,
    which is why it looked like a property of the node.

    Two guards now stand in front of that:

    - **Before the graph goes onto CUDA**, escpod opens cuBLAS the way cudarc
      will and compares the release of the cuBLAS and cuBLASLt files actually
      loaded. On a mismatch it reloads cuBLAS against the cuBLASLt shipped
      beside it and says so in a warning; if that is not possible, the GPU is
      refused.
    - **On real reads**, the first GPU batch of every run, then one batch in
      every 64 (`ESCAPEPOD_WAVEFORM_GPU_PARITY_EVERY`), is also scored on the
      CPU; any read differing by more than 1e-3 in P(charged) refuses the GPU.
      A healthy run sits around 1e-5.

    A refused GPU is an error under `--device gpu` and a warning plus a
    whole-run CPU fallback under `--device auto` — never a GPU run whose
    answers were not checked. Both the pairing that was resolved and the
    parity summary these two guards compute are recorded in the output BAM's
    own [`@PG` `DS` provenance](#provenance-pg-ds) (#425) — an after-the-fact
    audit no longer depends on a caller's own log capture.

## Notes

- A bundle's `abstain` rule is carried and **warned about, not applied** — if
  your bundle declares one, the calls here are unabstained and you should apply
  the rule downstream.
- Parity with the training-corpus implementation (`escapepod_models.charging`)
  is pinned by golden-vector tests, and the ONNX arm additionally by
  bit-exact reference-vector parity plus a real-weights run over a real corpus.
- BAM file I/O lives in the CLI; the classifier itself is the
  `escapepod-classify` crate, which owns every definition of the model's input.

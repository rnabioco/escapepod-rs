# Signal Processing

The `escapepod` package also exposes a few primitives from the
`escapepod-signal` crate: signal normalization, kmer level tables, and
signal-to-sequence map refinement (resquiggle).

## Normalization

Both functions apply median-MAD normalization (median-centered, scaled by the
MAD with the 1.4826 Gaussian factor, with a graceful fallback on constant
signal). They differ only in input dtype:

```python linenums="1"
import numpy as np
import escapepod

with escapepod.Reader("experiment.pod5") as reader:
    read = reader.reads()[0]
    adc = reader.get_signal(read)          # int16 raw ADC

# From raw int16 ADC:
norm = escapepod.normalize_signal(adc)     # -> float32

# From an already-float32 signal (e.g. picoamps):
pa = reader.get_signal_pa(read)
norm = escapepod.mad_normalize(pa)         # -> float32
```

## Kmer level tables

`KmerTable` loads a tab-delimited `kmer<TAB>level` file (gzip supported) — the
expected normalized signal level for each kmer — and looks levels up per kmer or
expands them along a sequence:

```python linenums="1"
table = escapepod.KmerTable.from_file("levels.txt.gz")

table.k                       # kmer length
table.get("AACGT")            # expected level for one kmer -> float
levels = table.extract_levels("AACGTACGT...")  # per-base expected levels -> float32
```

## Refining a signal-to-sequence map

`refine_signal_map` refines a base-to-signal boundary assignment (a
"resquiggle") against a level model using banded dynamic programming, and
returns updated rescaling parameters.

The input `signal` must already be normalized (see above). `expected_levels`
is typically produced by `KmerTable.extract_levels`. `seq_to_signal_map` is the
current per-base signal boundary indices.

```python linenums="1"
norm = escapepod.normalize_signal(adc)
expected_levels = table.extract_levels(sequence)

refined_map, scale, shift, drift = escapepod.refine_signal_map(
    norm,
    seq_to_signal_map,     # list[int], length == len(sequence) (+1)
    expected_levels,
    half_bandwidth=5,      # DP band half-width
    scale_iters=2,         # rescale refinement iterations
    dwell_target=None,     # None = per-read target from the input map's median dwell
    dwell_weight=None,     # None = the preset's 0.5
    seed=None,             # RNG seed for the Theil-Sen rescale sampling
)
```

The settings are escapepod's `RefineSettings::move_table_refinement` preset —
fixed banding, a least-squares rough rescale over the 0.05–0.95 quantiles
clipped 10 bases, a Theil-Sen inter-iteration rescale over at most 200 points,
and the asymmetric dwell penalty at weight 0.5. Rust callers wanting the same
refinement build the same preset, so the two paths cannot drift apart.
`dwell_target`/`dwell_weight` override the preset when set; leave them `None`
to use it.

The **dwell target is resolved per read** from the median dwell of the input
`seq_to_signal_map`. A constant suits exactly one chemistry at one
translocation rate — RNA004 at 130 bases/s and 4 kHz sits near 31
samples/base — and because the penalty is asymmetric (quadratic below target,
logarithmic above), a target set too low actively drags boundaries toward
dwells the pore never produced.

The return value is `(refined_seq_to_signal_map, scale, shift, drift)`. The
rescale parameters are returned **for inspection**; applying them is your
decision, and the refined map is not rescaled for you. They would be applied
as:

```python linenums="1"
matched = (norm - shift - drift * np.arange(len(norm))) / scale
```

!!! warning "The rescale fit can be weakly identified"
    A per-read affine fit estimated over a near-constant stretch of signal —
    a 3' adapter, a long homopolymer — is poorly constrained, and in practice
    returns wild or negative scales (observed: 15 to 1084, sign flips
    included). Pipelines that refine over such a region discard `scale`,
    `shift` and `drift` and keep their own normalization.

!!! note "Experimental"
    Resquiggle refinement is an evolving, lower-level API — the same one behind
    the experimental [`resquiggle`](../experimental/resquiggle.md) CLI command.
    Signatures here may change.

# escapepod resquiggle

Refine signal-to-base mapping using banded dynamic programming. Takes an input POD5 file (raw signal) and a BAM file with basecaller move tables, then produces a new BAM file with refined signal boundaries stored in auxiliary tags.

## Overview

Nanopore basecallers produce a **move table** that maps signal blocks to bases, but
this mapping is coarse (one decision per stride block). Resquiggle refines these
boundaries to sample-level resolution by aligning the raw signal against expected
kmer levels using banded dynamic programming.

```
  Raw signal (from POD5)
  ┌──────────────────────────────────────────────────────┐
  │  ╱╲    ╱╲╱╲      ╱╲                                 │
  │ ╱  ╲  ╱    ╲    ╱  ╲    ╱╲  ╱╲╱╲                   │
  │╱    ╲╱      ╲──╱    ╲──╱  ╲╱    ╲──╲╱╲──            │
  └──────────────────────────────────────────────────────┘
        ▲            ▲           ▲          ▲
  Move table      (coarse boundaries from basecaller)

        ▼  resquiggle refinement  ▼

  ┌──────────────────────────────────────────────────────┐
  │  ╱╲    ╱╲╱╲      ╱╲                                 │
  │ ╱  ╲  ╱    ╲    ╱  ╲    ╱╲  ╱╲╱╲                   │
  │╱    ╲╱      ╲──╱    ╲──╱  ╲╱    ╲──╲╱╲──            │
  └──────────────────────────────────────────────────────┘
       ▲           ▲          ▲          ▲
  Refined       (sample-level boundaries from DP)
```

## Usage

```bash
escapepod resquiggle <INPUT> -b <BAM> -k <KMER_TABLE> -o <OUTPUT> [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Input POD5 file or directory |

## Options

| Option | Description |
|--------|-------------|
| `-b, --bam <FILE>` | Input BAM file with move table (`mv` tag) (required) |
| `-k, --kmer-table <FILE>` | Tab-delimited kmer level table file (required) |
| `-o, --output <FILE>` | Output BAM file (required) |
| `--algo <ALGO>` | Refinement algorithm: `dwell-penalty` (default) or `viterbi` |
| `--iterations <N>` | Number of refinement iterations (default: 1) |
| `--half-bandwidth <N>` | Half bandwidth for banded DP (default: 5) |
| `--rescale <ALGO>` | Rescale algorithm: `theil-sen` (default) or `least-squares` |
| `--dwell-target <N>` | Target dwell time per base for dwell-penalty (default: 0 = auto from move table) |
| `--dwell-weight <W>` | Dwell penalty weight (default: 0.5) |
| `--normalize <MODE>` | Normalization mode for kmer levels (e.g., `mad`) |
| `-j, --threads <N>` | Number of threads for parallel processing |
| `-h, --help` | Print help |

## Input Requirements

### POD5 File

The input POD5 file (or directory of POD5 files) must contain the raw signal data for the reads present in the BAM file.

### BAM File

The BAM file must contain:

- **Read names** that are UUIDs matching POD5 read IDs
- **`mv` tag** (move table) from the basecaller, encoding stride and per-block move decisions
- **Sequence** from basecalling

The following optional BAM tags are used when present:

| Tag | Description |
|-----|-------------|
| `sm` | Signal mean (scaling) |
| `sd` | Signal standard deviation (scaling) |
| `sp` | Parent signal offset |
| `ts` | Trimmed signal length |
| `ns` | Subread signal length |

### Kmer Level Table

A tab-delimited file mapping kmers to expected signal levels. Each row contains
a kmer sequence and its expected pA level. The kmer size must be odd (e.g., 9).

```
  Kmer table file (TSV):

  AAAAAAAAA    0.958
  AAAAAAAAC    1.023
  AAAAAAAAG    0.912
  ...          ...
  UUUUUUUUU    0.445
```

The table is loaded into a flat lookup array indexed by 2-bit encoding
(A=0, C=1, G=2, T/U=3) for O(1) per-kmer lookup.

#### From sequence to expected levels

A sliding window of size k extracts the kmer at each position. Each kmer maps
to an expected signal level from the table. The level is assigned to the
**dominant base** position within the kmer (determined automatically by a
Kruskal-Wallis H test over all table entries to find which position most
influences the level).

```
  Basecalled sequence (from BAM):

  Position:   0   1   2   3   4   5   6   7   8   9  10  11
  Base:       A   C   G   U   A   G   C   U   A   G   C   A

  Sliding 9-mer window (k=9):
                                                     dominant
  pos 0:    [ A C G U A G C U A ]  ──▶  level[0]  ─▶  base
  pos 1:      [ C G U A G C U A G ]  ──▶  level[1]     pos
  pos 2:        [ G U A G C U A G C ]  ──▶  level[2]    │
  pos 3:          [ U A G C U A G C A ]  ──▶  level[3]  │
                                                        ▼
  Expected                                      (e.g., center
  levels:   L0  L1  L2  L3  L4  L5  L6  L7 ...   of the kmer)
```

#### How expected levels drive the DP

The DP aligns the raw signal against the expected level sequence. At each
candidate boundary, the cost is the squared error between the measured signal
samples and the expected level for that base. The DP finds the set of
boundaries that minimizes total squared error (plus any dwell penalty).

```
  signal ▲
         │    ╱╲
         │   ╱  ╲       ╱╲          expected level
   L0    │──╱────╲─────╱──╲── ─ ─ ─ ─ ─ ─ ─ ─ ─
         │ ╱      ╲   ╱    ╲
         │╱        ╲─╱      ╲
         │          ╲         ╲           ╱╲
   L1    │─ ─ ─ ─ ─ ╲─ ─ ─ ─ ╲─ ─ ─ ─ ╱──╲─ ─
         │            ╲        ╲       ╱    ╲
         │             ╲        ╲─────╱      ╲
   L2    │─ ─ ─ ─ ─ ─ ─╲─ ─ ─ ─╲─ ─ ─ ─ ─ ─╲─
         │               ╲       ╲╱            ╲
         └────────┬───────┬───────┬──────────┬──▶ time
                base 0  base 1  base 2     base 3

         cost = sum of (signal - expected_level)^2 within each base
```

When `--normalize mad` is used, the kmer levels are MAD-normalized before
alignment: `(level - median) / (MAD * 1.4826)`, centering them at zero with
unit dispersion.

## Output

The output BAM file contains all input records with refined signal-to-base mapping stored in auxiliary tags:

| Tag | Type | Description |
|-----|------|-------------|
| `rs` | `B:I` (uint32 array) | Refined signal boundaries per base (in full-signal coordinates) |
| `rc` | `f` (float) | Refined calibration scale |
| `ro` | `f` (float) | Refined calibration offset |

Records that could not be refined (missing POD5 data, no move table, etc.) are written through unchanged.

## Processing Phases

The command runs in three phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 1: LOAD                            │
│                                                                 │
│  ┌──────────┐   ┌───────────────┐   ┌──────────────────────┐   │
│  │ kmer     │   │ BAM           │   │ POD5                 │   │
│  │ table    │   │ read IDs +    │   │ signal extraction    │   │
│  │          │   │ move tables   │   │ (bulk, by read ID)   │   │
│  └────┬─────┘   └───────┬───────┘   └──────────┬───────────┘   │
│       │                 │                       │               │
│       ▼                 ▼                       ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Matched reads (POD5 ∩ BAM)                  │   │
│  └──────────────────────────┬───────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 2: REFINE (parallel)                   │
│                                                                 │
│  For each matched read:                                         │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  1. Initial scaling from POD5 calibration + BAM tags   │     │
│  │  2. Rough rescale (Theil-Sen on quantiles)             │     │
│  │  3. Iterative DP refinement + rescale loop:            │     │
│  │     ┌────────────────────────────────────────────┐     │     │
│  │     │  normalize signal ──▶ banded DP ──▶ rescale│──┐  │     │
│  │     └────────────────────────────────────────────┘  │  │     │
│  │              ▲                                      │  │     │
│  │              └──────────── repeat N iterations ─────┘  │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Phase 3: WRITE                            │
│                                                                 │
│  Output BAM with rs/rc/ro tags added to refined records         │
└─────────────────────────────────────────────────────────────────┘
```

## Algorithms

### Banded Dynamic Programming

The DP operates within a band around the initial move-table alignment to avoid
exploring the entire signal x sequence space. The `--half-bandwidth` parameter
controls how far from the initial alignment the DP can search.

```
  Signal position ──▶
  0         10        20        30        40        50
  ├─────────┼─────────┼─────────┼─────────┼─────────┤

  B  ░░░░░▓▓▓▓░░░░░
  a       ░░░░░▓▓▓▓░░░░░
  s            ░░░░░▓▓▓▓░░░░░
  e                 ░░░░░▓▓▓▓░░░░░
  ▼                      ░░░░░▓▓▓▓░░░░░

  ░ = band (allowed search region)
  ▓ = initial alignment from move table
  half_bandwidth = 5 in this example
```

The DP finds the lowest-cost path through the band, where cost is the squared
error between the measured signal and the expected kmer level at each position.

### Refinement (`--algo`)

- **`dwell-penalty`** (default) -- Banded DP with an asymmetric dwell penalty.
  Short dwells get a strong quadratic penalty to prevent degenerate single-sample
  bases; long dwells get a gentle logarithmic nudge that is easily overcome by
  good signal fit, preserving genuine long dwells.

- **`viterbi`** -- Standard Viterbi-style banded DP without dwell penalty.

```
  Dwell penalty (asymmetric)

  penalty
  ▲
  │▓
  │▓
  │ ▓                    quadratic        │    logarithmic
  │ ▓                   (short dwell)     │   (long dwell)
  │  ▓                                   │
  │  ▓                                   │
  │   ▓                                  │
  │    ▓▓                                │
  │      ▓▓▓                             │
  │         ▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▶
  └──────────────┬───────────────────────────────────────▶ dwell
                target
```

When `--dwell-target 0` (the default), the target is automatically set to the
median dwell time from the move table, adapting to the read's actual
signal-to-base ratio.

### Rescaling (`--rescale`)

After each DP iteration, the signal scale and shift are re-estimated to correct
for calibration drift between the basecaller model and the kmer level table.

- **`theil-sen`** (default) -- Robust Theil-Sen regression for estimating signal scale and shift, resistant to outliers.
- **`least-squares`** -- Least-squares regression with dwell filtering and level truncation.

```
  Rescaling: fit measured signal to expected levels

  measured ▲
  signal   │            ╱
           │          ╱  ○        ○ = per-base mean signal
           │        ╱ ○            vs expected kmer level
           │   ○  ╱
           │    ╱ ○
           │  ╱○               slope  = new scale
           │╱  ○               offset = new shift
           └──────────────────▶
                expected level
```

## Examples

### Basic Resquiggle

```bash
escapepod resquiggle reads.pod5 \
    -b basecalls.bam \
    -k kmer_levels.tsv \
    -o refined.bam
```

### With Multiple Iterations and Normalization

```bash
escapepod resquiggle pod5_dir/ \
    -b basecalls.bam \
    -k kmer_levels.tsv \
    -o refined.bam \
    --iterations 3 \
    --normalize mad
```

### Tuning Bandwidth and Algorithm

```bash
escapepod resquiggle reads.pod5 \
    -b basecalls.bam \
    -k kmer_levels.tsv \
    -o refined.bam \
    --algo viterbi \
    --half-bandwidth 10 \
    --rescale least-squares
```

### Custom Dwell Penalty Parameters

```bash
escapepod resquiggle reads.pod5 \
    -b basecalls.bam \
    -k kmer_levels.tsv \
    -o refined.bam \
    --algo dwell-penalty \
    --dwell-target 36 \
    --dwell-weight 0.3
```

### Using Multiple Threads

```bash
escapepod resquiggle reads.pod5 \
    -b basecalls.bam \
    -k kmer_levels.tsv \
    -o refined.bam \
    -j 8
```

## Algorithm References

The resquiggle algorithms are inspired by:

- **Banded DP refinement and signal rescaling**: Inspired by [fishnet](https://www.researchsquare.com/article/rs-8345719/v1) by Brickner et al., licensed under GPL-3.0.

## Notes

- Only reads present in both the BAM and POD5 input are refined; all BAM records are still written to the output.
- Signal data is bulk-extracted before the parallel refinement phase, so POD5 files are not accessed during refinement.
- Error diagnostics are printed at the end, showing aggregated counts of skip reasons.
- The `--half-bandwidth` parameter controls the width of the DP band around the initial move-table alignment. Larger values allow more deviation from the basecaller's initial mapping but increase computation.
- A `@PG` header record is added to the output BAM with the resquiggle command line parameters.

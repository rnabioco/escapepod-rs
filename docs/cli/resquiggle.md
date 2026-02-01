# escapepod resquiggle

Refine signal-to-base mapping using banded dynamic programming. Takes an input POD5 file (raw signal) and a BAM file with basecaller move tables, then produces a new BAM file with refined signal boundaries stored in auxiliary tags.

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
| `--normalize-levels` | Apply MAD normalization to kmer levels |
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

A tab-delimited file mapping kmers to expected signal levels. Each row contains a kmer sequence and its expected pA level.

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

1. **Load** -- Reads the kmer table, scans the BAM to collect read IDs, then indexes only matching POD5 reads and bulk-extracts their signal data.
2. **Refine** -- Runs banded DP refinement in parallel across all matched reads. Signal trimming tags (`sp`, `ts`, `ns`) are applied so refinement operates on the correct signal window.
3. **Write** -- Writes the output BAM with refined tags inserted.

## Algorithms

### Refinement (`--algo`)

- **`dwell-penalty`** (default) -- Banded DP with a penalty term for implausible dwell times, encouraging realistic signal-to-base assignments.
- **`viterbi`** -- Standard Viterbi-style banded DP without dwell penalty.

### Rescaling (`--rescale`)

- **`theil-sen`** (default) -- Robust Theil-Sen regression for estimating signal scale and shift, resistant to outliers.
- **`least-squares`** -- Least-squares regression with dwell filtering and level truncation.

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
    --normalize-levels
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

### Using Multiple Threads

```bash
escapepod resquiggle reads.pod5 \
    -b basecalls.bam \
    -k kmer_levels.tsv \
    -o refined.bam \
    -j 8
```

## Notes

- Only reads present in both the BAM and POD5 input are refined; all BAM records are still written to the output.
- Signal data is bulk-extracted before the parallel refinement phase, so POD5 files are not accessed during refinement.
- Error diagnostics are printed at the end, showing aggregated counts of skip reasons.
- The `--half-bandwidth` parameter controls the width of the DP band around the initial move-table alignment. Larger values allow more deviation from the basecaller's initial mapping but increase computation.

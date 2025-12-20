# escapepod demux

Barcode demultiplexing for Oxford Nanopore sequencing data. This command identifies barcodes in reads using signal-level analysis and splits reads into separate POD5 files by barcode.

## Overview

The demux workflow analyzes the raw nanopore signal to detect adapter regions, extract barcode fingerprints, classify reads, and optionally split them into separate files.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEMUX WORKFLOW OVERVIEW                               │
└─────────────────────────────────────────────────────────────────────────────┘

      POD5 Files                                             Demuxed POD5s
          │                                                       ▲
          ▼                                                       │
  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    │
  │    detect     │───▶│  fingerprint  │───▶│   classify    │────┤
  │  (LLR-based)  │    │ (t-test seg)  │    │ (DTW distance)│    │
  └───────────────┘    └───────────────┘    └───────────────┘    │
          │                    │                    │             │
          ▼                    ▼                    ▼             │
    boundaries.csv      fingerprints.csv   classifications.csv   │
                                                   │              │
                                                   ▼              │
                                           ┌───────────────┐      │
                                           │     split     │──────┘
                                           │ (by barcode)  │
                                           └───────────────┘

  ┌───────────────┐
  │     train     │──▶ reference.json (for classify --reference)
  │ (from known)  │
  └───────────────┘
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| [detect](#detect) | Detect adapter boundaries using LLR algorithm |
| [fingerprint](#fingerprint) | Extract signal fingerprints from adapter regions |
| [classify](#classify) | Classify reads by barcode using DTW distance |
| [split](#split) | Split reads into separate POD5 files by barcode |
| [train](#train) | Train reference fingerprints from known samples |

---

## detect

Detect adapter boundaries in reads using the Log-Likelihood Ratio (LLR) algorithm. This identifies where the adapter sequence starts and ends in the raw signal.

### Signal Structure (RNA Sequencing)

```
Signal Level
     │
high │  ╭──────╮                              ╭────────────
     │  │      │                              │
     │  │      ╰──────────────────────────────╯
     │  │  Open   Adapter      Barcode      RNA
low  │──╯  Pore   (detected    region       transcript
     └────────────────────────────────────────────────────▶
                                                        Time
              │◀─── Adapter Region ───▶│
          adapter_start            adapter_end
```

### LLR Algorithm

The LLR algorithm finds boundaries by maximizing the variance difference between adjacent segments:

```
                    LLR Boundary Detection
                    ─────────────────────

Signal:  ▁▁▁▁▁▁▁█████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
                ▲       ▲
                │       │
              Split   Split
              Point   Point

For each candidate position i:

  gain(i) = n × log(var[0,n)) - [n_head × log(var[0,i)) + n_tail × log(var[i,n))]
                ▲                      ▲                        ▲
                │                      │                        │
         Full variance          Head variance           Tail variance

The position with maximum gain indicates the best split point.
```

### Usage

```bash
escapepod demux detect <FILES>... -o <OUTPUT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<FILES>` | Input POD5 file(s) |

### Options

| Option | Description |
|--------|-------------|
| `-o, --output <FILE>` | Output boundaries CSV file (required) |
| `--min-adapter <N>` | Minimum adapter observations (default: 200) |
| `--border-trim <N>` | Border trim size (default: 50) |
| `-j, --threads <N>` | Number of threads (default: 4) |
| `-h, --help` | Print help |

### Output Format

The output CSV contains:

```csv
read_id,num_samples,adapter_start,adapter_end
a1b2c3d4-...,50000,1500,4200
b2c3d4e5-...,48000,1200,3800
```

| Column | Description |
|--------|-------------|
| `read_id` | Read UUID |
| `num_samples` | Total signal samples |
| `adapter_start` | Adapter start position (samples) |
| `adapter_end` | Adapter end position (samples) |

### Example

```bash
escapepod demux detect *.pod5 -o boundaries.csv --min-adapter 200 -j 8
```

---

## fingerprint

Extract barcode fingerprints from adapter regions using t-test segmentation. The fingerprint is a fixed-length feature vector representing the barcode signal pattern.

### Fingerprint Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FINGERPRINT EXTRACTION                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Raw Signal (adapter region only)
    │
    ▼
┌─────────────┐
│ Normalize   │  MAD normalization: (x - median) / MAD
│ (MAD)       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ T-test      │  Find N-1 changepoints using sliding window t-test
│ Segment     │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Compute     │  Mean signal level per segment
│ Means       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Normalize   │  Z-score, min-max, median, or none
│ Features    │
└─────────────┘
    │
    ▼
Fingerprint Vector [fp_0, fp_1, ..., fp_n]
```

### T-test Segmentation

The algorithm uses a sliding window t-test to find changepoints:

```
Window-Based Changepoint Detection
──────────────────────────────────

Signal: ████████▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄████████████████
              ◀──W──▶◀──W──▶
              Window1 Window2

At each position, compare adjacent windows:

  t_score = |mean₁ - mean₂| / √(var₁ + var₂)

        t-score
          ▲
          │        *
          │       * *
          │      *   *
          │  ···*     *···
          │ *           *
          └─────────────────▶ position
                   ▲
                   │
              Changepoint
              (local max)

Select top N changepoints with minimum separation.
```

### Resulting Segments

```
Segmented Signal with Means
───────────────────────────

Signal: ─────────────────────────────────────────────────
        ▁▁▁▁▁│████│▄▄▄▄▄│███│▁▁▁▁▁│▄▄▄▄│▁▁▁▁▁▁▁│████│▁▁
        seg 0│seg1│seg 2│seg3│seg 4│seg5│ seg 6 │seg7│...
        ──────────────────────────────────────────────────▶
                                                     samples

Fingerprint = [mean₀, mean₁, mean₂, mean₃, mean₄, mean₅, mean₆, mean₇, ...]
            = [-0.82,  1.23, -0.15,  0.95, -0.71,  0.12, -0.45,  1.08, ...]
```

### Usage

```bash
escapepod demux fingerprint <FILES>... --boundaries <CSV> -o <OUTPUT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<FILES>` | Input POD5 file(s) |

### Options

| Option | Description |
|--------|-------------|
| `--boundaries <FILE>` | Boundaries CSV from detect command (required) |
| `-o, --output <FILE>` | Output fingerprints CSV file (required) |
| `--num-segments <N>` | Number of fingerprint segments (default: 10) |
| `--window-width <N>` | T-test window width (default: 5) |
| `--normalize <METHOD>` | Normalization method: zscore, minmax, median, none (default: zscore) |
| `-j, --threads <N>` | Number of threads (default: 4) |
| `-h, --help` | Print help |

### Output Format

```csv
read_id,fp_0,fp_1,fp_2,fp_3,fp_4,fp_5,fp_6,fp_7,fp_8,fp_9
a1b2c3d4-...,-0.823451,1.234567,-0.156789,0.951234,...
b2c3d4e5-...,-0.712345,0.987654,-0.234567,1.123456,...
```

### Example

```bash
escapepod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv
escapepod demux fingerprint *.pod5 --boundaries boundaries.csv -o fp.csv --num-segments 12 --normalize median
```

---

## classify

Classify reads by barcode using Dynamic Time Warping (DTW) distance between fingerprints and reference barcodes.

### DTW Distance Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  DYNAMIC TIME WARPING (DTW)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Query fingerprint:     Q = [q₀, q₁, q₂, q₃, q₄, ...]
Reference fingerprint: R = [r₀, r₁, r₂, r₃, r₄, ...]

DTW finds the optimal alignment between sequences:

        r₀  r₁  r₂  r₃  r₄
       ┌───┬───┬───┬───┬───┐
   q₀  │ ● │   │   │   │   │   Legend:
       ├───┼───┼───┼───┼───┤   ● = optimal path
   q₁  │   │ ● │   │   │   │   ─ = allowed moves
       ├───┼───┼───┼───┼───┤
   q₂  │   │ ● │ ● │   │   │   D[i,j] = |qᵢ - rⱼ| + min(D[i-1,j],
       ├───┼───┼───┼───┼───┤                          D[i,j-1],
   q₃  │   │   │   │ ● │   │                          D[i-1,j-1])
       ├───┼───┼───┼───┼───┤
   q₄  │   │   │   │   │ ● │   DTW distance = D[n,m]
       └───┴───┴───┴───┴───┘

Sakoe-Chiba Band Constraint (--window):
────────────────────────────────────────
       ┌───┬───┬───┬───┬───┐
   q₀  │░░░│░░░│   │   │   │   ░ = valid region
       ├───┼───┼───┼───┼───┤       (within window)
   q₁  │░░░│░░░│░░░│   │   │
       ├───┼───┼───┼───┼───┤   Constraint: |i - j| ≤ window
   q₂  │   │░░░│░░░│░░░│   │
       ├───┼───┼───┼───┼───┤   Reduces time from O(nm) to O(n·w)
   q₃  │   │   │░░░│░░░│░░░│
       ├───┼───┼───┼───┼───┤
   q₄  │   │   │   │░░░│░░░│
       └───┴───┴───┴───┴───┘
```

### Classification Process

```
Classification Decision
───────────────────────

Query fingerprint ─┬─▶ DTW(query, barcode_01) ───▶ d₁ = 0.23
                   ├─▶ DTW(query, barcode_02) ───▶ d₂ = 0.87
                   ├─▶ DTW(query, barcode_03) ───▶ d₃ = 0.45
                   └─▶ DTW(query, barcode_04) ───▶ d₄ = 0.91

Best match:        barcode_01 (d₁ = 0.23)
Second best:       barcode_03 (d₃ = 0.45)

Confidence ratio = d_best / d_second_best = 0.23 / 0.45 = 0.51

If ratio < threshold (e.g., 0.8):
  → Assign to barcode_01 with confidence 0.51
Else:
  → Mark as "unclassified" (ambiguous)
```

### Usage

```bash
escapepod demux classify <FINGERPRINTS> --reference <CSV> -o <OUTPUT>
escapepod demux classify <FINGERPRINTS> --model <JSON> -o <OUTPUT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<FINGERPRINTS>` | Input fingerprints CSV |

### Options

| Option | Description |
|--------|-------------|
| `--reference <FILE>` | Reference fingerprints CSV (from train command) |
| `--model <FILE>` | WarpDemuX model JSON file |
| `-o, --output <FILE>` | Output classifications CSV (required) |
| `--window <N>` | DTW window size (Sakoe-Chiba band, optional) |
| `--threshold <F>` | Confidence threshold for classification (default: 0.8) |
| `-j, --threads <N>` | Number of threads (default: 4) |
| `-h, --help` | Print help |

### Output Format

```csv
read_id,barcode,confidence,best_distance,second_best_distance
a1b2c3d4-...,barcode_01,0.512,0.234,0.457
b2c3d4e5-...,barcode_03,0.723,0.156,0.216
c3d4e5f6-...,unclassified,0.912,0.345,0.378
```

| Column | Description |
|--------|-------------|
| `read_id` | Read UUID |
| `barcode` | Assigned barcode or "unclassified" |
| `confidence` | Distance ratio (lower = more confident) |
| `best_distance` | DTW distance to best match |
| `second_best_distance` | DTW distance to second best |

### Example

```bash
# Using reference fingerprints
escapepod demux classify fingerprints.csv --reference reference.csv -o classifications.csv

# Using WarpDemuX model
escapepod demux classify fingerprints.csv --model warpdemux.json -o classifications.csv --window 10

# With custom threshold
escapepod demux classify fingerprints.csv --reference reference.csv -o out.csv --threshold 0.7
```

---

## split

Split reads into separate POD5 files based on barcode classification.

### Usage

```bash
escapepod demux split <FILES>... --classifications <CSV> --output-dir <DIR>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<FILES>` | Input POD5 file(s) |

### Options

| Option | Description |
|--------|-------------|
| `--classifications <FILE>` | Classifications CSV from classify command (required) |
| `-d, --output-dir <DIR>` | Output directory for demuxed files (required) |
| `--prefix <STR>` | Output file prefix (default: none) |
| `--unclassified` | Include unclassified reads in separate file |
| `-j, --threads <N>` | Number of threads (default: 4) |
| `-h, --help` | Print help |

### Output Structure

```
output_dir/
├── barcode_01.pod5
├── barcode_02.pod5
├── barcode_03.pod5
├── barcode_04.pod5
└── unclassified.pod5  (if --unclassified)
```

### Example

```bash
escapepod demux split *.pod5 --classifications classifications.csv -d demuxed/
escapepod demux split experiment.pod5 --classifications class.csv -d out/ --prefix exp1_ --unclassified
```

---

## train

Train reference barcode fingerprints from known samples. Use this to create a custom reference for your barcode set.

### Training Workflow

```
Training Reference Fingerprints
───────────────────────────────

Input Option A: Directory structure
─────────────────────────────────
input_dir/
├── barcode_01/
│   ├── sample1.pod5
│   └── sample2.pod5
├── barcode_02/
│   ├── sample1.pod5
│   └── sample2.pod5
└── barcode_03/
    └── sample1.pod5

Input Option B: Assignments CSV
───────────────────────────────
read_id,barcode,pod5_file
a1b2...,barcode_01,sample1.pod5
b2c3...,barcode_01,sample1.pod5
c3d4...,barcode_02,sample2.pod5

Processing:
───────────
For each barcode:
  1. Extract fingerprints from all assigned reads
  2. Compute consensus (mean) fingerprint
  3. Compute standard deviation per feature

Output: reference.json
─────────────────────
{
  "barcodes": {
    "barcode_01": {
      "fingerprint": [0.12, -0.45, ...],
      "std_dev": [0.05, 0.08, ...],
      "read_count": 150
    },
    "barcode_02": { ... }
  },
  "metadata": {
    "num_segments": 10,
    "normalization": "zscore"
  }
}
```

### Usage

```bash
escapepod demux train --input-dir <DIR> -o <OUTPUT>
escapepod demux train --assignments <CSV> -o <OUTPUT>
```

### Options

| Option | Description |
|--------|-------------|
| `--input-dir <DIR>` | Directory with barcode subdirectories containing POD5 files |
| `--assignments <CSV>` | CSV with read_id, barcode, pod5_file columns |
| `-o, --output <FILE>` | Output reference JSON file (required) |
| `--num-segments <N>` | Number of fingerprint segments (default: 10) |
| `--window-width <N>` | T-test window width (default: 5) |
| `--normalize <METHOD>` | Normalization method (default: zscore) |
| `--min-adapter <N>` | Minimum adapter observations (default: 200) |
| `--border-trim <N>` | Border trim size (default: 50) |
| `-j, --threads <N>` | Number of threads (default: 4) |
| `-h, --help` | Print help |

### Example

```bash
# From directory structure
escapepod demux train --input-dir training_samples/ -o reference.json

# From assignments CSV
escapepod demux train --assignments known_barcodes.csv -o reference.json --num-segments 12
```

---

## Complete Workflow Example

```bash
# 1. Detect adapter boundaries in all POD5 files
escapepod demux detect *.pod5 -o boundaries.csv -j 8

# 2. Extract fingerprints from adapter regions
escapepod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv

# 3a. Train reference (if you have known samples)
escapepod demux train --input-dir training_data/ -o reference.json

# 3b. Or use a pre-trained WarpDemuX model
# (exported from Python using scripts/export_warpdemux_model.py)

# 4. Classify reads
escapepod demux classify fingerprints.csv --reference reference.json -o classifications.csv

# 5. Split into separate files
escapepod demux split *.pod5 --classifications classifications.csv -d demuxed/ --unclassified

# View classification summary
cut -d, -f2 classifications.csv | sort | uniq -c | sort -rn
```

## Algorithm References

The demux algorithms are based on:

- **LLR boundary detection**: Adapted from [ADAPTed](https://github.com/KleistLab/ADAPTed) by Wiep K. van der Toorn
- **T-test segmentation**: Based on the [Tombo](https://github.com/nanoporetech/tombo) algorithm used in WarpDemuX
- **DTW classification**: Standard Dynamic Time Warping with optional Sakoe-Chiba band constraint

## See Also

- [Signal Compression](../format/compression.md) - How POD5 stores signal data
- [Segmentation Algorithms](../format/segmentation.md) - Detailed algorithm descriptions

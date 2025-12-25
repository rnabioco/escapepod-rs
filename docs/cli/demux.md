# escapepod demux

Barcode demultiplexing for Oxford Nanopore sequencing data. This command identifies barcodes in reads using signal-level analysis and splits reads into separate POD5 files by barcode.

## Comparison with WarpDemuX

Escapepod demux is a pure Rust reimplementation of the signal-level barcode demultiplexing algorithms from [WarpDemuX](https://github.com/KleistLab/WarpDemuX) and [ADAPTed](https://github.com/KleistLab/ADAPTed). The key differences are:

| Feature | Escapepod | WarpDemuX/ADAPTed |
|---------|-----------|-------------------|
| Language | Pure Rust | Python + C |
| Dependencies | None (statically linked) | PyTorch, dtaidistance, pod5 |
| Adapter detection | LLR only | LLR + CNN + fallback |
| Classification | DTW (Rust) | DTW (dtaidistance) |
| Model format | JSON (native or WarpDemuX) | Scikit-learn pickle |

### Performance Benchmarks

Tested on RNA004 data with 5 barcodes (1000 reads total), 4 threads:

| Metric | Escapepod | WarpDemuX |
|--------|-----------|-----------|
| **Detection speed** | 14x faster | baseline |
| **Full pipeline** | ~0.5s | ~2.4s |
| **Throughput** | ~2000 reads/sec | ~400 reads/sec |

**Note:** For best classification accuracy, use WarpDemuX pre-trained models via the `classify --model` option. The escapepod training workflow is experimental and may not generalize well to new samples.

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
| [train-svm](#train-svm) | Train SVM model from fingerprints (requires `train` feature) |

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
| `--downscale <N>` | Downscale factor for signal processing (default: 1, use 10 for WarpDemuX compatibility) |
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
| `--segment-start <N>` | Start sample offset within adapter region (default: 1000) |
| `--segment-end <N>` | End sample offset within adapter region (default: 2000) |
| `--num-segments <N>` | Number of fingerprint segments (default: 10) |
| `--window-width <N>` | T-test window width (default: 5) |
| `--normalize <METHOD>` | Normalization method: zscore, minmax, median, none (default: zscore) |
| `-j, --threads <N>` | Number of threads (default: 4) |
| `-h, --help` | Print help |

**Note:** The `--segment-start` and `--segment-end` options define which region within the adapter to use for fingerprinting. The defaults (1000-2000) match the training parameters, ensuring consistency between training and classification.

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
| `--segment-start <N>` | Start sample for fingerprint region (default: 1000) |
| `--segment-end <N>` | End sample for fingerprint region (default: 2000) |
| `--num-segments <N>` | Number of fingerprint segments (default: 10) |
| `--window-width <N>` | T-test window width (default: 5) |
| `--normalize <METHOD>` | Normalization method (default: zscore) |
| `--min-adapter <N>` | Minimum adapter observations (default: 200) |
| `--border-trim <N>` | Border trim size (default: 50) |
| `-j, --threads <N>` | Number of threads (default: 4) |
| `-h, --help` | Print help |

### Output Format

The train command outputs a JSON file with consensus fingerprints for each barcode:

```json
{
  "barcodes": {
    "BC00": {
      "fingerprint": [0.12, -0.45, ...],
      "std_dev": [0.05, 0.08, ...],
      "read_count": 150
    }
  },
  "params": { "segment_start": 1000, "segment_end": 2000, "num_segments": 10 }
}
```

**Note:** For best classification accuracy, we recommend using WarpDemuX pre-trained models instead of training your own. The escapepod training workflow produces consensus fingerprints that may not generalize as well as WarpDemuX's SVM-based models.

### Example

```bash
# From directory structure
escapepod demux train --input-dir training_samples/ -o reference.json

# From assignments CSV
escapepod demux train --assignments known_barcodes.csv -o reference.json

# Use trained reference for classification
escapepod demux classify fingerprints.csv --reference reference.json -o classifications.csv
```

---

## train-svm

Train an SVM model from labeled fingerprints for probabilistic barcode classification. This command requires the `train` feature to be enabled.

**Note:** This creates a DTW-SVM model that provides probability outputs for each class, enabling more nuanced confidence thresholds.

### Usage

```bash
escapepod demux train-svm -f <FINGERPRINTS> -o <OUTPUT> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-f, --fingerprints <FILE>` | CSV file with fingerprints (read_id, barcode, feat1, feat2, ...) (required) |
| `-o, --output <FILE>` | Output JSON file for trained SVM model (required) |
| `--gamma <VALUE>` | RBF kernel gamma parameter (default: 1.0) |
| `--power <VALUE>` | Power to raise distances before exponential (default: 1.0) |
| `--c <VALUE>` | SVM regularization parameter C (default: 1.0) |
| `--window <N>` | DTW window constraint (Sakoe-Chiba band) |
| `--thresholds <VALUES>` | Per-class confidence thresholds (comma-separated) |
| `-h, --help` | Print help |

### Input Format

The fingerprints CSV should include barcode labels:

```csv
read_id,barcode,fp_0,fp_1,fp_2,...,fp_9
a1b2c3d4-...,BC00,-0.823,1.234,-0.156,...
b2c3d4e5-...,BC00,-0.712,0.987,-0.234,...
c3d4e5f6-...,BC01,-0.456,0.789,-0.321,...
```

### Example

```bash
# Train SVM with default parameters
escapepod demux train-svm -f fingerprints.csv -o model.json

# Train with custom hyperparameters
escapepod demux train-svm -f fingerprints.csv -o model.json --gamma 0.5 --c 10.0 --window 10

# Use trained SVM model for classification (with classify --model-svm)
escapepod demux classify fingerprints.csv --model-svm model.json -o classifications.csv
```

### Building with train feature

```bash
cargo build --release --features train
```

---

## Complete Workflow Example

### Basic Workflow (with pre-trained model)

```bash
# 1. Detect adapter boundaries in all POD5 files
escapepod demux detect *.pod5 -o boundaries.csv -j 8

# 2. Extract fingerprints from adapter regions
escapepod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv

# 3. Classify reads using a pre-trained model
escapepod demux classify fingerprints.csv --model warpdemux_model.json -o classifications.csv

# 4. Split into separate files
escapepod demux split *.pod5 --classifications classifications.csv -d demuxed/ --unclassified

# View classification summary
cut -d, -f2 classifications.csv | sort | uniq -c | sort -rn
```

### Training Your Own Reference (Experimental)

If you have known barcode samples, you can train a consensus-based reference:

```bash
# Create assignments CSV with known read-to-barcode mappings
cat > assignments.csv << EOF
read_id,barcode,pod5_file
a1b2c3d4-...,BC00,sample1.pod5
b2c3d4e5-...,BC00,sample1.pod5
c3d4e5f6-...,BC01,sample2.pod5
EOF

# Train reference fingerprints
escapepod demux train --assignments assignments.csv -o reference.json -j 8

# Use the trained reference for classification
escapepod demux detect *.pod5 -o boundaries.csv -j 8
escapepod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv
escapepod demux classify fingerprints.csv --reference reference.json -o classifications.csv
```

**Note:** For production use, we recommend using WarpDemuX pre-trained models which provide significantly better generalization.

### Using WarpDemuX Models

You can export WarpDemuX models using the provided script:

```bash
# Export a WarpDemuX model to JSON format
python scripts/export_warpdemux_model.py path/to/warpdemux_model.pkl -o model.json

# Use the exported model
escapepod demux classify fingerprints.csv --model model.json -o classifications.csv
```

## Algorithm References

The demux algorithms are based on:

- **LLR boundary detection**: Adapted from [ADAPTed](https://github.com/KleistLab/ADAPTed) by Wiep K. van der Toorn
- **T-test segmentation**: Based on the [Tombo](https://github.com/nanoporetech/tombo) algorithm used in WarpDemuX
- **DTW classification**: Standard Dynamic Time Warping with optional Sakoe-Chiba band constraint

## See Also

- [Signal Compression](../format/compression.md) - How POD5 stores signal data
- [Segmentation Algorithms](../format/segmentation.md) - Detailed algorithm descriptions

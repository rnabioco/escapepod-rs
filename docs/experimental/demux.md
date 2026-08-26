# escpod demux

Barcode demultiplexing for Oxford Nanopore sequencing data, in the default
`escpod` build — including CNN/TCN adapter detection (`--method cnn`, what
the published [escapepod-models](https://github.com/rnabioco/escapepod-models)
barcodes were trained against) and CTC-CRF basecalling. Reads are classified
from the raw signal and either split into per-barcode POD5 files or recorded
in the [`.p5s` sidecar](../format/sidecar.md) so nothing needs splitting at
all.

GPU acceleration and model training are opt-in builds — see
[GPU acceleration](#gpu-acceleration) below.

## Fused pipeline (recommended)

Running `escpod demux` with no subcommand streams the whole pipeline —
adapter detection, classification, and output — in one pass over the input:

```bash
# Fetch a model bundle once (networked node; compute nodes can't reach GitHub)
escpod demux models fetch crf_nbc16_rna004

# Demux into per-barcode POD5 files
escpod demux reads.pod5 --model ~/.cache/escapepod/demux_models/barcode_crf_nbc16_rna004@v0.3.1 -d out/

# Or demux into the sidecar only — no split files, no CSV
escpod demux reads.pod5 --model <bundle> --annotate
```

Key options (see `escpod demux --help` for the full set):

| Option | Description |
|--------|-------------|
| `--model <PATH>` | Model: DTW-SVM / GBM JSON, or a CTC-CRF bundle directory |
| `-d, --output-dir <DIR>` | Write per-barcode POD5 files (optional with `--annotate`) |
| `--annotate` | Record assignments in each input's `.p5s` sidecar |
| `--classifications <FILE>` | Also write a `read_id,barcode,confidence` CSV |
| `--prefix <STR>` | Split-file prefix (default: `barcode`) |
| `--method <cnn\|llr>` | Adapter detector (CRF bundles pin their own) |
| `--gpu` | GPU inference where a GPU feature is compiled in |
| `--info` | Describe the model and exit |

### Sidecar output

`--annotate` composes with the other outputs: alone it is *sidecar-only*
demux (the POD5 is never duplicated), with `-d` it writes split files *and*
the sidecar, and `--classifications` can keep the CSV too. Materialize
subsets later, on demand:

```bash
escpod demux split reads.pod5 --sidecar -d out/                    # all barcodes
escpod filter reads.pod5 --annotation barcode=nbc05 -o nbc05.pod5  # one group
```

The stepwise subcommands below remain available for running stages
individually or inspecting intermediates.

## Comparison with WarpDemuX

Escapepod demux is a pure Rust reimplementation of the signal-level barcode demultiplexing algorithms from [WarpDemuX](https://github.com/KleistLab/WarpDemuX) and [ADAPTed](https://github.com/KleistLab/ADAPTed). The key differences are:

| Feature | Escapepod | WarpDemuX/ADAPTed |
|---------|-----------|-------------------|
| Language | Pure Rust | Python + C |
| Dependencies | None (statically linked) | PyTorch, dtaidistance, pod5 |
| Adapter detection | LLR + CNN | LLR + CNN + fallback |
| Classification | DTW (Rust) | DTW (dtaidistance) |
| Model format | JSON (native or WarpDemuX) | Scikit-learn pickle |

### Performance Benchmarks

Early numbers (LLR + DTW path, before the CNN/CRF heads landed) on RNA004
data with 5 barcodes (1000 reads total), 4 threads:

| Metric | Escapepod | WarpDemuX |
|--------|-----------|-----------|
| **Detection speed** | 14x faster | baseline |
| **Full pipeline** | ~0.5s | ~2.4s |
| **Throughput** | ~2000 reads/sec | ~400 reads/sec |

**Note:** For best classification accuracy, use a published model bundle
(`escpod demux models list` / `fetch`) or a WarpDemuX-exported model via
`--model`. The escpod training workflow is experimental and may not
generalize well to new samples.

## Overview

The demux workflow analyzes the raw nanopore signal to detect adapter regions, extract barcode fingerprints, classify reads, and optionally split them into separate files.

```mermaid
flowchart LR
    pod5([POD5 Files]) --> fused
    fused["<b>demux</b> (fused)<br/><i>detect + classify</i>"] --> demuxed([Per-barcode POD5s])
    fused -. "--annotate" .-> sidecar[reads.pod5.p5s]
    sidecar -. "split --sidecar<br/>filter --annotation" .-> demuxed

    pod5 -.-> detect
    detect["<b>detect</b><br/><i>LLR / CNN</i>"] -.-> boundaries[boundaries.csv]
    boundaries -.-> fingerprint["<b>fingerprint</b><br/><i>t-test seg</i>"]
    boundaries -.-> basecall["<b>basecall</b><br/><i>CTC-CRF</i>"]
    fingerprint -.-> classify["<b>classify</b><br/><i>DTW / GBM</i>"]
    classify -.-> classifications[classifications.csv]
    basecall -.-> classifications
    classifications -.-> split["<b>split</b><br/><i>by barcode</i>"]
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| [detect](#detect) | Detect adapter boundaries (LLR or CNN) |
| [fingerprint](#fingerprint) | Extract signal fingerprints from adapter regions |
| [classify](#classify) | Classify reads by barcode using DTW distance |
| `basecall` | CTC-CRF barcode basecalling from a boundaries CSV; `--barcodes` assigns reads by edit distance |
| [split](#split) | Split reads into separate POD5 files by barcode (CSV or `--sidecar`) |
| `models` | List / fetch published model bundles (`list`, `path`, `fetch`) |
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
escpod demux detect <FILES>... -o <OUTPUT>
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
| `--downscale <N>` | Downscale factor for signal processing (default: 10, WarpDemuX-native; set 1 for full resolution) |
| `--method <cnn\|llr>` | Boundary detector; `cnn` is what the published models were trained against |
| `--cnn-model <FILE>` / `--cnn-model-name <NAME>` | Boundary-CNN ONNX (explicit path, or a fetched bundle by name) |
| `--gpu` | Run CNN detection on the GPU (`gpu` build) |
| `-t, -j, --threads <N>` | Number of threads for parallel processing (default: 16, capped at available CPUs) |
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
escpod demux detect *.pod5 -o boundaries.csv --min-adapter 200 -j 8
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
escpod demux fingerprint <FILES>... --boundaries <CSV> -o <OUTPUT>
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
| `-t, -j, --threads <N>` | Number of threads for parallel processing (default: 16, capped at available CPUs) |
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
escpod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv
escpod demux fingerprint *.pod5 --boundaries boundaries.csv -o fp.csv --num-segments 12 --normalize median
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
escpod demux classify <FINGERPRINTS> --reference <CSV> -o <OUTPUT>
escpod demux classify <FINGERPRINTS> --model <JSON> -o <OUTPUT>
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
| `--min-ratio <RATIO>` | Top-2 distance-ratio threshold below which a read is confident (default: 0.8) |
| `--model-name <NAME>` | Use a fetched model bundle by name instead of `--model` |
| `--probabilities` | Emit per-class probabilities (SVM models) |
| `--gpu` | GPU DTW batch classify (`gpu` build) |
| `-t, -j, --threads <N>` | Number of threads for parallel processing (default: 16, capped at available CPUs) |
| `-h, --help` | Print help |

!!! warning "`--gpu` DTW classify is experimental and usually slower"
    On a full node the CPU DTW beats the GPU: 113 s on 64 CPU cores vs 132 s
    with `--gpu` on an A30 (0.85x), plus ~2.2 GB more RSS, on a 1.22M-read
    DTW-SVM run. An apparent 1.67x speedup vanishes once the CPU is given the
    whole node instead of 16 of 64 cores. It may still help where cores are
    scarce relative to the DTW workload, and it does nothing for GBM models.
    The GPU paths that *do* pay off are CNN adapter detection and the CRF
    encoder — see [GPU acceleration](#gpu-acceleration).

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
escpod demux classify fingerprints.csv --reference reference.csv -o classifications.csv

# Using WarpDemuX model
escpod demux classify fingerprints.csv --model warpdemux.json -o classifications.csv --window 10

# With a custom confidence ratio
escpod demux classify fingerprints.csv --reference reference.csv -o out.csv --min-ratio 0.7
```

---

## split

Split reads into separate POD5 files based on barcode classification.

### Usage

```bash
escpod demux split <FILES>... --classifications <CSV> --output-dir <DIR>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<FILES>` | Input POD5 file(s) |

### Options

| Option | Description |
|--------|-------------|
| `--classifications <FILE>` | Classifications CSV from classify/basecall (or use `--sidecar`) |
| `--sidecar` | Read assignments from each input's `.p5s` sidecar instead of a CSV |
| `--annotation <NAME>` | Sidecar annotation to split by (default: `barcode`) — e.g. a design-derived `condition` |
| `-d, --output-dir <DIR>` | Output directory for demuxed files (required) |
| `--prefix <STR>` | Output file prefix (default: `barcode`) |
| `--classified-only` | Drop unclassified reads instead of writing them to their own file |
| `-f, --force` | Overwrite existing per-barcode output files |
| `-t, -j, --threads <N>` | Number of threads for parallel processing (default: 16, capped at available CPUs) |
| `-h, --help` | Print help |

### Output Structure

```
output_dir/
├── barcode_nbc01.pod5
├── barcode_nbc02.pod5
├── barcode_nbc03.pod5
└── barcode_unclassified.pod5   (omitted with --classified-only)
```

### Example

```bash
escpod demux split *.pod5 --classifications classifications.csv -d demuxed/
escpod demux split reads.pod5 --sidecar -d demuxed/                        # from .p5s
escpod demux split reads.pod5 --sidecar --annotation condition -d by_cond/ # by design variable
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
escpod demux train --input-dir <DIR> -o <OUTPUT>
escpod demux train --assignments <CSV> -o <OUTPUT>
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
| `-t, -j, --threads <N>` | Number of threads for parallel processing (default: 16, capped at available CPUs) |
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

**Note:** For best classification accuracy, we recommend using WarpDemuX pre-trained models instead of training your own. The escpod training workflow produces consensus fingerprints that may not generalize as well as WarpDemuX's SVM-based models.

### Example

```bash
# From directory structure
escpod demux train --input-dir training_samples/ -o reference.json

# From assignments CSV
escpod demux train --assignments known_barcodes.csv -o reference.json

# Use trained reference for classification
escpod demux classify fingerprints.csv --reference reference.json -o classifications.csv
```

---

## train-svm

Train an SVM model from labeled fingerprints for probabilistic barcode classification. This command requires the `train` feature to be enabled.

**Note:** This creates a DTW-SVM model that provides probability outputs for each class, enabling more nuanced confidence thresholds.

### Usage

```bash
escpod demux train-svm -f <FINGERPRINTS> -o <OUTPUT> [OPTIONS]
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
escpod demux train-svm -f fingerprints.csv -o model.json

# Train with custom hyperparameters
escpod demux train-svm -f fingerprints.csv -o model.json --gamma 0.5 --c 10.0 --window 10

# Use the trained SVM model for classification (--model auto-detects the JSON shape)
escpod demux classify fingerprints.csv --model model.json -o classifications.csv
```

### Building with train feature

```bash
cargo build --release --features train
```

---

## Complete Workflow Example

### Fused pipeline (recommended)

```bash
# One pass: detect + classify + output. --annotate keeps everything in the
# sidecar; add -d out/ to also write per-barcode files.
escpod demux reads.pod5 --model <bundle-or-model.json> --annotate

escpod inspect summary reads.pod5                 # barcode counts, sidecar state
escpod demux split reads.pod5 --sidecar -d out/   # materialize when needed
```

### Stepwise (inspectable intermediates)

```bash
# 1. Detect adapter boundaries in all POD5 files
escpod demux detect *.pod5 -o boundaries.csv -j 8

# 2. Extract fingerprints from adapter regions
escpod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv

# 3. Classify reads using a pre-trained model
escpod demux classify fingerprints.csv --model warpdemux_model.json -o classifications.csv

# 4. Split into separate files (unclassified reads get their own file by default)
escpod demux split *.pod5 --classifications classifications.csv -d demuxed/

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
escpod demux train --assignments assignments.csv -o reference.json -j 8

# Use the trained reference for classification
escpod demux detect *.pod5 -o boundaries.csv -j 8
escpod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv
escpod demux classify fingerprints.csv --reference reference.json -o classifications.csv
```

**Note:** For production use, we recommend using WarpDemuX pre-trained models which provide significantly better generalization.

### Using WarpDemuX Models

You can export WarpDemuX models using the provided script:

```bash
# Export a WarpDemuX model to JSON format
python scripts/export_warpdemux_model.py path/to/warpdemux_model.pkl -o model.json

# Use the exported model
escpod demux classify fingerprints.csv --model model.json -o classifications.csv
```

## GPU acceleration

One opt-in Cargo feature, `gpu`, enables every GPU path; the `--gpu` runtime
flag then uses whichever fits the model and stage:

| Stage | Commands | What runs on the GPU |
|-------|----------|----------------------|
| CNN adapter detection | `demux detect --method cnn --gpu`, fused `demux --method cnn --gpu` | CNN/TCN inference through onnxruntime CUDA. The GPU path that pays off most — detection is inference-bound, ~7× faster end-to-end on an A30. |
| CTC-CRF encoder | `demux basecall --gpu`, fused `demux --gpu` with a CRF model | The basecall encoder through onnxruntime CUDA, ~4× end-to-end. |
| DTW classification | `demux classify --gpu`, `demux train-svm --gpu` | Batched DTW distance. **Experimental and usually slower** than a full CPU node — see the warning in [classify](#classify). |

(The granular `cnn-gpu` / `crf-gpu` flags still exist for library consumers
of `escapepod-demux`; CLI users only need `gpu`.)

### Getting a GPU-capable binary

Each release ships one, `escpod-<ver>-x86_64-unknown-linux-gnu-gpu.tar.gz`
on the [releases page](https://github.com/rnabioco/escapepod-rs/releases) —
the only artifact built `--features gpu`, and the only dynamically linked
Linux one, because the GPU runtimes are `dlopen`ed and static musl cannot do
that. It needs glibc ≥ 2.28 (RHEL/Rocky/Alma 8+, Ubuntu 20.04+). Prefer it
over a local build where you need a *knowable* version: `escpod --version`
on a hand-built binary is a local fact, and `adapter_end` — which escpod
computes and which defines the training window — makes the binary part of a
model's definition.

To build it yourself instead, nothing CUDA-related is needed at **build**
time: the GPU runtimes load with `dlopen` at **run** time, so you can build
on any machine, and a gpu-built binary still runs fine on CPU-only nodes as
long as `--gpu` isn't requested:

```bash
cargo build --release --features gpu -p escapepod-cli
```

### Runtime libraries: the pixi environment (from a checkout)

At run time `--gpu` needs two things, and the repository's
[pixi](https://pixi.sh) `gpu` environment supplies both. This path needs a
checkout of this repository — the `gpu` environment, the `install-ort` task
and the activation script all live in it — but *not* a build: the released
`-gnu-gpu` binary runs inside the environment just as well as a local one. If
you would rather not clone, see [without a checkout](#runtime-libraries-without-a-checkout)
below.

1. **The CUDA 12 runtime libraries** — `libcublas`/`libcublasLt`,
   `libcudart`, `libcufft`, cuDNN 9 (for the onnxruntime CUDA execution
   provider) and `libnvrtc` (for the DTW kernels), from conda-forge.
2. **A CUDA-enabled `libonnxruntime`** for the CNN/CRF paths — fetched once
   by the `install-ort` task into `.pixi/ort/` (extracted from the
   `onnxruntime-gpu` wheel; nothing is pip-installed), version-pinned to
   match the `ort` crate the binary was built with.

```bash
# once, on a machine with network access (on clusters: the login node —
# this also creates the environment on first use)
pixi run -e gpu install-ort

# then, on a node with a visible NVIDIA GPU — no env vars needed
pixi run -e gpu ./target/release/escpod demux reads.pod5 --model <bundle> --gpu --annotate
```

Activating the environment (`pixi run -e gpu …` or `pixi shell -e gpu`) sets
`LD_LIBRARY_PATH` and `ORT_DYLIB_PATH` automatically. The only system
requirement on the node itself is an NVIDIA driver new enough for CUDA 12.2
(≥ 535) — the DTW kernels target that driver API. This works the same for the
released `-gnu-gpu` binary as for a local build; point the command at wherever
you unpacked it.

### Runtime libraries without a checkout

If you downloaded `escpod-<ver>-x86_64-unknown-linux-gnu-gpu.tar.gz` and have
no reason to clone the repository, the same two pieces fit in a standalone
pixi manifest. Drop this in an empty directory as `pixi.toml`:

```toml
[workspace]
name = "escpod-gpu-runtime"
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
# CUDA 12 runtime for the onnxruntime CUDA execution provider …
cuda-version = "12.*"
cuda-cudart = "*"          # libcudart.so.12
libcublas = "*"            # libcublas.so.12 + libcublasLt.so.12
libcufft = "*"             # libcufft.so.11
cudnn = ">=9,<10"          # libcudnn.so.9
# … and NVRTC for the GPU DTW kernels, which cudarc compiles at run time.
cuda-nvrtc = ">=12"        # libnvrtc.so.12
# Only used to fetch the onnxruntime wheel below; nothing is pip-installed.
python = "3.12.*"
pip = "*"

# conda-forge's CUDA packages ship no activation hook, so make $CONDA_PREFIX/lib
# visible to the dlopen paths (ort's CUDA EP and cudarc's libnvrtc).
[activation.env]
LD_LIBRARY_PATH = "$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# The wheel is only a container for the CUDA-enabled libonnxruntime that ort
# (load-dynamic) dlopens; nothing is installed into any Python environment.
[tasks]
install-ort = """
python -m pip download onnxruntime-gpu==1.28.0 --no-deps -d ort &&
python -c "import glob, zipfile; zipfile.ZipFile(sorted(glob.glob('ort/onnxruntime_gpu-*.whl'))[-1]).extractall('ort')" &&
ls ort/onnxruntime/capi/libonnxruntime.so.*
"""
```

Then, once on a networked machine (no GPU needed) and after that on any GPU
node:

```bash
pixi run install-ort
# ort/onnxruntime/capi/libonnxruntime.so.1.28.0

export ORT_DYLIB_PATH="$PWD/ort/onnxruntime/capi/libonnxruntime.so.1.28.0"
pixi run /path/to/escpod demux reads.pod5 --model <bundle> --gpu --annotate
```

Point `ORT_DYLIB_PATH` at the library **where it was unpacked** — onnxruntime
loads `libonnxruntime_providers_cuda.so` from the same directory, so copying
the one `.so` somewhere tidier costs you the CUDA execution provider and
demotes the run to CPU.

Unlike the repository environment, this manifest sets `ORT_DYLIB_PATH` nowhere:
a dangling value makes the ort paths hang silently at startup, whereas leaving
it unset until the library exists fails fast with an error that names the fix.

### Verifying the GPU is actually in use

At the default log level, escpod announces which device each stage runs on:

```
INFO Detecting adapter boundaries using boundary CNN (GPU)
INFO Encoder runs on: GPU (onnxruntime CUDA)
```

Those lines say what was *requested*; onnxruntime failures that demote the
work to CPU surface as **warnings** (visible by default), so a warning-free
run on a GPU node is a healthy one. For positive confirmation that the CUDA
execution provider loaded, raise the dependency log level — escpod pins
third-party logs at `warn` unless `RUST_LOG` overrides it:

```bash
RUST_LOG=ort=info escpod demux basecall --gpu … 2>&1 | grep CUDAExecutionProvider
# INFO [ort::ep] Successfully registered `CUDAExecutionProvider`
```

| Symptom | Cause |
|---------|-------|
| Warning that the execution provider *"may fall back to CPU"*, run is slow | A CUDA runtime library is missing (typically `libcublasLt.so.12` or `libcudnn.so.9`). Run inside the pixi `gpu` environment so `LD_LIBRARY_PATH` includes them. |
| Clear startup error: could not load onnxruntime | `ORT_DYLIB_PATH` is unset (e.g. `install-ort` was never run) or points at a CPU-only build of onnxruntime. |
| Process hangs at startup and prints **nothing**, not even a status line | `ORT_DYLIB_PATH` is set but points at a file that does not exist. The pixi activation only sets it when the library is present, so this normally means a stale manual override. |

!!! tip "On a cluster, redirect output to a file"
    When running under a scheduler, don't pipe the job's output through
    `tail` or similar — those buffer until EOF, so a healthy job looks
    identical to a hung one. Redirect to a log file and follow that instead.

### Manual setup (without pixi)

Reproduce what the environment provides:

- Put the CUDA 12 runtime on `LD_LIBRARY_PATH`: `libcublas.so.12`,
  `libcublasLt.so.12`, `libcudart.so.12`, `libcufft.so.11`,
  `libcudnn.so.9`, and `libnvrtc.so.12` for the DTW kernels.
- For the CNN/CRF paths, set `ORT_DYLIB_PATH` to a **CUDA-enabled**
  `libonnxruntime`, e.g. extracted from the `onnxruntime-gpu` wheel
  (`onnxruntime/capi/libonnxruntime.so.<version>`). The onnxruntime version
  must be compatible with the `ort` crate the binary was built with —
  current pins: onnxruntime 1.28.0 with `ort` 2.0.0-rc.13.

### HPC notes

- Anything that downloads (`pixi run -e gpu install-ort`, and the pixi
  environment creation it triggers) must run on a **networked** node —
  compute nodes typically cannot reach the internet. Neither step needs a
  GPU, so the login node is fine.
- Both the environment (`.pixi/envs/gpu`) and the onnxruntime download
  (`.pixi/ort`) live inside the project directory. On a shared filesystem
  the GPU nodes see them with no further staging.
- Model files are also fetched explicitly, never at run time — see
  `escpod demux models fetch` above.

## Algorithm References

The demux algorithms are based on:

- **LLR boundary detection**: Adapted from [ADAPTed](https://github.com/KleistLab/ADAPTed) by Wiep K. van der Toorn
- **T-test segmentation**: Based on the [Tombo](https://github.com/nanoporetech/tombo) algorithm used in WarpDemuX
- **DTW classification**: Standard Dynamic Time Warping with optional Sakoe-Chiba band constraint

## See Also

- [Signal Compression](../format/compression.md) - How POD5 stores signal data
- [Segmentation Algorithms](../format/segmentation.md) - Detailed algorithm descriptions

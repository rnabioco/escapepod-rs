# Signal Segmentation & DTW Algorithms

This document describes the signal processing algorithms used for barcode demultiplexing in escapepod.

## Overview

The demux pipeline uses three core algorithms:

1. **LLR (Log-Likelihood Ratio)** - Detects adapter boundaries
2. **T-test Segmentation** - Extracts fingerprints from adapter regions
3. **DTW (Dynamic Time Warping)** - Classifies barcodes by signal similarity

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SIGNAL PROCESSING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

Raw Signal (i16)
    │
    ▼
┌─────────────────┐
│  Normalization  │  MAD: (x - median) / MAD
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  LLR Boundary   │  Find adapter start/end positions
│    Detection    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Extract        │  Isolate adapter region signal
│  Adapter        │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  T-test         │  Find changepoints within adapter
│  Segmentation   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Compute        │  Mean value per segment = fingerprint
│  Segment Means  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  DTW Distance   │  Compare to reference barcodes
└─────────────────┘
    │
    ▼
Classification Result
```

---

## LLR Boundary Detection

The Log-Likelihood Ratio algorithm detects abrupt changes in signal variance, which indicate boundaries between different signal regions (open pore, adapter, RNA).

### Mathematical Foundation

For a signal segment, the LLR gain at position `i` measures how much better the data is explained by two separate distributions vs. one:

```
LLR Gain Formula
────────────────

gain(i) = n × log(σ²_full) - [n_head × log(σ²_head) + n_tail × log(σ²_tail)]

Where:
  n       = total samples in segment
  n_head  = samples before split point
  n_tail  = samples after split point
  σ²_full = variance of entire segment
  σ²_head = variance of [0, i)
  σ²_tail = variance of [i, n)
```

### Efficient Variance Computation

Using cumulative sums enables O(1) variance calculation for any segment:

```
Precomputed Cumulative Sums
───────────────────────────

Signal:     s₀   s₁   s₂   s₃   s₄   s₅   ...
           ─────────────────────────────────────
cumsum:     s₀   s₀+s₁  Σs₀₋₂  ...
cumsum_sq:  s₀²  s₀²+s₁²  Σs₀₋₂²  ...

Variance of [start, end):
  mean = (cumsum[end-1] - cumsum[start-1]) / (end - start)
  var  = (cumsum_sq[end-1] - cumsum_sq[start-1]) / (end - start) - mean²
```

### Three-Split Strategy

The algorithm uses a three-split strategy to identify adapter regions:

```
Three-Split Boundary Detection
──────────────────────────────

Step 1: Find primary split (adapter end)
────────────────────────────────────────
Signal: ████████████████████████▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
                              ▲
                              x_first (max gain)

Step 2: Split left segment (find adapter start)
───────────────────────────────────────────────
Signal: ▁▁▁▁▁▁▁▁████████████████████▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
              ▲                   │
              x_head              x_first

Step 3: Split right segment (refinement)
────────────────────────────────────────
Signal: ▁▁▁▁▁▁▁▁████████████████████▄▄▄▄▄▄▄▄████████████████████████
              │                   │       ▲
              x_head              x_first x_tail

Final boundaries determined by median analysis:
  - Compare median levels of all 4 segments
  - Adapters show characteristic "dip" in pA level
  - Select adapter_start and adapter_end accordingly
```

### Visualization

```
LLR Gain Curve Example
──────────────────────

gain
  │                    ╭─╮
  │                   ╱   ╲
  │                  ╱     ╲
  │                 ╱       ╲
  │           ╭────╯         ╰────╮
  │      ╭───╯                     ╰───╮
  │  ───╯                               ╰───
  └──────────────────────────────────────────▶ position
                       ▲
                       │
                  Best split
                  (max gain)

Signal at same positions:
  │  ▁▁▁▁▁▁▁▁███████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
  └──────────────────────────────────────────▶
```

---

## T-test Segmentation

After isolating the adapter region, t-test segmentation divides it into discrete levels (events) to create a fingerprint.

### Windowed T-test

The algorithm slides two adjacent windows across the signal, computing a t-score at each position:

```
Sliding Window T-test
─────────────────────

        Window 1       Window 2
       ◀────w────▶   ◀────w────▶
Signal: ...████████████▄▄▄▄▄▄▄▄▄▄▄▄...
              ▲           ▲
             m₁          m₂

T-score formula (simplified):

  t = |m₁ - m₂| / √(var₁ + var₂)

Where:
  m₁, m₂   = means of windows 1 and 2
  var₁, var₂ = variances of windows 1 and 2
```

### Finding Changepoints

```
Changepoint Selection Algorithm
───────────────────────────────

Step 1: Compute t-scores for all positions
──────────────────────────────────────────

t-score
  │           ╭╮        ╭╮
  │          ╱  ╲      ╱  ╲
  │    ╭╮   ╱    ╲    ╱    ╲   ╭╮
  │   ╱  ╲ ╱      ╲  ╱      ╲ ╱  ╲
  │ ──    ─        ──        ─    ──
  └──────────────────────────────────────▶ position
         ▲          ▲         ▲
         │          │         │
        p₁         p₂        p₃   (local maxima)


Step 2: Sort by t-score (descending)
────────────────────────────────────
Candidates: [p₂, p₁, p₃, ...]  (sorted by score)


Step 3: Select with minimum separation
──────────────────────────────────────
Selected: []
Blacklist: {}

For p₂: not in blacklist → SELECT
  Selected: [p₂]
  Blacklist: {p₂-sep, ..., p₂+sep}  (nearby positions)

For p₁: not in blacklist → SELECT
  Selected: [p₂, p₁]
  Blacklist: {p₁-sep, ..., p₁+sep, p₂-sep, ..., p₂+sep}

For p₃: not in blacklist → SELECT
  Selected: [p₂, p₁, p₃]

Final: sort by position → [p₁, p₂, p₃]
```

### Segment Mean Computation

```
From Changepoints to Fingerprint
────────────────────────────────

Changepoints: [p₁, p₂, p₃]
Boundaries:   [0, p₁, p₂, p₃, n]

Signal:
  │ ████│▄▄▄▄▄▄▄│██████│▁▁▁▁▁▁▁│███████████
  │ seg0│ seg1  │ seg2 │ seg3  │   seg4
  └──────────────────────────────────────────▶

Fingerprint computation:
  fp[0] = mean(signal[0:p₁])
  fp[1] = mean(signal[p₁:p₂])
  fp[2] = mean(signal[p₂:p₃])
  fp[3] = mean(signal[p₃:n])

Fingerprint = [fp₀, fp₁, fp₂, fp₃]
            = [1.23, -0.45, 0.87, -0.12]
```

---

## Dynamic Time Warping (DTW)

DTW measures the similarity between two sequences that may have different lengths or temporal variations.

### Basic Algorithm

```
DTW Cost Matrix
───────────────

Given:
  Query Q = [q₀, q₁, q₂, q₃]
  Reference R = [r₀, r₁, r₂]

Build cost matrix D where D[i,j] = min cost to align Q[0:i] with R[0:j]:

        r₀    r₁    r₂
      ┌─────┬─────┬─────┐
  q₀  │  ●──┼─────┼─────┤  D[i,j] = cost(qᵢ, rⱼ) + min( D[i-1,j-1],  ← match
      │  │╲ │     │     │                                D[i-1,j],    ← insertion
      ├──┼─╲┼─────┼─────┤                                D[i,j-1] )   ← deletion
  q₁  │  │  │  ●──┼─────┤
      │  │  │  │╲ │     │  cost(qᵢ, rⱼ) = |qᵢ - rⱼ|
      ├──┼──┼──┼─╲┼─────┤
  q₂  │  │  │  │  │  ●──┤
      │  │  │  │  │  │╲ │
      ├──┼──┼──┼──┼──┼─╲┤
  q₃  │  │  │  │  │  │  ●  DTW distance = D[n-1, m-1]
      └──┴──┴──┴──┴──┴──┘

Allowed moves from (i,j):
  ↘ (i+1, j+1)  diagonal (match/substitute)
  ↓ (i+1, j)    vertical (insertion in Q)
  → (i, j+1)    horizontal (insertion in R)
```

### Memory-Efficient Implementation

```
Two-Row DTW Computation
───────────────────────

Only need previous row to compute current row:

Iteration i=0:
  prev: [0, ∞, ∞, ∞]
  curr: [∞, d₀₀, d₀₁, d₀₂]

Iteration i=1:
  prev: [∞, d₀₀, d₀₁, d₀₂]
  curr: [∞, d₁₀, d₁₁, d₁₂]

Iteration i=2:
  prev: [∞, d₁₀, d₁₁, d₁₂]
  curr: [∞, d₂₀, d₂₁, d₂₂]

Space complexity: O(m) instead of O(nm)
```

### Sakoe-Chiba Band

The Sakoe-Chiba constraint limits the warping path to a diagonal band, reducing computation:

```
Sakoe-Chiba Band Constraint
───────────────────────────

Full DTW matrix (window = ∞):        With window = 1:
      ┌───┬───┬───┬───┬───┐          ┌───┬───┬───┬───┬───┐
  q₀  │░░░│░░░│░░░│░░░│░░░│      q₀  │░░░│░░░│   │   │   │
      ├───┼───┼───┼───┼───┤          ├───┼───┼───┼───┼───┤
  q₁  │░░░│░░░│░░░│░░░│░░░│      q₁  │░░░│░░░│░░░│   │   │
      ├───┼───┼───┼───┼───┤          ├───┼───┼───┼───┼───┤
  q₂  │░░░│░░░│░░░│░░░│░░░│      q₂  │   │░░░│░░░│░░░│   │
      ├───┼───┼───┼───┼───┤          ├───┼───┼───┼───┼───┤
  q₃  │░░░│░░░│░░░│░░░│░░░│      q₃  │   │   │░░░│░░░│░░░│
      ├───┼───┼───┼───┼───┤          ├───┼───┼───┼───┼───┤
  q₄  │░░░│░░░│░░░│░░░│░░░│      q₄  │   │   │   │░░░│░░░│
      └───┴───┴───┴───┴───┘          └───┴───┴───┴───┴───┘

  Time: O(n × m)                     Time: O(n × w)

Constraint: |i - j| ≤ window

Benefits:
  - Prevents excessive warping
  - Faster computation
  - Often improves classification accuracy
```

### DTW for Classification

```
Classification with DTW
───────────────────────

Query fingerprint: Q

References:
  barcode_01: R₁ = [0.5, -0.3, 0.8, ...]
  barcode_02: R₂ = [0.2, 0.4, -0.1, ...]
  barcode_03: R₃ = [-0.4, 0.7, 0.2, ...]
  barcode_04: R₄ = [0.6, -0.5, 0.9, ...]

Compute distances:
  d₁ = DTW(Q, R₁) = 0.23
  d₂ = DTW(Q, R₂) = 0.87
  d₃ = DTW(Q, R₃) = 0.45
  d₄ = DTW(Q, R₄) = 0.91

Sort: [d₁, d₃, d₂, d₄] = [0.23, 0.45, 0.87, 0.91]

Confidence ratio:
  ratio = d_best / d_second_best
        = 0.23 / 0.45
        = 0.51

Decision (threshold = 0.8):
  ratio (0.51) < threshold (0.8)
  → Confident: assign to barcode_01
```

---

## Signal Normalization

### MAD (Median Absolute Deviation)

```
MAD Normalization
─────────────────

Given signal: [s₀, s₁, s₂, ..., sₙ]

Step 1: Compute median
  median = median([s₀, s₁, s₂, ..., sₙ])

Step 2: Compute MAD
  deviations = [|s₀ - median|, |s₁ - median|, ...]
  MAD = median(deviations)

Step 3: Normalize
  normalized[i] = (s[i] - median) / MAD

Properties:
  - Robust to outliers (unlike z-score)
  - Centers signal around 0
  - Scales by robust spread measure
```

### Fingerprint Normalization

```
Normalization Methods for Fingerprints
──────────────────────────────────────

Z-score:
  μ = mean(fp)
  σ = std(fp)
  normalized[i] = (fp[i] - μ) / σ

Min-Max:
  normalized[i] = (fp[i] - min(fp)) / (max(fp) - min(fp))

Median:
  med = median(fp)
  MAD = median(|fp[i] - med|)
  normalized[i] = (fp[i] - med) / MAD

None:
  normalized[i] = fp[i]  (no transformation)
```

---

## Performance Considerations

### Parallel Processing

```
Parallel DTW Distance Matrix
────────────────────────────

For N queries and M references:

Sequential:
  for i in 0..N:
    for j in 0..M:
      D[i,j] = dtw(Q[i], R[j])

  Time: O(N × M × n × m)

Parallel (rayon):
  (0..N).par_iter().flat_map(|i| {
    (0..M).map(|j| dtw(Q[i], R[j]))
  })

  Time: O(N × M × n × m / num_threads)

Block-based parallel (for very large matrices):
  - Divide into blocks of size B × B
  - Process blocks in parallel
  - Better cache locality
```

### Early Stopping in LLR

```
Early Stopping Optimization
───────────────────────────

Monitor derivative of gain curve:
  if mean_derivative(window) < 0:
    stop_searching

         gain
           │     ╭─╮
           │    ╱   ╲
           │   ╱     ╲  ← derivative becomes negative
           │  ╱       ╲
           │ ╱         ╲
           └──────────────────▶ position
                 ▲
                 │
            Stop here (past peak)

Benefits:
  - Avoids unnecessary computation
  - Useful for very long signals
  - Configurable window and stride
```

---

## Library API

### LLR Module

```rust
use escapepod::segmentation::{LlrTrace, detect_adapter};

// Create trace from signal
let trace = LlrTrace::new(&signal, stride);

// Compute gains for a range
let gains = trace.compute_gains(start, end, min_obs, border_trim);

// Find best split
let (position, gain) = trace.best_split(start, end, min_obs, border_trim)?;

// Full adapter detection
let (adapter_start, adapter_end) = detect_adapter(&signal, min_obs, border_trim);
```

### T-test Module

```rust
use escapepod::segmentation::{
    windowed_ttest,
    find_changepoints,
    compute_segment_means,
    segment_signal,
};

// Compute t-scores
let t_scores = windowed_ttest(&signal, window_width);

// Find changepoints
let changepoints = find_changepoints(&signal, window_width, num_changepoints, min_separation);

// Get segment means
let segments = compute_segment_means(&signal, &changepoints);

// All-in-one
let segments = segment_signal(&signal, window_width, num_changepoints, min_separation);
```

### DTW Module

```rust
use escapepod::dtw::{dtw_distance, dtw_distance_matrix};

// Single distance
let distance = dtw_distance(&query, &reference, Some(window));

// Distance matrix (parallel)
let matrix = dtw_distance_matrix(&queries, &references, Some(window));
```

---

## References

1. **ADAPTed**: van der Toorn, W.K. et al. "Adapter and poly(A) Detection And Profiling Tool"
2. **Tombo**: Oxford Nanopore Technologies. Signal-level analysis for modified bases
3. **DTW**: Sakoe, H. & Chiba, S. (1978). "Dynamic programming algorithm optimization for spoken word recognition"
4. **WarpDemuX**: Barcode demultiplexing using signal-level analysis

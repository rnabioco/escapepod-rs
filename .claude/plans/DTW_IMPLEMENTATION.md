# DTW Module Implementation Summary

## Overview

This implementation adds a complete Dynamic Time Warping (DTW) distance computation module to the escapepod library, inspired by WarpDemuX for nanopore barcode demultiplexing.

## Module Structure

The DTW module is located at `escapepod/src/dtw/` and consists of three main components:

### 1. Core DTW Algorithm (`dtw.rs`)

**Key Functions:**
- `dtw_distance(a, b, window)` - Compute DTW distance between two sequences with optional Sakoe-Chiba band constraint
- `dtw_distance_matrix(queries, references, window)` - Parallel computation of full distance matrix using rayon
- `dtw_distance_matrix_blocked(queries, references, window, block_size)` - Block-based parallel computation for very large matrices

**Features:**
- Classic DTW recurrence relation: `D[i,j] = dist(a[i], b[j]) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])`
- Sakoe-Chiba band constraint for faster computation when window is specified
- Memory-efficient implementation using two rows instead of full matrix
- Parallel computation using rayon for distance matrices
- Block-based parallelization following the WarpDemuX approach

**Complexity:**
- Time: O(n*m) without window, O(n*w) with window width w
- Space: O(m) for single distance computation

### 2. Kernel Conversion (`kernel.rs`)

**Key Functions:**
- `distance_to_kernel(distances, gamma, power)` - Convert distance matrix to RBF kernel
- `distance_to_kernel_auto(distances, power)` - Auto-estimate gamma using median heuristic

**Features:**
- RBF kernel: `K[i,j] = exp(-gamma * D[i,j]^power)`
- Median heuristic for automatic gamma estimation: `gamma = 1 / (2 * median(distances)^2)`
- Suitable for use with SVM classifiers

### 3. Fingerprint Utilities (`fingerprint.rs`)

**Key Types:**
- `Fingerprint` - Represents a barcode fingerprint with normalized feature values
- `NormMethod` - Enumeration of normalization methods (ZScore, MinMax, Median, None)

**Key Functions:**
- `normalize_fingerprint(fp, method)` - Normalize fingerprint using specified method

**Features:**
- Z-score normalization: `(x - mean) / std`
- Min-max normalization: `(x - min) / (max - min)`
- Median normalization: `(x - median) / MAD`
- Helper functions for statistical computations (mean, std, median, MAD)

## Dependencies Added

- `ndarray = "0.16"` - For efficient 2D array operations in distance matrices

## Testing

The implementation includes comprehensive unit tests covering:

### DTW Algorithm Tests:
- Identical sequences return distance of 0
- DTW is symmetric
- Known distance examples verify correctness
- Window constraint works correctly
- Empty sequences return infinity
- Distance matrix computation
- Block-based matrix computation matches standard approach
- Alignment with different sequence lengths

### Kernel Conversion Tests:
- Correct RBF kernel computation
- Power parameter effects
- Gamma parameter effects
- Kernel values in valid range (0, 1]
- Diagonal elements are 1.0 for zero distances
- Kernel matrix symmetry
- Auto gamma estimation

### Fingerprint Tests:
- Z-score normalization (mean ≈ 0, std ≈ 1)
- Min-max normalization (range [0, 1])
- Median normalization
- No normalization preserves values
- Empty fingerprints handled correctly
- Constant values handled correctly
- Statistical helper functions (mean, std, median, MAD)

**Test Results:**
- All 25 DTW module tests pass
- All 82 escapepod library tests pass
- All 11 documentation tests pass
- Zero compiler warnings
- Clean cargo clippy output

## API Examples

### Basic DTW Distance
```rust
use escapepod::dtw::dtw_distance;

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![1.0, 2.0, 3.0, 4.0];
let distance = dtw_distance(&a, &b, None);
assert_eq!(distance, 0.0);
```

### DTW with Window Constraint
```rust
use escapepod::dtw::dtw_distance;

let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let distance = dtw_distance(&a, &b, Some(2)); // Sakoe-Chiba band width = 2
```

### Distance Matrix Computation
```rust
use escapepod::dtw::dtw_distance_matrix;

let queries = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
let references = vec![vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0]];
let matrix = dtw_distance_matrix(&queries, &references, None);
```

### Kernel Conversion
```rust
use escapepod::dtw::{dtw_distance_matrix, distance_to_kernel};

let queries = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
let references = vec![vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0]];

let distances = dtw_distance_matrix(&queries, &references, None);
let kernel = distance_to_kernel(&distances, 1.0, 1.0);
```

### Fingerprint Normalization
```rust
use escapepod::dtw::{Fingerprint, normalize_fingerprint, NormMethod};
use uuid::Uuid;

let mut fp = Fingerprint::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Uuid::nil());
normalize_fingerprint(&mut fp, NormMethod::ZScore);
```

## Integration with Escapepod

The DTW module is exposed as a public module in `escapepod/src/lib.rs`:
```rust
pub mod dtw;
```

All public functions and types are re-exported from `escapepod/src/dtw/mod.rs`:
```rust
pub use dtw::{dtw_distance, dtw_distance_matrix, dtw_distance_matrix_blocked};
pub use fingerprint::{normalize_fingerprint, Fingerprint, NormMethod};
pub use kernel::{distance_to_kernel, distance_to_kernel_auto};
```

## Performance Characteristics

- **Parallel Computation**: Uses rayon for parallel distance matrix computation
- **Memory Efficiency**: DTW distance computation uses O(m) space instead of O(n*m)
- **Window Constraint**: Sakoe-Chiba band reduces time complexity from O(n*m) to O(n*w)
- **Block-Based Processing**: Enables efficient computation of very large distance matrices

## Reference Implementation

This implementation is inspired by WarpDemuX:
- Parallel distance computation pattern from `warpdemux/parallel_distances.py`
- Kernel conversion from `warpdemux/models/dtw_svm.py`
- Classic DTW algorithm with Sakoe-Chiba band constraint

## Future Enhancements

Potential improvements for future versions:
1. GPU acceleration for very large distance matrices
2. Additional DTW variants (weighted DTW, subsequence DTW)
3. Integration with machine learning libraries for classification
4. Optimized C/SIMD implementation for core DTW loop
5. Streaming DTW for online barcode classification

## Files Created/Modified

**New Files:**
- `escapepod/src/dtw/mod.rs` - Module definition and exports
- `escapepod/src/dtw/dtw.rs` - Core DTW algorithm (317 lines)
- `escapepod/src/dtw/kernel.rs` - Kernel conversion utilities (208 lines)
- `escapepod/src/dtw/fingerprint.rs` - Fingerprint normalization (243 lines)
- `examples/dtw_example.rs` - Comprehensive usage examples (133 lines)

**Modified Files:**
- `escapepod/Cargo.toml` - Added ndarray dependency
- `escapepod/src/lib.rs` - Exposed dtw module

**Total New Code:** ~900 lines including tests and documentation

# Resquiggle Integration Test: Progress Notes

## Overall Plan (4 tasks)

1. **Generate fishnet reference output** - DONE (`data/drna/fishnet_yeast_trna_query.parquet`, 179 reads)
2. **Write Rust integration test** - DONE (97.4% mean boundary match, passes >=95% threshold)
3. **Create benchmarking script** - NOT STARTED
4. **File GitHub issue for DNA resquiggle support** - NOT STARTED

## Task 2: Integration Test — COMPLETE

**File**: `escapepod/tests/test_resquiggle.rs`

### Final result
- 179 reads compared, 1 skipped, 0 panics
- **97.4% mean boundary match** (threshold: 95%)
- Min: 0.0% (ccb15034, pathological read) — excluded from failure by mean threshold
- All tests pass: `cargo test --release -p escapepod --test test_resquiggle`

### Root causes found and fixed

#### 1. Missing rough rescaling (1.3% → 76.1%)
**Problem**: `RoughRescaleAlgo` defaulted to `None`. Without rough rescaling, the initial
normalization produced signal values ~-6.0 while kmer levels were ~0-1 (raw pA space).
The DP had no discriminating power.

**Fix**: Changed `RoughRescaleAlgo::default()` to `TheilSen` with fishnet-matching
parameters (quantiles 0.05-0.95, clip_bases=10, use_base_center=true). After rough
rescaling, signal values align with level ranges (~[-1, 2]).

**Files changed**: `escapepod/src/resquiggle/types.rs`

#### 2. Too few refinement iterations (76.1% → 97.4%)
**Problem**: `n_refinement_iters` defaulted to 1. Fishnet's CLI defaults to 2 iterations.
With 2 iterations, the first DP provides a rough alignment, then rescaling adjusts
shift/scale using that alignment, and the second DP produces a much more accurate result.

**Fix**: Changed `n_refinement_iters` default to 2 in `RefineSettings::default()` and
the CLI argument.

**Files changed**: `escapepod/src/resquiggle/types.rs`, `escapepod-cli/src/commands/resquiggle.rs`

#### 3. DP overflow bug at dp.rs:353
**Problem**: `path[base_idx] = sig_lookup_pos - (next_sig_offset as usize)` overflowed
when traceback value was -1 (invalid sentinel). In debug mode: panics. In release: wraps
to garbage values.

**Fix**: Added check for `next_sig_offset >= 0` before the subtraction, falling back to
`band_start` for invalid traceback cells.

**File changed**: `escapepod/src/resquiggle/dp.rs`

### Investigation that ruled out other causes
- **Level orientation**: Fishnet does NOT reverse the query sequence for RNA. Levels are
  extracted from the original BAM sequence. The test approach was already correct.
- **BAM sm/sd tags**: Present and correct (sm=778.445, sd=116.091). The normalization
  formula is identical to fishnet.
- **Kmer dominant_base**: Both implementations use the same Kruskal-Wallis H test with
  identical encoding conventions. Fishnet's dominant_base=6 was from a different test table.
- **DP scoring**: Unit tests confirm scores match fishnet's expected values exactly.
- **Band computation**: Same algorithm, same half_bandwidth=5, same adjust_band_min_size=2.

## Key files
- `escapepod/tests/test_resquiggle.rs` - the integration test
- `escapepod/src/resquiggle/dp.rs` - DP with overflow fix
- `escapepod/src/resquiggle/refine.rs` - refinement pipeline
- `escapepod/src/resquiggle/types.rs` - default settings (rough rescale + 2 iterations)
- `data/drna/fishnet_yeast_trna_query.parquet` - fishnet reference (179 reads)
- `data/drna/yeast_trna_reads.pod5` - POD5 test data (180 reads)
- `data/drna/yeast_trna_mappings.bam` - BAM test data

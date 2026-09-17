# escapepod-signal

## Purpose and layering
- Signal algorithms (DTW, resquiggle, segmentation, chunk assembly, LSTM kernel, coordinate mapping, k-mer tables) over `escapepod-pod5`, which it depends on and re-exports as `pod5` plus the type surface.
- Consumed by demux, classify, cli, python — and out-of-tree by **leech** (`rust/Cargo.toml` pins this crate by git tag). Leech-load-bearing but unused in-workspace: `rough_rescale_quantile`, `ref_to_signal`, `span_stats`, `extract_levels`, `banded_dp_with_penalty_table`, `sequence_bases_with_context`, `encode_signal_kmer`/`KmerContext`, `encode_signal_kmer_batch[_into]`/`SignalKmerBatch` (#380). Grep leech before deleting any `pub` item.

## Module map
- `dtw/` — `dtw_distance*`; lane-parallel `dtw_distances_batch` (SoA, `DTW_LANES=32`); `Fingerprint`; `cuda/` (`gpu`): `GpuDtwContext` DTW+SVM kernels, prep kernels experimental.
- `resquiggle/` — `refine_signal_map` (rough rescale → banded DP → rescale loop); `dp/` fill + `fill_simd`; `adaptive_dp`; `bands`; `rescale`; `types` (`RefineSettings::move_table_refinement` is *the* shared preset); `kmer_table` (`KmerTable`, fishnet); `kmer_levels` (Remora/leech); `dp_raw_penalty` (leech parity DP).
- `segmentation/` — `llr::detect_adapter`, `ttest::segment_signal`, `normalize::downscale_normalize_into` + MAD helpers.
- `chunk.rs` — `process_read` → `read_rows` → `cut_chunk`; channel lists in `ChunkSpec`; `signal_kmer_inputs` (pub, encoding-independent — a caller of `cut_chunk` recovers `sig_start`/`sig_end` from `Chunk::focus_signal_pos` + `spec.signal_context`, but must then run that pair through `placed_window` before calling `signal_kmer_inputs`: `place_window` centre-crops when the requested width exceeds `spec.signal_len`, and the raw pair only equals the placed window when it doesn't — rnabioco/escapepod-rs#388).
- `features.rs` — `span_stats` + `SpanConfig` (fill/bounds/median named per call).
- `lstm.rs` — `LstmWeights::from_onnx`, `run_scalar`/`run_avx2`/`run_*_batch::<N>`, `LstmBackend::best_for`, `tanh_slice`.
- `mapping.rs` — `seq_to_signal_from_moves`, `ref_to_signal`. `seq_encoding.rs` — `base_to_int`, `KmerContext`, `encode_signal_kmer[_into]`, `encode_signal_kmer_batch[_into]` (`SignalKmerBatch`, CSR-packed, rayon).
- `stats.rs` — `median_via_select`, `median_and_mad[_with_scratch]`: every median/MAD routes here.

## Build, test, bench this crate alone
Cluster/allocator/linker policy: root CLAUDE.md "Build baseline and SLURM builds".
```bash
srun -p rna -c 32 --mem=32G pixi run cargo nextest run -p escapepod-signal
srun -p rna -c 32 --mem=32G pixi run cargo test --doc -p escapepod-signal      # nextest skips doctests
srun -p rna -c 32 --mem=32G pixi run cargo bench -p escapepod-signal --bench hot_paths -- dtw
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 pixi run -e dev-gpu cargo nextest run --features gpu -p escapepod-signal
```
- `hot_paths` env: `ESCAPEPOD_BENCH_THREADS` (matrix-bench pool), `ESCAPEPOD_BENCH_SAMPLES`.
- Self-skipping (green ≠ ran): `tests/test_resquiggle.rs` needs repo-root `data/drna/*`; `kmer_table` unit tests need repo-root `data/kmer_models/` — check nextest's count, a skip is silent. Parity tests use committed `tests/fixtures/*.json`.
- GPU-only (`cfg(feature = "gpu")`): `tests/gpu_{dtw,llr_detect,svb16}.rs`, `benches/hot_paths_gpu.rs`.

## Invariants and traps
- LSTM kernels are `#[inline(never)]`; dispatch keeps two *separate* lone-read arms, never an or-pattern (AVX2 width-2 miscompiled once) → `lstm::tests::batched_matches_single_bit_for_bit` under every `ESCAPEPOD_LSTM_BACKEND` cap.
- SIMD only via runtime `is_x86_feature_detected!`/`supported()`; a baseline bump SIGILLs rna/Alpine.
- `run_avx2` ≡ `run_avx2_batch::<N>` ≡ `run_avx512_batch::<N>` bit-for-bit; scalar differs (Cephes exp, 1e-4) → `avx2_matches_scalar`.
- DTW batch lanes bit-identical to scalar at every width/window/penalty → `test_dtw_batch_matches_scalar`, `test_dtw_distances_batch_windowed_matches_scalar`.
- Dwell-DP sweep exact vs `prefix_scalar` under every `DpBackend` → `transposed_matches_prefix_scalar_exactly`; only 3e-3-equal to the pre-#356 reference.
- `dp_raw_penalty`: tie `<` (stay wins), horizon = table length. `dp_step_buffered`: `<=` (move wins), horizon `target.ceil().clamp(4,256)`. Leech parity — not foldable.
- `KmerTable` (f32, strict parse, Kruskal-Wallis dominant base: RNA004 9-mer = 3, not 4) vs `kmer_levels` (f64, explicit centre, unknown → 0) → `rna004_dominant_base_is_not_the_midpoint`, `packed_extraction_matches_the_map_path`.
- `MedianConvention::SortPartialCmp` propagates NaN (numpy); `SelectTotalCmp` returns finite → `the_median_conventions_disagree_on_a_nan_span`.
- `kmer_levels` quantiles subtract in f32 before the f64 lerp (numpy dtype rule); `rescale::calculate_quantiles` is a different lerp — not interchangeable → `test_numpy_quantile_probe_parity`.
- `ref_to_signal` is integer knots, not the `np.interp` float chain → `ref_to_signal_differs_from_the_float_chain`.
- `KmerContext.before/after` not interchangeable; spans intersect, never clamp-after-cast → `before_and_after_are_not_interchangeable`, `spans_are_intersected_with_the_window_at_both_ends`.
- `chunk` row order is the spec's, never fixed → `row_order_follows_the_spec`.
- `downscale_normalize_into` is downscale-then-MAD: LLR-invariant, **not** bit-identical to normalize-then-downscale; validate with a boundary parity run.

## Env levers
| var | effect | default | pinned |
|---|---|---|---|
| `ESCAPEPOD_LSTM_BACKEND=scalar\|avx2\|avx512` | caps LSTM kernel | best supported | tests sweep `available()` |
| `ESCAPEPOD_LSTM_BATCH=N` | caps lockstep width (≤3 AVX2, ≤8 AVX-512) | max | no |
| `ESCAPEPOD_DP_BACKEND=scalar\|avx512` | caps dwell-DP sweep (`avx2` warns, ignored) | best supported | tests sweep |
| `ESCAPEPOD_DTW_AVX512=1` | AVX-512 batch DTW, slower everywhere | off | yes; scheduled removal |
| `ESCAPEPOD_BENCH_THREADS/SAMPLES` | `hot_paths` pool / sample count | rayon / criterion | bench only |

## Known debt
- `(backend, N)` dispatch table written thrice: `lstm.rs:1088-1230`, `classify/fnn_lstm.rs:437-505`, `demux/crf/encoder_native.rs:385-552` → one `lstm::run_batch`.
- `normalize.rs:101-170` `normalize_downscale_into` reachable only at factor 1 (`:217-221`) — fold into `downscale_normalize_into`.
- `dp/fill.rs:437` `_weight`, `dp/mod.rs:42` `dwell_weight` dead in production; `DpContext` Option+unwrap → enum; `fill.rs:74-92` `dp_step` test-only.
- `lstm.rs:593-607` unreachable tail (`h%8==0 ⇒ g%32==0`); `lstm.rs:394-502` `run_avx2` vs `::<1>` bench-gated (`charging`, `ESCAPEPOD_LSTM_BATCH=1`).
- `fingerprint.rs:190-212` re-implements `median_and_mad_with_scratch`; `chunk.rs:993-1007` re-spells the alphabet, `:580` allocates a String per read, `:952` ignores `encode_signal_kmer_into`.
- **Scheduled removal, minor bump, leech re-verify:** `dtw/kernel.rs`; `dtw_distance_matrix_blocked` (`distance.rs:785-835`); `dtw_distance_penalty`; `Fingerprint::{to_feature_vector,to_interleaved_features,has_dwell_times}`; `mad_normalize_with_clipping`, `normalize_dwell_times_mad`; `llr.rs:103-205` gains fns; `ttest.rs:343-423` `segment_signal_with_dwell`/`SegmentationResult`; `sequence_ints_with_context`; `ESCAPEPOD_DTW_AVX512` (`distance.rs:410-455`).
- GPU prep kernels (`cuda/mod.rs:323-756` + four `*_kernel.rs`) will be feature-gated and compiled lazily; today `new_on_device:132-154` NVRTC-compiles all six modules on every production `GpuDtwContext::new()`.

## Do not
- Bump `target-cpu`/`target-feature`: SIGILL on rna and Alpine.
- Inline a SIMD kernel or merge lone-read arms: miscompile evidence.
- Change `DTW_LANES` without the 851/3404-ref measurement: latency-bound, measured.
- Delete a `pub` item on in-workspace evidence alone: leech pins by tag.
- Add a second median/MAD/quantile: parity tests route through `stats`.
- Hard-code chunk channel order: the bundle declares it.
- Re-try hand-AVX2 dwell DP, AVX-512 DTW, GPU prep offload: measured losses.
- Copy pod5/bam/parquet fixtures under this crate: root storage rules.

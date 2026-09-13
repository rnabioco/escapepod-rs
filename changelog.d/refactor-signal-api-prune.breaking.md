- `escapepod-signal`'s public API surface is reduced: removed
  `dtw::kernel` (`distance_to_kernel`/`distance_to_kernel_auto`) — demux has
  its own equivalent (`svm::kernel::distances_to_kernel_into`), and the only
  in-workspace caller was a demonstration in `examples/dtw_example.rs`, now
  updated; removed `dtw::distance::dtw_distance_matrix_blocked` — superseded
  by the parallel `dtw_distance_matrix` and exercised only by its own test;
  removed `dtw::distance::dtw_distance_penalty` — a thin wrapper over
  `dtw_distance_bounded_penalty` exercised only by in-crate tests; removed the
  `ESCAPEPOD_DTW_AVX512` lever (`dtw::distance`'s AVX-512 batch DTW kernel) —
  measured slower than the AVX2 baseline on every cluster CPU and referenced
  nowhere else; removed `dtw::Fingerprint::{to_feature_vector,
  to_interleaved_features, has_dwell_times}` — unused conversions superseded
  by DTW distance directly on `Fingerprint::values`; removed
  `segmentation::{mad_normalize_with_clipping, normalize_dwell_times_mad}` —
  unused MAD-clipping/dwell-normalization variants; removed
  `segmentation::LlrTrace::{compute_gains, compute_gains_with_early_stop}` —
  superseded by `best_split`, which tracks the argmax inline instead of
  materializing a gains vector; removed
  `segmentation::{segment_signal_with_dwell, SegmentationResult}` — demux
  derives dwell times itself from `segment_signal`'s `(start, end, mean)`
  tuples; removed `seq_encoding::sequence_ints_with_context` — leech (the
  out-of-tree, git-tag-pinned consumer) uses only the bases form
  (`sequence_bases_with_context`) and composes `sequence_to_int` itself when
  it needs ints. Verified against leech's checkout at
  `~/devel/rnabioco/leech` (pinned to this crate at tag `v0.24.3`): none of
  the removed items appear in its `rust/` sources or Python bindings, and
  every item on `escapepod-signal/CLAUDE.md`'s leech-load-bearing exclusion
  list (`rough_rescale_quantile`, `ref_to_signal`, `span_stats`,
  `extract_levels`, `banded_dp_with_penalty_table`,
  `sequence_bases_with_context`, `encode_signal_kmer`/`KmerContext`) is
  confirmed still in use there and was left untouched.

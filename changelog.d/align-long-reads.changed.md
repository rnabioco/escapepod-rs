- **`escpod align` no longer stalls on very long reads.** A tRNA sample's
  handful of 50–400 kb reads used to take seconds each on one thread while the
  in-order writer — and, once the permit window filled, the whole pool —
  waited for them. Reads of 1,024 nt or more are now scored by a
  reference-major ("transposed") layout of the SIMD score kernel, whose state
  is one panel row (cache-resident at any read length) instead of one read
  column (50 MB per lane group for a 395 kb read): 2.3× faster on that read on
  one core. A read of 16 kb or more also has its panel scored across the pool,
  one lane group per task: 0.29 s wall for that read, against 3.15 s. On a
  490 k-read sample: 19.2 → 15.2 s wall on 16 rna cores (avx512), and
  11.1 → 7.7 s with `--device gpu` on an A30 — within 0.2 s of the same run
  without its long reads. Output is byte-identical;
  `ESCAPEPOD_ALIGN_ROW_MAJOR_ONLY=1` restores the old path for A/B
  measurement. `escapepod_align::Aligner` gains `score_groups`/`score_group`
  (per-lane-group scoring for a caller with threads to spare), `kernel_for`
  and `with_transposed_min_len` (#415).

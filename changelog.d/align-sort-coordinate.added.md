- **`escpod align --sort coordinate` writes a coordinate-sorted BAM itself**,
  so the `samtools sort` pass after it goes away (#417). The order is exactly
  `samtools sort`'s — reference, position, forward before reverse, then input
  order, unmapped reads last — and the header differs from the unsorted
  output only in `SO:coordinate` and the `@PG CL`. It is a per-reference
  bucket sort over the records the workers already encode, with no merge:
  past `--sort-memory` (default 4G) the largest buckets spill to one unlinked
  temporary file in `--tmp-dir` (default beside the output), and peak memory
  is the budget plus the largest single reference's records. The default
  stays `--sort unsorted`; no `.bai` is written.

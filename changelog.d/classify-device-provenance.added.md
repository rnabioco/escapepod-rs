- **`escpod classify`'s output BAM now records which device scored it.** The
  `@PG` `DS` JSON gains a `device` object: `requested` (`"cpu"`/`"gpu"`, the
  device that *actually* scored the reads — never merely the one a GPU
  scorer loaded for, whether that's because the run was smaller than one GPU
  batch or because a real-batch parity check triggered a full CPU rescore),
  and, whenever the GPU was involved at all, the cuBLAS/cuBLASLt pairing
  `ensure_cublas_pairing()` resolved (#416, including the physical library
  paths, not just versions) and the run's GPU/CPU parity check summary
  (batches scored, batches checked, worst `|ΔP|`). All of this used to be
  computed and then only `tracing::debug!`'d, which meant an after-the-fact
  audit of whether a given output could have been hit by #416 depended
  entirely on a caller's own log retention or Slurm accounting, neither of
  which escpod controls (#425). The DS JSON also gains `positive_class`
  (which class `cl`/`operating_point.probability` are stated on), and `@PG`
  `CL` now records the real invoked command line, mirroring
  `align`/`resquiggle` (#409), instead of a hand-templated
  `--model X (cl = ...)` string that never varied with `--device` and never
  said which class it scaled.

- **`escpod align` scores on a CUDA GPU** (#401). In a `--features gpu` build,
  `--device` now places `align`'s panel scoring (`Stage::Align`): a CUDA kernel
  in `escapepod-align` (feature `gpu`, cudarc/NVRTC like the DTW kernel, nothing
  CUDA needed at build time) scores a whole batch of reads against every
  reference, one GPU thread per (read, reference) pair, with the panel in shared
  memory at 4 bits a base and each read walked in 16-row register stripes. Only
  the scores come from the device: the tie set, the tracebacks, `MD`/`NM` and
  the records are the CPU path's own code (`Aligner::map_reads_scored`), so
  `--device gpu` output is byte-identical to `--device cpu`, and the kernel is
  tested for equality against the scalar oracle. Reads over 4,096 nt are scored
  on the CPU inside the same run; a panel with a reference over 3,072 nt cannot
  use the GPU. On one gpu node (`-c 16`, A30) a 3.3 M-read sample aligns in
  ~70 s and ~700 CPU-seconds against 184 s and ~2,900 on the CPU, so
  `--device auto` uses the GPU when one is visible; `--device gpu` is no longer
  refused.

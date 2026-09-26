- **`escpod classify --device gpu` no longer returns wrong charging
  probabilities on nodes with a host CUDA toolkit in the `ldconfig` cache.**
  cudarc opens the unversioned `libcublas.so` first; a CUDA runtime
  environment does not ship that name, so the host's cuBLAS (12.8 on
  compgpu03) was loaded and paired with the environment's cuBLASLt (12.9).
  That mix raises no error and computes wrong GEMMs — the TCN's GPU output
  correlated 0.009 with the CPU's (13.1% charged against 1.5%). The GPU
  loader now checks which release each loaded file belongs to and reloads
  cuBLAS against its own cuBLASLt, or refuses the GPU when it cannot (#416).
- **The windowed charging classifier's GPU path now checks itself against
  the CPU on real reads.** The first GPU batch of every run, and one in every
  64 after it (`ESCAPEPOD_WAVEFORM_GPU_PARITY_EVERY`), is re-scored on the
  CPU; a read off by more than 1e-3 in P(charged) refuses the GPU —
  an error under `--device gpu`, a warning and a whole-run CPU fallback under
  `--device auto`.

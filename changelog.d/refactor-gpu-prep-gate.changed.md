- **`escapepod-signal`'s GPU prep kernels (SVB16 decode, t-test fingerprint,
  LLR adapter detect) now require a new `gpu-prep-experimental` feature**,
  which the existing `gpu` feature does not imply. Every production
  `GpuDtwContext::new()` call site (escapepod-demux's `train.rs`,
  `svm/gpu.rs`, `classify.rs`; escapepod-cli's `demux/run.rs`,
  `demux/classify.rs`) NVRTC-compiled all six kernel modules on every call —
  DTW, SVM, and three prep modules with no production caller anywhere in the
  workspace — despite only ever using the DTW + SVM ones. `new_on_device`
  now compiles just the DTW + SVM modules under plain `gpu`, and additionally
  the three prep modules when `gpu-prep-experimental` is also enabled. Public
  API used by production code (`GpuDtwContext` and its DTW/SVM methods) is
  unaffected under plain `gpu`. `escapepod-demux` gained a same-named feature
  forwarding to `escapepod-signal/gpu-prep-experimental` (kept separate from
  its atomic `gpu` flag on purpose) for `tests/gpu_fingerprint.rs` and
  `examples/gpu_detect_timing.rs`; `escapepod-signal`'s
  `tests/gpu_{svb16,llr_detect}.rs` now require the same feature. Measured
  on an A30: `GpuDtwContext::new()` drops from ~280ms to ~181ms warm (~35%),
  ~1.12s to ~535ms cold.

- **Three deprecated CLI aliases, scheduled for removal, are gone.** Each had
  a replacement that has worked since the alias was introduced:
  - `escpod signal classify` (deprecated since 0.19.0) — use `escpod classify`.
  - `--gpu` (deprecated since 0.17.0) — use `--device gpu`.
  - `demux classify --svm-model` — use `demux classify --model` (auto-detects
    `DtwSvmModel` vs the legacy `WarpDemuxModel` JSON shape).

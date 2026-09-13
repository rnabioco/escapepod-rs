- `escapepod-demux::train`'s binary and multiclass SVM-fit code paths are
  collapsed into one `fit_from_labels`, and the dead `demux train-svm --c`
  flag is removed. Both fit paths used to build `DtwSvmModel` with
  `use_kernel_weighted: true`, and every reader of that flag
  (`SvmPredictor::decision_function`/`decision_function_into`, `svm::gpu`'s
  host-side coefficient table) branches on it *before* looking at
  `dual_coef`, computing class scores from `training_labels` alone — so the
  binary path's single uniform `dual_coef` row and the multiclass path's
  per-pair OvO layout were two constructions of a value nothing downstream
  distinguishes; one construction, generalized to `n_classes >= 2`, replaces
  both. `--c`/`TrainConfig.c` was stored and echoed back in the CLI's own
  log line but never read by the fit — the `train` feature's SVM fit is a
  label-only stub with no `linfa-svm` (or other) dependency left to feed a
  regularization parameter to, so the flag could not have done anything
  since that dependency was dropped. `demux train-svm` now also logs a
  warning that the SVM fit is experimental and less maintained than the
  CRF/GBM demux paths.

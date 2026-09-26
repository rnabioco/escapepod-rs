- **Setting `ESCAPEPOD_FNN_TRACT`, `ESCAPEPOD_WAVEFORM_HOIST`,
  `ESCAPEPOD_WAVEFORM_GPU_HOIST`, `ESCAPEPOD_CRF_TRACT` or
  `ESCAPEPOD_CRF_DEBUG_RECOGNIZER` to `0` or `false` now turns it off
  (previously on).** Each used to test `var_os(..).is_some()`, so any set
  value — including `=0` — counted as "on", the opposite of what every other
  `ESCAPEPOD_*` switch and this repo's own documentation say `=1`/`=0` mean.
  All five, plus every other boolean switch and positive-integer knob, now
  route through the new `escapepod_pod5::env::flag`/`positive_usize`, which
  also warns once on a value that does not parse instead of silently falling
  back to the default with no trace.

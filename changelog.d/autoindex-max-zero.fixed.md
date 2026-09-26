- **`ESCAPEPOD_AUTOINDEX_MAX=0` skips read-index warm-up again.** 0.29.x's
  move of the knob onto `env::positive_usize` (#412) rejected `0` as a typo
  and fell back to the 5,000,000 default, so `=0` — the documented way to
  turn off speculative warm-up in `ReaderCache`, `DatasetCache` and the Python
  `Reader` context manager — quietly warmed every file again. It now reads
  through the new `escapepod_pod5::env::usize_allow_zero`, which accepts `0`
  and otherwise keeps `positive_usize`'s contract (unset/empty → default, a
  value that does not parse warns once and falls back). The Python `Reader`
  gains a read-only `index_resident` property, so the skip is now asserted,
  not just its read ids (#421).

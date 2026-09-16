- `escapepod_pod5::Dataset` / `DatasetCache` / `cached_dataset`: a pure-Rust,
  multi-file POD5 collection — open a file, a directory (recursive,
  glob-suffix filterable), or a mix of both as one random-access-by-read-id
  source, re-exported from `escapepod-signal`. Every file routes through the
  existing `ReaderCache`/`cached_reader`, so a file reachable both directly
  and via a directory scan shares one `Arc<Reader>` and one warmed index. The
  directory-scan and read-id-routing logic that used to exist only in
  `escapepod-python`'s `DatasetReader` (pyo3-specific, unreachable from a
  plain Rust dependency) is now reachable from `escapepod-pod5`/
  `escapepod-signal` directly; `PyDatasetReader` is refactored to wrap it
  instead of duplicating it (#384).

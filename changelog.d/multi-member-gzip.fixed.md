- **A bgzipped or concatenated `.gz` k-mer table, boundaries CSV, or reference
  CSV no longer reads as truncated after its first gzip member.**
  `flate2::read::GzDecoder` stops at the end of the first member; switched to
  `MultiGzDecoder` in `escapepod-signal`'s `KmerTable::from_file` and
  `load_kmer_table`, and in `escpod demux`'s boundaries-CSV reader
  (`crates/escapepod-cli/src/commands/demux/utils.rs`), so every member of a
  multi-member `.gz` input is read (#409).

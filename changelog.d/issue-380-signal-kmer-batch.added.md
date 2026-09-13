- `seq_encoding::encode_signal_kmer_batch`/`_into`: a batched entry point over
  `encode_signal_kmer`, CSR-packed so leech can marshal a whole training
  batch across the pyo3 boundary once (and pay one `(rows, channels, L)`
  allocation) instead of once per chunk. Every row is exactly
  `encode_signal_kmer`'s output, computed in parallel with rayon.
- `chunk::signal_kmer_inputs` is now `pub`, so a caller of `cut_chunk` can get
  the chunk-local `seq_to_signal` map and `N`-padded k-mer context sequence
  even when the chunk itself was cut with a different `SeqEncoding` (#380).

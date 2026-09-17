- **`cut_chunk`'s `SeqEncoding::SignalKmer` arm no longer misaligns its k-mer
  map when the requested `signal_context` window is wider than `signal_len`**
  (#388). `place_window` centre-crops such a window before copying it into
  `signal`, but the call into `signal_kmer_inputs` still used the raw,
  pre-crop `sig_start`/`sig_end` — so the emitted `map`'s offsets, and the
  `SignalKmer` sequence tensor built from them, were shifted by the crop
  amount relative to where the real signal actually landed. `place_window`'s
  crop arithmetic is now shared through a new `pub fn placed_window`
  (`chunk.rs`), which `cut_chunk` runs the window through before calling
  `signal_kmer_inputs`, so the two can no longer drift apart. Nothing in this
  repo's own test suite or shipped bundle geometry crosses the `signal_context
  > signal_len` threshold, so no existing output changes; the bug was only
  live via `leech`'s per-read `--signal-context-bases`.

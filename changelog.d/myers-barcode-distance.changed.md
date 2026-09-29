- **Barcode edit distance no longer depends on `fqxv-align`.** `demux basecall
  --barcodes` and the fused CRF demux path assign a decode to its closest
  reference with an in-tree Myers bit-vector Levenshtein instead of a
  pinned-rev git dependency that was used only for the distance. Distances are
  exact for every input (references longer than 64 nt or holding non-`ACGT`
  bytes take a DP fallback), so calls, margins and tie-breaking are unchanged.
  Drops the last git dependency, which blocked `cargo publish` and a Bioconda
  build.

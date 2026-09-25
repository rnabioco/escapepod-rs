- **`escpod align`: all-against-all alignment of reads to a small reference
  panel (tRNA), and the `escapepod-align` crate behind it** (#395). Every read
  is scored against every reference with an affine-gap local or overlap
  (`--mode semiglobal`) DP, the best reference is the primary, and references
  that tie with it are reported in `XA` (and, with `--secondary`, as 0x100
  records) with MAPQ 0 — isodecoder ambiguity as output, rather than bwa's 99%
  MAPQ-0 reads. Takes dorado's uBAM (or FASTA/FASTQ, plain or gzip), writes an
  input-ordered BAM with every input tag carried byte for byte (`mv`, `ns`,
  `ts`, `MM`/`ML`, `RG`, …), and writes `MD`/`NM` exactly as `samtools calmd`
  would — reference `N` included, measured and pinned by a golden — so it
  replaces the aa-tRNA-seq pipeline's `bwa mem` + `samtools calmd` steps
  without the FASTQ-comment tag round trip or the `-H` header hack. There is
  no seed index: memory is one batch of reads, not the 78–90 GiB bwa `-k 6`
  held on this panel. Scoring runs on inter-sequence AVX2/AVX-512BW kernels
  (one lane per reference, exact i16), the winners' tracebacks on a second
  pair whose lanes each carry their own read and reference, every kernel
  tested for equality against a scalar Gotoh that is itself pinned against
  parasail; `ESCAPEPOD_ALIGN_BACKEND=scalar|avx2|avx512` caps the choice.
  `--scoring M,X,O,E` (a gap of length k scores `O + (k-1)·E`; default
  `2,-1,-10,-1`), `--min-score`, `--max-ties`, `--strand forward|both`,
  `--read-ids`, `--batch-size`, and `--device` (no GPU stage yet — `gpu` is
  refused). The `escapepod-align` crate is std-only and has no I/O.

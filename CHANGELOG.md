# Changelog

## Unreleased

### Added

- **`escpod demux models` — pinned manifest, verified cache, offline resolution**
  (#158 item 1). Boundary and barcode model binaries are gitignored upstream and
  distributed through GitHub Releases, so there was previously **no supported way
  to obtain them from escpod at all**. Same shape as `escpod resquiggle models`
  does for k-mer tables:

  ```
  escpod demux models fetch wdx4_rna004    # on a networked login node
  escpod demux models list                 # what's known, what's cached
  escpod demux detect --method cnn --cnn-model-name adapter_rna004 …
  escpod demux classify --model-name barcode_wdx4_rna004 …
  ```

  `--cnn-model-name` / `--model-name` sit alongside the existing path flags,
  mirroring `--kmer-table` / `--kmer-model`. **Resolution never touches the
  network**: on this project's HPC target the compute nodes generally cannot
  reach GitHub, so a lazy fetch would hang a job rather than fail it. A missing
  model errors immediately and names the exact command to run.

  **The fetch unit is a bundle, not a model.** The barcode GBM is trained
  against a specific boundary model's output; using LLR boundaries instead costs
  17.2 points of balanced recall, and even swapping between two *good* boundary
  models costs 0.0059 (McNemar p=3.8e-08) unless the GBM is retrained. Fetching
  the matched release as a unit makes that coupling impossible to break by
  accident — which is most of what #158 item 2 asks for, obtained structurally
  rather than by a check. Members stay individually addressable for resolution;
  they just cannot be *fetched* apart.

  Each member's sha256 is verified after extraction against the value pinned in
  the manifest — the same value the release's `BUNDLE.json` and `provenance.json`
  publish, independently re-hashed from the artifacts before being recorded. The
  archive's own checksum is deliberately not pinned: re-packing the zip would
  change it without changing a model byte, so it would produce false failures
  while adding nothing.

  **Requires a GitHub token today.** `escapepod-models` is currently a *private*
  repository, so release assets 404 for anonymous requests — GitHub's advertised
  `browser_download_url` is unusable without credentials, and a private repo
  answers 404 rather than 401, which makes the bare failure deeply misleading.
  Fetching goes through the REST asset endpoint and sends a bearer token from
  `$GITHUB_TOKEN` or `$GH_TOKEN`; the anonymous 404 is rewritten to say exactly
  this. The same endpoint serves public repositories anonymously, so if the
  repository is opened up later this keeps working with no token and no code
  change. escpod's own test suite never fetches, so CI needs no secret.

### Changed

- **The default `escpod` binary now links a TLS stack** (`ureq`/rustls) and a
  zip reader, via the new `model-fetch` feature. Measured cost: **32.76 MB →
  34.74 MB stripped (+1.98 MB, +6.0%)**. This was previously avoided on purpose,
  and it is a deliberate reversal: without it `escpod demux models fetch` cannot
  exist in a released binary, and the models it serves are the
  difference between 0.9928 and 0.8196 balanced recall. rustls (not OpenSSL)
  keeps the static-musl release self-contained. `model-fetch` is separate from
  the existing `models-download` precisely so the demux fetch could ship by
  default without dragging in `experimental`, which gates the whole resquiggle
  command.

- **CTC-CRF barcode basecalling — `escpod demux basecall`.** escapepod-rs can
  now run a bonito-style CTC-CRF barcode model end to end: ONNX encoder
  inference plus a native lattice decode. Previously the decode did not exist
  in Rust at all, so the Stage-1 pipeline
  (rnabioco/escapepod-models#27) had to drop into Python between
  `escpod demux detect` and `escpod demux split`.

  The split follows what ONNX can express. The encoder (convolutions → 5 stacked
  LSTMs → `LinearCRFEncoder`) exports cleanly and runs through tract; the decode
  is a sparse forward/backward over 256 states × 5 edges that standard ONNX ops
  cannot express and that bonito itself only implements as hand-written CUDA
  (`koi`). So the encoder ships as an ONNX file and `escapepod_demux::crf`
  owns the decode — in portable Rust, with **no GPU requirement and no
  dependencies**, which also means CI exercises it.

  Correctness is pinned against bonito rather than asserted. `crf_golden.rs`
  replays a score tensor defined by a closed-form expression through the real
  `CTC_CRF` on a GPU and checks the Rust decode reproduces the decoded
  sequence, the per-timestep argmax edge, *and* the path exactly, on every
  backend the host supports. Two details differ from what #27 recorded and are worth carrying
  forward: the score width is **1280**, not 1024 (1024 is the linear layer;
  `LinearCRFEncoder` expands blanks to one per state), and the output is
  **time-major** `[T, batch, n_score]`, so it cannot reuse the boundary CNN's
  batch-major shape probe and gets its own.

  Standardisation constants come from the export's `metadata.json` sidecar, not
  from bonito's `config.toml` — that file carries SeqTagger's unrelated
  `mean = 80.876 / stdev = 17.270`, and using it applies a ~1.8 sigma shift and
  a 1.7× scale error that degrades the decode with nothing to indicate it. The
  loader refuses to guess.

  With `--barcodes` (a `name,sequence` CSV) each read is also assigned to its
  closest reference by edit distance, and the output carries `read_id,barcode` —
  exactly what `escpod demux split` reads. So `detect → basecall → split` runs
  end to end with **no Python in the middle**, which is what this was for.
  Verified on 300 real reads: **300/300 barcode calls identical** to
  `demux_stage1.py`, and `split` produced 14 per-barcode POD5s whose counts match
  the classifier exactly. Without `--barcodes`, only decoded sequences are
  emitted.

  Alignment is `fqxv-align`'s wavefront implementation (a git dependency on
  `rnabioco/fqxv`, pinned by rev — a zero-runtime-dependency leaf crate
  extracted for exactly this in rnabioco/fqxv#252, so it brings no dependency
  cone). WFA rather than a DP because its work scales with the edit *distance*:
  a decode sits ~4 edits from its own reference and ~10+ from the other 95, so
  most of the comparisons abandon almost immediately. Confidence is the
  edit-distance margin to the second-best reference — the definition the model's
  published precision-at-recovery numbers were computed with, kept identical so
  a recovery threshold still means what it meant at evaluation time. Ties
  resolve to the lowest reference index with a margin of 0, which is the honest
  signal that a read is ambiguous rather than a silent coin flip.

  Note the git dependency would block `cargo publish`. Neither crate is
  published today, and shipping barcode assignment in the default binary is
  worth more than keeping that door open — a `basecall` that emits sequences
  nobody can act on repeats the mistake `cnn-detect` was promoted to fix.

- **AVX2 and AVX-512 kernels for the CRF decode**, runtime-dispatched with a
  scalar fallback. The decode is transcendental-bound — ~770k `exp` and ~260k
  `ln` per 200-timestep read — and started out as *half* the total CPU cost,
  not a rounding error:

  | decode backend | per read | vs scalar |
  |---|---|---|
  | scalar | 12.14 ms | — |
  | AVX2 | 1.92 ms | 6.3× |
  | AVX-512 | 1.19 ms | 10.2× |

  (tract's encoder, for scale, is 13.9 ms/read.) Reproducible via
  `cargo bench --bench crf_decode`, so the claim stays checkable.

  This gates the GPU path being worth anything: with the encoder on the device
  the decode *is* the runtime, so a scalar decode would have capped `--gpu` at
  roughly no gain at all. Vectorised `exp`/`ln` are polynomial approximations
  and the softmax denominator is reassociated across lanes, so the contract is
  "same decoded sequence, floats within a tight tolerance" rather than
  bit-identity — enforced by an equivalence test that runs *every* backend the
  host supports against scalar, and by the bonito parity test doing the same.
  Both SIMD paths break argmax ties by flat index rather than lane order, so
  they cannot disagree with scalar on a tie.

  Per the repository's build policy AVX-512 is runtime-detected, not a baseline
  bump — the pinned `target-cpu=x86-64-v3` stays portable across the Broadwell
  login node, Cascade Lake `rna`, and Ice Lake `gpu`.

- **`crf-gpu` feature** — CRF encoder inference through onnxruntime's CUDA
  execution provider (`escpod demux basecall --gpu`), sharing the exact ONNX
  graph and decode with the CPU path. The lattice decode deliberately stays on
  the CPU: it is sequential in time with a 256-wide inner dimension, a poor fit
  for the device next to the encoder's dense matmuls, and it overlaps across
  rayon workers while the GPU runs the next batch.

  Across 300 real reads, CPU, GPU, and the Python pipeline disagree on only two
  reads, and the pattern says the disagreement is in the *encoder*, not the
  decode: on one read CPU and GPU agree with each other while Python differs
  (and Python flips that read's answer across `--batch-size` 1→256, agreeing
  with both Rust backends at 8–128), and on the other the GPU agrees with
  Python exactly while tract differs by one base. Both are single-base indels
  in the model's least-confident regions, and all three assign the same barcode
  for all 300 reads.

- **`cnn-detect` is now part of the default `cli` feature**, so released
  binaries can run `escpod demux detect --method cnn` (and the fused
  `escpod demux --method cnn`) without a rebuild. The barcode models published
  in escapepod-models are trained against the CNN/TCN boundary detector, so a
  binary without it silently fell back to the LLR detector and produced
  materially worse classification. Unlike the 0.6.3 `demux` promotion this does
  add a dependency (tract-onnx: ~10 MB → ~31 MB stripped, ~3.7 MB → ~11 MB
  compressed), but only for the binary — library consumers reach the layers
  through `default-features = false`, which never enabled `cli`. `cnn-gpu`,
  `gpu`, and `train` remain opt-in.

- **`demux detect --method cnn --emit-llr-delta`** runs the LLR detector
  alongside the CNN and adds `llr_adapter_start`, `llr_adapter_end`, and
  `end_delta` columns, so the two independent detectors can be compared per
  read. Boundary quality is otherwise only checkable against EDX-derived
  labels, which production demultiplexing does not have; detector disagreement
  is the gate that works without them. A summary line reports the median, p95,
  and max of `|end_delta|` — percentiles rather than a "within N samples" count,
  since any such N would be invented here.

  Opt-in because of I/O, not compute. LLR is nearly free next to CNN inference
  (~0.25 s vs ~77 s per 20k reads), but it normalizes over the whole read, so
  the CNN path can no longer decode only each read's leading `max_obs_trace`
  samples — the saving that matters on long mRNA reads. Running LLR on the
  CNN's prefix instead would have been nearly free, and was rejected: those are
  not the boundaries `--method llr` reports, so the delta would measure the
  truncation rather than the disagreement. For the same reason the flag is
  refused with `--gpu`, whose producer only ever decodes a prefix.

  `end_delta` is left empty unless *both* detectors found a boundary, because
  `adapter_end == 0` is a shared sentinel for "no adapter", "too short", and
  "inference failed" — differencing against it would report a large
  disagreement that neither detector has.

### Fixed

- **`demux detect --method cnn` honours `-t/-j` again.** It ran a full-width
  rayon pool for every value of the flag — 18 threads and ~1550% CPU at `-j 1`
  just as at `-j 16` — so a Slurm job could not be held to its allocation.
  Sizing the pool happened *after* a read-counting `par_iter()` had already
  built rayon's global pool at `available_parallelism()`; the resulting
  `build_global()` error was discarded, making the ignored flag invisible. The
  pool is now built once in `main` before command dispatch, so no command can
  reorder itself into the same trap, and a failure to size it warns instead of
  passing silently (#155).
- **`--gpu` CNN detection bounds onnxruntime too.** Its intra-op pool was left
  at onnxruntime's default width and spawned alongside rayon's, so `--threads`
  did not bound the process even once the rayon half was fixed (#155).

### Changed

- **`-t/--threads` now defaults to 16 everywhere, capped at the CPUs actually
  available** (`available_parallelism()`, which respects cgroup quota and CPU
  affinity — under `srun -c 8` the default is 8). An explicit `-t/-j` is never
  capped. This replaces two different defaults: the block-copy commands
  (`merge`, `filter`, `subset`, `index`) used a fixed 8, and the `demux` and
  `resquiggle` commands used all CPUs.

  Three behaviour changes worth planning for:
  - `demux` and `resquiggle` **drop from all CPUs to 16** by default. On a
    wide allocation that is a real throughput loss — pass
    `-j $SLURM_CPUS_PER_TASK` to keep the old behaviour.
  - the block-copy commands rise from 8 to 16.
  - `view`, `summary`, `repack`, and `bam-filter` have no `--threads` flag and
    were previously unbounded; they now get the same 16-thread default.

  `RAYON_NUM_THREADS` still applies when no flag is given. `-t 0` is now an
  error rather than a silent "let rayon decide".
- `-t` and `-j` are accepted interchangeably by every command that takes a
  thread count; previously `merge`/`filter`/`subset`/`index` took only `-t`
  and `demux classify` only `-j`.
- **Dependency logs no longer appear at the default verbosity.** The tracing
  filter is now scoped to escpod's own crates with dependencies held at `warn`;
  previously `demux detect --method cnn` printed a dozen lines of tract SIMD
  kernel-probe output before doing any work. `RUST_LOG` still overrides
  everything, and `-q` still silences all but errors.

## 0.6.3 (2026-07-26)

### Added

- **`demux` is now part of the default `cli` feature**, so the standard
  `escpod` build ships barcode demultiplexing without a rebuild. It pulls in
  zero new third-party crates (ndarray/serde_json/rayon/uuid were already in
  the graph). The accelerator features `train`, `gpu`, `cnn-detect`, and
  `cnn-gpu` remain opt-in because they need extra toolchains or a CUDA
  runtime (#150).

### Fixed

- **`mad_normalize` no longer aborts on a constant signal.** A dead pore or
  flat read produced a zero MAD that panicked; since release builds use
  `panic = "abort"`, a single bad read killed a multi-hour job (#150).
- **Short fingerprints are no longer emitted silently.**
  `extract_fingerprint_from_signal` treated `keep_last` as a maximum rather
  than an exact width, so a truncated fingerprint could reach the classifier.
  GBM errored out, but DTW/SVM scored the truncated query and reported a
  confident *wrong* barcode. Observed in real data at 1 ragged row in 9,894
  (#150).
- **`read_query_csv` validates row width.** Malformed cells were dropped
  silently; the labeled loader had guarded this for a while, the query loader
  never did (#150).
- **A GBM classify failure no longer discards its whole chunk.** One failing
  read dropped all 1,024 reads in the chunk instead of just itself (#150).
- **`filter`/`subset` write real read IDs into the signal table.** The
  `read_id` column was zero-filled, diverging from ONT's own tooling and from
  the schema's documented "UUID for consistency checking" purpose. Files still
  loaded and round-tripped losslessly because the reads table is the authority
  for the read→signal mapping, so nothing read the column — but any tool that
  used it for the check it is named for would have rejected every file
  `escpod filter`/`subset` ever produced (#151).
- **`demux train` output is reproducible.** Grouping iterated a `HashMap`, so
  `std_dev` summation order — and its low bits — varied run to run, and
  `TrainingOutput::barcodes` was a `HashMap`, so serde emitted barcodes in a
  different order each process. Results are now sorted by read index and the
  map is a `BTreeMap`; three consecutive runs are byte-identical (#152).

### Performance

- **Fused `demux` pipeline is 2.3× faster on GBM models** (151.0 s → 65.1 s on
  1.22M reads / 10.4 GB, 48 cores; CPU utilization 511% → 1273%), and 1.27×
  on DTW-SVM. The pipeline was neither compute- nor I/O-bound — it ran faster
  at 8 threads than at 48, because block fill never overlapped with processing
  and the dominant barcode's writer channel blocked rayon workers that could
  not then be stolen from. Per-barcode counts are unchanged, and the staged
  `detect`/`fingerprint` subcommands still produce byte-identical output
  (#150).
- **DTW-SVM classify is faster, bit-identically.** `DTW_LANES` 16→32 and
  `decision_function` now skips support vectors that cannot contribute to a
  class pair: per-read predict 150.1 → 110.4 µs (−26.4%), and the 20-class
  decision function 366.9 → 26.7 µs (−92.7%), which matters for wider barcode
  designs (#150).
- **`filter` is 2.29× faster and `subset` 1.74×** (51.3 s → 22.4 s and 37.9 s
  → 21.8 s on 1.22M reads / 10.4 GB, 48 CPUs). The signal-table copy ran
  inline with the IPC writes on one thread, so every page fault stalled the
  thread that then issued the write — 75% of profile samples in `memmove` plus
  kernel page-fault time at 24% CPU. Batches now build across rayon and are
  written in order, with lookahead bounded to 16 batches in flight. `merge` is
  deliberately untouched: it copies whole Arrow IPC blocks and patches footer
  offsets, so it never rebuilds batches (#151).
- **`demux train` is 8.7× faster** (231.5 s → 26.5 s on 200k assigned reads,
  48 cores; CPU utilization 86% → 1199%). `extract_fingerprints` parallelized
  only *across files*, so the common single-POD5 training run was effectively
  serial regardless of `-t`. It now walks files sequentially — one ascending
  mmap sweep per Arrow batch, preserving the single-stream I/O — and
  parallelizes inside each batch. Consensus fingerprints are bit-identical
  (#152).
- **`demux train-svm` no longer computes an all-pairs DTW distance matrix and
  its RBF kernel matrix.** Both were discarded: their only consumer took the
  kernel as an ignored parameter, so ever since the SMO fit was removed the
  emitted model has been a function of the labels alone. At N=50,000 the
  command now runs in 0.27 s / 25 MB, where the matrices alone would have
  wanted ~40 GB — the memory wall that `--max-per-class` exists to work around
  (#152).

### Changed

- **The `gpu` DTW-classify feature is now marked experimental.** On a full
  node it measured *slower* than the CPU: 113.0 s on 64 cores versus 132.4 s
  with `--gpu` on an A30 (0.85×), plus ~2.2 GB more RSS. The apparent win
  disappears once the CPU gets the whole node instead of 16 of 64 cores. GPU
  **CNN adapter detection** (`--method cnn --gpu`) remains the GPU path that
  pays off (#150).
- **`demux train-svm --window` and `--gpu` now warn** that they do not affect
  the fit, instead of silently appearing effective (#152).
- `compute_distance_matrix`, `distance_to_kernel_matrix`, and
  `train_svm_from_distances` stay public and tested, so a real SVM fit can be
  wired back in without a signature change (#152).

### Build / Tooling

- The signal `RecordBatch` builder is shared between `filter`/`subset` and the
  incremental `Writer` (`build_signal_batch` + `SignalRow` in
  `utils::table_builders`) rather than duplicated (#151).
- New `test_table_conformance` guardrail in the CI compat suite snapshots all
  three embedded Arrow tables from a pod5-written and an escapepod-written
  file and diffs schema plus contents, resolving signal `read_id` through each
  read's `signal_rows`. Validated as a guardrail: it fails against a pre-fix
  binary. Adds an `ESCPOD_BIN` override so the suite can target an arbitrary
  build (#151).
- Dependency bumps: `clap` 4.6.4, `libc` 0.2.189, and `actions/setup-python` 7.

## 0.6.2 (2026-07-22)

### Build / Tooling

- **mold is now the linker full time inside pixi.** The base `[activation]`
  block exports `LD_PRELOAD=$CONDA_PREFIX/lib/mold/mold-wrapper.so` (+
  `MOLD_PATH`) for every environment, so any `pixi run [-e <env>] cargo …` (and
  maturin) links with mold — no more `mold -run` wrappers or `-e dev`-only mold.
  Works with the stock system gcc 11.5 on the compute nodes (no `-fuse-ld=mold`
  support needed) via mold's `mold-wrapper.so` interposer, and needs no
  glibc-static. CI release builds (musl, outside pixi) are unaffected.

### Added

- **Auto-warmed read-id index on Python context-manager entry.** Entering a
  `Reader` or `DatasetReader` in a `with` block now builds the in-memory
  read-id index, so repeated `reads(selection=…)` lookups take an O(k) indexed
  path instead of re-scanning the reads table each call (~2× faster
  random-access selection, on par with the `pod5` package). Plain `open()` is
  unchanged — the index stays lazy for single-pass streaming. Size-gated by
  read count (default 5,000,000, overridable via `ESCAPEPOD_AUTOINDEX_MAX`) to
  bound memory; this is the in-memory index, not the `.p5i` sidecar, so no
  sidecar file is written (#97).

- **PyPI publishing** for the `escapepod` Python package. The `release.yml`
  workflow now builds abi3 wheels (CPython 3.9+) for Linux (x86_64/aarch64,
  manylinux + musllinux) and macOS (x86_64/arm64) plus an sdist, and publishes
  them to PyPI via Trusted Publishing (OIDC) on each `v*` tag. `pip install
  escapepod` will work once the first tagged release lands. The bindings crate
  gained `abi3-py39` and complete PyPI metadata (readme, license, URLs,
  classifiers, type stubs).

### Performance

- **Resquiggle hot paths** (band construction and refinement) iterate with
  `array_windows` instead of manual index arithmetic, dropping redundant bounds
  checks from the inner loops (#61).

### Fixed

- **Status output no longer escapes ANSI.** The `tracing` formatter was
  escaping ANSI control sequences, so styled status/progress output rendered as
  literal escape codes; it now emits proper styling (#142).
- **Quiet `SIGPIPE` on piped output.** Piping CLI output into a consumer that
  closes early (e.g. `| head`) no longer produces a broken-pipe error — the CLI
  exits cleanly on a closed downstream pipe (#140).

## 0.6.1 (2026-07-20)

### Fixed

- POD5 archives are now written **atomically**: output is staged to a
  temporary file and renamed into place on completion, so an interrupted or
  failed write can no longer leave a truncated/partial `.pod5` at the target
  path.

### Build / Tooling

- Dependency bumps: `noodles-bam` 0.91, `noodles-sam` 0.86, `noodles-bgzf`
  0.48, `noodles-csi` 0.57, `sha2` 0.11, and the grouped cargo minor/patch
  and GitHub Actions updates.

## 0.6.0 (2026-07-11)

### Added

- **Native GBM (gradient-boosted tree-ensemble) barcode classifier** for
  `demux classify` / fused `demux`, alongside the existing DTW+SVM path.
  Fingerprints are scored directly by a distilled tree ensemble (no DTW),
  loaded from the same auto-detected model surface. Fused-pipeline support
  is included, so `escpod demux --model gbm.json …` routes reads end-to-end.
- **Architecture-agnostic GPU CNN/TCN adapter detection** (`demux detect
  --method cnn --gpu`, feature `cnn-gpu`): runs any `[B,1,L] -> [B,2,L]`
  ONNX graph batched through onnxruntime's CUDA execution provider via the
  `ort` crate (`load-dynamic` — needs a CUDA-enabled `libonnxruntime` on
  `ORT_DYLIB_PATH` and a visible GPU at run time, nothing at build time).
  TCN detect is inference-bound, so this pays off (~7.6× end-to-end on an
  A30 at 20k reads). This is the ONNX-generic backend anticipated by the
  removal of the old arch-locked CUDA kernel below — it replaces it without
  hardcoding a topology.
- **pod5-parity Python API**: a multi-file `DatasetReader`, `to_dict` /
  `to_pandas` / `to_polars` exporters, `missing_ok` handling,
  `calibrate_signal_array`, `Writer.add_reads`, and per-read `byte_count`,
  bringing the `escapepod` bindings closer to the reference `pod5` package.
- **Python bindings for the signal layer**: `normalize`, a `KmerTable`
  wrapper, and `refine_signal_map` are now exposed from `escapepod-signal`,
  so the resquiggle/normalization primitives are usable from Python.
- `-t` / `--threads` on `filter`, `subset`, and `merge` (matching `demux`)
  to cap the worker pool; these commands now default to a bounded pool of 8
  threads instead of grabbing every CPU on a shared node.
- **POD5 reads-table schema V5**: the `expected_open_pore_level` and
  `selected_read_level` fields (both `float32`), introduced upstream in pod5
  0.3.44, are now read, written, merged/filtered, surfaced in `view`/`inspect`
  (and as selectable output fields), and exposed on the Python `ReadData` /
  `Writer.add_read` API. Files escapepod writes are now stamped `pod5_version`
  0.3.44 and verified readable by the reference ONT `pod5` reader; existing
  V0–V4 files still read, with the new fields defaulting to 0.0.
- Defensive pre-mmap probe when opening a POD5 file: the header and footer are
  read through ordinary I/O before the file is memory-mapped, so a truncated
  file or an archive "stub" (unresident data on HSM/tape-backed filesystems)
  surfaces a recoverable error instead of an uncatchable SIGBUS on first page
  fault. Mirrors upstream pod5 0.3.37; set `POD5_DISABLE_MMAP_OPEN=1` to skip.
- `escpod demux <file> --model M -d OUT` now runs a **fused, streaming
  pipeline** by default: each read's signal is decoded once, then detect +
  fingerprint + classify run in a single pass and reads are routed
  (block-level compressed copy) into per-barcode POD5s — no intermediate
  boundaries/fingerprints/classifications files. The granular
  `detect`/`fingerprint`/`classify`/`split`/`train` subcommands remain for
  advanced use. `--classifications` writes the per-read CSV only when asked.
- Experimental GPU primitives (behind `--features gpu`) for the demux signal
  chain — SVB16 decode, t-test fingerprint, LLR detect — kept as validated,
  reusable kernels. They are **not** used by `escpod demux`: measurement shows
  the prep stages run faster on a multi-core CPU than on the GPU.

### Removed

- The **GPU CNN adapter-detection path** (`demux detect --method cnn --gpu`,
  plus the `--cnn-weights` flag and `scripts/dump_adapter_cnn_weights.py`). Its
  hand-written CUDA kernels were hardcoded to ADAPTed's `BoundariesCNN`
  topology (3× Conv1d + ConvTranspose1d, fixed K=7/C=64) and could not run any
  other architecture — including escapepod-models' replacement TCN. CNN
  detection (`--method cnn`) now runs **only** through the architecture-agnostic
  tract-onnx CPU path (`adapter_cnn.rs`), which accepts any `[B,1,L] -> [B,2,L]`
  ONNX graph. This is not a regression at typical scales: `detect` is dominated
  by POD5 read + signal prep, not CNN compute, so the CPU path is as fast or
  faster (the GPU flag's own help already said as much). Removing it also drops
  the CC-BY-NC `.weights` dumper. If a GPU CNN path is ever needed again, add an
  ONNX-generic backend (e.g. ORT CUDA EP) rather than a per-architecture
  kernel — which is exactly what the new `cnn-gpu` path (see **Added**) does.

### Changed

- **The CLI crate is renamed `escapepod` → `escapepod-cli`**, matching its
  `escapepod-{pod5,signal,demux}` siblings and making its role explicit. The
  `escpod` binary name is unchanged, and installation is unchanged
  (`cargo install --git …`). Library consumers of the umbrella crate now
  import it as `escapepod_cli` (e.g. `use escapepod_cli::signal`) instead of
  `escapepod`. (The `escapepod` name on PyPI already belongs to the Python
  bindings, so this also removes the crate-name overlap.)
- **CLI output split into logs vs. data.** All status/progress/warning output
  now flows through `tracing` to **stderr** (`timestamp LEVEL [target] message`),
  while command *data* (TSV/CSV rows, `inspect`/`summary` reports, ID lists)
  stays on **stdout**, so it can be piped/redirected independently of logs.
  Default level is `info`; control it with `-v`/`-vv`/`-q` or `RUST_LOG`.
  Progress bars auto-hide under `-q`.
- **Minimum supported Rust version raised to 1.95** (from 1.92).
- The **resquiggle module is relicensed to MIT** as an independent
  implementation, bringing it in line with the rest of the workspace.
- The Theil–Sen rescale **subsample seed is now configurable** (`resquiggle
  --seed`), making refinement runs bit-for-bit reproducible.
- The LLR detect `--downscale` default is now **10** (the WarpDemuX-native
  mode) for `demux` and `demux detect`, up from 1. This makes detect — the
  dominant prep stage — ~5× faster, with ~98% barcode agreement versus
  full-resolution (ds=1). Pass `--downscale 1` to restore full-resolution
  detect.
- Dependency bumps (no behavior change): Arrow ecosystem `arrow` + `parquet`
  58 → 59, `tabled` 0.20 → 0.21, the `noodles-*` BAM stack (`bam` 0.90,
  `sam` 0.85, `bgzf` 0.47, `core` 0.20, `csi` 0.56), `ndarray` 0.16 → 0.17,
  `cudarc` 0.12 → 0.19, and `tract-onnx` 0.21 → 0.23.

### Performance

- **Demux classify DTW is much faster, bit-identical.** Per-call DTW
  allocation eliminated from the SVM classifier (~1.8×), then extended with
  lane-parallel SIMD DTW (8 signals/vector) covering windowed/penalty models
  as well — together a large end-to-end speedup on the DTW-bound path.
- **GBM classify ~3.1× faster** (compact tree arena + 8-read batched walk),
  bit-identical.
- **Cold-read demux I/O fixed.** Signal is now read in a single sequential
  memory-mapped stream with O(1) Arrow-batch seeks and dictionary pre-scan /
  readahead, instead of per-read random access across many threads (which
  degraded to demand-paging on cold files). Removes the large-POD5 startup
  stall and the cold-read throughput cliff.
- **Zero-copy single-read signal fetch** with adaptive prefix decode
  (#94): the reader's per-read signal path avoids materializing intermediate
  buffers, and fingerprint prep streams only the needed prefix.
- **Faster Python read iteration**: per-batch column resolution for the lazy
  `for rd in reader:` iterator and `reads()`, plus numpy-backed metadata
  columns for `to_dict` / `to_pandas` / `to_polars`.
- **Parallelized bulk-file copies**: `filter`, `subset`, and `merge` now
  copy reads across threads (bounded default pool of 8; `-t` to override),
  and the experimental `index` command adopts the same bounded default
  instead of inheriting rayon's all-CPU global pool.
- **Single-pass `demux split`**: reads are routed to per-barcode outputs in
  one pass, and the `--unclassified` flag now works correctly.
- Codebase-wide optimization/refactor sweep (#86), all bit-identical output:
  - **Resquiggle adaptive banded DP ~31% faster** — the per-base traceback no
    longer heap-allocates a `Vec` per base; the whole read shares one flat
    buffer.
  - **O(1) POD5 read-batch access** — `read_batch(i)` / `read_ids_from_batch(i)`
    now seek via the Arrow IPC footer instead of decoding every preceding batch.
    Iterating a many-read-batch file (e.g. the Python `Reader` read iterator)
    drops from O(B²) to O(B) batch decodes: ~10× faster random batch access and
    ~2.6× faster full-file iteration on a 1.65M-read / 166-batch file.
  - **Signal median computations are O(n) instead of O(n log n)** — the SVM
    kernel γ-heuristic, Theil–Sen rescale, and resquiggle dwell median now use
    `select_nth_unstable` instead of a full sort.
  - Smaller per-read allocations on the demux/classify and fingerprint hot
    paths (MAD-normalization scratch reuse, Platt coupling workspace sized once).
- Internal consolidation with no behavior change: six duplicated median impls
  unified into `escapepod-signal::stats`; the SVM RBF-kernel mapping and the
  CPU/GPU CNN batch packing/scatter each live in one shared helper.

### Fixed

- Resolved a PyPI name collision: both the `escapepod` CLI crate and the
  `escapepod-python` bindings crate declared `name = "escapepod"`. The PyPI
  `escapepod` distribution is the **Python `Reader` bindings**
  (`escapepod-python`); the `escpod` CLI now ships via `cargo install
  escapepod` and GitHub release binaries only, so its maturin `pyproject.toml`
  (a `bindings = "bin"` wheel) has been removed. This reverses the 0.5.1 note
  about `pip install escapepod` installing the CLI.

## 0.5.1 (2026-06-14)

### Changed

- The CLI now ships from the `escapepod` crate (renamed from
  `escapepod-cli`), so `cargo install escapepod` installs the `escpod`
  binary. The same crate doubles as an umbrella library: with
  `default-features = false` plus `pod5` / `signal` / `demux`, it
  re-exports each layer (e.g. `escapepod::signal`) without pulling in the
  CLI's dependency tree. `demux` stays opt-in until it stabilizes.
- The maturin binary wheel is published as `escapepod` (was
  `escapepod-cli`) so `pip install escapepod` matches `cargo install`.

### Fixed

- Packaging `readme` pointed at a nonexistent path, which made
  `cargo package` fail; the workspace now points every publishable crate
  at the root `README.md`. `escapepod-python` is marked `publish = false`.
- `demux fingerprint` (test fixture): labeled-Parquet temp files lacked a
  `.parquet` suffix, so format detection read them as CSV and the parquet
  loaders failed with an "invalid UTF-8" error.

### Build / Tooling

- Gated the `train`-only labeled-fingerprint loaders behind
  `#[cfg(feature = "train")]`, removing dead-code warnings from
  `--features demux` builds.
- Bumped GitHub Actions to current majors (checkout v6, upload-artifact
  v7, download-artifact v8, setup-python v6, setup-pixi v0.9.6,
  action-gh-release v3, actions-netlify v4), clearing the Node 20
  deprecation warnings.

## 0.5.0 (2026-04-27)

### Added

- `demux fingerprint`: Parquet output when `-o` ends in `.parquet`, plus
  an `--emit-dwell` flag that adds per-segment dwell-time features.
- `demux classify` (CLI): `fp_io` module reads fingerprint inputs from
  both Parquet and CSV (gzipped CSV included); new flags
  `--gpu-chunk-cells` and `--threads`, with model auto-detection so
  `--model` accepts any supported format.
- `escapepod-demux`: `AnyModel` enum and `load_any_model()` for
  format-agnostic SVM/DTW model loading.
- `escapepod-signal`: SVM helper CUDA kernels exposed via function
  handles for downstream GPU pipelines.

### Performance

- `demux classify` (GPU): on-GPU RBF + OvO decision pipeline
  (`GpuSvmContext`) keeps SVM evaluation on the device; producer/
  consumer pipeline parallelizes the consumer side and bumps the
  default chunk to 4G cells with channel depth 2 for better GPU
  utilization on long runs. Per-chunk indicatif progress bar surfaces
  throughput.
- `escapepod-demux`: RBF kernel fast paths for `power == 1.0` and
  `power == 2.0` skip the generic `powf` call.
- `escapepod-pod5`: filter and merge hot paths reworked; remaining
  `reader.reads()` callers now batch-amortize the schema/footer parse,
  and a `PoreType` newtype removes per-read string churn.

### Fixed

- `train` (multiclass OvO): dropped an unused SMO solve path that ran
  during training without contributing to the final model.

### Build / Tooling

- Pixi `dev` env wires `mold -run` for fast local links (system gcc
  11.5, no glibc-static needed); release artifacts in CI continue to
  build against musl.
- Docs: benchmark page leads with bulk operations (`merge`, `filter`,
  `subset`, `repack`); `inspect` and `view` demoted to a secondary
  section.

## 0.4.0 (2026-04-22)

### Performance

- `demux fingerprint`: nested `par_iter` streams signals across files and
  reads so fingerprinting a 48-file run drops from ~32 min to ~9.8 s on
  the rna partition.
- `demux classify` / `train-svm`: reusable per-thread `SvmWorkspace` and
  a streaming (rayon fan-out + single writer) output path cut RSS by
  ~37% and remove a serialize-then-write stall.
- `svb16`: AVX2 decode path (16 samples/iter), preferred at runtime over
  SSSE3 when available.
- `dtw`: split the inner band loop so the trailing segment auto-
  vectorizes under AVX2; the x86-64-v3 baseline (AVX2 + FMA + BMI2 +
  POPCNT + F16C) is now pinned in `.cargo/config.toml` for portability
  across Broadwell/Cascade Lake/Ice Lake cluster nodes.
- `segmentation::llr`: allocation-free `best_split` and an opt-in
  early-stop variant.
- CLI: progress-bar updates throttled out of hot paths.

### Changed

- Moved from CLI into libraries (additive for library consumers):
  - `ReadBoundaries` and fingerprint types/helpers now live in
    `escapepod-demux`.
  - `normalize_signal(&[i16])` and the CLI's `downscale_signal` now
    live in `escapepod-signal` (the CLI's duplicate was removed).
- Docs: recommend `srun -c 48` for throughput-sensitive demux runs on
  the rna partition (fills one socket without crossing NUMA).

## 0.3.1 (2026-04-21)

### Added

- `resquiggle::banded_dp_with_penalty_table` — banded Viterbi DP variant
  that accepts a caller-supplied short-dwell penalty table and uses its
  length as the check horizon. Tie-break is strict (`<`), matching the
  Remora-compatible refinement pipeline. Complements the existing
  `banded_dp` which builds the asymmetric penalty internally.
- `segmentation::mad_normalize_robust` — median-MAD normalization with
  the 1.4826 Gaussian scale factor and graceful fallback (returns
  `signal - median` without dividing) when MAD is essentially zero.
  Complements `mad_normalize`, which panics on constant signals.

### Performance

- Hot-path audit across fingerprint MAD, SVM prediction, `view`, and
  `merge`. Fingerprint MAD uses a single-pass median/MAD with an
  in-place partition; SVM prediction reuses per-thread scratch buffers
  and avoids redundant kernel evaluations on the OvO dual path; CLI
  `view` streams reads with reused formatting buffers; `merge` switches
  to mmap-backed readers where possible to cut per-file overhead.

### Fixed

- `escapepod-python` cdylib now links cleanly under a plain
  `cargo build` on macOS. A `build.rs` emits the pyo3
  extension-module link args (equivalent to
  `pyo3_build_config::add_extension_module_link_args()`), scoped to the
  cdylib, so the build no longer fails with undefined `_Py*` symbols
  when maturin is not driving the build. macOS is now in the CI matrix
  for `check`, `test`, and `clippy` to catch regressions.

### Changed

- Workspace crates moved under `crates/` (no public-API change).
- Docs reorganised with an "experimental tools" section; `resquiggle`
  and `index` CLI subcommands are gated behind their respective Cargo
  features.

## 0.3.0 (2026-04-20)

### Breaking

- Barcode demultiplexing moved out of `escapepod-signal` into a new
  `escapepod-demux` crate. The `escapepod_signal::demux` module is gone;
  downstream code should depend on `escapepod-demux` directly and
  import from `escapepod_demux::...` (model loaders, `classify_read`,
  SVM predictor/trainer, Platt scaling, GPU batch classify, ADAPTed
  CNN adapter detection). The `escpod demux` CLI surface is unchanged,
  but `escapepod-cli`'s `demux` Cargo feature now pulls in the new
  crate; the `train`, `gpu`, and `cnn-detect` features forward to it.

### Added

- GPU-accelerated DTW for demux, opt-in via `--features gpu` on
  `escapepod-signal` and `escapepod-cli`. Wires up `escpod demux classify
  --gpu` (WarpDemuX model, CSV reference, and SVM model paths) and
  `escpod demux train-svm --gpu`. CUDA kernel is NVRTC-compiled at
  runtime, so no `nvcc` is required at build time — only the CUDA driver
  and `libnvrtc` at run time. On the lab cluster, `pixi run -e gpu …`
  provisions `cuda-nvrtc` via conda-forge and sets `LD_LIBRARY_PATH`.
  Anti-diagonal kernel with shared-memory-cached fingerprints; single-
  warp blocks with `__syncwarp()` and `__launch_bounds__(32, 64)`.
  Measured ~7.7× speedup over CPU rayon on A30 at 1024×40×110 and
  8192×40×110 workloads (criterion, band w=10).
- `GpuDtwContext`, `dtw_distance_matrix_gpu`, `classify_reads_gpu`,
  `classify_with_svm_batch_gpu`, `compute_distance_matrix_gpu`,
  `train_svm_gpu` public API on `escapepod-signal` (all `gpu`-gated).
- CPU `classify_read` now uses `dtw_distance_bounded` with the running
  second-best squared distance as an upper bound, skipping DTW work for
  training fingerprints that cannot change the top-2. Safe for both
  ratio and kernel threshold paths.

### Fixed

- **Behavior change for windowed DTW.** The 2-row banded DP in
  `dtw_distance` / `dtw_distance_bounded` was leaving stale
  `curr[j_start - 1]` values from earlier rows, letting the recurrence
  read an out-of-band predecessor and occasionally find a shorter-than-
  valid path. The classical Sakoe-Chiba band treats those cells as
  unreachable; we now re-seed the boundary to `INF` at the top of each
  row and also short-circuit to `INF` when `|n − m| > w` (the endpoint
  itself is outside the band and the DP would otherwise propagate a
  stale in-band value through the trailing empty rows). Only affects
  callers that pass `Some(window)`; unwindowed DTW is unchanged. In
  practice the difference is small but non-zero on real data — any
  downstream classify output produced with a band constraint may shift
  slightly, with GPU and CPU now agreeing bit-for-bit up to f32
  summation order.

## 0.2.0 (2026-04-20)

### Breaking

- Workspace split into two library crates. POD5 format I/O (reader, writer,
  VBZ compression, merge/filter/repack/subset, schema, footer, types, errors)
  now lives in the new `escapepod-pod5` crate. The crate formerly called
  `escapepod` has been renamed to `escapepod-signal` and contains the
  signal-processing algorithms (DTW, resquiggle, segmentation) layered on
  top of `escapepod-pod5`. Downstream consumers depending on `escapepod`
  by name must rename to `escapepod-signal`; the pod5 surface is
  re-exported from `escapepod-signal` so most `use escapepod::...` paths
  translate to `use escapepod_signal::...` with no other changes.
- Barcode demultiplexing is now opt-in. The `escapepod-signal::demux`
  module and the `escpod demux` CLI subcommand require building with
  `--features demux`; the `train` feature now implies `demux`.

### Added

- `escapepod-pod5` crate for POD5 format I/O.
- `demux` Cargo feature on both `escapepod-signal` and `escapepod-cli`.

### Changed

- README no longer advertises barcode demultiplexing as a shipped feature;
  `docs/cli/demux.md` carries an experimental admonition.
- CLI now declares demux and resquiggle commands as experimental in the
  commands index.

### Removed

- Empty `escapepod-vortex/` directory (content preserved on the
  `escapepod-vortex` branch).
- Stale `PROGRESS.md` and the top-level `examples/test_ipc.rs` scratch
  file; `examples/dtw_example.rs` moved under
  `escapepod-signal/examples/`.

## 0.1.3 (2026-04-20)

### Added

- Tracing-based runtime verbosity (`-v`/`-vv`/`-q`, `RUST_LOG`)
- `release-with-debug` and `profiling` build profiles; phase timer
- Criterion microbenches covering audit hot paths (`escapepod/benches/hot_paths.rs`)

### Changed

- SSSE3 SIMD encode/decode for SVB16 (~2× vs scalar on x86_64)
- Audit-driven hot-path optimizations across reader, DTW, demux, DP
- Dropped `escapepod-vortex` workspace member
- `[profile.bench]` pinned to inherit from release

### Fixed

- Clippy lints: `unnecessary_sort_by`, `needless_range_loop`

## 0.1.0 (2026-03-19)

First stable release of escapepod-rs.

### Added

- **index**: `.p5i` sidecar read index for fast UUID lookup (`escpod index`), with zstd-compressed entry blocks, sorted-vec binary search, and file size checksum validation
- **filter**: Sample count and end reason filters, stdin support for read IDs, fast `reads_by_ids()` path for UUID-only filtering
- **subset**: Accelerated subsetting via indexed batch lookup
- **merge**: Parallel I/O optimization

### Fixed

- Include ZSTD content size in VBZ frames for Dorado/pod5 compatibility
- POD5 forward compatibility with Python pod5 library
- Correct pore count in summary table

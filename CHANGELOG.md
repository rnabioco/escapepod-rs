# Changelog

## Unreleased

### Fixed

- **Rescaling no longer returns a wild scale when the fit does not identify
  one.** All three estimators return `scale / slope`, and all three only
  rejected an *exactly* zero slope. A slope merely close to zero is not "no
  change" — it is a scale multiplied by `1/slope`.

  This is reached by ordinary data, not a pathological input. When the window
  being rescaled sits in a constant adapter or a homopolymer the expected levels
  carry almost no spread, the fitted slope collapses toward zero, and the caller
  gets a scale orders of magnitude off. On real tRNA chunks the returned scales
  ranged from 15 to 1084 and were **frequently negative** — a sign-flipped read.
  Applying that transform destroys the per-base levels: a measured-vs-expected
  k-mer correlation that should sit near +0.8 came out at -0.03.

  `least_squares`, `theil_sen` and `least_squares_with_drift` now require a
  finite, positive slope of at least 0.01 (already a 100x rescale) before using
  it, and otherwise return the caller's parameters unchanged — the same
  "no information, leave it alone" behaviour the exactly-singular case already
  had. `theil_sen` returns them rather than erroring, so refinement still runs.

  Note for callers: rejection is necessarily per-read, so a caller feeding
  already-normalized signal is better off keeping its own normalization than
  applying a mix of fitted and unfitted transforms across its reads.

- **A long `--gpu` run no longer exhausts device memory and stalls.**
  onnxruntime's default `kNextPowerOfTwo` arena strategy doubles the CUDA
  arena on every extension and never returns it. Demux gives `Session::run`
  a different batch shape constantly — a full `batch_rows` mid-file, a short
  tail at every POD5 boundary, whatever the halve-and-retry leaves after an
  OOM — so each unseen shape forces an extension, and the arena ends holding
  the whole device in bins too small to serve the next request. The CUDA EP
  now asks for `kSameAsRequested`, which extends by exactly what was
  requested, so fragmentation stops compounding.

  The failure scales with how **long** the stream is, not how big the batch
  is, which is why it survived testing: on a 24 GB A30 with RNA004 nbc16,
  1.0M- and 1.8M-read runs finish clean while a 4.88M-read run wedged 61% in
  after 409 failed ~780 MB `Reshape` allocations. It does not surface as an
  error — allocations fail, the process keeps running, and `nvidia-smi`
  shows 100% memory at 0% utilisation, which reads as a slow job rather than
  a dead one. Measured at the same point in the same run: 411 MiB and 0
  failures, against 24,145 MiB and 409. No numerics change — same graph,
  same inputs, same arithmetic, same calls.

### Added

- **k-mer level primitives moved down from leech** (#204).
  `escapepod_signal::resquiggle` gains `load_kmer_table` (lenient
  Remora-convention table parse, gz-aware, `f64` levels), `extract_levels`
  (expected level per base with an explicit center index; unknown k-mers and
  short sequences yield zeros, not errors), and `rough_rescale_quantile`
  (the quantile fit that puts observed signal into k-mer level units).
  These are the primitives the tRNA charging classifier's k-mer *residual*
  feature is defined against, so leech and `escpod` must share one
  implementation; parity with leech's NumPy references is pinned at the bit
  level by golden-vector tests (including NumPy's dtype-dependent quantile
  interpolation). Distinct from the existing `KmerTable`, which keeps its
  fishnet conventions for the `resquiggle` command.

## 0.9.0 (2026-08-09)

### Changed

- **One GPU flag.** The CLI's `gpu` Cargo feature now enables every GPU
  path — CNN adapter detection and the CTC-CRF encoder (onnxruntime CUDA)
  as well as GPU DTW classify (cudarc) — so `cargo build --features gpu`
  is the whole story and `--gpu` at runtime uses whichever path fits the
  model and stage. Nothing is needed at build time, and a gpu-built binary
  still runs on CPU-only nodes. The granular `cnn-gpu` / `crf-gpu` flags
  remain for library consumers of `escapepod-demux` and still work on the
  CLI for backward compatibility.

- **The `.p5i` read-index sidecar is retired in favor of `.p5s`** — same
  locator data, now one combined companion file whose read index and
  annotations coexist. `.p5i` files are no longer read; delete them and rerun
  `escpod index` (the index is a rebuildable cache, nothing is lost).
  `Reader.has_index` / `build_index()` now target `.p5s`.

### Added

- **`.p5s` sidecar: per-read annotations without touching the POD5.**
  `escpod annotate -a demux.csv reads.pod5` (experimental) records a read →
  barcode mapping in `reads.pod5.p5s`; the POD5 itself is never modified, so
  raw sequencer output stays byte-identical and checksummable. The sidecar is
  a plain Arrow IPC table (`read_id`, `batch_idx`, `row_idx`, plus one
  dictionary-encoded column per named annotation — `--name` for more than
  one), readable directly with pyarrow/polars, and is bound to its POD5 by
  file-identifier UUID + byte size (checked before any data is decoded), so a
  stale or misplaced sidecar fails loudly instead of describing the wrong
  reads. Writes are atomic and column-preserving in both directions:
  `escpod index` rebuilds the locator columns without dropping annotations,
  `escpod annotate` adds a column without dropping the index. `demux split
  --sidecar` splits from the sidecar instead of a classifications CSV
  (verified read-for-read identical to `--classifications` on a 50k-read
  production subset), so per-barcode POD5s become something you materialize
  on demand rather than store. Python: `Reader.annotation()` /
  `Reader.annotation_names()`.

- **Experimental-design tables in the `.p5s` sidecar.** `escpod annotate
  --design samplesheet.csv` records a mapping from annotation labels — or
  combinations of them (`ldx,edx`) — to experimental variables
  (`condition`, `replicate`, …). The table itself lives as JSON in the
  sidecar's Arrow schema metadata (`escapepod:design`), and each variable is
  materialized as a derived per-read dictionary column by joining across the
  key annotations, so `demux split --sidecar --annotation condition`,
  pyarrow filtering, and `Reader.annotation("condition")` need no join
  logic. Key columns are auto-detected (CSV columns naming an existing
  annotation; `--keys` to override). Rewriting a key annotation re-derives
  its dependent columns automatically; writing a derived column directly is
  refused, keeping the design the single source of truth. Python:
  `Reader.design()`.

- **Sidecar usability across the CLI.** `escpod demux … --annotate` records
  assignments straight into each input's `.p5s` during the fused pipeline —
  with no `-d` it is the only output, so demultiplexing no longer has to
  duplicate the POD5 or produce a CSV at all. `escpod filter --annotation
  NAME=LABEL` materializes one group on demand (repeatable: same name =
  any-of, different names = all-of, `--ids` intersects). `escpod view
  --include read_id,barcode,condition` joins sidecar columns into the TSV.
  `escpod inspect summary` shows the sidecar (index, annotations with label
  and read counts, design). `escpod annotate --list` / `--remove NAME` /
  `--remove-design` inspect and prune without pyarrow. `escpod index` now
  rebuilds a *stale* sidecar without `--force` instead of "skipping" a file
  the reader refuses to load.

- **Progress bars show live throughput** — a `12,847/s` figure between the
  position and the ETA. `demux` ticks per read, so that is live reads/s;
  `merge`/`filter`/`repack` show their own tick unit per second.

## 0.8.1 (2026-08-09)

### Fixed

- **Signal batches are uniform again — they encode a stride** (#195). A read's
  `signal` column holds GLOBAL row indices, and readers resolve one to a
  position by assuming a constant batch stride. `add_read` and
  `add_read_with_compressed_signal` appended all of a read's signal chunks and
  only then tested the flush threshold, so a read whose chunks straddled the
  boundary emitted an oversized batch — and every index after it then pointed
  somewhere else.

  Found in production: `escpod demux` wrote 16 per-barcode POD5s, 6 with an
  oversized batch (57x1000, one 1003, last 410). **dorado silently skipped the
  reads it could not resolve** — no error, no non-zero exit — dropping 97,790
  reads, 10.6% of a 1.0M-read run, and it presented as a biological result until
  the batch row counts were dumped.

  The loud failure is the lucky one: `Queried signal row N is outside the
  available rows (M in batch)` only fires when the shifted index runs past the
  END of a batch. An index that lands inside a batch returns another read's
  signal, silently. **Any POD5 written by escpod before this release should be
  treated as suspect, not just ones that error** — the fault is data-dependent
  (it needs a read whose chunks straddle a boundary), so it moves between runs
  rather than tracking a version.

### Added

- **Non-portable signal batches are detected and reported** (#196).
  `Reader::signal_batch_row_counts` and `Reader::nonuniform_signal_batch` (one
  Arrow IPC footer parse, no signal decode), surfaced by `escpod inspect` and
  warned wherever the CLI resolves inputs.

  This crate resolves such a file correctly — `get_signal` walks cumulative
  per-batch row counts rather than assuming a stride — which is exactly why #195
  hid for so long: the corrupt files passed every escpod check and failed only
  in dorado. Reported as NOT PORTABLE rather than corrupt, because that is the
  true statement and it says what to do about it (`escpod repack`).

- **`escpod filter` takes multiple files and/or directories** (#196), matching
  `pod5 filter`. Inputs are resolved and de-duplicated, preserving the order
  given, so overlapping arguments cannot feed the same read twice. A caller
  whose run is split across per-flowcell directories must pass them in one
  invocation or filter against only part of the run.

## 0.8.0 (2026-08-08)

### Added

- **The bundle's boundary input contract is read, not assumed** (#190,
  escapepod-models#56, closing #187's root cause). A CRF bundle's sidecar can
  now declare the input tensor its pinned boundary CNN consumes
  (`boundary.input`: window, downscale, fixed `input_len`, pad value —
  written by escapepod-models from the same `DataConfig` that framed every
  training example), and the detector preps from that declaration instead of
  hardcoded constants. `input_len` is declared redundantly and cross-checked:
  a self-inconsistent block is a load error, not a silent preference. Bundles
  without the block keep the legacy rna004 defaults, which is what their
  models trained with. `--info` prints the declared geometry.

- **The pinned boundary model's sha256 is verified at load** (#190,
  escapepod-models#56). The pinned adapter copy is the one bundle file the
  registry manifest does not hash, so the sidecar now declares it
  (`boundary.sha256`) and `demux` refuses to run pinned weights that do not
  match, naming both hashes. Applies only when the pinned file is what loads —
  an explicit `--cnn-model` already chose different weights deliberately.

- **The CTC-CRF runtime bundle is fetchable**: `escpod demux models fetch
  crf_nbc16_rna004` (#191), pinned at `barcode_crf_nbc16_rna004@v0.3.1` — the
  release whose metadata carries the input contract and sha declaration above.
  A CRF bundle is a directory, not a file, so the manifest grows `extras`:
  non-model files (metadata sidecar, pinned adapter copy, provenance)
  downloaded and sha256-verified exactly like members, into the member's cache
  directory. The cached directory is a complete `escpod demux --model <dir>`
  argument, and `models list` prints the exact command.

- **Batched CTC-CRF lattice decode on the GPU** (`crf::lattice_gpu`, `crf-gpu`).
  The same two passes as `crf::lattice`, the same edge indexing and the same
  tie-breaking, with the batch axis mapped onto the device. On by default
  wherever the kernels load; `ESCAPEPOD_CRF_GPU_DECODE=0` forces the CPU decode.

  **This reverses a documented decision.** `encoder_gpu` said the lattice decode
  was "sequential in time with a 256-wide inner dimension, a poor fit for the
  device". That is true of decoding *one read* and wrong for this pipeline: a
  device batch holds hundreds of independent reads, so a timestep is
  `batch * n_states` lanes (131 072 at batch 512) and only the `t_len` sweep is
  sequential — as it is on the CPU too. Profiling settled it: with the encoder on
  the device the AVX-512 decode was 66% of all remaining CPU cycles and the
  pipeline had stopped scaling with cores. bonito agrees; its own forward/backward
  scores come from koi's `ctc.{fwd,bwd}_scores_cu_sparse` CUDA kernels.

  **What it actually buys is cores, not wall time.** 40 k real RNA004 reads, one
  A30, same binary both ways:

  ```text
  threads     CPU decode   GPU decode
     2          51.8 s       30.5 s
     4          36.8 s       30.2 s
     8          29.5 s       29.8 s
    16          30.6 s       28.7 s
  ```

  Flat from 2 to 16 threads: the run now reaches full speed on **2 cores where it
  previously needed 8**, so the demux rule can hand a dozen cores back to the
  cluster. At 16 cores it is only 2.4% faster, because the bottleneck has moved
  off the CPU entirely (see the known limitation below).

  Correctness is exact: **40 001/40 001 identical barcode calls** against the CPU
  decode on the same binary — not merely the same sequences, the same calls.
  `tests/gpu_crf_lattice.rs` checks the kernels against `crf_golden.json`, the
  fixture produced by running real `bonito.crf.model.CTC_CRF` through koi's CUDA
  kernels, plus batch-invariance, an all-tied lattice (which exercises the argmax
  tie-break end to end), a `-inf` column, and time-major/batch-major equivalence.

  Fusions worth noting, since they shape the kernel layout:
  - **No transpose pass.** The CPU decode transposes each timestep into
    `[edge][dest]` for unit-stride SIMD; the GPU wants the opposite, so the
    kernels read the encoder's native `[dest][edge]` directly. That drops a
    kernel, a second `t_len * n_score` device buffer, and its round trip through
    memory — and makes the Viterbi tie-break key the flat index itself, because
    native order *is* bonito's `dest * n_edges + edge`.
  - **Forward and backward share one launch** (`blockIdx.y` picks the direction).
    They only read `scores`, so fusing them doubles the blocks in flight.
  - **onnxruntime's time-major output is decoded as it stands.** Rows are
    addressed by stride, so `[T][batch][n_score]` needs no host de-interleave —
    removing 1.5 MB per read of pure memory movement that previously had to
    complete before the decode could start.

- **The encoder's scores are decoded in place in device memory.** onnxruntime's
  output is bound to CUDA through `IoBinding::bind_output_to_device` and the
  decode kernels read it where it lies, so the largest object in the pipeline —
  `t_len * n_score` floats, 1.5 MB per read — never crosses PCIe in either
  direction. Previously onnxruntime copied it to the host and the decode uploaded
  it straight back: ~122 GB of round-trip traffic across 40 k reads, and the
  binding constraint once both the encoder and the decode were on the device.

  ```text
  40k reads, one A30, --method cnn --gpu
                         16 threads    4 threads    2 threads
  scores via host           27.3 s       29.8 s          —
  scores decoded in place   16.5 s       17.3 s       18.0 s
  ```

  1.65x on top of the GPU decode, and **4.3x against 0.7.0** (70.7 s -> 16.5 s).
  Peak RSS drops 4.4 GB -> 3.6 GB with the host score buffer gone. Two cores now
  beat what 0.7.0 did with sixteen, by 3.9x.

  Bit-identical: **40 001/40 001 identical rows** against the copying path on the
  same binary — full rows, not just the barcode calls. `ESCAPEPOD_CRF_GPU_ZEROCOPY=0`
  restores the copying path.

  The device pointer is only used after checking that the output actually landed
  on CUDA (`MemoryInfo::allocation_device`) and has the expected shape; if an
  execution provider declines the binding and returns a host tensor, this errors
  and names the escape hatch rather than handing a host pointer to a kernel.
  `IoBinding::synchronize_outputs` retires onnxruntime's stream before the decode
  kernels run on the lattice context's own stream.

- **`escpod demux --gpu` now runs the CTC-CRF encoder on the GPU.** The fused
  pipeline previously always took `produce_cpu_crf`, so a CRF bundle got tract on
  the CPU no matter what `--gpu` said; the `crf-gpu` encoder existed but was only
  reachable from `demux basecall --gpu`. Since the encoder is ~91% of that head's
  cost, `--gpu --method cnn` left the device essentially idle — measured on a real
  1M-read run, 0–6% GPU utilisation while 11.6 of 16 cores were pinned.

  The new `produce_gpu_crf` mirrors the DTW-SVM `produce_gpu`: parallel CPU prep
  (decode, batched detect, window standardisation) feeds a dedicated encoder
  thread over a bounded channel, so prep for the next block overlaps inference on
  the current one instead of the two alternating.

  Measured on 40 k real RNA004 reads, one A30, 16 cores (the allocation the
  production rule requests), against 0.7.0 with the same model, same input and
  the same `--method cnn --gpu`:

  ```text
                        wall     reads/s   GPU util (mean / median / p90)
  0.7.0 (CPU encoder)   70.7 s      566     1.0% /  0% /  0%
  this change           29.5 s     1355    22.1% / 25% / 40%
  ```

  2.4x end-to-end, and the device goes from idle in 91% of samples to busy in
  82% of them. Barcode calls agree with the CPU encoder on 99.97% of reads
  (39,989/40,001); the residue is tract-vs-onnxruntime numerics in the encoder,
  the same single-base-indel disagreement already documented for
  `demux basecall --gpu`, and confidence margins differ on 0.4% of rows.

  `ESCAPEPOD_CRF_GPU_BLOCK` (default 4096 reads) sizes the host-side handoff.
  It is deliberately not the device batch — `ESCAPEPOD_CRF_GPU_BATCH_ROWS` is
  that — and measuring 1024/4096/16384 showed under 5% between them, so it is a
  memory knob rather than a throughput one.

- **Adapter detection gets GPU 0 to itself when more than one is visible**, and
  the encoder pool takes the rest. Previously both roles round-robined over all
  devices, so device 0 carried detection *plus* a share of the encoding while the
  others carried only encoding. On 1 M reads across two A30s: 447.6 s -> 431.2 s,
  and 510.6 s -> 431.2 s against a single GPU (1.18x). Collapses to shared
  placement on a single-GPU host, so nothing changes there.

- **`ESCAPEPOD_CRF_GPU_TRACE=1` now splits adapter detection into its host and
  device halves**, which is what finally located this pipeline's bottleneck.
  On 1 M reads:

  ```text
  producer   detect 406.1s (host 4.8s + device 401.0s)  prep 0.9s  BLOCKED on send 0.0s
  2 workers  encode+decode 153.8s  match 11.0s  route 0.7s  BLOCKED on recv 683.8s
  wall 425.4s — worker busy 19% of its wall, producer busy 96%
  ```

  **Detect is 401 s of device time in a 425 s run — 94% of the wall**, and 99% of
  detect is on the device rather than in host prep. The encoder workers idle 82%.
  Every previous attempt to explain the idle GPU as a scheduling problem was
  aimed at a stage that was already idle; this is the measurement that says so.

  For scale: CPU detection through tract is 0.915 ms/read on 16 threads against
  the GPU's 0.240, so moving detection to the host is not an option either — it
  is 3.8x slower already, and there is only 0.2 s of host-side prep to
  parallelise.

- **A pool of CRF encoder workers, spread over every visible GPU**
  (`ESCAPEPOD_CRF_GPU_WORKERS`, default two per device, capped at 8). Each worker
  holds its own onnxruntime session and lattice context pinned to one ordinal,
  and they pull from a single shared channel rather than a partition, so a slow
  device cannot leave one worker holding the tail.

  Device placement is dynamic and needs no flag: the count comes from the
  *visible* devices, so `CUDA_VISIBLE_DEVICES` and SLURM `--gres=gpu:1` collapse
  it to a same-device pool automatically. Nothing changes for single-GPU users.

  **Measured gain was small when this landed, and the reason is worth keeping**:
  17.6 s → 15.7 s on 40 k reads with two A30s (1.12x), not the ~2x the device
  count suggests. The pipeline was not device-bound — the GPU idled ~75% of the
  time — so a second GPU mostly added idle capacity. Both causes of that idling
  have since been fixed (the boundary-prep shape churn, and reader threads
  starving the pool; see *Fixed* and *Performance*), and the device now runs at
  90%+ for three quarters of a 1 M-read run on one A30. The two-GPU numbers
  predate those fixes and have not been re-measured.

  Note the pool must be sized against device memory, not just device count — see
  the `DEVICE_ROW_BUDGET` entry under *Fixed*.

  Output is unaffected: 40 001/40 001 identical rows at one, two and four workers.

### Fixed

- **The boundary CNN was never prepped the way it was trained** (#187).
  escapepod-models builds every training example through
  `dataset.py::prepare_signal`, which pads to a fixed
  `input_len = (max_obs_trace - min_obs_adapter) / downscale` = 1500 with
  `score_excl = -5.0`; its own inference path does the same. escpod instead
  truncated to `search_window + ESCAPEPOD_CNN_MARGIN` (550 + 256 = 806) and
  padded with nothing, so the model has been running off-distribution since the
  CNN detector shipped. The truncation's premise did not hold either — the
  graph's receptive field is wider than the margin assumed, so an output at the
  far edge of the search window depended on input the truncation removed.

  Scored against escapepod-models' move-table gold labels (20,303 reads, the
  same yardstick used to rank boundary architectures), matching the training
  convention is an improvement on every metric:

  ```text
  prep    MAE    median   +/-200   +/-500
  old    111.8    30.0    0.9146   0.9654
  new    111.1    30.0    0.9149   0.9658
  ```

  Of the 38 labelled reads where the two preps disagree, 24 are closer to gold
  under the new prep and 14 under the old.

  It was also the fused GPU pipeline's bottleneck. The window clamps to the read
  end, so every read shorter than `max_obs_trace` got its own length — 680
  distinct shapes over one production run's 527 k reads, so a block issued one
  large onnxruntime call plus ~679 tiny ones, each paying fresh cuDNN plan
  selection. Detection measured **401 s of device time in a 425 s wall (94%)**
  against 0.009 ms/read for the same kernel at a steady shape. One fixed length
  fixes both: **401 s -> 20.2 s**.

  Expect adapter positions to move on a minority of reads. That is the point of
  the change, and the gold comparison above is the evidence for its direction.

- **`ESCAPEPOD_CRF_GPU_WORKERS` could kill or hang a run.** Every encoder worker
  sharing a device allocates its LSTM activations from that device's VRAM, but
  each took a full `ESCAPEPOD_CRF_GPU_BATCH_ROWS` batch, so asking for more
  workers multiplied device memory instead of dividing the work. Four workers on
  one 24 GB A30 hit 49 allocation failures and then died with a generic
  `CudaCall` error on a `Reshape`, followed by `corrupted double-linked list` —
  the halve-and-retry recognised all 49 but could not save the run, because by
  then the CUDA context was wedged.

  Both failure modes are bad in different ways. One run aborted with exit 134
  after writing **997,211 of 1,001,307 reads**, so a caller that checked only the
  output and not the exit code would have lost a block of reads silently. Another
  did not abort at all: it hung holding the full 24 GB for as long as it was left
  running, pinning the device against every other job on the node.

  Rows are now a **per-device budget** (`DEVICE_ROW_BUDGET`, 1024) split across
  the workers on that device, so total in-flight rows stay flat as the pool
  grows. The default two-workers-per-device configuration is unchanged at 512
  rows each; an explicit `ESCAPEPOD_CRF_GPU_BATCH_ROWS` still wins and is still
  stated per worker. The pipeline logs the resolved rows/call alongside the
  worker count.

  The configuration that previously aborted now completes: four workers on one
  A30, 1 M reads, 256 rows/call, 112.0 s, zero allocation failures, all
  1,001,307 reads written.

- **`escpod demux --gpu` aborted at exit roughly half the time**, after writing
  every read correctly. onnxruntime's CUDA provider reads freed memory during
  onnxruntime's *own* at-exit teardown; valgrind, on a run whose output was
  complete:

  ```text
  Invalid read of size 8
     at  libonnxruntime_providers_cuda.so
     by  libonnxruntime.so.1.27.1
     by  <Arc<ort::environment::Environment>>::drop_slow
     by  ort::environment::release_env_on_exit
     by  _dl_fini
  Address .. is 1,258,744 bytes inside an unallocated block of size 1,360,656
  ```

  Ten errors, five contexts, all at teardown; **none in the processing loop**.
  glibc notices at the last `free` and aborts with `corrupted double-linked
  list`. Measured 5 runs in 10 against 0/10 for 0.7.0 — which never hit it
  because it ran the CRF encoder on CPU tract and only gave the CUDA EP the
  small adapter CNN, so there was far less provider state to unwind.

  Output was never affected (every trial wrote all reads), but the process
  exited non-zero, which a workflow engine reads as a failed job.

  `release_env_on_exit` calls `ReleaseEnv` only when the last
  `Arc<Environment>` drops, and every live `Session` holds one, so the fix is to
  keep our sessions alive past `main`: the faulty path never runs. Verified back
  to **0/10**. Narrow by construction — our own destructors still run, writers
  are already joined and outputs renamed before that point, and it only applies
  when a GPU path created ORT sessions. Worth revisiting after an `ort` or
  onnxruntime bump; the code comment carries the trace needed to re-test.

- **The zero-copy binding named device 0 regardless of which device its session
  was on.** Harmless while there was only ever one device; with a worker pool it
  failed the run outright — onnxruntime resolves a bound output's allocator
  against the session's device and reports `Failed to find allocator for device`.
  Each encoder now carries its ordinal and binds against it.

- **`is_oom` missed two of the three wordings a device out-of-memory arrives in**,
  so the halve-and-retry both GPU paths depend on never fired and a batch that
  only needed splitting killed the run. onnxruntime's BFC arena says "Failed to
  allocate memory for requested buffer" and never "out of memory"; the driver
  says `CUDA_ERROR_OUT_OF_MEMORY`, whose underscores the old substring missed.
  Reproduced at `ESCAPEPOD_CRF_GPU_BATCH_ROWS=2048`, which now halves and
  completes. The `lattice_gpu` unit test covering the driver spelling had never
  run — `crf-gpu` is not in the workspace default features, so it was compiled
  out of every suite invocation, and its assertion had been wrong since it was
  written.

- **The GPU CRF decode's time-major transpose was serial, and it — not the
  device — was the bottleneck.** `split_time_major` de-interleaves onnxruntime's
  `[T, batch, n_score]` output into per-read buffers: 786 MB in and 786 MB out
  per 512-read batch at the RNA004 geometry, single-threaded. It capped the fused
  GPU path at ~700 reads/s on *any* thread count (16/32/48 all within 5%) while
  the CPU-encoder path kept scaling to 1150. Parallelising it over the batch axis
  (order-preserving, so read alignment is unaffected) took the same workload from
  57.2 s to 27.9 s — 2x, on top of the win above. `demux basecall --gpu` uses the
  same code and gets the same speedup.

- The fused CRF path no longer materialises the whole transpose before decoding:
  it gathers each read straight into the decode's input buffer, one buffer per
  rayon worker rather than one 1.5 MB allocation per read. Wall time is unchanged
  — the parallel transpose above is what mattered — but peak RSS no longer
  carries a second full copy of the score batch, which is what made raising
  `ESCAPEPOD_CRF_GPU_BATCH_ROWS` expensive (that knob still scales onnxruntime's
  own output tensor: 4.33 GB peak at 512 rows, 7.06 GB at 2048).

### Changed

- `escpod demux --gpu`'s "no effect here" warning for a CRF model no longer
  requires `--method` to be something other than `cnn`. The old gate silenced the
  warning for exactly the combination that looks most accelerated and is not —
  `--gpu --method cnn` against a CPU-only encoder — which is how a production run
  spent its time CPU-bound on a GPU node without a word of warning. Binaries
  built with `crf-gpu` no longer warn at all, because the flag now does something.

- `--gpu` is available on `demux` whenever any of `gpu`, `cnn-gpu` **or**
  `crf-gpu` is compiled in; previously a `crf-gpu`-only build had no such flag.

## 0.7.0 (2026-08-04)

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

- **The fused `escpod demux` pipeline drives the CRF head**, so
  `escpod demux in.pod5 --model <bundle> -d out/` produces barcoded POD5s with
  nothing else on the command line. Previously the CRF was reachable only as
  `demux basecall`, which takes boundaries as input, so using it meant
  `detect → basecall → split`: two intermediate files and three passes over the
  POD5. `--model` now accepts a CRF bundle directory — sniffed by the sidecar's
  `format` key rather than the path — and runs detect → prep → basecall → match
  → route, decoding each read once.

  Verified against the 3-step path on 4,000 reads with the same detector,
  encoder, and references: all **3,993 shared reads get an identical call** and
  every barcode bin matches exactly. The fused path emits **4,000 rows to the
  3-step path's 3,993** — `basecall` drops reads with no usable window so
  `split` never sees them, whereas the fused path routes them `unclassified`
  like the other heads. Output now reconciles with input.

- **CRF bundles describe themselves, and `--info` interrogates one.**
  `metadata.json` gains optional `barcodes`, `boundary`, `model`, and `metrics`
  keys, so a bundle carries its own references and pinned detector instead of
  needing `--barcodes`, `--method`, and `--cnn-model` on every invocation.

  Carrying references in the bundle is not just ergonomics. The CRF has
  `state_len=4` and emits `target[4:]`, so a hand-written CSV of full-length
  targets still calls the same barcode but inflates every edit distance by 4 and
  **compresses the confidence margin** that `--min-margin` and `--recovery` rank
  on. Measured on 20,000 reads with the shipped weights: median `best_dist` 4 →
  0, margin median 10 → 12/13, distinct margin values 12/11 → 15/14. Deriving
  them at bundle-build time from the encoder's own `state_len` removes the
  failure mode rather than documenting it; `--barcodes` remains as an override.

  `--info` prints identity, geometry, references with their minimum pairwise
  distance, the pinned detector, published metrics, and caveats, then exits
  without touching a POD5.

### Fixed

- **The two CRF entry points computed picoamps differently.** `demux basecall`
  and `escapepod_python::adc_to_pa` use a fused `adc.mul_add(scale, offset *
  scale)`; the fused `demux` pipeline used an unfused `(adc + offset) * scale`
  despite a comment claiming it matched the reference — two roundings instead of
  one, ~1 ulp apart. Both now use the reference form. `demux basecall` is
  unaffected; the fused pipeline's encoder input shifts by ~1 ulp, which moved
  the reported confidence on 1 of 992 and 7 of 9943 reads in a parity run and
  changed **no** barcode call.
- **The router's memory budget is honoured up to 768 barcodes, not 192.**
  `ROUTER_TOTAL_SLOTS` is meant to cap queued-read memory regardless of barcode
  count, but the per-barcode depth was clamped to a floor of 256, so past 192
  barcodes the total scaled with the barcode count again — the exact behaviour
  the budget replaced. The CRF head takes its references from a user-supplied
  CSV, so the count is unbounded, and a 384-plex set sat at roughly twice the
  budget. The floor is now 64 and any overshoot is logged. No shipping design
  changes: the floor does not bind until 768 barcodes.
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

### Performance

- **The CRF producer fans out per read, not per 256-read chunk.**
  `produce_cpu_crf` chunked and then walked each chunk with a serial loop — a
  shape copied from `produce_cpu_gbm`, where it is correct because
  `predict_many` is a genuinely batched kernel. tract has no batched LSTM, so
  the CRF chunk only serialized its reads. At ~14 ms per read one chunk is
  ~3.6 s of work a starved worker cannot steal, and a block with fewer than
  `256 × threads` reads could not fill the machine at all (1,000 reads produced
  four tasks for 32 cores). `for_each_init(CrfScratch::new, …)` gives per-read
  work stealing and moves the scratch from per-chunk to per-worker: **2.25× at
  1k reads** (4.91 s → 2.19 s) and **1.36× at 10k** (23.3 s → 17.0 s) on 32
  cores. The gap narrows as chunk count catches up to core count, which is the
  predicted shape.

- **Barcode matching abandons early, as it was always documented to.**
  `edit_distance` passed WFA `cap = a.len() + b.len()` — the largest score any
  alignment can reach — so the `max_score` guard could never fire and every
  reference ran to its true optimum *and* did a full traceback, defeating the
  reason WFA was chosen over a DP. Each comparison is now capped at the running
  runner-up: both branches of the selection loop already discard any
  `d >= second`, and `second` is monotonically non-increasing, so a reference
  that cannot beat it never needs its exact distance. Per comparison, 7.74 µs →
  5.65 µs at 16 references and 7.74 µs → 2.47 µs at 96 (**3.1× per read**,
  ~743 → 237 µs) — the cap tightens faster with more references, so this scales
  *with* the barcode design rather than against it. Output is unchanged field
  for field, pinned by a 2,400-case test against the previous implementation.

- **The encoder no longer copies 1 MB per read for nothing.**
  `basecall_prepped` decoded from an owned `Vec` that `encode` filled element by
  element (`t_len * n_score` floats, 1 MB for RNA004), which `decode_with`
  immediately transposed into `CrfScratch` and dropped. It now decodes straight
  out of tract's output tensor, and the decode backend is resolved once at load
  instead of re-probed per read.

- **Only the encoder's window is calibrated.** `prep` needs `chunk` samples of
  pA ending at `adapter_end`, but callers converted the entire decoded prefix
  first — 16,000 samples under the CNN detector (8× the window) and the whole
  read under LLR, which sets no decode bound. `prep_adc_into` fuses calibration
  and standardisation into one pass over exactly the 2,000 samples the encoder
  sees, into a per-worker buffer.

- **`basecall_batch` no longer retains every read's scores.**
  `CrfEncoderGpu::basecall_batch` encoded the whole caller batch before decoding
  any of it. `ESCAPEPOD_CRF_GPU_BATCH_ROWS` bounds *device*-side activations,
  not the host, and the scores coming back are 1 MB per read for RNA004 — so an
  Arrow batch of a few thousand reads retained gigabytes of host RAM regardless
  of that knob. Encode and decode now alternate one device batch at a time,
  capping host high-water at `batch_rows` reads' worth. Chunk boundaries are
  unchanged, so the scores are identical. This is a memory fix, not an overlap
  of encode with decode.

### Changed

- **Breaking: `escpod demux` and `escpod demux detect` require `--method`**
  when the model does not pin one. `--method` previously defaulted to `llr`,
  and LLR boundaries cost **17.2 points** of barcode recall against the same
  classifier (0.9928 → 0.8196) while failing silently — the run succeeds and
  the output looks plausible. A bundle may now pin its detector, supplying both
  method and weights; an explicit `--method` overrides that pin, except that a
  bundle pinned to `cnn` **refuses** `--method llr`; with neither, the command
  errors out naming the tradeoff instead of quietly picking the worse detector.
  Scripts relying on the implicit default must add `--method llr` to keep their
  current behaviour — which is the point, since that default was silently
  costing 17.2 points.

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

### Build / Tooling

- `ort` moves to `2.0.0-rc.13` and the CUDA execution provider is configured in
  one place, shared by the `cnn-gpu` and `crf-gpu` paths instead of being set up
  independently in each.
- noodles bumps: `noodles-bam` 0.92, `noodles-sam` 0.87, `noodles-csi` 0.58
  (together, since their APIs move as a set), and `noodles-bgzf` 0.49.
- CI clippies the opt-in feature builds (`train`, `gpu`, `cnn-gpu`, `crf-gpu`)
  and the CLI, which previously went unlinted — `crf-gpu` in particular did not
  compile under `-D warnings` on `main`.

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

- **Barcode matching abandons early, as it was always meant to.** `edit_distance`
  passed WFA a cap of `a.len() + b.len()` — the largest score any alignment can
  reach — so the `max_score` guard could never fire and every reference ran to
  its true optimum plus a full traceback. The module was written around WFA
  precisely because its work scales with edit *distance*, so the cap was
  defeating the reason for the algorithm choice. Each comparison is now capped at
  the running runner-up, which both branches of the selection loop already treat
  as a discard threshold, so the result is unchanged field for field: per
  comparison 7.74 µs → 5.65 µs at 16 references and → 2.47 µs at 96 (the cap
  tightens faster with more references, so a 96-plex read goes ~743 µs → 237 µs,
  3.1×). Pinned by a 2400-case test against the previous implementation.
- **The CRF encoder no longer copies its scores out.** `basecall_prepped`
  decoded from an owned `Vec` that `encode` filled element by element —
  `t_len * n_score` floats, 1 MB per read for RNA004 — which the decode's first
  loop then immediately transposed into `CrfScratch` and dropped. It now decodes
  straight out of tract's output tensor. `encode` still exists for callers that
  want to own the scores. The decode backend is also resolved once at load
  instead of re-probed per read.
- **Only the model's window is calibrated, not the whole decoded prefix.**
  `prep` needs `chunk` samples of pA ending at `adapter_end`, but callers
  converted every decoded sample to pA first — `max_obs_trace` (16 000, 8× the
  window) under the CNN detector, and the *entire read* under LLR, which sets no
  decode bound at all. `prep_adc_into` fuses calibration and standardisation into
  one pass over exactly the 2000 samples the encoder sees, writing into a
  per-worker buffer.

- **Fused `demux` CRF head is 1.36–2.25× faster** (10k reads 23.3 s → 17.0 s;
  1k reads 4.91 s → 2.19 s, 32 cores), bit-identically: per-read barcode calls
  and every per-barcode POD5 are unchanged. `produce_cpu_crf` fanned out with
  `.chunks(256)` and then ran the chunk *serially*, mirroring `produce_cpu_gbm`
  — but that head chunks because `predict_many` is a genuinely batched kernel,
  whereas tract has no batched LSTM, so the CRF chunk only ever serialized its
  reads. At ~14 ms per read (13 ms encode + 1.2 ms decode, measured on rna) one
  chunk is ~3.6 s of work no starved worker can steal, and a block with fewer
  than `256 × threads` reads cannot fill the machine at all — 1000 reads made
  just 4 tasks for 32 cores. Now `for_each_init`, which also moves `CrfScratch`
  from per-chunk to per-worker.
- **`CrfEncoderGpu::basecall_batch` no longer retains every read's scores.**
  Encode and decode now alternate one device batch at a time instead of
  encoding the whole caller batch first. `ESCAPEPOD_CRF_GPU_BATCH_ROWS` bounds
  only the *device*-side activations; the scores coming back are `t_len *
  n_score` floats — 1 MB per read for the RNA004 geometry — so a several-
  thousand-read Arrow batch held gigabytes of host memory regardless of that
  setting. Host high-water is now `batch_rows` reads' worth. The prepped
  windows are also borrowed rather than cloned on the way to the device,
  dropping one 8 KB copy and one allocation per read.

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

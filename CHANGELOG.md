# Changelog

## Unreleased

### Fixed

- **The reads table is written in batches again, instead of one batch per file**
  (#297). `filter`, `merge`, `subset` and `split` build the reads table with
  `build_reads_table{,_remapped}` rather than `Writer`, and those wrote the
  whole table as a single Arrow record batch however large it was — so
  `read_batch_size` on `FilterOptions`/`MergeOptions`/`SubsetOptions` was
  declared, defaulted, and never read. `escpod filter` asking for 10,000 rows
  per batch wrote 40,000 reads as **one**.

  Measured, for scale: MinKNOW writes ~10,000 reads per batch (1,575,748 in
  158) and the pod5 Python package exactly 1,000 (17,919,658 in 17,920).
  Neither writes a whole file as one batch.

  It is not cosmetic for anything that reads escpod's output back. `demux`
  shards its reader threads by batch index and only emits a block at a batch
  boundary, so a single-batch file is read by exactly one thread whatever
  `ESCAPEPOD_DEMUX_FILLERS` says, and nothing reaches the GPU until the entire
  file has been decoded. On a 100k-read escpod-written file that was 11.2 s of
  a 19 s stage with the card idle, and it made every tuning knob look flat.
  Fixing it took that stage to 11.3 s with the first block arriving in 0.3 s
  and GPU utilisation going from 22-30% to ~60%.

  Every default is now 1,000, and the five CLI sites that overrode it (merge at
  100,000, the rest at 10,000) inherit it, so the geometry has one definition.
  Note this never affected MinKNOW input, which was already many-batched:
  measured on a real 1.3M-read run, GPU utilisation is ~94% and the first block
  arrives in 0.4 s both before and after.

### Performance

- **The CRF encoder pool uses every visible GPU instead of reserving one for
  adapter detection** (#297). Detection is ~5% of device time since #187, so
  holding a whole card for it left that card 12% busy on two GPUs and 27% on
  four while the lone encoder device pinned at 93%. Measured on a real 1.3M-read
  run, interleaved, 2 reps:

  ```text
  GPUs   reserve device 0            encode everywhere        vs 1 GPU
    1    162.5 s  (gpu0 88%)         same policy              --
    2    136.6 s  (12% / 93%)         80.8 s  (88% / 88%)     1.19x -> 2.01x
    4     56.7 s  (27% / 75%)         55.3 s  (72% / 58-66%)  2.86x -> 2.94x
  ```

  Two GPUs go from 1.19x to **2.01x**. Four are a wash — the pool was already
  wide enough there.

- **`--ref-scores` runs its reference scan on the GPU instead of on the host**
  (#297). The flag that gates production demuxing was the one configuration
  with no GPU decode path: `try_run_and_decode_with_refs` always copied the
  whole score tensor back and ran the CPU lattice decode, where the plain path
  decodes on the device and never copies at all. Over the isolated CRF stage on
  100k reads that was 19.7 s at 656% CPU against 12.4 s at 114% without the
  flag — +57% wall for five and a half extra cores, with the card idle.

  The constrained scan is now a CUDA kernel (the one #241 left unwritten): one
  block per read, a grid-stride loop over chain cells, both alpha buffers
  double-buffered in shared memory, and `logZ_full` reduced on the device so
  only `n_refs` floats per read come back rather than a strided gather over a
  157 MB alpha buffer. It runs between the two decode passes by necessity, not
  by choice — it needs the raw scores that pass 1 overwrites in place.

  End to end on one A30 over 100k reads, arms interleaved in one allocation:
  **38.7 s → 31.5 s wall (1.23x), and 5.9 cores → 1.3 cores.** Barcode calls are
  identical for all 100,654 reads; `crf_logp`, `crf_margin` and `mean_logpost`
  agree to 2e-4, which is the output's own print precision.

  Note this does **not** close #297. GPU utilisation is unchanged at ~22%, so
  removing 4.5 cores of host work from the critical path bought only 20% wall.
  That is evidence for the issue's "overlap-bound, not throughput-bound"
  framing: the pipeline is waiting on something structural, not on compute.

  A panel that does not fit the kernel's shared memory, or whose fan-in exceeds
  the fixed accumulator, falls back to the CPU scan — slower, never wrong.

## 0.17.2 (2026-08-29)

### Fixed

- **Every command that takes POD5 input accepts a directory, and refuses a
  path that does not exist** (#293). `escpod demux basecall <run>/pod5` logged
  one WARN, wrote a header-only CSV and **exited 0** — a result nothing
  downstream can tell from a run where no read passed. In the report that
  found this, the empty table became a per-barcode share over zero rows and
  reported `0.00x enrichment`, which is a *meaningful value* in that analysis:
  a failed run rendered as a clean negative result, caught only by noticing it
  finished in 85 s.

  `demux fingerprint` had the same failure one stage earlier (and without even
  the warning), and `demux detect`, `demux split`, the fused `demux` pipeline
  and `subset` died on a directory with a bare `No such device (os error 19)`
  from the mmap. None of the six called `resolve_pod5_inputs`, which
  `merge`/`view`/`index`/`annotate`/`repack`/`filter`/`resquiggle`/`signal
  classify` have always used; they all do now, so a directory expands to the
  `*.pod5` under it, a missing path is `Path does not exist:`, and an empty
  directory is `No POD5 files found in directory:`. No `-r` flag: escpod's
  directory expansion is recursive everywhere already, and one command
  needing an opt-in would be the odd one out.

  `subset` gains multi-input support as a consequence, via the `subset_files`
  that `demux split` already used: a group whose reads span several files of a
  run comes out as one output rather than needing a `merge` afterwards.

  A POD5 that fails to open or decode *after* that validation is now fatal in
  `demux fingerprint` and `demux basecall` rather than skipped. Those paths
  swallowed a truncated or corrupt file and wrote a short table with a zero
  exit; `demux detect` has always propagated the same three errors, and the
  point of #293 is that the stages of one pipeline should not disagree about
  what counts as a failure.

## 0.17.1 (2026-08-26)

Recovers the v0.17.0 release. 0.17.0's GPU artifact failed to build, which
skipped the `Create Release` job — so 0.17.0 reached PyPI but has no GitHub
Release and no binaries. PyPI does not allow re-uploading a version, so
v0.17.0 is superseded rather than corrected: **0.17.1 is the first 0.17.x
with downloadable binaries**, and carries the same changes as 0.17.0 plus
the two fixes below. The `v0.17.0` tag was deliberately not moved, so it
still points at exactly what PyPI was built from.

### Build / Tooling

- **PyPI can no longer publish a version whose binaries failed to build.**
  This is the second half of the v0.17.0 failure, and the part that made it
  unrecoverable. `publish-pypi` needed only `[wheels, sdist]`, so it was
  independent of the CLI builds by design — the stated intent being that a
  wheel failure should never block the GitHub Release. Run in the other
  direction that same independence meant the GPU binary could fail, the
  `release` job be skipped, and PyPI publish regardless. 0.17.0 is therefore
  installable with `pip` but has no GitHub Release and no binaries, and since
  PyPI refuses re-uploads, that half cannot be withdrawn or corrected — only
  superseded.

  The rule now is that **the irreversible step goes last**: `publish-pypi`
  waits on `release`, which waits on `build`, `build-gpu`, `wheels` and
  `sdist`. A GitHub Release can be deleted and re-created and the workflow
  re-run; a PyPI upload cannot, so it happens only once everything that could
  still fail already has not. Under this graph v0.17.0 would have published
  nothing at all, which is the correct outcome.

  Making `release` wait on the wheels fixes a second, quieter bug. The wheels
  and sdist are attached to the GitHub Release as assets, but the job
  downloads artifacts with no pattern — so the asset list was decided by a
  race, and a wheel job slower than the binaries would have been silently
  omitted. It never happened, which is exactly why it was worth closing.
  `SHA256SUMS.txt` now covers the wheels as well, which it never did.

  `skip-existing` is set on the publish step. With PyPI last, re-running a run
  that failed *after* publishing would otherwise die on a duplicate upload
  with the GitHub Release still broken — the re-run needs to be able to reach
  the thing that failed.

### Fixed

- **The v0.17.0 GPU release artifact failed to build, and no GitHub Release
  was published.** `ort` was declared without `default-features = false`, so
  its default `download-binaries` came along and dragged in `ureq` with
  `native-tls`, and therefore `openssl-sys`. Under `load-dynamic` — which is
  how this crate uses `ort`, and the reason nothing CUDA is needed at build
  time — that machinery never downloads or links anything, so the whole TLS
  stack was dead weight in the graph. It became fatal in #277's new artifact
  job, which builds in a `manylinux_2_28` container carrying no
  `openssl-devel`: `openssl-sys`'s build script failed, and since `release`
  needs `build-gpu`, the GitHub Release for v0.17.0 was skipped entirely.
  (The wheels were unaffected and 0.17.0 is on PyPI.)

  `ort` now takes `default-features = false` and restates the four defaults
  actually used (`std`, `ndarray`, `tracing`, `api-27`) alongside `cuda` and
  `load-dynamic`. `api-27` is kept explicitly rather than dropped, so the
  required onnxruntime API version is unchanged. This removes 19 packages
  from `Cargo.lock` — `openssl*`, `native-tls`, `schannel`,
  `security-framework`, and the rest of that subtree — with no additions and
  no version changes, and no code change of any kind.

  **CI could not have caught this**, which is the more interesting half. The
  `gpu` feature *is* built on every PR, but on `ubuntu-latest`, which has
  `libssl-dev` — so `openssl-sys` compiled happily there and the break
  surfaced only in the release container, on a tag, after PyPI had already
  published. The feature-builds job now asserts `openssl-sys` is unreachable
  under both `gpu` and `models-download`, over `--target all -e all` so a
  target-specific or build-dependency edge counts too. Every TLS user in this
  workspace is meant to be rustls: the musl artifacts are statically linked
  and the GPU artifact builds in manylinux, and neither has an OpenSSL to
  find.

## 0.17.0 (2026-08-26)

### Added

- **A GPU-enabled Linux release artifact (#270).** Every release from v0.10.0
  to v0.16.1 published four `escpod` archives and none of them could carry
  `--features gpu`: the Linux builds are static musl, and every GPU path
  `dlopen`s its runtime (the CUDA driver and `libnvrtc` for DTW, a CUDA-enabled
  `libonnxruntime` for CNN detection and the CTC-CRF encoder). So the feature
  was unreachable from a release by construction, and the only way to get it was
  `cargo install --git`, which needs a Rust toolchain and — the part that
  actually bites — produces a binary whose version is a local fact. escpod
  computes `adapter_end`, `adapter_end` defines the training window, so the
  binary is part of a model's definition; a downstream consumer pinning
  `escpod_version` found four hand-built binaries on one machine, three of them
  referenced by config and none matching the pin.

  Releases now also publish `escpod-<ver>-x86_64-unknown-linux-gnu-gpu.tar.gz`,
  built `--features gpu`. musl stays the portable default and the thing an
  unattended installer should fetch; this is the one dynamically linked Linux
  artifact, and the name says so. Without `--gpu` it behaves exactly like the
  musl build.

  Two decisions worth reviewing. It builds inside a `manylinux_2_28` container
  rather than natively on the runner, because a runner build links against
  Ubuntu 24.04's glibc 2.39 and the RHEL 9 compute nodes this exists for are on
  2.34 — it would not start on the machines that asked for it. The image name
  fixes the floor at 2.28, and a build step then asserts it off `objdump -T`
  rather than trusting the image, so a base bump fails the release instead of
  shipping a binary nobody can run. And the CUDA expectations are appended to
  *every* release's notes by the workflow rather than written into each
  CHANGELOG entry: the required CUDA major is fixed by the `ort`/`cudarc` pins
  the binary was built with, is not discoverable from the file name, and a
  mismatch demotes the work to CPU with a warning rather than failing outright.

  Prebuilt binaries are now documented in the README and the installation guide,
  which previously described only `cargo install`.

### Changed

- **Docs: `demux`, `signal classify` and `index` are documented as shipped
  commands, not experimental ones.** All three are in the default build —
  `demux` for several releases, `signal classify` since #206, and `index` as of
  #283 — but the docs site still filed `demux` under *Experimental* (a section
  headed "may change without notice"), still told readers `index` needed
  `--features experimental`, and never mentioned `signal classify` at all: the
  whole `escapepod-classify` crate had zero hits across `docs/` and the README.

  `demux.md` moves to the CLI reference, `index` and `signal classify` get pages
  of their own, and *Experimental* is now exactly what its name says —
  `annotate`, `resquiggle`, `repack`.

  Swept up in the same pass, all of it drift since the last docs audit
  (2026-08-09):

  - The README's performance table cited numbers matching no run in
    `benchmarks/README.md`, against a `pod5` two point releases old. Replaced
    with the recorded 2026-07-26 run, and the "up to 9x" headline — true of a
    2026-03 measurement — with the 3–5x (bulk) / 20–50x (metadata) the current
    numbers support.
  - `--gpu` prose in the README and installation guide, left over from before
    #281 made it a hidden deprecated alias for `--device gpu`.
  - `api/index.md` said the workspace has five crates (six), and its `gpu` row
    read "Implies `gpu` + `gpu`, which remain individually selectable" — a
    rename artifact from #282 asserting the opposite of what that PR did.
  - The POD5 version history stopped at "4 | Current version"; escapepod has
    read V6 and deliberately written V5 since #267. The reads-table field list
    it sat under omitted every V1–V5 column.
  - `span_statistics`, `span_statistics_batch` and `AnchoredReads` are exported
    from the Python module and were undocumented.

- **The `.p5s` sidecar now caches the POD5 signal table's batch geometry, and a
  scattered fetch stops paying for it.** Row count is the one field an Arrow IPC
  footer does not carry — the footer's `Block` records offset, metadata length
  and body length, and nothing else — so recovering it means reading every
  record batch's own message header. That is one scattered mmap touch per batch,
  measured at 15–24 ms *each* cold on BeeGFS, and it is paid on every process
  start rather than once: a 33 GB file with 8866 signal batches spent 4.95 s
  there, **78% of a cold 5000-read scattered fetch**, warm or cold alike.

  `escpod index` now walks it once and records the answer in the sidecar under
  `escapepod:signal_batch_rows`, run-length encoded (a conformant file is one
  run and a short tail — `"8865x100,1x37"`, about 15 bytes however many batches
  it has). Measured on real data, footer parse drops from **4.95 s to 4.82 ms**
  on that file (8866 batches) and from 149 ms to 2.96 ms on a 3.7 GB file with
  266 batches, with byte-identical results — the same 226,235,036 samples over
  the same 5000 reads either way. Loading the read index from the sidecar
  instead of building it improves in the same pass (249 ms → 59 ms).

  **The POD5 is neither modified nor read differently.** The cheap fix would
  have been to read batch 0 and assume every batch matches — which is what the
  official `pod5` library and dorado do, and what `Reader::nonuniform_signal_batch`
  exists to catch them out on. Recording the *measured* counts costs a handful
  of bytes, is exact for a non-uniform file too, and means escapepod never makes
  that bet. The cached geometry is then checked rather than believed: it is used
  only if it has one entry per batch the footer describes, and the first and
  last batches are read for real and compared (the last is the short one, so a
  geometry from a file with a different read count disagrees there). Any
  mismatch logs a warning and falls back to the full walk, so a stale cache
  costs time and never correctness.

  Adding the key is not a `.p5s` version bump, on the same footing as the
  provenance keys: it is optional on read, a sidecar written before it existed
  still loads, and an older escpod ignores it. A sidecar bound to another POD5
  is rejected by the existing identity check before the geometry is looked at.
  A sidecar written by an earlier version gains the geometry on its next write
  of any kind; one that already has it carries it through untouched.

- **Every `.p5s` write records the geometry, and `escpod index` is no longer
  gated.** As first written, only `Reader::build_and_write_index` recorded it —
  and that made the cache close to unreachable, for two compounding reasons.
  `escpod index` is behind `--features experimental`, and **no release binary is
  built with it**, so nobody on a prebuilt `escpod` could run the command at all;
  meanwhile `demux --annotate`, which is the documented and default-build route
  to a sidecar, went through `write_columns` and never measured. The common
  workflow therefore produced a sidecar with a read index and no geometry, and
  then re-walked every batch header on each subsequent run.

  Three changes close that. `Reader::write_sidecar` is now the funnel every
  sidecar write goes through and records the geometry when the sidecar lacks it,
  so `demux --annotate`, `escpod annotate` and `escpod annotate --design` all
  produce a complete sidecar and a later path cannot forget. `escpod index` is
  ungated — it builds caches that are always rebuildable from an untouched POD5,
  which is a different proposition from `escpod annotate` (still experimental)
  writing data products that exist nowhere else. And `escpod index` no longer
  skips a sidecar that has an index but no geometry: its "already indexed" check
  predated the second cache, so the obvious remedy for a slow file reported
  success and did nothing. Annotations and scores are preserved by the rebuild,
  as before.

  A failed re-measure can no longer erase a recorded geometry, either.
  `Sidecar::set_signal_batch_rows` treated an empty vec as a value, and
  `Reader::measure_signal_batch_rows` returns an empty vec for every failure it
  has — no signal table, a slice past EOF, an unparseable footer. So one failed
  measure during `escpod index --force` would drop a geometry an earlier run had
  recorded correctly, and silently, since the sidecar stays valid and merely
  gets slow again. Empty now means "could not measure" and leaves the existing
  value alone; discarding one requires the new `clear_signal_batch_rows`.

- **A sidecar that will not load is no longer treated as a sidecar worth
  discarding.** Every write path answered a failed load the same way: `escpod
  annotate --force`, `demux --annotate` with overwrite, and `write_design` all
  matched `Err(_) if overwrite => Sidecar::new(scan)`, and
  `Reader::build_and_write_index` went further still with `Ok(None) | Err(_) =>
  Sidecar::default()` — no `--force` required. That collapses two situations
  that are not alike. `--force` means *this sidecar belongs to a POD5 that has
  since been replaced*, and replacing it loses nothing that applies here. It
  does not mean *this file is truncated*, or *this file was written by a newer
  escpod*, where the barcode and score columns a demux run spent hours on are
  still sitting there intact and the rebuild would overwrite them with an empty
  column set — quietly, since the result is a perfectly valid sidecar.

  The new `SidecarLoad` / `load_sidecar_for_write` split the cases: `Absent` and
  `Foreign` may be replaced, `Unreadable` is refused, and `--force` now only
  licenses the first two. Nothing is lost by refusing — the POD5 is untouched
  and the sidecar is still on disk, so a version this build cannot read stays
  readable by the build that can.

  `read_sidecar_metadata` gained the version gate `read_sidecar_file` already
  had. The two readers disagreeing is what makes check-then-write unsound:
  `escpod index` uses the cheap metadata read to decide whether a rewrite is
  needed, so a file the cheap read accepts and the full read rejects gets
  reported as fine and then fails once the rewrite is already committed to.

- **A concurrent sidecar update is no longer silently discarded.** The atomic
  write made a `.p5s` update all-or-nothing, which rules out a torn file and
  says nothing about a second writer. The case that actually happens is not
  corruption: a `demux --annotate` run that takes hours is still going when
  someone runs `escpod index` on the same file to make it faster. Index reads
  the sidecar as it is, demux finishes and writes its barcodes, index finishes
  and renames over them. Nothing errors, nothing is corrupt, and the
  classification is gone.

  `SidecarStamp` fingerprints the file (size + mtime) when it is read and
  re-checks immediately before the rename; `write_sidecar_file_checked` refuses
  the write if it changed. It is not a lock and does not pretend to be — two
  writers interleaving inside stat granularity can still race, and the remedy
  for that is not to run two writers — but an update that visibly landed in
  between is never thrown away without a word. `write_sidecar_file` keeps its
  unconditional behaviour for callers that did not read first.

- **`escapepod-pod5` no longer re-parses the signal footer inside every bulk
  fetch.** `get_signal_bulk_prefix`, `get_compressed_signal_bulk` and
  `signal_extractor` each called `ArrowIpcFooter::parse` on entry, bypassing the
  parse the `Reader` already had cached. Also sorts rows within a batch in
  `extract_signal_rows`, which previously visited them in request order — and a
  request assembled by iterating a `HashSet<Uuid>`, which is what `reads_by_ids`
  hands down, has no order at all. Neither change is measurable on the
  workloads tested (repeat walks are cheap once the region is mapped; a sparse
  pick lands under one target per batch), and both are kept as strictly less
  work. The cost they were suspected of is the one the geometry cache above
  actually removes.

- **BREAKING: `cnn-gpu` and `crf-gpu` are gone; `gpu` is the single GPU Cargo
  feature.** Anyone building with `--features cnn-gpu` or `--features crf-gpu`
  must switch to `--features gpu`, on `escapepod-cli` and `escapepod-demux`
  alike. There is no deprecation shim: a stale flag is a hard "feature does not
  exist" error at `cargo` resolution, not a silent behaviour change. `gpu` is a
  superset of both, so the fix is always the same substitution, and any build
  that already said `--features gpu` is unaffected — it already enabled all
  three.

  The granular flags were kept for "library consumers and CI isolation builds".
  Neither survives inspection. `escapepod-demux` depends on `fqxv-align` by git
  URL, so it and `escapepod-cli` cannot be `cargo publish`ed and have no
  external consumers to serve; the isolation builds only ever proved that a
  partially-GPU binary *compiles*, and nobody ships one.

  Their real cost was in the fused pipeline. `commands/demux/run.rs` carried
  three runtime warnings and their `cfg(all(not(…), any(…)))` guards whose sole
  job was to describe those builds — "this binary has `crf-gpu` but not `gpu`,
  so `--gpu` does nothing for DTW-SVM classify", and two variations on it. Once
  `gpu` is atomic those builds cannot exist, so the warnings are deleted rather
  than ported: `--gpu` now always means every stage with a device path uses it.
  The one warning that survives is the one about a *runtime* combination — GBM
  classify with `--method llr`, where the GPU genuinely has nothing to do.

  `escapepod-signal`'s `gpu` feature is deliberately **unchanged**: cudarc DTW
  only, no onnxruntime, and that crate has no git dependency blocking
  publication. CI's feature-build job drops from seven clippy steps to three —
  `escapepod-signal --features gpu`, `escapepod-demux --features gpu`,
  `escapepod-cli --features gpu` — covering the builds people actually produce
  instead of combinations they do not. No algorithm, numeric behaviour or CLI
  flag semantics changed; this is a feature-graph and `#[cfg]` change.

- **The pixi `gpu` environment now supplies CUDA itself, instead of borrowing
  the host's (#278).** `install-ort` pip-downloaded `onnxruntime-gpu` into
  `.pixi/ort/`, which was the only input to the GPU environment that `pixi.lock`
  did not describe — no version record, no checksum, re-fetched per clone. That
  is where a silent bug lived: PyPI's default build moved to **CUDA 13** while
  `[feature.gpu.dependencies]` pinned CUDA 12, so not one of the provider's
  sonames could be satisfied from `$CONDA_PREFIX/lib`.

  It never failed. Verified on an A30: `--gpu` worked, because the loader fell
  through to a system `/usr/local/cuda-13.0` our GPU nodes happen to carry. The
  conda packages were inert and the whole path rested on a CUDA install we
  neither chose nor control — which would have broken on any node without one.

  `libonnxruntime` is now an ordinary conda-forge package
  (`onnxruntime-cpp`, build-pinned to `*cuda129`), so the lockfile describes it
  like everything else. Deleted along the way: the `install-ort` task,
  `scripts/unpack_ort_gpu.py`, `scripts/gpu_env_activation.sh`, the `.pixi/ort`
  directory, and the "fetch this once on a networked node" step. Setup is
  `pixi run install-gpu`. `ORT_DYLIB_PATH` is now a static path the lock
  guarantees exists, so the script that used to export it *only if the file was
  there* — guarding a dangling value that hangs escpod silently at startup — has
  nothing left to guard.

  The CUDA-major mismatch is now a **solve error rather than prose**: the `gpu`
  environment targets a `linux-64-cuda` platform (`cuda = "12"`), which excludes
  the `_cuda130` build. This needs pixi ≥ 0.77, whose named platform variants
  scope `__cuda` to one feature; on 0.76 the only available spelling re-keyed the
  workspace's single platform and dragged every environment onto it, which is why
  the wheel was fetched out of band until now.

  Two things worth knowing. The explicit `libcublas` / `cuda-cudart` / `cudnn`
  pins are **not** redundant and were not removed: `onnxruntime-cpp`
  under-declares its dependencies, so dropping them puts the provider back to
  failing-as-a-slowdown. `libcufft` and `libcurand` *are* gone — this build,
  unlike the wheel, references neither. And `pixi install -e gpu` no longer works
  on a GPU-less login node, which on a cluster is where you install; the
  `install-gpu` task exists so that is one obvious command rather than a solver
  error about virtual packages.

  Non-GPU environments are byte-identical after the change; only `gpu` and
  `dev-gpu` move.

- **`--device auto|cpu|gpu` replaces the `--gpu` boolean (#270, #278).** The old
  flag failed in both directions at once, and the second one is what makes this
  worth doing rather than a rename.

  *Silent CPU.* GPU was opt-in, and the flag only **existed** when the matching
  Cargo feature was compiled in. Forget it — or run a release binary that has no
  GPU code at all — and you got the CPU path with nothing said. Boundary-CNN
  detection is ~7x slower there; a downstream consumer measured **37 minutes** on
  one flowcell before working out why.

  *Silent GPU fallback.* `--gpu` was a *request*. onnxruntime registers its CUDA
  execution provider best-effort and commits to the CPU provider when it cannot,
  so a broken runtime — a CUDA 13 `libonnxruntime` against a CUDA 12 library set,
  which only ever "worked" because the GPU nodes happen to carry a system
  `/usr/local/cuda-13.0` — produced a correct, slow run that looked accelerated.

  What is new:

  - `--device auto` (the default) puts a stage on the GPU only where the GPU
    actually wins, and only when the relevant feature is compiled in *and* a CUDA
    device is visible. **DTW classification stays on the CPU under `auto`**, on
    purpose: the CPU is faster for it (113 s on 64 cores against 132 s on an A30
    for 1.22M reads, plus ~2.2 GB more RSS). CNN/TCN adapter detection (~7x) and
    the CTC-CRF encoder (~4x) are the two that go to the device.
  - `--device gpu` is a **requirement**. A missing Cargo feature, an absent
    device, or an onnxruntime that cannot register its CUDA execution provider
    each fail the run with a message naming the cause, instead of falling back.
    It is also how you opt DTW onto the GPU, which is worth doing only when CPU
    cores are scarce; the "experimental and usually slower" warning still fires.
  - `--device cpu` forces CPU everywhere, device or no device.
  - `--gpu` survives as a hidden, deprecated alias for `--device gpu` (it warns,
    including about the change in meaning), and `--cpu` is a new alias for
    `--device cpu`. Both conflict with `--device`.
  - **`--device` exists in every build**, including the musl release artifacts
    that contain no GPU code. `--device gpu` there explains that the feature is
    not compiled in rather than dying on an unknown argument.

- **A GPU-capable stage that runs on the CPU now says so, at startup.** This is
  the higher-value half of the change: it is what would have told the consumer
  above at second one rather than after 37 minutes. The line names the cost
  (`~7x slower than GPU end-to-end`) and distinguishes the two causes, because
  they need different fixes — `this build has no cnn-gpu feature` versus
  `no CUDA device is visible`. It is emitted from code that is compiled in every
  build; feature detection uses `cfg!(…)`, never `#[cfg(…)]`, so a binary
  without the feature can still explain that it lacks it.

- **onnxruntime CUDA registration failure is fatal under `--device gpu`.**
  Centralised in `escapepod-demux`'s `ort_ep`, the one module that knows how
  `ort` spells CUDA: `--device gpu` arms `error_on_failure()` on the EP dispatch
  once, before any session is built, so registration failure surfaces as an
  ordinary session-build error. Under `auto` it still warns and falls back — that
  is what `auto` is for.

- **A GPU CNN detect run where *every* read fails inference is now an error, not
  a boundaries CSV of `adapter_end=0`.** Found while verifying the above, and it
  is the same failure the rest of this entry is about wearing a different hat.
  Registering the CUDA execution provider only appends a factory to the session
  options — which is exactly the step `error_on_failure` guards — while the
  libraries the kernels need are dlopened later, inside the first `Conv`. A
  missing `libcudnn` therefore gets past a satisfied `--device gpu` and fails
  per node, and the old code warned about the model's `[B,1,L] -> [B,2,L]`
  contract (the wrong cause), wrote a file that was uniformly zero, and exited 0.
  It now errors before `File::create`, so nothing is left behind to mistake for a
  real result, and it names the likely cause. GPU only: on CPU, tract runs one
  read at a time and an all-fail can legitimately be a property of the input, so
  that path keeps the warning it has always had.

### Fixed

- **The "rebuild for GPU" hint named a Cargo feature that no longer exists.**
  #282 folded `cnn-gpu` and `crf-gpu` into the single atomic `gpu` feature and
  updated `Stage::compiled_in` to match, but `Stage::feature` — the accessor
  three lines above it that supplies the *name* in every message about a stage
  — was missed. So the default release binary, which has no GPU code by
  construction, told anyone running `demux detect --method cnn` that "this build
  has no `cnn-gpu` feature", pointing them at a flag `cargo` would reject. Both
  the CPU-fallback warning and the hard `--device gpu` error carried it.

  `feature()` now returns `gpu` for all three stages, and the two messages drop
  the parenthetical that existed only to explain why the feature they named and
  the flag they told you to pass were different strings. It stays a per-stage
  accessor rather than a constant so the messages keep one source of truth and a
  future stage whose feature genuinely differs has somewhere to say so.

- **`CLAUDE.md` pointed GPU tests at an environment with no test runner.** It
  showed `pixi run -e gpu cargo nextest run …`, but `cargo-nextest` is in the
  `dev` feature only, so that command could never have run. It is `-e dev-gpu`.

### Removed

- **`escpod demux train-svm --gpu`.** The flag never did anything: the
  `train_svm_gpu` it called was byte-identical to the CPU `train_svm` (both
  forward straight to `fit_from_labels`), because the distance matrix it used to
  compute on the device fed only a kernel matrix the current label-only fit
  discards. The command already warned the flag had no effect and then called it
  anyway. It is removed rather than renamed to `--device gpu`, since a device
  flag on a command with one implementation is the same lie in newer syntax —
  `train-svm` is now honestly CPU-only and takes no `--device` at all. The
  `escapepod_demux::train_svm_gpu` library entry point is gone with it;
  `compute_distance_matrix_gpu`/`_with_ctx` (which do real device work) stay.

## 0.16.1 (2026-08-25)

### Added

- **`sequence_bases_with_context`: the k-mer context window as bases (#274).**
  #272 moved the signal-level k-mer *encoding* upstream and leech now calls it
  (rnabioco/leech#222), but one overlap survived, and it was the same shape of
  problem: leech kept its own copy of the **context windowing**, because
  `sequence_ints_with_context` returns ints and a training corpus *serialises*
  the context — `sequence_with_kmer_context` is a string in the chunk format,
  which `data merge`/`load_chunks` read back as one. So the downstream caller
  needed bases, and deriving them from the ints by hand would have been a third
  copy of the window rather than the end of the second.

  The new form is the same cut, padded with `UNKNOWN_BASE_CHAR` (`N`), and it
  composes exactly: `sequence_to_int` of the bases *is*
  `sequence_ints_with_context`, padding included, because
  `base_to_int(UNKNOWN_BASE_CHAR)` is `UNKNOWN_BASE`. That equivalence is what
  makes this a refactor rather than a new rule, and a test sweeps it over five
  contexts and every offset from the start of the sequence to past its end, so
  a divergence between the two forms fails rather than ships. The windowing
  arithmetic itself is now one private helper (`context_range`) that both
  public forms call.

  Worth doing because this is the step where `KmerContext`'s halves are *not*
  interchangeable — swap them and every k-mer is read from a window displaced
  by `before - after` bases, silently, and `encode_signal_kmer` cannot detect
  it because it only ever sees their sum. A second copy of precisely that rule
  is the one this crate least wants to keep.

## 0.16.0 (2026-08-25)

### Added

- **`escapepod-signal` owns the signal-level k-mer encoding
  (`seq_encoding`, #271).** `mapping` (#262) already *produced* a base→signal
  map; the primitive that *consumes* one — scattering the one-hot k-mer context
  along the signal axis, the 36-channel `sequence` input of a leech
  `seq_encoding="signal_kmer"` model — lived downstream in leech, inside a
  `cdylib` Python extension module that Rust cannot link. Since that tensor is
  computed in the dataset it is **not** in leech's exported ONNX graph
  (rnabioco/leech#220), so a Rust runtime has to build it before it can call
  the model at all, and "call leech-core" is not an option. The choice was to
  transcribe the rule or not to run those models — which is how
  `KmerTable::extract_levels` ended up with two centring conventions and how
  `escapepod-classify` reproduced a superseded feature definition for two
  months.

  The new module is `encode_signal_kmer` (plus an `_into` form for a hot loop
  that would otherwise allocate per chunk), `sequence_ints_with_context` for
  cutting the context window a chunk needs, and the `A/C/G/T=U` alphabet
  (`base_to_int`, `sequence_to_int`) that both take —
  `resquiggle::kmer_table` now shares that one definition rather than carrying
  its own copy. `KmerContext` names the `(before, after)` pair, since
  transposing it displaces every k-mer window by `before - after` bases and
  still returns a correctly shaped tensor, and it is where `channels()` (36 for
  the usual `(4, 4)`) is computed rather than in each caller.

  Parity with leech's NumPy reference is pinned bit-exactly over 35 cases
  (`tests/signal_kmer_parity.rs`, regenerate with
  `tests/fixtures/gen_signal_kmer_golden.py`) — the encoding is exactly zeros
  and ones, so there is no tolerance to argue about. The golden is generated
  from the **NumPy** path deliberately: leech's own compiled extension
  disagrees with its own fallback on a span whose start is negative, because it
  clamps *after* an `as usize` cast, so the start lands on `signal_len`, the
  span comes out empty and the base disappears. Measured against
  `leech_core` 0.8.0 on a 3-base window with a map of `[-8, 10, 20, 30]`: 60
  hot samples from the extension against 90 from NumPy, and for
  `[-30, -20, 40, 60]` a span covering the entire window vanishes to 0. This
  crate keeps the surviving tail, which is both the readable definition and
  what a reference-anchored map — whose entries legitimately go negative once
  the aligned region is cropped — needs.

- **POD5 V6 files are readable (upstream 0.3.46).** V6's only change is that
  the reads-table `channel` column is retyped from `uint16` to `uint32` — same
  name, same position, so nothing about the container moves. But because it
  retypes an *existing* column rather than appending new ones the way V4 and V5
  did, it is not a change a narrow reader can ignore: pinned to `uint16`, every
  V6 file fails outright rather than degrading. `channel` is now resolved to
  whichever width the file carries and widened to `u32`, on the per-row path,
  the bulk columnar path, and the row extractor alike. V0–V5 files are
  unaffected.

  `ReadData.channel`, `ReadColumns.channel`, and the Python `ReadData.channel`
  / `Writer.add_read` parameter are `u32` accordingly, matching upstream's own
  C++ `ReadData`. `to_dict`/`to_pandas`/`to_polars` hand back a `uint32` column
  where they used to give `uint16`.

### Changed

- **Written files stay V5; an unrepresentable channel now fails the write.**
  Emitting V6 today would make every file escpod produces unreadable by every
  installable reader: the newest `pod5` on PyPI is 0.3.44, and it rejects a
  `uint32` `channel` with `Schema field 'channel' is incorrect type: 'uint32'`
  (verified against an escpod-written V6 file). The trade would be a hard break
  with the deployed ecosystem in exchange for channel numbers no flow cell
  produces — PromethION tops out at 3000. So the emitted column stays `uint16`
  and files stay stamped `0.3.44`, while reading stays lossless at both widths.

  The one input that would lose data — a channel above `u16::MAX`, which can
  only come from a genuine V6 file — is refused with an error naming V6 rather
  than silently written as `channel % 65536`. `escpod inspect summary`'s
  channel statistics widen to match.

  This flips once ONT publishes v6-capable wheels: `narrow_channel` in
  `escapepod-pod5::schema::reads` is the single site, and
  `emitted_channel_width_matches_the_stamped_version` pins the schema width and
  `POD5_VERSION` together so they cannot drift apart.

## 0.15.0 (2026-08-23)

### Build / Tooling

- **The POD5 compat job stops rebuilding its dependency graph every run.**
  0.14.0 moved it off `--release` onto an optimised-but-not-LTO profile, which
  did not work: `Swatinem/rust-cache` derives its key from `Cargo.lock` and the
  toolchain, **not** from the cargo profile, and GitHub refuses to overwrite an
  existing cache key. The key therefore still held the old `release` artefacts,
  so every run restored artefacts it could not use, rebuilt everything, and
  then declined to save because the key already existed. Measured on main:
  363 s of a 400 s job was `cargo build`, the compat test itself was **1 s**,
  and the cache post-step wrote nothing. Warm cost went 316 s → 400 s — the
  change cost more than it saved.

  Two fixes, both needed. The cache key now carries an explicit suffix that is
  bumped whenever the profile changes, so a profile switch can actually be
  saved. And the job builds the **dev** profile rather than an optimised one,
  because the suite round-trips a *five-read* fixture and the binary's
  throughput is irrelevant to it; third-party crates still compile at
  `opt-level = 2` through `[profile.dev.package."*"]`, so only escapepod's own
  crates drop to `-O0`, and those have to rebuild on any source change anyway.
  The now-unused `ci-bin` profile is removed.

### Added

- **A cache of open, indexed readers, so the read-id index is built once per
  *file* instead of once per *reader* (#258).** `Reader` caches its index in a
  `OnceLock` on the instance, so a consumer that opens a reader per batch
  throws the index away and rebuilds it on the next batch. That is not a small
  constant: on a 145 GB POD5 on a network filesystem it was minutes of
  uninterruptible sleep in `folio_wait_bit_common` per batch at ~0.6% of one
  core — the 10–80x data-preparation regression in rnabioco/leech#176. #251
  fixed the other half of it (the scan variants are gone and lookups index
  unconditionally), but "one reader per file per process" was left to every
  consumer, and each consumer that did not write it silently got the slow path.
  leech wrote it in Rust, and then wrote the same idea again, independently, in
  Python.

  Both shapes ship, because they answer different questions. `cached_reader()`
  is the process-global convenience, and it is the one that makes consumers
  actually stop hand-rolling this. `ReaderCache` is the owned type underneath
  it, for a library that needs the lifetime bounded or a process where one
  stage must not share readers with another; `global_reader_cache()` reaches
  the global's `len()` / `clear()`.

  The value is in the ordering and the failure semantics rather than in the
  `static`, so those are the parts worth stating:

  - **The file is opened outside the lock**, which guards only the map. A slow
    open on one path never blocks a lookup on another, and the lock is never
    held across I/O, so this cannot deadlock. Two threads racing the same path
    cost one redundant open and both get the winner's `Arc` — publication goes
    through `entry`, not `insert`, so a race can never leave two live readers
    (and two indexes) for one file.
  - **The index is warmed before the entry is published**, so N workers hitting
    their first batch together find it built instead of piling up inside one
    lazy init. The warm-up respects `autoindex_max()`: above that read count it
    is skipped, because warming is a *guess* that random access is coming and a
    huge file that is only iterated should not pay for an index nobody asked
    for. Skipping only defers the build to the first lookup that demands one,
    and because the reader is now shared that build still happens once per file
    rather than once per batch — the cache keeps its whole value above the
    threshold, it just stops guessing.
  - **A failed index build is logged, not propagated.** `Reader::open` failing
    *is* an error, because there is no reader to hand back. An un-indexable
    POD5 is still a perfectly good reader for iteration, metadata, and signal
    access, and failing an open for a caller that may never do a lookup is
    worse than the slowdown; a caller that does demand a lookup sees the same
    error then, from the call that needs it. (One correction to the issue's
    framing: after #251 such a file is not "readable, just slowly" — the error
    surfaces from `reads_by_ids` rather than degrading to a scan. The reader
    stays usable; lookups by read id do not.)

  Keys are canonicalized, falling back to the path as given if that fails, so
  `reads.pod5`, `./reads.pod5`, and a symlink to it are one entry rather than
  three readers with three indexes. The reader is opened on the canonical path
  too, so `.p5s` sidecar resolution does not depend on which spelling happened
  to arrive first. What stays resident is the index and not the file —
  ~24 bytes/read, so a few tens of MB even for a multi-million-read POD5 — and
  entries are never evicted, with `clear()` as the escape hatch for a process
  that walks an unbounded set of files.

- **`Reader::read_index_if_built()`** — the non-committing half of
  `read_index()`: it never loads a sidecar and never scans, so it is the only
  way to ask whether a reader is warm without making it warm. Without it the
  warm-before-publish ordering above is unobservable, and a test that "checked"
  it by calling `read_index()` would only be asserting its own side effect.

- **`escapepod_signal::mapping`: the two Oxford Nanopore coordinate
  conventions that produce a resquiggle's input.** `refine_signal_map` has
  always *taken* a sequence→signal map; nothing in the workspace *produced*
  one. So every consumer wrote its own eight lines off the `mv`/`ns`/`ts` tags
  and its own CIGAR walk — three copies in this repo alone (the charging
  classifier's anchoring, the `resquiggle` command, a test helper), plus the
  ones downstream. Each is a shifted map away from answering a different
  question than the caller thinks, with no error to show for it, which is the
  same argument that moved the k-mer level primitives here.

  - `seq_to_signal_from_moves(moves, stride, trim_offset, num_samples)` —
    Remora's `query_to_signal = np.nonzero(mv)[0] * stride`, returned in
    **trimmed-signal coordinates** with `num_samples - trim_offset` as the
    closing boundary, because that is the frame the move table is in and the
    frame `refine_signal_map` is handed. A caller indexing the untrimmed POD5
    array adds `trim_offset` back; the charging anchoring now does that
    explicitly instead of folding `+ ts` into the map's construction, where
    the frame was invisible.
  - `ref_to_signal(query_to_signal, cigar)` — reference→signal by the Remora
    knot convention: trailing non-match ops stripped, knots at the start and
    `end - 1` of each match block (not `end`, which stretches every gap by a
    position), exact 1:1 integer lookup inside a block, and linear
    interpolation only across indel gaps.

  The CIGAR arrives as a local `CigarOp { kind, len }` rather than the
  `(op, len)` integer pair the convention is usually written with: the crate
  takes no alignment-library dependency for this, and a bare pair of integers
  is exactly what a caller transposes without the compiler noticing.

  `ref_to_signal` is integer arithmetic throughout except the one ratio each
  gap position needs — deliberately not the `ref → float query → float signal`
  chain that a pair of `np.interp` calls performs. Both interpolations there
  evaluate `slope * (x - x0) + y0` with a pre-rounded slope, and the result is
  floored, so a one-ulp difference in the intermediate query coordinate
  becomes a one-sample difference in the answer: with the map `[0, 7, 8]` and
  a CIGAR of `1M 6D 1M`, the float chain puts reference position 5 at sample 4
  instead of 5. It is rare — a 200 000-case sweep of realistic random CIGARs
  found no difference at all, and it takes a long deletion spanned by short
  dwells — which is precisely what makes it expensive to find once two
  consumers have each written their own version. It is pinned by a test here
  rather than rediscovered downstream.

### Changed

- **`features::span_stats` gains a median, a range, a fill policy and an
  out-of-range policy, and takes a `SpanConfig` instead of a bare
  `Normalization` (#260).** The reduction was already the right one — one pass
  over the covered region with `f64` prefix sums, O(1) per span, spans supplied
  by the caller — but three of its choices were baked in, and a consumer that
  disagreed with any of them could not use the function at all. leech therefore
  carried its own copy, then a *second* copy, and the two disagreed on exactly
  one of those choices (rnabioco/leech#200): the Python fast path skipped a
  span with a negative start and left zeros, the Rust pipeline computed over
  the truncated span. Same read, different features, depending on which path
  reached it. The payoff is not line count; it is that the numbers stop
  depending on which code ran. Precedent: #204, where the rule that decides
  what a model sees was moved to the crate that owns the reduction rather than
  re-derived in each caller.

  The three gaps, all now named fields on `SpanConfig` rather than assumptions.
  `SpanStatsOut` grows optional `median` and `range` buffers, built through
  `SpanStatsOut::new(..).with_median(..).with_range(..)` — optional because
  neither can come from the prefix sums (each needs its own pass over the span,
  and the median a select or a sort on top), so a caller wanting only
  dwell/mean/sd does not pay for them. `SpanFill { Nan, Zero, Value(f32) }`
  chooses what an unresolved span gets: `Nan` stays the default and stays the
  honest answer — an unresolved base has no observation, and a substituted
  value is indistinguishable from a real one — but that argument does not
  survive contact with a neural network, where one `NaN` poisons the forward
  pass, so the alternatives exist for a caller feeding these arrays to a model.
  `SpanBounds { Skip, Clamp }` chooses what happens to a span hanging off the
  end: `Skip` (the default, and the old behaviour) treats an out-of-range
  coordinate as evidence the map is broken; `Clamp` intersects with
  `[0, len)` and summarises what survives, which is what a reference-anchored
  map needs once the aligned region is cropped and entries can legitimately go
  negative while the truncated span still carries real signal. Under `Clamp`,
  `dwell` is the **clamped** length, not the requested width — every other
  output is computed from exactly those samples, and pairing a sample count
  with a mean not taken over that many samples would be a contradiction a model
  could read.

  **Both median conventions ship, rather than one being chosen silently.**
  `MedianConvention::SelectTotalCmp` (the default) is
  `stats::median_via_select`, i.e. `select_nth_unstable` with `total_cmp` — the
  convention every other median in escapepod-signal already uses.
  `MedianConvention::SortPartialCmp` is a full sort with `partial_cmp` plus
  numpy's own `NaN` check, reproducing `numpy.median` over a `float32` array
  exactly, for a consumer that cross-checks against a Python reference. Both
  average the two middle order statistics on an even-length span and take the
  middle one on an odd-length span; neither picks one middle and discards the
  other. Measured, the two are *bit-identical* over any span of finite values,
  including the even-length ulp-separated `f32` spans where a ~1e-7 split was
  expected — `total_cmp` and `partial_cmp` induce the same order on non-`NaN`
  values, and `numpy.median`'s `float32` two-element mean is bit-for-bit
  `(a + b) / 2.0`. Where they genuinely diverge is a span containing `NaN`:
  `SelectTotalCmp` sorts it to the high end and returns a finite median from
  the values below, `SortPartialCmp` propagates it. That is not exotic — a
  caller padding a window with `NaN` hits it on every padded base — and the
  propagating answer is the one consistent with `mean`, which is already `NaN`
  there. Both behaviours are pinned by tests, the numpy arm against goldens
  generated from numpy 2.5.1.

  **The API break is behaviour-preserving by construction and proven so.**
  `SpanConfig::default()` is the old behaviour exactly (`Nan` fill, `Skip`
  bounds, no normalisation), so `span_stats(sig, spans, SpanConfig::new(norm),
  ..)` is the old call. A test keeps the pre-`SpanConfig` implementation
  verbatim as an oracle and asserts the default path is bit-for-bit identical
  to it across three normalisations and six signal/span fixtures — including
  with the optional outputs requested, since the point of making them optional
  is that they cannot perturb the prefix-sum path. The guardrail was checked
  for teeth by flipping each default in turn and confirming it fails.

  The Python binding exposes the same knobs, keyword-only and all defaulting to
  the historical behaviour: `median=True` / `range=True` append a fourth and
  fifth array to the returned tuple, `fill=<float>` replaces the `NaN`
  sentinel, and `bounds` / `median_convention` take the policy by name. Callers
  that unpack three arrays are unaffected.

### Fixed

- **One named refinement preset, with a per-read dwell target** (#257). The
  settings block for refining a basecaller move table existed twice — once in
  escapepod's own Python binding (`py_refine_signal_map`) and once in a
  downstream Rust consumer — each carrying a comment asserting that it matched
  the other. The binding's docstring went further and promised the two paths
  matched "bit-for-bit". They did not: `dwell_target` had drifted, a fixed
  `4.0` in the binding against the `0.0` sentinel that asks escapepod to
  resolve the target from the read's own move-table median dwell.

  That one field is not cosmetic. The dwell penalty is asymmetric — quadratic
  below target, logarithmic above — so a target set too low does not merely
  weaken the prior, it actively drags boundaries toward dwells the pore never
  produced. RNA004 at 130 bases/s and 4 kHz sits near **31 samples/base**, so a
  target of `4.0` treated every base as roughly 8x too long. Measured across
  the two backends on the same reads with the same flags: **max |signal delta|
  3.44 in normalized units, every dwell different, max |feature delta| 3.57**.
  The two paths refined the same data to different boundaries for four
  releases, and the comment saying they agreed was there the whole time.

  `RefineSettings::move_table_refinement(half_bandwidth, n_iters, seed)` is now
  that configuration, as a value rather than a convention: fixed banding, a
  least-squares rough rescale over the 0.05–0.95 quantiles clipped 10 bases
  with `use_base_center`, a Theil-Sen inter-iteration rescale over at most 200
  points, level normalization off, and the asymmetric dwell penalty at weight
  0.5 with the per-read target. The sentinel gets a name —
  `RefineAlgo::PER_READ_DWELL_TARGET` — because `0.0` at a call site does not
  say what it means, and this is the field where that cost something.
  `RefineAlgo`'s shape is unchanged, so nothing downstream has to move.

  **Behaviour change for `escapepod.refine_signal_map`.** `dwell_target` and
  `dwell_weight` become `Optional[float]`, default `None`, meaning "use the
  preset"; passing a number still overrides it. The old default of `4.0` is
  gone rather than preserved. It is simply wrong for RNA004, it silently
  corrupted a production corpus, and a caller who wants it back can pass it
  explicitly — which is a better trade than making every future caller inherit
  a known-wrong number for bug-compatibility.

  The docstring stops promising bit-for-bit parity, since that promise is not
  enforceable from inside a docstring and was false when written; it now names
  the preset both paths construct, which is checkable. It also settles what
  `(scale, shift, drift)` are for. The return tuple is unchanged, and it had
  instructed callers to apply the rescale as `(signal[i] - shift - drift*i) /
  scale` while the downstream Rust path deliberately discarded those same
  values. Both readings were defensible because escapepod never said which it
  intended. It now does: the values are returned **for inspection**, applying
  them is the caller's decision, and the failure mode is documented — a
  per-read affine fit estimated over a near-constant stretch of signal (a 3'
  adapter, a homopolymer) is weakly identified, with observed scales ranging
  from 15 to 1084 and frequently negative.

  Three tests pin this. The preset's fields are asserted one by one, including
  the rescale filter constants and the quantile grid, so a future edit to any
  default cannot quietly redefine the preset. Refining an RNA004-like synthetic
  read under the preset must reproduce refining it under an explicitly named
  target equal to the input map's median dwell — "per-read" stated as something
  observable rather than as prose. And the same read must refine *differently*
  under a fixed `4.0`; restoring the old default fails that test, which was
  confirmed by restoring it.

## 0.14.0 (2026-08-23)

### Build / Tooling

- **CI audit: duplicate compile passes removed, cache churn stopped.** Measured
  per-job on a warm run (PR #252) and a cold one (a Dependabot lockfile bump),
  then cut by what the numbers showed rather than by what looked redundant.

  - **`check` is gone; `clippy` covers it.** `cargo clippy --workspace
    --all-targets` runs the same rustc front-end over the same unit graph as
    `cargo check --workspace --all-targets` and then adds lints, so a green
    clippy already implied a green check. The workspace was being compiled
    twice per PR for no additional signal. The same pair ran in `release.yml`'s
    macOS gate, where it cost 10x — GitHub bills macOS minutes at ten times
    Linux.

  - **Doctests moved into the `test` job.** nextest does not run doctests, but
    `cargo test --doc` needs the same toolchain, profile and unit graph that
    nextest has just built. As its own job it rebuilt the whole dependency
    graph from scratch — 406 s cold, against roughly 2 s of actual doctest
    execution — and held one of the largest caches in the repo. Sharing the
    target directory makes it nearly free.

  - **The POD5 compat suite stops building a shipping artefact.** It was
    `cargo build --release`, i.e. fat LTO and `codegen-units = 1`, to produce a
    binary that round-trips small fixtures. That made it the longest job in the
    workflow (316 s warm, 480 s cold) and therefore the critical path of every
    PR. It now builds the new `ci-bin` profile — `opt-level = 3` kept,
    whole-program optimisation and single-threaded codegen dropped — and points
    the suite at it through the `ESCPOD_BIN` override the harness already had.

  - **Coverage runs the instrumented suite once instead of twice.** The job
    ran `cargo llvm-cov nextest` in full for lcov and then again in full for
    HTML. It now runs once with `--no-report` and renders lcov, the threshold
    summary and HTML from that single set of profiles.

  - **Two feature builds dropped as duplicates.** The `features` job exists
    because "the default jobs never compile the opt-in features" — but
    `cnn-detect` and `crf-decode` are both in escapepod-cli's *default* `cli`
    feature, so workspace feature unification already compiled escapepod-demux
    with them in the ordinary `clippy` job. The genuinely opt-in ones (gpu,
    cnn-gpu, crf-gpu, models-download) are untouched.

  - **PR runs no longer write to the Actions cache.** The repo held 11 GB
    across 34 entries against GitHub's 10 GB per-repository ceiling, so it sat
    in permanent LRU eviction — which is why nominally warm jobs still showed
    cold timings. Every `Swatinem/rust-cache` step now carries
    `save-if: github.ref == 'refs/heads/main'`, so PRs restore from main's
    cache but never add to it, and cache entries stop multiplying per branch.

  - **Prose-only changes no longer start a Rust build.** `paths-ignore` on
    `docs/**`, `**/*.md`, `LICENSE*` and `.gitignore`. A commit touching both
    prose and code still runs everything — `paths-ignore` suppresses a run only
    when every changed path matches — and with no branch protection on `main`
    there are no required checks for a skipped run to block.

### Fixed

- **A targeted lookup no longer scans the whole reads table when there is no
  `.p5s` sidecar** (#251). `reads_by_ids`, `find_signal_rows_by_ids` and
  `find_signal_rows_with_calibration_by_ids` chose between an indexed path and
  a full scan with a `has_index()` helper that answered *"is there a sidecar on
  disk?"* — a different question from *"can I use the index?"*, and one with a
  different answer whenever a sidecar is absent, which is the common case. So
  every call scanned the file, and none of them built the index that would have
  made the next call a seek, even though `read_index()` was already there,
  self-caching, and cheaper than the scan being chosen instead.

  The fallback was never the cheap option. The index *is* a scan, projected to
  the `read_id` column, so building it moves strictly fewer bytes than one
  execution of the path it declined — against 22 columns for `reads_by_ids`,
  2 and 4 for the signal lookups — and it is then cached for every later call.
  The early exit that justified the scan ("stops once all targets are found")
  does not fire in the access pattern that matters: targets arrive in BAM
  order, unrelated to POD5 storage order, so the last one sits near EOF and the
  scan runs to the end of the file. Per call. Downstream this cost
  `rnabioco/leech` roughly 10–80x on data preparation, silently; on a 145 GB
  merged POD5 one call for 1000 ids had not returned after 13.5 minutes.

  All three entry points now go through `read_index()`. The scan variants are
  gone rather than kept behind a size threshold: what decides whether a scan
  could win is *where the targets land*, not how many there are, so a threshold
  on `target_ids.len()` cannot detect the one case it would be for.

### Changed

- **`autoindex_max()` moved from the Python bindings into `escapepod-pod5`**,
  where the decision it encodes actually lives (#251). It was reachable only
  from Python — two call sites there warmed the index on context-manager entry
  to route around the scan described above — so no Rust caller could reach the
  policy, and the workaround had to be written again in every consumer.
  `ESCAPEPOD_AUTOINDEX_MAX` and the 5,000,000-read default are unchanged.

  Its meaning is now narrower and honest: it gates *speculative* indexing only.
  Entering a Python reader as a context manager still checks it, because that
  is a guess that random access is coming and a large file that is only
  iterated should not pay for an index nobody asked for. A caller that has
  actually asked for random access always gets an index, whatever the file
  size — above the threshold the build is reported at `warn` naming
  `escpod index`, not traded for a scan. It was never a memory guard in any
  case: loading a `.p5s` sidecar has always built the same in-memory entry
  table with no cap at all, so the same file with a sidecar already holds what
  the threshold claimed to prevent.

- Dependency bumps (lockfile only, no behavior change): the Arrow ecosystem
  `arrow` + `parquet` 59.1 → 59.2 and the `tract-*` stack 0.23.4 → 0.23.5,
  plus `bit-vec` 0.9 → 0.11 and `bitflags` 0.19.8 → 0.19.9 (#250).

### Added

- **`escapepod-pod5` can log.** The format crate — the layer every other one
  sits on — had no `tracing` dependency at all, which is why a per-call rescan
  of a 145 GB file was indistinguishable from slow I/O and cost a day of
  profiling to find. Building a read index now says that it is happening, why
  (no sidecar), and what it cost (reads, batches, elapsed).

- A `reads_by_ids` group in `io_hot_paths`, covering the three arms that
  matter: index loaded from a sidecar, index built on the first call, and the
  warm steady state. Its doc states the limit of what it can prove — at fixture
  scale a scan and an indexed seek cost the same, so the guard against taking
  the wrong path is the invariant test that asserts the index is built exactly
  once for two lookups, not the benchmark.

## 0.13.0 (2026-08-23)

### Fixed

- **The `.p5s` read index is no longer trusted past the read it names.** A
  sidecar's `(batch_idx, row_idx)` locators were dereferenced unchecked, so an
  index that passed the file-level identity guard but held wrong offsets
  returned a **different real read, correctly self-labelled** — which nothing
  downstream could detect. The signal paths were worse: they projected
  `read_id`, never read it, and stamped the *queried* UUID onto whatever row
  they landed on. Every indexed lookup now confirms the row's `read_id` before
  using it (a 16-byte compare against the cost of decoding a read; the column
  was already resolved) and reports which read the locator actually pointed at.

  An out-of-range `row_idx` previously reached an Arrow accessor and **panicked**
  (`Trying to access an element at index 10000 from a PrimitiveArray of length
  25`); it is now bounds-checked into an error. Both failures are pinned by
  tests that forge an identity-valid sidecar with permuted and out-of-range
  locators — a round trip cannot reach this code, because a sidecar escapepod
  wrote for a file is correct by construction.

- **`escpod annotate` no longer succeeds silently on the wrong CSV.**
  Assignments are intersected with each file's own reads, so a classifications
  CSV from another run dropped every row, wrote a valid but empty column, and
  exited 0 — reporting `0 of 50000 reads assigned` at info level, which `-q`
  hides. Zero overlap on one input now warns, zero across *all* inputs is an
  error naming the CSV and pointing at `annotate --remove`, and the summary
  line reports how many assignments matched.

### Added

- **The sidecar records where it came from** — `escapepod:source_name` (the
  POD5's base name), `escapepod:read_count` and `escapepod:writer`, surfaced in
  a mismatch error and in `escpod inspect summary`. Identity remains
  `file_identifier` + `pod5_size` and nothing else; these are descriptive and
  **never compared**, because matching a filename would break every legitimate
  rename. They exist for the moment identity fails, when the error otherwise
  knows only that two UUIDs differ — precisely when a filename is what you
  want. All three are optional on read, so this is not a format-version bump:
  older sidecars still load and an older escpod ignores the new keys. No write
  timestamp, since the sidecar file's mtime already records it.

- **`AnchoredReads.coords()` (Python) — junction geometry without reading
  POD5.** The BAM scan in `AnchoredReads::new` already decides every
  coordinate the extractor uses, but the only way to read one back was
  `extract()`, which pulls signal: ~136 GB of POD5 per flowcell, I/O-bound
  (4.18 of 48 allocated cores), to answer a question about *geometry*.
  `coords()` runs the same finalization over the scanned reads and returns the
  coordinates alone — 87 s to scan 13.8 M BAM records plus 2.6 s to finalize
  12.5 M reads, against 20–30 min for a full extraction — and it is exact, not
  an approximation: all seven geometry columns agree with a corpus `extract()`
  had just written on all 7,997,543 shared reads. That makes sizing a rebuild,
  auditing anchor choice by class, and checking that a new anchoring rule never
  displaces an already-exact read cost 90 seconds instead of half an hour.

  Every value is a numpy array, so the result saves as-is with
  `np.savez(path, **reads.coords())`. `read_id` is a flat ASCII buffer
  (`.view("S36")`) because 15.9 M Python strings would be ~1.3 GB of PyObject
  built and thrown away. `anchor_source`/`mask_source` are indices into the
  module lists, and `MASK_SOURCES` is now exported alongside `ANCHOR_SOURCES`
  so a saved npz decodes standalone; both enums are `#[repr(u8)]`, since an npz
  stores the bare integer and reordering a variant would silently relabel every
  stored read.

### Changed

- `escpod index` says what a rebuild discards. Replacing a stale sidecar drops
  annotations and scores that exist nowhere else; the warning now says so, and
  carries the identity error's description of what the sidecar was built from.

- **The charging junction is anchored from flanks the modification does not
  damage.** `ref_to_query` resolved a mis-called junction by
  nearest-aligned-neighbour backfill within `slop=2`, probing `r, r+1, r-1,
  r+2, r-2` — entirely inside the band the aminoacyl adduct disturbs, and
  upward first, so the misplacement went toward the adapter. The adduct
  mis-calls the junction it attaches to: **51.9% of charged reads carry a CIGAR
  indel across `CCAGGC` against 2.4% of uncharged** (23x, construct-matched),
  the unaligned-base rate peaks at reference offset +5 (18.3%), and both
  classes are clean at ≤ −8 and ≥ +19.

  `flank_anchored_qj` therefore takes flanks at (−10, +20) — outside the damage
  band on both sides — requires **both** donor-exact, and interpolates the
  junction in query-base space. It fires only when the junction is not already
  donor-exact, so a read the aligner placed directly is never touched, and the
  anchor is installed only if its flank context is constant across every
  reference record: 47/47 on the edx panel, 0/164 on the divergent v2 adapters,
  where offsets +17..+24 are the library-identifying 13-mer and anchoring on it
  would be worse than not anchoring at all. Verified read-for-read against
  `escapepod_models.charging.flank_anchored_qj` on 40,000 charged reads:
  `anchor_source` and `junction_sig` agree 100.0000%.

  It moves 1.48% of reads, exclusively into `flank_interp` — zero previously
  `exact` reads are displaced. This is a **placement** fix, not a yield fix:
  corpus composition is identical before and after (n = 15,872,877, same
  per-class counts) and the trained model moved 0.9906 → 0.9903 AUROC, within
  the run-to-run spread. The correctness argument stands on its own; no
  accuracy claim is being made.

### Documentation

- `docs/format/sidecar.md` had drifted from the format it specifies: it gave the
  version as "currently `1`" (`2` ships whenever a score column is present) and
  omitted `Float32` score columns from the layout table entirely. It now also
  states what identity *means* — a UUID and a byte length, with no content hash
  — and why: a v4 `file_identifier` is a 122-bit per-file token that answers
  "same file?" better than a checksum, while hashing a multi-gigabyte POD5 on
  every open would cost more than the scan the sidecar exists to avoid. The
  documented pyarrow escape hatch is now marked as bypassing that check.

## 0.12.0 (2026-08-20)

### Performance

- **AVX2 and AVX-512 kernels for the reference-scoring scan — 3.3x** (#241).
  `--ref-scores` cost +25% on `demux basecall` when it landed; it now costs
  +7.6%, and +3.6% on the fused `demux`.

  Two measurements shaped this, both against the obvious guess. The scan was
  never loop-bound: specialising its two fan-in classes into separate scalar
  loops moved it 25.7% → 25.1%. It was transcendental-bound exactly as the
  decode is, and the difference was simply that it called scalar `exp`/`ln_1p`
  while the decode ran the crate's Cephes kernels. And vectorising only the
  fan-in-1 cells — 65% of the lattice — capped the whole scan at 2.15x, almost
  exactly the Amdahl bound, because the head is a tenth of the cells but a
  third of the work at five terms per cell against two.

  The kernels rest on a reordering: cells are now partitioned by fan-in, so a
  cell's own `alpha` is a unit-stride load and its result a unit-stride store,
  and only the score indices need gathering. AVX2 has no scatter at all, so an
  unordered lattice could not have been vectorised on it. The head's moves are
  additionally stored transposed to `[edge][cell]`, so each edge's indices are
  a unit-stride load rather than a gather feeding a gather.

  Checked against the scalar scan on 20k real reads: 20,000/20,000 identical
  barcode calls, largest `crf_logp` difference 0.0002 nats.

### Added

- **The CRF can now say how sure it is: `escpod demux basecall --ref-scores`**
  (#241). Demultiplexing reports `confidence` as the edit-distance margin to
  the runner-up, and on a designed panel that measures how far apart the
  references are, not how sure the model is. Measured over one production
  16-plex flowcell (1,001,307 reads): 99% of classified reads take one of three
  margin values, so sweeping `--min-margin` does nothing and then falls off a
  cliff, and 90% of the reads two independently trained bundles disagree about
  are *exact* matches to a reference. There was no way to buy precision at any
  price.

  The lattice has an opinion and it was never surfaced. Every path through a
  CTC-CRF emits exactly one string, so restricting the forward recursion to the
  paths that emit a given reference and normalising by the full partition
  function gives a real probability:

  ```text
  crf_logp = logZ_target(reference) - logZ_full = log P(reference | signal)
  ```

  `--ref-scores` computes that for every reference in one shared lattice —
  references with a common prefix share their cells — and adds four columns:
  `crf_logp` (the called barcode's log-probability), `crf_best` (the reference
  the lattice itself prefers, which need not be the one edit distance called),
  `crf_margin` (log-odds in nats against the runner-up), and `mean_logpost`
  (the decoded path's mean per-timestep log-posterior, which the Viterbi pass
  was already computing and discarding).

  Over 20k RNA004 reads: `confidence` takes 15 distinct values with 98.7% of
  reads in three of them; `crf_margin` takes 14,818 over 16,747 reads. 98.4% of
  reads match a reference exactly, and *within that group* `P(barcode | signal)`
  still spans below 0.1 (26 reads) through 0.1–0.5 (828, 5.0%) to 0.9–0.99
  (91.8%) — reads a clean decode cannot tell apart and the lattice can.

  The scan is folded into the decode rather than bolted on after it, because it
  reads the raw scores and pass 1 overwrites them in place with log-posteriors.
  Correctness is pinned against exhaustive enumeration of every path through a
  small lattice, including the marginalisation over the `state_len` bases a
  bundle's references do not carry, plus the identity that the probabilities of
  all possible emissions sum to 1.

  Opt-in, and cheap: +7.6% on `demux basecall`, +3.6% on the fused `demux`.

- **A precision/recall dial that actually turns: `--min-crf-margin` and
  `--min-crf-prob`** (#241), on both `escpod demux` and `escpod demux
  basecall`. Below the threshold a read becomes `unclassified` rather than a
  possibly-wrong assignment.

  `crf_margin` is the *called* barcode's log-odds against its best alternative,
  not the lattice's own top-two gap, which is what makes one threshold enough:
  it is positive when the lattice agrees with the call by that much and
  **negative when the lattice prefers something else**, so any positive
  threshold rejects both the ambiguous reads and the ones the lattice actively
  disagrees with. Measured over 20k RNA004 reads, all 218 reads whose
  `crf_best` differs from their call have a negative margin, and
  `--min-crf-margin 0.5` removes every one of them for 4.4% of recall:

  ```text
  gate                  classified   recall   lattice disagreements
  none                      16,747  100.00%                     218
  --min-crf-margin 0.5      16,015   95.63%                       0
  --min-crf-margin 2.3      15,393   91.91%                       0
  --min-crf-margin 4.6      14,939   89.20%                       0
  --min-crf-prob 0.5        15,688   93.68%                       0
  ```

  `--min-crf-prob` is a different cut, not a rescaling of the same one: it asks
  whether the model is confident in absolute terms rather than whether it can
  tell the call from the next reference. A read can be certain it is not any of
  the other 15 barcodes and still put little mass on any reference at all.

  The fused `demux --classifications` gains the same four columns
  (`crf_logp,crf_margin,crf_best,mean_logpost`), which it did not have at all —
  it emitted `read_id,barcode,confidence` and nothing else. A gated row keeps
  its scores rather than blanking them, so `unclassified` says which gate
  dropped it and by how much. On both commands a gate implies `--ref-scores`,
  rather than silently doing nothing without it.

  The gate is reached through separate code in the two commands, so they are
  checked against each other on real data: over 19,980 shared reads they agree
  on 100.00% of barcodes, gated and ungated, with a largest `crf_logp`
  difference of 0.

  Sidecar-only demux keeps the scores too — see the `.p5s` entry below.

- **`.p5s` sidecars carry numbers, not only labels** (#241). Every annotation
  column was a dictionary-encoded utf8 label, which is the wrong shape for a
  per-read score: a continuous value over a million reads is a million distinct
  "labels", past the 65535 limit, with a dictionary larger than the data it
  indexes. Sidecar-only demux (`--annotate` with no `-d`) writes no CSV either,
  so a scored run computed `crf_logp` and then had nowhere to put it.

  A column is now labels (`Dictionary(Int32, Utf8)`) or scores (`Float32`), and
  a reader dispatches on the Arrow type rather than on a convention. Null means
  the read has no value — for a score, absence rather than a sentinel, since
  every `f32` is a possible answer. `NaN` is refused on the way in for the same
  reason: it is the one float that already means "no value".

  `demux --annotate --ref-scores` now records `barcode`, `crf_best`,
  `crf_logp`, `crf_margin` and `mean_logpost` in **one** read-modify-write
  (`write_columns`), rather than five that would leave four intermediate
  sidecars on disk describing a run that never happened. `escpod view --include
  crf_logp`, `escpod inspect`, and the Python `Reader.score()` /
  `.score_names()` all read them back.

  **The version bump is gated on content**: a sidecar with only label columns
  still declares `1`, so an escpod that predates this reads it exactly as
  before; only one that actually carries a numeric column declares `2`. Bumping
  every write would have made older binaries reject barcode-only sidecars they
  handle perfectly well, and without any bump they would have failed on a score
  column with a message about it not being "dictionary-encoded utf8" — true,
  but not something a user can act on.

  Verified end to end on 20k reads: sidecar and classifications CSV agree on
  all 20,000, and the POD5 is byte-identical as always.

- **`escpod demux detect --method cnn --gpu --profile` reports a per-stage
  breakdown** (#239). The GPU path is a producer (decode + prep on the rayon
  pool) feeding a batched onnxruntime consumer through a bounded channel, and
  until now the only way to ask which of them was the constraint was to sample
  `nvidia-smi` from outside and guess. `--profile` now prints, alongside the
  existing phase total:

  ```text
  GPU pipeline
    index (reads table)                0.31s
    read + decode (cpu-time)          16.82s (summed)
    prep (cpu-time)                   20.91s (summed)
    producer block                     2.36s
    producer blocked on GPU            2.38s
    GPU starved for blocks             0.00s
    GPU inference                      3.64s
  ```

  `read + decode` and `prep` are summed across the workers that ran them, so
  they are CPU time and exceed the producer's wall-clock; the rest is wall.
  The two waiting rows are the point: `GPU starved for blocks` is the idle GPU
  #239 measured from outside, and `producer blocked on GPU` is the opposite
  case, so the pair names the bottleneck instead of implying one. Output is
  bit-identical and the added timing costs ~1% (measured over three interleaved
  warm reps on 150k reads).

- **Bounded signal reads: `max_samples` on the Python reader API** (#237).
  `Reader.get_signal{,_pa}`, `Reader.get_signals{,_pa}` and the same four on
  `DatasetReader` take an optional `max_samples`, returning exactly what
  `[:max_samples]` of the full read would give without paying to decode the
  tail. Default `None` is today's behaviour; a read shorter than `max_samples`
  comes back whole. Backed by `Reader::get_signal_prefix` and
  `Reader::get_signal_bulk_prefix` in `escapepod-pod5`.

  **What it saves, honestly.** Decode is 79% ZSTD / 21% SVB16 on tRNA-length
  reads (87/16 on mRNA), and ZSTD inflates a whole 128 KiB block into its
  window before emitting any of it. A chunk that fits in one block — every read
  up to ~110k samples — therefore cannot skip ZSTD work at all, only SVB16. The
  sample ratio is not the time ratio: end to end against 0.11.0, asking for
  5,400 of 9,828 samples is **1.26x**, not the 1.8x the ratio suggests. Smaller
  windows and longer reads do better — 1.41x at a 10% tRNA prefix, 1.85x at a
  10% mRNA prefix, and once a read is long enough to span several ZSTD blocks
  the streaming path takes over: on 1M-sample chunks a 10k-sample prefix is
  **9.9x** a full decode.

### Changed

- **The boundary-CNN CUDA session no longer runs a 16-thread onnxruntime pool**
  (#239). The graph runs on the device, so onnxruntime's intra-op pool has
  nothing to compute — but it was sized to `--threads` and spawned *on top of*
  rayon's, and it did not sit idle. Profiling `demux detect --method cnn --gpu`
  over 150k reads, 15 pool threads accounted for **~35% of all CPU samples** in
  the process, next to 4% for the preprocessing they were starving. The session
  is now pinned to one non-spinning intra-op thread.

  Both halves are needed and neither works alone (warm, three interleaved reps):
  16 threads spinning 7.34 s, 1 thread still spinning 7.42 s, 16 threads without
  spinning 7.37 s, **1 thread without spinning 6.93 s**. One thread removes the
  per-op fan-out and join; disabling the spin stops that thread burning a core
  between calls.

  Against 0.11.0, on 150k reads with eight interleaved reps: **6.13 s -> 5.95 s**
  (non-overlapping ranges), process CPU **340% -> 274%**, threads **36 -> 21**.
  Replicated on a second node (7.29 -> 7.02 s, 6.28 -> 5.96 s). The fused
  pipeline shares the loader and also improves slightly (19.7/18.4 -> 19.1/18.0 s).
  Output is bit-identical throughout — 150,001 of 150,001 boundaries, and the
  fused pipeline's classifications likewise.

  This also finishes what #155 started: `--threads 16` on a 16-CPU allocation now
  means 16 worker threads, not 16 plus onnxruntime's own 16.

  **API**: `AdapterCnnGpu::load_with_threads` is gone and `load_with_config`
  /`load_with_config_on_device` lose their `intra_threads` parameter. The thread
  count was never the caller's to choose — it is a property of running on CUDA —
  and a parameter that is accepted and ignored is worse than no parameter.

- **VBZ decode reuses a per-thread ZSTD context.** `zstd::decode_all` built a
  fresh `Decoder` per call — a `DCtx`, its window buffer, and a 32 KB
  `BufReader` — which at POD5 chunk sizes is a large share of the decode.
  Measured **1.13x on tRNA-length reads and 1.49x on mRNA**, for every caller
  of `decompress_signal`, with no API change. Buffers are sized from the ZSTD
  frame's content size rather than the SVB16 worst case (~1.75x over what real
  signal encodes to), and the thread-local scratch is capped at 1 MiB so one
  pathological read cannot pin memory per rayon worker.

- `Reader::get_signal_bulk` now errors when the signal table returns fewer
  chunks than rows requested. It previously carried on and mis-assigned the
  remaining chunks to the wrong reads — reachable only from a malformed file,
  but silent when it happened. `get_compressed_signal_bulk` already guarded
  this.

### Fixed

- **`decompress_signal_prefix` no longer loses to a full decode.** It gated the
  streaming path on a sample fraction (`n*4 >= total`), which put the crossover
  where streaming is *slowest*: a 25% prefix measured **0.80x** — slower than
  decoding the whole read. Two causes, both fixed. The gate was on the wrong
  quantity: what governs early exit is whether a whole 128 KiB ZSTD block can
  be skipped, which at tRNA read lengths is never, so streaming ran and saved
  nothing. And the SVB16 half ran a scalar decoder while the full path
  dispatches to AVX2 — forced at a 75% prefix, that path cost 0.43x a full
  decode. Both branches now share the SIMD decode via the new
  `svb16::decode_split`, which accepts a chunk-sized key section; the
  scalar-only `svb16::decode_prefix` is gone, superseded by it. `escpod demux
  detect`/`fingerprint`, which already read prefixes, pick this up unchanged.

## 0.11.0 (2026-08-17)

### Fixed

- **`escpod signal classify` applies the bundle's abstain rule, and every
  unscored read now says why.** A charging bundle can name reads the model must
  not be asked about; the block was parsed and then dropped, so those reads got
  a confident `cl` like any other (#230). The rule is now evaluated, and one it
  cannot evaluate is a load error rather than a silent pass — accepting an
  unknown rule would score exactly the reads the bundle excludes.

  **What the rule catches is a distinct population, not a scoring failure.**
  Measured on a 1.06M-read edx07 corpus it fires on **0.85%** of scoreable
  reads. On those reads the alignment stops exactly at the CCA-adapter
  junction with a median 81-101 nt of unaligned sequence after it, at *higher*
  mapq than the reads that were called, and that 3' sequence is the reverse
  complement of the common arm 51.8% of the time (the common partner oligo is
  the arm's revcomp, so the plurality are reads of the wrong strand of the
  duplex), poly(A) 4.2%, the arm-but-unaligned 1.4%, other 42.5%. They are
  reads of something else, so the `reason` column names the population
  (`no_aligned_arm`) rather than the mechanism that caught them.

  The bundle's own rationale is stale and worth correcting upstream: it cites
  23-34% of charged-library reads, measured under the *aligner*-derived span
  rule on the yeast/v2 adapters. Under the counting anchor on this adapter
  family the geometry it was written for has largely been fixed, which is the
  outcome `rnabioco/aa-tRNA-seq-pipeline#110` is after.

  **Every anchored read is now accounted for.** `--tsv` gains a `reason`
  column and emits a row per unscored read — `no_aligned_arm`, `no_signal`,
  `ns_mismatch` — where before they simply vanished and were visible only as
  an aggregate warning. That is the same failure #110 raises against remora
  one layer up: a 12% drop that had to be inferred from the gap between two QC
  rows. The column is empty for a call, so the file still reads as
  `read_id, reference, p, cl` for anything that only wants calls. No-called
  reads get no `cl` tag on the BAM, per the bundle's `emit` contract.

- **The counting anchor falls back to a resolved arm base, not straight to the
  junction (#226).** When the counted boundary base runs off the end of the move
  table, the reference implementation falls back as the aligner mode does — any
  resolved arm base first, since it over-masks least, and the junction only when
  nothing of the arm resolved at all. Counting mode went straight to the
  junction, masking more of the window than it should. Found by comparing a real
  corpus rather than the fixture: 4 reads in 842 came back `junction_fallback`
  against the reference's `arm_fallback`, which moved their mask boundary and so
  changed the features. All 19 fixture reads resolve as `counted`, so the golden
  could not catch it; both modes now share one fallback chain.

- **`escapepod-classify` matched the charging feature definition as it stood on
  2026-08-10, not the current one (#222).** Two commits landed in
  escapepod-models on 2026-08-13 and neither was carried over, so every charging
  model trained since computed different features from the ones `escpod
  classify` would score it with — silently, because the recipe recorded nothing
  that could catch it. Regenerating the golden from the current reference showed
  it stale by **116 / 1900 features (6.1%)** in NaN pattern with **zero**
  finite-value mismatches: the arithmetic always agreed, *which offsets resolve*
  did not. The Rust matched the stale golden, so the parity test was green
  against a superseded definition.

- **The charging feature-model CNN scores 6.1× faster — 305 → 50 µs/read —
  by hoisting the convolution padding out of the graph at load.** A `Conv`
  lowers to im2col + matmul in tract, and tract's im2col has a fast block-copy
  path that it abandons the moment `pads != 0`, falling back to a per-element
  bounds-checked loop. On the shipped `charging_fnn_ldx16x_rna004` the second
  convolution (96→96, k=3, over 33 offsets) spent **257 µs of the model's 310 µs
  building a 9,504-float buffer** — ~26 ns per element, about 100× a memcpy.
  The matmul it feeds is fine at 63 GFLOP/s; only the packing was broken.

  Zero padding *is* a concatenation of zeros, so `FeatureNet::load` now rewrites
  each padded `Conv` into `Concat(zeros) + Conv(pads=0)` on the ONNX proto
  before tract sees it. Bit-identical by construction and in fact: exactly equal
  logits over 200 random inputs against the original graph, and
  `tests/charging_fnn_parity.rs` — which pins the fixture bundle, the same
  two-padded-conv architecture, against golden vectors bit-exactly — runs
  through the rewrite.

  Two other spellings were measured and rejected. An ONNX `Pad` node before an
  unpadded `Conv` is fused straight back into the convolution by tract's
  optimizer, restoring the slow path (272 µs); and a larger batch amortises
  nothing, because the cost is per row, not per call (252 µs/read at batch 64).
  `Concat` is used precisely because it survives optimization.

  This lives in the loader rather than in escapepod-models' export on purpose.
  The natural PyTorch spelling (`ConstantPad1d` + `padding=0`) emits `Pad`, so
  an export-side fix would need the *same* protobuf surgery — but applied to a
  published, sha256-pinned artifact, which would then carry zero-concat nodes
  that exist only because of one Rust runtime, and would leave every
  already-shipped bundle slow. Here it fixes bundles that already exist, and if
  tract's padded im2col is ever fixed the rewrite degrades to a no-op that costs
  one graph node — never a wrong answer.

  The rewrite refuses anything it cannot be sure of rather than guessing: a
  single spatial axis only, the default ONNX domain, `group = 1`, explicit
  non-negative `pads`, a weight that is a graph initializer, and never when
  `auto_pad` is computing the padding.

  For scale: the LSTM arm is untouched (it has no convolutions) at 468 µs/read,
  and the GBM arm is microseconds. At 2M reads on 32 cores the CNN scorer now
  costs ~3 s rather than ~19 s.

### Performance

- **`demux basecall` reads ahead one batch, so the encoder is not idle (#218).**
  The per-Arrow-batch stages ran in lockstep — read signal → prep → encode →
  match → write — with nothing overlapping. On a live production run (136 GB /
  20 files, A30, `--gpu`) that measured **62% mean GPU utilisation with 20% of
  samples below 5%**, the dead windows being the reader stalled *inside a page
  fault* demand-paging off the network filesystem at ~2.4 MB/s, with only ~1.4
  of 16 allocated cores busy.

### Added

- **Python: `AnchoredReads` — motif-anchored windows and per-offset statistics
  (#223, #224, #225, #227).** `escapepod-classify` already scans a BAM, indexes
  POD5 and reduces each read to per-offset statistics in parallel; there was no
  way into it from Python, so the training side reimplemented the loop in NumPy
  on one core. Nothing in the binding names an assay: it takes a reference
  motif, base offsets, a mask rule and optionally a k-mer table, and returns the
  signal window plus `(dwell, mean, std, resid)` per offset, so a corpus build
  and `escpod signal classify` compute features from the same code rather than
  two ports that agree until they don't.

  `extract` returns the **full corpus row** rather than five of its fields
  (#224) — `JunctionCoords` now carries `cca_a_sig`, `cca_a_dwell`,
  `junction_dwell`, `arm_resolved_depth`, `aligner_arm_depth`, `polya_mid_sig`
  and `body_mid_sig`, computed once where the spans are, because recomputing
  them in the caller is how two implementations of one rule start.

  The scan's own bookkeeping is exposed too (#225): `records_scanned`, `skips`
  keyed by a stable snake_case reason, and `orientation_votes`. The corpus
  builder's retention audit is a mandatory gate that compares rejection rates
  run to run — a rate that differs by class is a label-correlated filter, which
  is how two separate bugs got into this pipeline.

  `storage_order(read_ids)` (#227) reorders a selection the way `extract` will
  read it. `extract` already sorted *within* a batch (199 s → 55 s on 60k
  reads); a caller that shuffled 8M ids into 250k-read batches still swept all
  20 POD5 files per batch — ~32 passes over 136 GB on a network filesystem,
  presenting as "huge RAM, no CPU" with `read_bytes` flat at zero, because POD5
  is mmap'd and its page faults never appear as `read()` bytes.

- **`escapepod-signal` gains two model-agnostic primitives (#221).**
  `features::span_stats` reduces caller-supplied `[start, end)` intervals to
  dwell, mean level and sd via prefix sums over the spanned region — one pass
  plus O(1) per span, rather than a pass per span. How a base maps to signal is
  the caller's business; the reduction is identical either way. K-mer level
  centring is now explicit rather than assumed.

- **`escpod signal classify` loads the per-base-feature ONNX charging model
  (`feature_model`).** The `escapepod-charging-classifier/1` format names three
  models; the runtime implemented one of them, the GBM, and refused the others
  with `missing field \`gbm\``. The one worth shipping is the third: a small
  network over the *same* per-base features, which on a held-out flowcell
  (three paired seeds) scores AUROC 0.9621 ± 0.0001 against the GBM's
  0.9475 ± 0.0001 and MCC 0.8399 against 0.7928 — and calls **0.727 of reads at
  99% precision against the GBM's 0.449**, so the gap is ~28 points of usable
  yield, not a decimal on a summary statistic.

  Almost nothing had to change to run it. The two models read one feature
  space, so anchoring, spans, the k-mer residual, the mask and the column
  selection are shared verbatim; a bundle now carries `gbm` **or**
  `feature_model` and the whole model-specific part of the pipeline is a
  two-armed `ChargingScorer`. Which arm a directory holds is a property of the
  bundle, never a flag, and `escpod signal classify` names it in its startup
  line.

  Three rules stand between the flat feature vector the runtime already builds
  and the graph, and each fails *silently* if guessed: the fold from
  offsets-outer columns to a `[channel, offset]` tensor, the per-channel
  standardisation (constants fitted on the training split and shipped, never
  recomputed per batch), and missingness — `NaN` is never handed to the
  network, whose value channel is zeroed while a paired observed channel
  carries the indicator. All three are declared in the bundle and reproduced
  from what it declares. The fold in particular is checked against
  `features.order` at load: the declared channels must reproduce the declared
  column names, because folding the other way transposes every input and still
  scores.

  Refusals stay explicit rather than positional. A bundle declaring **both**
  scorers, or **neither** (the raw-signal CNN variant, a different input space
  this runtime does not implement), is rejected with its own message; the ONNX
  is pinned by sha256 like every other bundle dependency; and a load-time shape
  probe insists on `[1, 2]` logits, so a differently-headed graph fails at load
  with the file named rather than downstream as a wrong probability on every
  read.

  No new third-party crate: `tract-onnx` was already in the binary via
  `cnn-detect`. Library consumers select it with `escapepod-classify`'s
  `fnn-onnx` feature, which the CLI's `classify` feature turns on; without it
  such a bundle is refused with a rebuild hint.

  Parity is pinned two ways. `tests/charging_fnn_parity.rs` folds the
  *reference implementation's own* feature vectors so the input tensor
  comparison is **bit-exact**, then runs the full pipeline for the probability
  (max |ΔP| 2.1e-7 over the fixture reads) — a golden built only end to end
  could not tell a wrong rule from the feature grid's known 1e-4 rounding
  headroom. `examples/verify_feature_model.rs` plus
  `scripts/dump_feature_model_reference.py` do the same against a real bundle
  on real weights and a real corpus, where the shipped feature set is a subset
  (2 statistics of 4, 33 offsets) and `select_columns` is not the identity:
  24,576 reads over two corpus slices, max |ΔP| 2.4e-7, median 1.1e-8, none
  above 1e-5.

### Changed

- **A charging bundle's metadata is now a closed schema: a key this runtime
  does not implement is refused, not ignored.** `metadata.json` was parsed
  leniently, so any block the loader did not model was dropped silently at
  parse time — and every key in that file is a *rule the model was built
  with*. Two were already being dropped, both of them undetectable downstream
  because a bundle scored against the wrong one produces exactly the output it
  should:

  - `abstain` — which reads must not be scored at all. The shipped GBM bundle
    has always declared `aligner_arm_depth == 0`, measured at balanced
    accuracy 0.4993 with **100% of the uncharged library called charged**, and
    the runtime scored those reads anyway.
  - `features.feature_set` — whether the dwell columns were divided by the
    read's own median before training (`rel_dwell`, `all_rel`). The transformed
    columns keep their plain names, so `features.order` cannot express the
    difference and nothing further down can catch it.

  Every block that can carry a rule is now `deny_unknown_fields`, which means
  the schema *names* the builder's prose (`anchor.description`,
  `features.per_base`, `feature_model.input.fold`, …) rather than allowing it
  by omission: documenting a rule stays free, introducing one does not.
  `provenance`, `metrics` and `caveats` are deliberately exempt and stay
  free-form — nothing under them can change what the model sees, and that is
  where new documentation with no natural home belongs. The trade is explicit:
  a bundle from a *newer* builder now fails to load rather than loading with
  its new rule quietly ignored, and refusing to answer is the recoverable half
  of that.

  Two named rules are refused outright, because the runtime could so nearly
  run them: a non-empty `refinement.opts` (a banded DP that re-fits the
  signal-to-base mapping *before* the features are taken — this runtime's
  spans come straight from the move table), and a `features.feature_set`
  that transforms its columns per read. An unrecognised feature-set name is
  refused too rather than assumed harmless; a bare offset rule (`arm_le24`,
  `collapse_safe`) is matched by shape, so widening the cap is not a new name
  to teach. `abstain` is named and *carried* instead —
  `ChargingBundle::abstain` hands it to the caller, and `escpod signal
  classify` warns at startup that it is not applying it (#230), which is the
  honest state until that lands.

  Deciding the variant now happens before the strict parse, which also fixes
  the raw-signal CNN refusal for older bundles: `charging_cnn_rna004@v0.1.0`
  predates `classes` and spells its k-mer table `path`, so it used to fail
  with `missing field \`classes\`` — the exact "go looking for a corrupt file"
  failure the by-name refusal was written to prevent. Verified against the
  real shipped bundles: `charging_gbm_ldx16x_rna004@v0.1.0` loads unchanged
  (132 columns, abstain rule carried), and the raw-signal CNN gets its own
  message.

- **`escpod classify` is now `escpod signal classify`.** The bare top-level
  spelling sat one word away from `escpod demux classify` while meaning
  something else entirely: `demux classify` assigns a barcode from a DTW/GBM
  adapter fingerprint, whereas the charging classifier scores a read-level
  model against raw signal anchored on the CCA–aa junction in *reference*
  coordinates (different inputs, different output, different failure modes).
  Two commands that share a verb and share nothing else is a trap in a shell
  history or a pipeline script, so the charging classifier moved into a
  `signal` group named for what it operates on.

  The old spelling still works: it is a hidden alias that logs
  ``warn: `escpod classify` is deprecated; use `escpod signal classify`.``
  and forwards to the same runner — verified by an end-to-end test that
  asserts the two invocations produce byte-identical calls, so the alias
  cannot drift into a second implementation.

  `signal` has no default action, so unlike `demux` and `resquiggle` it is a
  plain required subcommand enum with no flattened run-args struct.

- **`escapepod_classify::feature_grid` takes a `FeatureRecipe`, not a
  `ChargingBundle`.** The grid only ever read three things out of the bundle —
  the offsets, the span mode, and the k-mer levels the residual is taken
  against — but taking the whole bundle meant anything that wanted features
  had to own weights, an operating point and a set of verified checksums it
  had no use for. The caller that most needs the features is the corpus
  builder, which by definition has no model yet, so it was left choosing
  between a fake bundle and its own copy of the definition. This repo has
  already shipped two divergent feature definitions once.

  `FeatureRecipe` is a borrowed view (`bundle.recipe()`, or
  `FeatureRecipe::from(&bundle)`), so there is no second copy to drift.
  `ChargingBundle` still owns parsing and verifying `metadata.json`.
  `KmerLevels` moves from `bundle` to the new `recipe` module and is
  re-exported at the crate root. A new `feature_grid_at` takes coords the
  caller already resolved, so a caller wanting both a window and the features
  runs `finalize` once instead of keeping its own residual computation.

- **The signal-window rule moved out of the pyo3 binding** into
  `escapepod_classify::window` (`signal_window`, `BaseJustify`). Anchoring
  inside the junction base, the `[anchor-left, anchor+right)` slice, `NaN`
  padding and the mask of everything earlier than the common-arm start are
  model contract — the Rust inference path has to reproduce them exactly —
  but the only implementation lived in `escapepod-python`, where that path
  could not see it. That is the same structural mistake that left this
  pipeline with two divergent feature definitions once already. The binding
  now calls it, and the rule is pinned by unit tests in the crate that owns
  it: padding at both ends, each justification, and the mask boundary landing
  on `common_start_sig` exactly (first masked / first surviving sample).

  Two things the move made visible, both preserved deliberately rather than
  fixed: the reject rule counts real samples across the *whole* window, not
  samples after the anchor (a read short on the right is kept if the left
  makes up the count) — the docs said the latter; and the justification shift
  uses `JunctionCoords::junction_dwell` rather than recomputing the dwell from
  the move table, which is the same number and cannot be resolved against
  different coords than the window is cut from.

## 0.10.0 (2026-08-14)

### Added

- **`escpod classify` — the tRNA charging (aminoacylation) classifier**
  (#204 §2/§3), in a new `escapepod-classify` crate. Takes POD5 *and* an
  aligned BAM with move tables (the input pair `remora infer
  from_pod5_and_bam` takes): the model anchors on the CCA–aa junction,
  which only exists in reference coordinates. The chain — CCAGGC junction
  location, CIGAR ref→query, Remora-convention move-table mapping, per-run
  frame-orientation detection (voted from the data, never assumed),
  per-base dwell/mean/std plus the z-scored k-mer *residual*, divergent
  region masked — is a parity-tested port of the training-corpus
  implementation in escapepod-models. The recipe (feature order, offsets,
  mask rule, k-mer table pinned by sha256, recommended operating point)
  comes from the model bundle's `metadata.json`, not flags: a caller
  computing features differently gets a wrong answer, not an error.
  Output is the input BAM with `cl = round(P(charged)·255)` (uint8)
  written directly onto every record of each classified read — no modbase
  `ML`→`cl` round-trip — plus an optional TSV; the summary reports against
  the bundle's operating point rather than the legacy hard-coded 200.
- **Binary (sigmoid-head) GBM support** in `escapepod-demux`'s native GBM
  runtime and `scripts/export_gbm_model.py`: sklearn binary
  `HistGradientBoostingClassifier` models (one tree per iteration, raw
  score = logit of class 1) now export and run alongside the multiclass
  softmax layout, discriminated by a `head` field. NaN feature routing
  (`missing_go_to_left`) was already supported and applies to both.
- **`bam-filter` takes `-t/--threads`.** It fans out over a POD5 directory
  through the same `filter_files` core as `filter` and `subset`, both of which
  have always accepted the flag, but `bam-filter` was listed among the commands
  that "run on the default pool" — so its width was pinned to
  `available_parallelism()` with no way to raise it.

  That is the wrong default for this workload. Extraction reads signal rows
  through an mmap, so on a network filesystem the run is bound by page-fault
  latency rather than CPU or bandwidth — and page-ins do not show up in
  `read_bytes`, so a run in this state looks stalled even while it progresses.

  Pulling 97,386 reads out of 3.8M across a 42 GB / 8-file directory on a
  network filesystem, same host and same input, only `--threads` varying:

  | threads | wall | CPU |
  |---|---|---|
  | 2 | 503s | 11s |
  | 16 | 237s | 9s |

  2.1x faster for the same 9-11s of CPU — the work is waiting, not computing,
  and only concurrency moves it. Selection is unaffected: both runs emit the
  same 97,386 reads with byte-identical signal. On a larger 208 GB / 66-file
  directory the same extraction spends 28m47s wall for **25s of CPU** (1.5%
  utilization), so the pool wants to be well above the core count.

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

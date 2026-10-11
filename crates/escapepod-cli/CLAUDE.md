# escapepod-cli

## Purpose and layering
- Builds `escpod` (`[[bin]] required-features = ["cli"]`) plus an umbrella library (`escapepod_cli::{pod5,signal,demux,classify}`) for `default-features = false` consumers — none exist in the workspace.
- Depends on `escapepod-signal` (re-exports `pod5`), `escapepod-demux`, `escapepod-classify`. BAM/FASTA/FASTQ I/O (noodles, hand-rolled FASTX) lives here, not in the libraries.

## Module map
- `main.rs` — clap tree, `EscpodFormatter`, `-q/-v` → `EnvFilter`, `requested_threads`, signal/SIGPIPE handlers, dispatch, arg-parse unit tests.
- `lib.rs` re-exports; `build.rs` `ESCPOD_VERSION`; `progress.rs` bars (hidden below INFO); `commands/profile.rs` `PhaseTimer`; `style.rs` ANSI gated on stderr TTY + `NO_COLOR`; `threads.rs` the one rayon `init`; `device.rs` `--device` + `Stage` placement; `util.rs` input/output resolution, BAI; `test_env.rs` env lock for tests.
- `commands/`: `view inspect summary merge filter bam_filter subset index annotate sidecar classify` (`sidecar` only nests `index`/`annotate` and owns the hidden `escpod index`/`escpod annotate` forwarders), `signal` (deprecated alias), and under `experimental`: `repack resquiggle resquiggle_models`.
- `commands/demux/` — see `commands/demux/CLAUDE.md`.

## Feature flags
| Feature | Gates |
|---|---|
| `cli` (default) | the binary; implies `signal demux classify cnn-detect crf-decode demux-models model-fetch` |
| `mimalloc` (default) | global allocator |
| `pod5 signal demux classify` | library layers → `escapepod-*`; `classify` also forwards `fnn-onnx`,`waveform-onnx` |
| `experimental` | `repack resquiggle`; implies `classify` |
| `model-fetch` | ureq/sha2/zip; `models-download` = `resquiggle models fetch`, implies `experimental`+`model-fetch` |
| `demux-models` | ≡ `demux` today |
| `gpu` | every GPU path → `escapepod-demux/gpu`, `escapepod-classify/cuda` |
| `train` | → `escapepod-demux/train` |

Planned: `experimental` ≡ `classify` — fold or split; drop the `demux-models` alias; `models-download` must stop dragging in `classify`.

## Output contract
- stdout = data only (`println!`); stderr = everything else via `tracing` (the formatter prints the level — never hand-prefix `Warning:`).
- Progress bars auto-hide below INFO; multi-line report blocks gate on `tracing::enabled!(Level::INFO)`.
- Colour gates on the stream it writes to (`NO_COLOR` + that stream's TTY) — stderr-gated colour on stdout breaks `| less` and `> file`.

## Build, test this crate alone
Use the `srun -p rna` form from the root CLAUDE.md. Slowest crate: it alone links tract + noodles.
```
pixi run cargo build --release -p escapepod-cli
pixi run cargo build -p escapepod-cli --no-default-features --features cli   # system allocator A/B
pixi run -e dev cargo nextest run -p escapepod-cli && pixi run cargo test --doc -p escapepod-cli
```
`tests/`: `classify_e2e` (golden parity + alias warning), `sidecar_group` (new vs hidden-legacy spelling byte-for-byte), `dir_inputs`, `thread_pool` (spawns the binary) — all read `../escapepod-classify/tests/fixtures/`, no `ext/`. `main.rs` unit tests pin parsing; `cli_definition_is_internally_consistent` is clap's `debug_assert` and the only thing that catches a bad `#[command(flatten)]`.

## Invariants and traps
- `requested_threads` is an exhaustive match on purpose — the #155 tripwire.
- `threads::init` has one call site, before dispatch; an earlier `par_iter` pins the pool silently.
- `device.rs` uses `cfg!` not `#[cfg]` so a musl build can say the feature is missing.
- `collect_pod5_inputs` dedupes but keeps argument order — `merge`/`repack`/`index` output order depends on it.
- `experimental` stubs (`repack`, `resquiggle`) exist so `escpod repack` errors with a rebuild hint, not "unknown subcommand". `annotate` has none: it ships by default, and a leftover stub would shadow the real command.
- `IndexArgs` / `AnnotateArgs` are the one definition behind `escpod sidecar index|annotate` **and** the hidden top-level `escpod index|annotate` forwarders (#488): a flag is added once, and `sidecar::run_legacy_*` only `warn!`s and calls the same `run`. `requested_threads` names all four paths — a deprecated spelling that loses `-j` is the #155 bug again. `legacy_index_and_annotate_are_hidden_but_work` fails if a forwarder is un-hidden or stops warning. Every user-facing message says `escpod sidecar index`; the old spelling appears only in the forwarders, their test and changelog history.
- `build.rs` must name the real `HEAD` path (`git rev-parse --git-path HEAD`): a nonexistent `rerun-if-changed` target reruns the script every build.
- `annotate --from-table`/`-a` look before they write: a pre-pass over every input's `read_ids()` refuses (error, nothing written) when no read matches. With the fill on, every read counts as assigned, so nothing the writers report can catch a table from another run, and an in-place replace would overwrite a good column with all-`unclassified` → `tests/annotate_table.rs::a_table_for_another_run_is_refused_before_anything_is_written`. The fill (`apply_fill`) is CLI-side and must keep `codes.len() == index.len()` per column — the writers `get` a slot and silently drop one with no code.
- `util::open_maybe_gzip` sniffs `1f 8b` and uses `MultiGzDecoder`; `GzDecoder` silently truncates a multi-member table → `a_multi_member_gzip_is_read_whole`.
- Every command resolves POD5 reads through the read index and processes them in storage order (the #334 rule); BGZF pools are sized from the rayon pool — `MultithreadedReader::new` is one worker.

## Known debt
- `main.rs:447–517,686–701,964–978`: `cfg(not(feature = "demux"|"classify"))` stubs unreachable (`cli` implies both).
- `main.rs:151–654`: `threads/force/profile` ×6 → flattened `CommonRunArgs`; three stubs → one.
- `classify.rs:175–207`/`:381–435`: scan-report-index duplicated in `run`/`run_waveform`.
- `resquiggle.rs:132–187`, `classify.rs:97–114`: sentinel `value_parser`s → `ValueEnum`.
- `inspect.rs:30,259,315`: open-with-warning ×3; `util::OpenResult` → `Result<Option<_>>`.
- `merge.rs:47–152`: re-implements `PhaseTimer`. `filter.rs:213`/`bam_filter.rs:98`: same callback boilerplate.
- `subset.rs:127–264`, `view.rs:141–229`: tests of `escapepod_pod5` fns already tested there.
- `lib.rs` docs stale (no `classify`, `version = "0.5"`); `profile.rs:12` links nonexistent `PhaseTimer::finish`.

## Do not
- Size a rayon pool in a command — `threads::init` is the only builder.
- Add a `cfg(not(feature))` stub for anything `cli` implies — dead code.
- `println!` status or colour stdout — breaks `| tool` and `> file`.
- Sort in `collect_pod5_inputs` — reorders users' output.
- Add a GPU flag/feature — extend `device::Stage` under the atomic `gpu`.
- Add fixtures here — reuse `escapepod-classify/tests/fixtures/`.
- Restate `index`/`annotate` args or logic in `commands/sidecar.rs` — it nests the existing structs and handlers; a copy is how the two spellings drift apart.

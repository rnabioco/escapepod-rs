# escapepod-pod5

## Purpose and layering
- POD5 container I/O only: mmap reader, writer, VBZ, byte-level Arrow IPC parser, `.p5s` sidecar, block-level merge/filter/subset/repack. No signal algorithms. No cargo features.
- Depends on nothing in the workspace. `escapepod-signal` re-exports the whole surface as `pod5`; `escapepod-python` consumes it through that re-export — grep `crates/escapepod-python` before calling anything unused.

## Module map
- `reader/` — `Reader::open`; `ReaderCache` (one indexed reader per file per process); `ReadIndex`; `SignalExtractor`; `dataset.rs` `Dataset`/`DatasetCache`/`cached_dataset` (one file, a directory, or a mix, as one random-access-by-read-id collection; every file routes through `ReaderCache`/`cached_reader`, never a second per-file cache — rnabioco/escapepod-rs#384).
- `writer/` — `Writer::create`; `atomic.rs` `AtomicFile` (temp + rename).
- `arrow_ipc.rs` — `ArrowIpcFooter::parse_with_row_counts`, `extract_signal_rows` (zero-copy).
- `arrow_helpers.rs` — `ReadsBatchView` (columns resolved once per batch), `ReadColumns`, `verify_index_row`.
- `sidecar.rs` — per-file + collection `.p5s`; `load_sidecar_for_write`, `write_sidecar_file_checked`.
- `operations/` — filter+subset share `assemble_output`; `repack`; `annotate` (`resolve_sidecar` merges own + collection).
- `merge.rs` — `merge_files`, copies signal IPC blocks byte-for-byte.
- `compression/` — `vbz::decompress_signal_prefix` is the one decode path; `svb16/` dispatch only in `encode`/`decode_split`.
- `footer.rs` `parse_footer`; `schema/` `narrow_channel`; `utils/table_builders` (every output table); `utils/pod5_assembler::write_post_signal_sections`; `flatbuffers_gen/` generated.
- `env.rs` — `flag`/`positive_usize`/`usize_allow_zero`, the one parse for every `ESCAPEPOD_*` boolean switch and integer knob workspace-wide (rnabioco/escapepod-rs#410; `usize_allow_zero` for knobs where `0` means something, #421). Lives here, the lowest crate, so every other crate can reach it without a new dependency edge.

## Build, test, bench this crate alone
Root CLAUDE.md "Build baseline and SLURM builds" for the `srun` form.
- `srun -p rna -c 32 --mem=32G pixi run cargo nextest run -p escapepod-pod5`
- `srun -p rna -c 32 --mem=32G pixi run cargo test --doc -p escapepod-pod5` (nextest skips doctests)
- `srun -p rna -c 32 --mem=32G pixi run cargo clippy -p escapepod-pod5 --all-targets`
- `srun -p rna -c 32 --mem=32G pixi run cargo bench -p escapepod-pod5 --bench io_hot_paths`
- Silently skipped without `data/drna/` or `ext/`: `arrow_ipc::tests::test_parse_real_signal_table` and the seed corpus of `test_parser_never_panics_on_malformed_input`. All `tests/*.rs` write their own POD5.
- Compat suite (official `pod5` reads our output) is NOT in nextest: `pixi run -e python-test test-compat`.

## Invariants and traps
- Sidecar never writes the POD5 → `annotate_writes_sidecar_and_pod5_is_untouched`.
- `parse_with_row_counts` checks a supplied geometry against first+last batch, never trusts it → `a_wrong_geometry_is_detected_and_the_file_still_reads_correctly`.
- Every signal batch but the last holds exactly `signal_batch_size` rows (flush per chunk) — other readers assume the stride and return the wrong read → `signal_batches_are_uniform_with_multi_chunk_reads`.
- `MADV_SEQUENTIAL` stays: measured seq≈normal, random ~10× worse; no test pins it.
- Read `channel` u16 (V3–V5) or u32 (V6); always write V5/u16 — PyPI `pod5` ≤0.3.44 rejects uint32 → `v6_compat`, `emitted_channel_width_matches_the_stamped_version`, `oversized_channel_is_refused_not_truncated`.
- arrow-ipc `FileWriter` errors on any dictionary change between batches → multi-batch `Writer` output needs `PredefinedDictionaries` → `dictionary_columns_consistent_across_batches`.
- `extract_signal_rows` returns mmap slices (chunks borrow the `Reader`), batches in file order, rows sorted within a batch → `test_signal_roundtrip`, `parallel_row_counts_keep_their_batch`.
- By-id lookups always go through `read_index()`, sort work by `(file, first signal row)`, confirm every locator → escapepod-classify `tests/charging_index_order.rs` (rotated locator must fail).
- `signal_ipc_footer()` is parsed once per `Reader`, seeded from the sidecar; `ArrowIpcFooter::parse` re-walks every batch header (minutes cold on BeeGFS).
- `read_sidecar_metadata`/`read_sidecar_file` share `check_version`; `load_sidecar_for_write` classifies Absent/Foreign/Unreadable — only the first two may be replaced.
- `Sidecar::set_signal_batch_rows(vec![])` is a deliberate no-op.
- Reads table must be multi-batch (default 1000) — demux shards by batch → `*_splits_the_reads_table_into_batches`.

## Env levers
| var | effect | default | pinned |
|---|---|---|---|
| `ESCAPEPOD_AUTOINDEX_MAX` | reads above which an in-memory index build warns and `ReaderCache` skips warm-up; `0` skips warm-up always (via `env::usize_allow_zero`); never disables indexing, not a memory cap | 5_000_000 | `=0` → `test_autoindex_max`, Python `test_reader_threshold_disables_autobuild` |
| `ESCAPEPOD_POD5_FOOTER_SERIAL` | `1/true/yes/on` (via `env::flag`) forces the serial batch-header walk | parallel at ≥16 batches | serial≡parallel pinned; lever untested |
| `POD5_DISABLE_MMAP_OPEN` | any value skips the pre-mmap SIGBUS probe | probe on | untested |

## Known debt
- Reads-table builder ×3: `utils/table_builders.rs:431-627` ≡ `:646-829`; `writer/file_writer.rs:618-761`.
- Writer duplicates post-signal assembly: `file_writer.rs:764-880` ≡ `table_builders.rs:103-217`; `:883-934` ≡ `build_pod5_footer`; `finish` ≡ `write_post_signal_sections`. Dead raw-signal API `file_writer.rs:245-248, 426-576`.
- Hand-parsed footer `footer.rs:148-420` despite `flatbuffers_gen::Footer` accessors.
- Indexed lookups ×3 `reader/file_reader.rs:1527-1702`; 7 copies of "slice embedded table".
- Sidecar mirror: `sidecar.rs:1034-1134`≡`:1861-1956`, `:683-762`≡`:1640-1698`.
- `utils/run_info.rs` not in `utils/mod.rs` (never compiles); `utils/dictionary.rs` dead.

## Do not
- Copy `.pod5` into a worktree/test dir — root CLAUDE.md storage rules; symlink.
- Put fixtures under `tests/data/` — bare `data/` is gitignored.
- Read signal in caller order — 60× slower, invisible in output.
- `ArrowIpcFooter::parse` a table a `Reader` already opened — per-batch header walk.
- Write `channel` uint32 or bump `POD5_VERSION` alone — every installed reader breaks.
- Emit multi-batch reads tables without predefined dictionaries — arrow-ipc rejects replacement.
- Rebuild a sidecar that merely failed to load — annotations exist nowhere else.
- `stream_position()` on the 128 MiB `BufWriter` — forces a flush.
- Bump the SIMD baseline — runtime dispatch only (root CLAUDE.md).

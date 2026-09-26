# escapepod-align

## Purpose and layering
- All-vs-all read-to-reference alignment for small panels (tRNA) — the engine behind `escpod align` (#395). Every read is scored against every reference with an affine-gap DP, the best-scoring reference(s) win, ties are reported as ties, and a traceback is computed only for the winners. No seed index: memory is one read's DP, runtime is `reads × references × lengths`. Intended range **≤ ~10k references of ≤ ~1kb**; a bigger panel is simply proportionally slower.
- Written clean-room from the issue, with gpu-tRNA-mapper as the *functional* model only (every read against every reference, ties explicit, tracebacks for winners only) — none of its source was read, and its semi-global definition is deliberately not reproduced.
- A leaf crate on purpose: **std only, no noodles, no git dependency**, so unlike `escapepod-demux` it stays `cargo publish`-able. All BAM/FASTA/FASTQ I/O is the CLI's (`escapepod-cli/src/commands/align.rs`), the same split `escapepod-classify` keeps.
- Depends on nothing in the workspace (only `cudarc`, optional, under `gpu`). Consumed by `escapepod-cli` (`escpod align`).

## Module map
- `alphabet` — 2-bit A/C/G/T with U folded to T, case ignored; **every other letter is a wildcard that scores as a match**.
- `scoring.rs` — `Scoring`/`Mode`: a gap of length *k* costs `gap_open + (k-1)·gap_extend` (parasail's convention with signs, so bwa's `-O1 -E1` is `-2,-1` here); `Local` = Smith–Waterman, `SemiGlobal` = overlap (leading/trailing gaps of *either* sequence free = parasail `sg`).
- `scalar.rs` — the Gotoh oracle: score-only and with traceback. Every other path is tested against it by equality — integer scoring throughout, no tolerance anywhere.
- `simd/` — `avx2.rs`/`avx512.rs` score-only inter-sequence kernels (one reference per i16 lane) plus `trace_pairs` (one (read, reference) pair per lane); `mod.rs` owns `Backend` (`available`, `best`, `cap_from_env`) and `fits_i16`.
- `sam.rs` — CIGAR strings and the `MD`/`NM` pair, reproducing `samtools calmd` including at reference ambiguity codes.
- `panel.rs` — `Panel`: the reference set in file order, duplicate names refused.
- `mapper.rs` — `Aligner` (panel + dispatch + tie rules): `map_reads` (CPU, scores then traces) and `map_reads_scored` (takes a precomputed `ScoreMatrix`, e.g. from the GPU, and does everything after scoring identically); `Hit`, `ReadMapping`, `MapOptions`.
- `cuda/` (`gpu`) — `GpuScorer`: the same score, one CUDA thread per (read, reference) pair, batched. `mod.rs` orchestration, `kernel.rs` the CUDA source string.
- `error.rs` — `AlignError` (`FromStr` for `Scoring`/`Mode` parse errors, surfaced by the CLI's `value_parser`).

## Feature flags
| Feature | Gates |
|---|---|
| `gpu` | `dep:cudarc` (`dynamic-loading`) + `cuda/`: the CUDA score kernel. Nothing CUDA is needed to build; at run time it needs the CUDA driver and libnvrtc. `escapepod-cli`'s atomic `gpu` forwards here. |

## Build, test this crate alone
Use the `srun -p rna` form from the root CLAUDE.md; GPU tests need `srun -p gpu -A gpu_rbi --gres=gpu:1` (root CLAUDE.md "Build Commands").
```
pixi run cargo nextest run -p escapepod-align
pixi run cargo test --doc -p escapepod-align
pixi run cargo clippy -p escapepod-align --all-targets
pixi run -e dev-gpu cargo nextest run --features gpu -p escapepod-align --test gpu_parity
```
- `tests/parasail_golden.rs` — the scalar oracle against parasail: scores *and* coordinates, which only works because the tie-breaks were measured and matched (end cell = smallest reference end then smallest read end; traceback prefers diagonal > insertion > deletion, extend > open; local stops at the first zero). CIGARs are checked by re-scoring, since optimal paths are not unique.
- `tests/sam_fields.rs` — `MD`/`NM` against a golden measured on samtools 1.23.1 by `tests/fixtures/gen_calmd_golden.py`.
- `tests/gpu_parity.rs` (`gpu`) — `scalar::score` by equality: the SIMD tests' random generator in both modes and all three schemes, every read length 1–40 around the stripe, a 300-reference panel that splits into groups, the fixture reads against both fixture panels (108,624 pairs). Skips with a message where there is no device.
- `examples/gpu_score_probe.rs` (`gpu`) — times `score_batch` alone on real reads; keep it re-runnable (2.4e11 cells/s on an A30, ALU-bound).
- `simd/mod.rs`'s own equality tests (`avx2_matches_scalar_random`, `avx512_matches_scalar_random`, `simd_matches_scalar_on_fixture_reads`, `traceback_kernels_match_scalar`) sweep `Backend::available()`, not the dispatch's pick — run on `rna` (Cascade Lake) or the AVX-512 kernels never execute.

## Invariants and traps
- **Two kernel families, both inter-sequence.** The *score* kernels (`simd::{avx2,avx512}::score_group`) put one **reference** per i16 lane (16 / 32), references sorted longest-first into lane groups with a per-group profile `s(read code, ref[j])`; `j` outer, read `i` inner, so a column's profile vectors stay in L1, with a per-column lane mask to keep padding out of the best. The *traceback* kernels (`trace_pairs`) put one **(read, reference) pair** per lane, so a chunk's unique winners and one read's tie-set members fill lanes alike (`Aligner::map_reads`; the CLI feeds it 64 reads at a time); they write the scalar path's traceback byte per lane and hand every lane to the *same* `scalar::walk`, so their output is the oracle's by construction. The traceback family is not a nicety: on a real sample ~5% of reads are adapter-only and tie with all references of an adapter-flanked panel, and tracing ties then unique winners one reference at a time cost 65%, then 19%, of total CPU before it existed — the score kernel is now ~77%.
- **i16 is exact, not approximate.** Every real cell value is bounded by `min(n, L)·max(|match|,|mismatch|)` plus one gap; `simd::fits_i16` checks that per read and falls back to scalar; `i16::MIN` is a −∞ that saturation keeps −∞.
- Dispatch is `escapepod_signal::lstm`'s pattern: runtime `is_x86_feature_detected!` (never a baseline bump), kernels `#[inline(never)]`, `ESCAPEPOD_ALIGN_BACKEND=scalar|avx2|avx512` caps it.
- **The CUDA score kernel (`gpu`, `cuda/`, #401)** mirrors `escapepod_signal::dtw::cuda`: kernel source as a string, NVRTC at `GpuScorer::new`, which checks `is_culib_present` for the driver *and* NVRTC first so a missing library is an error rather than cudarc's panic (release aborts on panic). One thread per (read, reference) pair; the panel lives in shared memory at 4 bits a base (A/C/G/T one bit, anything else `0xF`, so "masks intersect" *is* the wildcard rule); the read is walked in 16-row stripes held in registers, only the stripe's bottom row leaving them. **Caps:** a read over `MAX_READ_LEN` = 4096 nt or outside `fits_i16` comes back unscored (the CPU scores it) — a *scheduling* bound, not a memory one, since one block owns a read serially; a reference over `MAX_REF_LEN` = 3072 (one warp's panel in 48 KB) refuses the scorer.
- **`MD`/`NM` follow `samtools calmd`, not the scoring** (`sam.rs`): a reference `N` (or any IUPAC code) is an `MD` mismatch against every read base, read `N` included, and counts toward `NM`; letters are upper-cased (`n`→`N`) but otherwise verbatim (a reference `U` stays `U`, and `U` equals `T` for matching). `escpod classify`'s windowed bundle rebuilds the read's reference from `MD` — the #306 reference-`N` trap from the other side. The scorer still calls that `N` a match: `AS` and `MD` disagree about it on purpose.
- `Aligner::map_reads_scored` runs the *same* tie/traceback/`MD`/`NM` code as `map_reads`, just fed a precomputed `ScoreMatrix` — this is what keeps `escpod align --device gpu` byte-identical to `--device cpu` (the CLI's own invariant, not this crate's to re-derive).

## Do not
- Add a noodles, git, or other non-`std` runtime dependency (besides `cudarc` under `gpu`) — this crate stays `cargo publish`-able on purpose.
- Read or write BAM/FASTA/FASTQ here — that is `escapepod-cli`'s job.
- Reproduce gpu-tRNA-mapper's semi-global definition, or read its source — the functional model only, clean-room.
- Bump the SIMD baseline — runtime dispatch only (root CLAUDE.md).
- Inline a SIMD kernel or merge dispatch arms without re-running the equality tests on `rna` — AVX-512 only executes there.
- Widen `MAX_READ_LEN`/`MAX_REF_LEN` without re-checking the shared-memory and scheduling bounds they exist for.

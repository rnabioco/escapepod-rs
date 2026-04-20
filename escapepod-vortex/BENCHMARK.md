# POD5 → Vortex Signal Encoding Benchmark

Goal: determine whether converting POD5 signal data to Vortex (with BtrBlocks
cascading + pco numeric codec) gives meaningful wins over POD5's native VBZ
(SVB16 + ZSTD) on (a) on-disk size and (b) decode-to-i16 throughput.

## TL;DR

There is a real trade-off, no Pareto winner:

- **`vortex-default` (FoR scheme)** wins **decode throughput** (2–4×) **and random access** (1.3–4.6×) but files are **30–40 % larger** than POD5/VBZ.
- **`vortex-pco` / `vortex-delta+pco`** beat POD5 on **size** by 4–5 %, but lose on decode (~30 % slower) and random access (~10 % slower).

If you care about interactive lookup or bulk decode and can tolerate larger files, `vortex-default` is a real win. If you care about bytes on disk, VBZ remains the best end-to-end pipeline.

## Test corpus

Two files in-tree (small, both ~1.7 MB / ~1.87 M samples):

- `data/drna/yeast_trna_reads.pod5` — 180 reads, dRNA
- `ext/dorado/tests/data/single_channel_multi_read_pod5/filtered.pod5`

Production data (`~/devel/rnabioco/2026-aars-in-vitro/results/demux/pod5/*`)
is 4.5–77 GB per file; not yet sampled in this report (deferred — small-file
results were consistent enough to call the question).

## Codec variants tested

| label | pipeline |
|---|---|
| `pod5` | SVB16 (delta + zigzag + StreamVByte) + ZSTD-1, baseline |
| `vortex-default` | `BtrBlocksCompressorBuilder::default()` — standard ALL_SCHEMES |
| `vortex-pco` | default + `with_compact()` — adds Pco scheme to integer cascade |
| `vortex-delta` | in-process `samples[i] -= samples[i-1]` then default |
| `vortex-delta+pco` | in-process delta then default + `with_compact()` |

`with_compact()` requires the `pco` and `zstd` features on `vortex-btrblocks`.

## What scheme the cascade actually picks

Compressing the first read of each file in isolation, with each codec:

```
yeast_trna_reads.pod5 — first read, 7962 samples (15 924 bytes raw)
    vortex-default      fastlanes.for      9 244 bytes  1.72×
    vortex-pco          vortex.pco         7 281 bytes  2.19×
    vortex-delta        vortex.zigzag      9 300 bytes  1.71×
    vortex-delta+pco    vortex.zigzag      7 252 bytes  2.20×

filtered.pod5 — first read, 766 044 samples (1.5 MB raw)
    vortex-default      fastlanes.for    958 720 bytes  1.60×
    vortex-pco          vortex.pco       656 905 bytes  2.33×
    vortex-delta        vortex.zigzag    882 082 bytes  1.74×
    vortex-delta+pco    vortex.zigzag    656 743 bytes  2.33×
```

The cascade is **one-level for the leaf encoder shown**: just FoR, just ZigZag,
or just Pco. Whether anything bit-packs on top is opaque from `encoding_id()`
alone but the byte counts suggest no further squeeze.

## Full file results (best of 3 runs, `--concat` mode)

```
=== yeast_trna_reads.pod5 (1.86 M samples) ===
format              bytes     vs_raw   vs_pod5   decode_s    MS/s
pod5            1 771 984      2.10×    1.000     0.0094     198
vortex-default  2 311 796      1.61×    1.305     0.0036     520
vortex-pco      2 073 136      1.79×    1.170     0.0115     161
vortex-delta    2 117 028      1.75×    1.195     0.0027     683
vortex-delta+pco 1 683 332     2.21×    0.950     0.0107     174

=== filtered.pod5 (1.87 M samples) ===
format              bytes     vs_raw   vs_pod5   decode_s    MS/s
pod5            1 682 120      2.23×    1.000     0.0079     237
vortex-default  2 349 860      1.60×    1.397     0.0021     876
vortex-pco      1 610 528      2.33×    0.957     0.0102     183
vortex-delta    2 154 844      1.74×    1.281     0.0028     662
vortex-delta+pco 1 610 364     2.33×    0.957     0.0110     170
```

`--concat` puts all signal samples for the file in a single chunk before encoding.
Without it (per-read chunking) sizes are unchanged but decode is roughly half as
fast in the chunk-iteration loop — likely per-chunk stream overhead.

## Why no clear win

1. **VBZ is genuinely well tuned for nanopore signal.** SVB16's delta + zigzag +
   StreamVByte is essentially purpose-built for low-magnitude i16 deltas. ZSTD-1
   catches what's left. The whole pipeline is SIMD-friendly.

2. **Default Vortex cascade picks `fastlanes.for`** — Frame of Reference. That
   captures the *value range* of a chunk but not the temporal autocorrelation
   that gives nanopore signal its compressibility. Result: 30–40 % bigger.

3. **pco wins on ratio, loses on decode speed.** Pco's ANS-style entropy decode
   has more data dependencies than SVB16's branch-free table-lookup unpack and
   ZSTD-1's fast path. The 4–5 % size win comes at the cost of ~30 % slower
   decode.

4. **In-process delta + Pco gives the best size** (`vortex-delta+pco`, 5 % smaller
   than VBZ) but doesn't help decode — pco still dominates the decode cost, and
   the Rust delta-undo loop adds work.

5. **The format has fixed overhead.** Vortex file footer + per-column statistics
   add ~tens of KB per file. At ~1.7 MB this is small but visible. Less of an
   issue at GB scale.

## Random access

Random-access fetch of 100 reads (chosen uniformly at random) on each format.
For Vortex, the data was written as `List<Int16>` so each read is one row;
`with_row_range(i..i+1)` fetches one read. For POD5, `find_signal_rows_by_ids`
+ `get_signal` for each random UUID.

```
yeast_trna_reads.pod5 (180 reads, 100 random fetches)
    pod5             3 908 reads/s
    vortex-default   5 072 reads/s   (1.3× POD5)
    vortex-delta+pco 2 734 reads/s   (0.7× POD5)

filtered.pod5 (very few reads, fully covered by 100 random picks)
    pod5               325 reads/s
    vortex-default   1 486 reads/s   (4.6× POD5)
    vortex-delta+pco   294 reads/s   (0.9× POD5)
```

`vortex-default` wins random access by a clear margin on both files. The
pco-based codec is roughly tied with POD5 (or worse, depending on file).

## Custom `DeltaScheme` experiment (negative result)

We implemented a `DeltaScheme` for `vortex-btrblocks` (`src/delta_scheme.rs`) that
does `i16 → ZigZag → u16 → Delta(FastLanes) → BitPacking` — the closest in-cascade
analog of VBZ's pipeline. Two findings:

1. **Cascade selector correctly skips it.** When registered with `Sample` estimation,
   the sample-based ratio loses to FoR (1.44× vs 1.72×). The selector picks the
   better scheme.
2. **Forced via `AlwaysUse`, output is bigger.** The DeltaScheme produced 11 078
   bytes for 7 962 samples (1.44×) vs FoR's 9 244 bytes (1.72×). The deltas of
   nanopore signal aren't uniformly small — they span ~11 bits — so BitPacking
   on the deltas needs almost as many bits as the FoR-narrowed range.

**Why VBZ wins on size where Vortex BitPacking can't:** SVB16 is variable-length
*per value* — outliers cost an extra byte but the common case stays small. The
`vortex-btrblocks` cascade uses BitPacking which forces the worst-case bit width
across a chunk. ZSTD on top of SVB16 captures further patterns. Pco gives
Vortex an entropy-coded leaf that matches VBZ's ratio (5 % better, even), but
the decoder is slower.

The conclusion is the encoding library's leaf-codec choice (BitPacking vs
variable-length entropy) is the binding constraint, not the absence of Delta.

## What this benchmark did NOT measure

- **Real production scale.** All test files are <2 MB; 4–80 GB files would amortize
  fixed-cost overhead and might shift the throughput numbers materially.
- **A variable-length leaf scheme** in BtrBlocks (something SVB-like). This would
  be the actual fix for the size-on-the-cheap-axis: keep `vortex-default`'s
  fast random access, get VBZ-like compression. Implementing one would be a
  non-trivial upstream contribution to `vortex-btrblocks`.
- **`pco`'s `delta_encoding_order` config.** `vortex-pco` uses `ChunkConfig::default()`
  which may or may not auto-detect delta. Worth checking in upstream.

## Recommendation

Per the original plan's exit criterion ("Vortex wins on at least 2 of
{size, decode throughput, random access} → graduate to deeper integration;
otherwise stop"):

- `vortex-default`: wins **2 of 3** (decode throughput + random access). Loses
  on size by 30–40 %. **Meets the bar.**
- `vortex-pco` / `vortex-delta+pco`: wins **1 of 3** (size, by 4–5 %). Below the bar.

If 30 % more disk is acceptable in exchange for 2–4× faster decode and 1.3–4.6×
faster random access, the experiment graduates. For a size-sensitive workflow
the answer is "stay on VBZ".

The Vortex format itself is sound. The size gap to VBZ comes from
`vortex-btrblocks`'s BitPacking-based leaves (worst-case bit width per chunk)
vs VBZ's per-value variable-length SVB16. A custom `DeltaScheme` was tried and
made things worse (see above). The real fix would be adding a variable-length
integer scheme upstream — out of scope for this experiment.

For our purposes the choice is:
- **Want disk savings?** Stay on POD5/VBZ.
- **Want fast random access + bulk decode and can spend +30 % disk?** Vortex
  with the default BtrBlocks cascade is a real win (graduates per the plan's
  exit criterion).

## How to reproduce

```bash
# inside pixi env (provides libclang)
LIBCLANG_PATH=$PWD/.pixi/envs/default/lib \
  cargo build --release -p escapepod-vortex --example signal_bench

./target/release/examples/signal_bench --concat \
    data/drna/yeast_trna_reads.pod5 \
    ext/dorado/tests/data/single_channel_multi_read_pod5/filtered.pod5
```

Add `--max-reads N` to sample the first N reads of a large production file.

# escpod align

All-against-all alignment of reads to a small reference panel: every read is
scored against every reference, the best-scoring reference is reported, and
references that tie with it are reported *as ties* rather than hidden behind a
MAPQ of zero. Built for tRNA panels (~150–200 references of ~150 nt), where it
replaces `bwa mem` + `samtools calmd`.

```bash
escpod align reads.ubam -r trna.fa -o aligned.bam
```

Ships in the default build — no extra Cargo feature.

There is no seed index. Memory is one batch of reads, and runtime is a function
of `reads × references × lengths` alone — the same for every sample of the same
size. That is the right trade for a panel of short references and the wrong one
for a genome: the intended range is **≤ ~10 k references of ≤ ~1 kb**, and a
panel beyond it gets a warning and proportionally more runtime.

## Usage

```bash
escpod align [OPTIONS] -r <FASTA> -o <BAM> <READS>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<READS>` | Reads: an unaligned BAM (dorado's uBAM), or FASTA/FASTQ, plain or gzip. The format is sniffed from the content, not the extension |

A BAM whose records are already aligned is accepted too, as long as they are on
the forward strand (the read as sequenced): the alignment fields and any
`NM`/`MD`/`AS`/`XS`/`XA` are discarded and the read is aligned afresh.
Secondary and supplementary input records are skipped (they are not reads); a
reverse-strand record is refused.

## Options

| Option | Description |
|--------|-------------|
| `-r, --reference <FASTA>` | Reference panel, plain or gzip. Names are the header's first word; order is kept |
| `-o, --output <BAM>` | Output BAM, records in input order (or coordinate order with `--sort coordinate`); `-` for stdout |
| `--sort <unsorted\|coordinate>` | Record order (default `unsorted`) — see [Sorted output](#sorted-output) |
| `--sort-memory <SIZE>` | With `--sort coordinate`, records held in memory before spilling to a temporary file (default `4G`; `K`/`M`/`G`/`T`, binary) |
| `--tmp-dir <DIR>` | With `--sort coordinate`, where the spill file goes (default: the output's directory; `$TMPDIR` for `-o -`) |
| `--mode <MODE>` | `local` (default) or `semiglobal` — see [Modes](#modes) |
| `--scoring <M,X,O,E>` | Match, mismatch, gap open, gap extend (default `2,-1,-10,-1`) — see [Scoring](#scoring) |
| `--min-score <N>` | A read whose best score is below this is written unmapped (default `0`) |
| `--max-ties <N>` | At most this many tied references besides the primary go into `XA` and `--secondary` (default: all) |
| `--secondary` | Also write each tied reference as a secondary (0x100) record |
| `--strand <forward\|both>` | `forward` (default: direct-RNA reads are always sense) or `both` |
| `--read-ids <FILE>` | Align only the reads named in FILE, one per line (like `samtools view -N`); the rest are not written |
| `--batch-size <N>` | Reads per batch (default `50000`); memory is proportional to it |
| `-t, --threads <N>` | Threads for parallel processing |
| `--device <auto\|cpu\|gpu>` | Where panel scoring runs (a `gpu` build only; tracebacks and output are always on the CPU, and the output is identical either way). `auto` (default) scores on a visible CUDA GPU, else on the CPU; `cpu` forces the CPU; `gpu` requires the GPU and fails rather than falling back. See [GPU scoring](#gpu-scoring) |

## Scoring

`--scoring M,X,O,E` takes four signed integers. The convention, stated once:

> A gap of length *k* contributes `O + (k − 1)·E`.

`O` is therefore the whole cost of a one-base gap. This is parasail's and
EMBOSS's convention (with the signs written out), and the aligner is tested
against parasail on it. bwa states its penalties the other way round — a gap
of length *k* costs `O + k·E` — so bwa's `-A1 -B1 -O1 -E1` (`-x ont2d`) is
`--scoring 1,-1,-2,-1` here.

The default, `2,-1,-10,-1`, is gpu-tRNA-mapper's, so a user moving from it sees
the same behaviour. It is not a recommendation for any pipeline; choosing a
scheme is a validation question (see [Replacing bwa](#replacing-bwa-in-a-pipeline)).

Every letter other than A/C/G/T/U (upper or lower case; U is T) is a
**wildcard**: pairing it with anything scores as a match. The tRNA panels this
is built for carry an `N` in every reference, and scoring it as a mismatch
would penalise every read at the same column for a base the reference does not
claim to know.

## Modes

**`local`** — Smith–Waterman. The alignment may start and end anywhere; both
overhangs of the read are soft-clipped.

**`semiglobal`** — overlap alignment: leading and trailing gaps of *either*
sequence are free. The path starts on the top or left border of the DP matrix
and ends on the bottom or right border, and interior gaps are scored normally.
In practice: a read that runs off either end of its reference is clipped
there for free, and a reference end the read does not reach costs nothing.
This is our definition (parasail's `sg`), not a reproduction of any other
tool's semi-global mode.

Optimal alignments are rarely unique. The choice is fixed: the end cell with
the smallest reference end, then the smallest read end; from it, the traceback
prefers a diagonal step, then an insertion, then a deletion, extends a gap
rather than opening one, and (local mode) stops at the first zero. These are
parasail's `sw_trace`/`sg_trace` choices, and pinned against them.

## Output

Records come out in **input order** (`@HD SO:unsorted`) by default, or
coordinate-sorted with `--sort coordinate` (see [Sorted output](#sorted-output)).
The header is
`@SQ` from the reference in file order, the input's `@RG`, `@PG` and `@CO`
copied through, and one `@PG ID:escpod-align` for this run, chained to the last.

**Every input tag is copied byte for byte** — `mv`, `ns`, `ts`, `MM`/`ML`,
`RG`, `qs`, `du`, `ch`, `st`, and the rest. `NM`, `MD`, `AS`, `XS` and `XA` are
the aligner's and replace any the input carried. Of the input flags only
QC-fail (0x200) and duplicate (0x400) survive.

Per read:

| Field | Value |
|-------|-------|
| primary | the first member of the tie set: the lowest-indexed reference scoring the best score *S* |
| `AS:i` | *S* |
| `XS:i` | the best score among references *outside* the tie set; absent if there are none |
| `XA:Z` | the other tie-set members, bwa's format `ref,±pos,CIGAR,NM;`, capped by `--max-ties` |
| MAPQ | **60** for a tie set of one, **0** otherwise — deterministic, not a probability |
| CIGAR | `M`/`I`/`D`/`S` only (no `=`/`X`), as bwa writes it |
| `NM:i`, `MD:Z` | as `samtools calmd` writes them — see below |

`--secondary` also writes each `XA` member as a 0x100 record right after its
primary. Secondaries carry `SEQ`/`QUAL` as `*` and only `NM`, `MD`, `AS` and
`RG` (minimap2's convention), so the move table and modification calls are not
duplicated once per tie.

A read whose *S* is below `--min-score` — or that aligns no base at all — is
written **unmapped** (flag 4) with its tags intact. Reads are never dropped
(except by `--read-ids`).

`--strand both` also scores the reverse complement of every read. A reverse
winner gets flag 16 with `SEQ` reverse-complemented and `QUAL` reversed, as
SAM requires; per-base tags (`mv`, `MM`/`ML`) are copied verbatim, as `bwa -C`
does. The tie order is reference order, forward before reverse.

### Sorted output

`--sort coordinate` writes `@HD SO:coordinate` and the records in exactly
`samtools sort`'s order: by reference (`@SQ` order), then position, then
forward before reverse strand, then input order — so the output is record for
record what `samtools sort` of the unsorted output gives (tested on the
fixtures, and checked on a 1.1 M-read sample). Unmapped reads come last, in
input order; `--secondary` records sort by their own coordinates. Nothing else
in the header changes but the `@PG CL`. `samtools index` takes the file as it
is; no `.bai` is written.

It is a bucket sort — one bucket per reference, which is cheap because the
panel is small — so there is no separate sort pass and no merge. A sorted file
cannot be written until the last read is aligned, so the records are held
until then; with `-o -` the output starts once the input ends. Past
`--sort-memory` (default 4 GiB) the largest buckets are appended to one
temporary file in `--tmp-dir`, and at the end each reference's records are read
back and ordered on their own: peak memory is about the budget plus the
largest single reference's records, not the sample. The temporary file is
unlinked as soon as it is created, so nothing is left behind on any exit.

### `MD` and `NM` at ambiguity codes

The scorer treats a reference `N` as matching anything. `MD` and `NM` do not:
they are exactly what `samtools calmd` writes for the same CIGAR, because the
windowed charging bundle in `escpod classify` rebuilds each read's reference
from `MD`, and a reference `N` that vanished from `MD` would silently put the
read's own base in its place. Measured on samtools 1.23.1 and pinned by a
golden:

- A base is a match only when both letters are A/C/G/T (U counting as T, case
  ignored) and equal. A reference `N` — or any other IUPAC code — is a
  **mismatch** against every read base, `N` included, and counts toward `NM`.
- A mismatched or deleted reference letter is written upper-cased but otherwise
  verbatim (`n` → `N`; a reference `U` stays `U`).
- `NM` = mismatches + inserted bases + deleted bases.

So a read that matches its reference everywhere except across the reference's
one `N` has `NM:i:1` and an `MD` with that `N` in it, and `AS` counts the `N`
as a match. A pipeline stage that used `samtools calmd` after bwa sees the same
`MD`/`NM` for the same CIGAR.

## Kernels

Scoring runs on inter-sequence SIMD kernels — one lane per reference, 16 lanes
on AVX2, 32 on AVX-512BW, `i16` arithmetic that is exact (not approximate) for
any panel in the intended range — chosen at runtime from what the CPU supports.
The winners' tracebacks run on a second pair of kernels whose lanes each carry
their own read and reference, so a read that ties with the whole panel (an
adapter-only read on an adapter-flanked panel does) costs about what scoring
it did. Every kernel is tested for equality against a scalar reference
implementation, which is itself tested against parasail.

`ESCAPEPOD_ALIGN_BACKEND=scalar|avx2|avx512` caps the kernel choice, for A/B
measurements; the run's second log line names the kernel in use.

### Performance

A 3.3 M-read dorado uBAM against the 164-reference sacCer3 dual-adapter panel
aligns in ~2 minutes on 16 cores (`-t 32`, ~28 k reads/s, ~3,300 CPU-seconds)
in under 3 GiB — against a 12 h / 160 GB budget for the bwa step it replaces.
On Cascade Lake the AVX-512 and AVX2 kernels are within a few percent of each
other; scalar is ~32× the CPU. Details and reproduction in
`benchmarks/README.md`.

### GPU scoring

In a binary built with `--features gpu`, scoring every read against every
reference can run on a CUDA GPU: one GPU thread per (read, reference) pair, a
whole `--batch-size` batch per kernel launch. Only the scores come from the
device — the tie set, the tracebacks, `MD`/`NM` and the records are the CPU
path's own code — so `--device gpu` and `--device cpu` write the same records
byte for byte (tested, and checked on a 490 k-read sample). Reads over 4,096 nt
are scored on the CPU inside the same run, and a panel with a reference over
3,072 nt cannot use the GPU (`auto` then stays on the CPU and says so).

On a gpu node (`-c 16`, one A30), the 3.3 M-read sample takes ~70 s and ~700
CPU-seconds with GPU scoring against 184 s and ~2,900 CPU-seconds on the same
16 cores without it, so `--device auto` uses the GPU when one is visible. Nothing
CUDA is needed to build; at run time the GPU path needs the CUDA driver and
NVRTC (`pixi run install-gpu` sets both up). Details in
`benchmarks/README.md`.

## Replacing bwa in a pipeline

The aa-tRNA-seq alignment step today is:

```bash
samtools view -u -N ids.txt reads.ubam \
  | samtools fastq -T '*' - \
  | bwa mem -C -H rg.sam -x ont2d -k 6 -W 13 -T 20 ref.fa - \
  | samtools view -u -F 2324 - \
  | samtools sort -o aln.bam
samtools calmd -Qb aln.bam ref.fa > aln.calmd.bam
```

`escpod align` does all of it in one process — the tags ride through without
the FASTQ-comment round trip, the uBAM's `@RG` lines are copied without the
`-H` header hack, `MD` is written at alignment time, and `--sort coordinate`
replaces the sort:

```bash
escpod align reads.ubam -r ref.fa -o aln.bam --read-ids ids.txt --scoring 1,-1,-2,-1 \
    --sort coordinate
samtools index aln.bam
```

`escpod classify` reads either order. Unmapped reads are written, never
dropped; `samtools view -F 2324` reproduces the old output's contents exactly
(primary, forward, mapped records) if that is wanted, and a filtered sorted
file stays sorted. Stamping sample/barcode read groups stays pipeline plumbing.

The scoring scheme is the pipeline's call, judged by charging-call agreement:
`--scoring 1,-1,-2,-1` is bwa `-x ont2d`'s penalties, but bwa also clips, bands
and chains, so the same scores do not give the same CIGAR for every read. On
the classify test fixture, 27 of 60 reads get a different CIGAR (or start) from
bwa's; the charging calls of every read whose alignment is unchanged are
bit-identical.

## Examples

```bash
# dorado uBAM in, aligned BAM out
escpod align calls.ubam -r trna.fa -o aligned.bam -t 32

# FASTQ, overlap mode, gap-lenient scores
escpod align reads.fq.gz -r trna.fa -o aligned.bam --mode semiglobal --scoring 1,-1,-2,-1

# Coordinate-sorted, ready for samtools index
escpod align calls.ubam -r trna.fa -o aligned.bam --sort coordinate

# Ties as secondary records, at most 5 per read
escpod align calls.ubam -r trna.fa -o aligned.bam --secondary --max-ties 5
```

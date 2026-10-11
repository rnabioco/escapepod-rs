# dorado-compat: can dorado read what escpod writes?

A manual, GPU-needing compatibility check: run `escpod merge | filter | subset |
demux split` over real POD5 files that end in a **short last signal batch**, basecall
every output with dorado, and compare the parent read ids in dorado's BAM against the
reads in each POD5.

## Why this exists

dorado and the official `pod5` library assume a **constant signal-batch stride**: row
`r` of the signal table lives in batch `r / batch_size`. escpod's own reader does not
assume that, it walks real cumulative row counts. So a POD5 with a stride break
(a short batch followed by a full one, which is what a naive merge of N files makes) is
read *correctly* by escpod and *incorrectly* by everything else. Nothing escpod can
test on its own notices; only an independent reader does. History: #195 (writer
emitted non-uniform strides) and 0.31.1 (`merge` rebuilt at one uniform stride).

Why not `cargo nextest`: it needs a GPU, a ~5 GB dorado install, a ~100 MB model and real
signal data that is not redistributable. It sits next to the benchmarks because it is
run by hand, on a cluster, before a release or after touching the writer/merge path.

## What it established (escpod 0.31.2+b499be7, dorado 2.1.2, `rna004_fast@v6.0.0`)

| output | signal batch rows | parents missing vs POD5 | dorado rc | python pod5 |
|---|---|---|---|---|
| 4 short-last-batch inputs | `1000,906` etc. | 0 | 0 | OK |
| `merge` | `1000x7,542` | 0 | 0 | OK |
| `filter` (all inputs) | `1000x2,511` | 0 | 0 | OK |
| `subset` (2 groups) | `1000x3,783` / `1000x3,759` | 0 | 0 | OK |
| `demux split` (3 barcodes) | `1000x2,...` | 0 | 0 | OK |

dorado 2.1.2 reads every escpod output. (Records exceed parents because dorado 2.x
splits reads; the parent id is the `pi` tag, which `bam_parents.py` uses.)

### Negative control, and the finding it forces

`negative_control.sh` re-batches one input's signal table to `500,1000,<rest>` rows
without escpod (pyarrow plus a footer patch), then runs every probe. Result on dorado 2.1.2:

- `escpod inspect summary`: `NOT PORTABLE - signal batch 1 has 1000 rows, expected 500`
- python `pod5`: reads 1375 of 1865, then `IndexError`
- **dorado: exit code 0**, 704 `POD5 Failed to get read signal - 'Invalid: Row outside batch bounds'`
  errors on stderr, 1161 reads basecalled, **704 of 1865 reads silently dropped.**

**dorado exits 0 while dropping reads, so its exit code is not a check.** The only
reliable signal is comparing BAM parent ids to the POD5, which is what `compare.py`
does. A check that passes on rc alone would have passed this broken file.

## Usage

Prerequisites: `pixi run install-pod5` (the check's python needs `pod5`; `pysam` is in
the default env), network for the first dorado/model download, Slurm.

    INPUT_DIR=/path/to/real/pod5s benchmarks/dorado-compat/run_check.sh [--sha REF | --src DIR] [--rebuild] [--keep-work]
    benchmarks/dorado-compat/run_check.sh --help

Run it from a login node and in the foreground; it `srun`s the build, CPU and GPU
stages itself. It builds escpod (`cargo build --release -p escapepod-cli`) from a clean
`origin/main`, or a detached worktree of it, or `--sha REF`. The report lands in
`$DC/results/last_report.txt`; `$WORK` is deleted afterwards unless `--keep-work`.
Use `NO_SRUN=1` to run every stage in the current shell (e.g. inside your own allocation).

Then, with `--keep-work` from a previous run:

    benchmarks/dorado-compat/negative_control.sh        # prep, dorado, report

Stages can also be run alone (`build_inputs.sh`, `make_outputs.sh`, `run_dorado.sh`,
`report.sh`), each under `srun` or `NO_SRUN=1`.

### Inputs

`INPUT_DIR` (required; env or `--input-dir`) is a directory searched recursively for
`*.pod5`. They are symlinked, never copied, and no `.p5s` is written beside them (the
script compares the `.p5s` set before and after). The first four files (sorted) are
used, each cut to its first N reads with `escpod filter -i`, N from `COUNTS`
(default `1865 2431 1207 1733`). Each file needs at least that many reads, and a
count that is a multiple of 1000 would not leave a short last batch. `RAW_INPUTS` picks
files explicitly (globs relative to `INPUT_DIR`). There is no data in this repository.

### Configuration (all environment variables; defaults in `config.sh`)

| variable | default |
|---|---|
| `SCRATCH_DIR` / `DC` | `$HOME/scratch` / `$SCRATCH_DIR/dorado-check` (tools, models, results, target, work) |
| `DORADO_VERSION` | `DORADO_VERSION` in `pixi.toml`, else `2.1.2` |
| `DORADO_BIN` | else `$PIXI_PROJECT_ROOT/.pixi/tools/dorado-$VER-linux-x64/bin/dorado`, else `$DC/tools/...` (downloaded there if absent; the version is checked, no fallback) |
| `MODEL_NAME` / `MODELS_DIR` | `rna004_fast@v6.0.0` / `$DC/models` (downloaded if missing) |
| `CPU_PARTITION`, `BUILD_PARTITION`, `GPU_PARTITION`, `GPU_ACCOUNT` | `rna`, `=CPU_PARTITION`, `gpu`, `gpu_rbi` (site-specific) |
| `CPU_SRUN`, `GPU_SRUN`, `BUILD_SRUN` | full `srun` prefixes, if the partition names are not enough |
| `ESCPOD`, `SRC`, `REPO`, `WORK` | built binary, checkout to build, checkout for pixi python, bulky work dir |

## Reading the report

Per output: signal batch row counts (run-length encoded), python `pod5` full-signal read,
`escpod inspect` `NOT PORTABLE` grep, dorado rc and error-line count, and BAM parents
missing against the POD5 and against the unmodified controls' union. Expect all zeros /
OK. Any `NONUNIFORM_BEFORE_LAST=True` on an escpod *output* is a bug in escpod.

## Files

`run_check.sh` entry point; `config.sh` parameters; `build_inputs.sh`, `make_outputs.sh`,
`run_dorado.sh`, `report.sh` stages; `negative_control.sh` + `mk_nonuniform_raw.py` +
`total_rows.py` the control; `batches.py`, `verify_pod5.py`, `compare.py`, `bam_parents.py`,
`pod5_ids.py`, `mkids.py`, `mkbarcodes.py` helpers; `pod5_layout.py` dumps a container's
footer entries (debugging).

## Superseded

`tests/test_dorado_compat.sh` (removed) called `dorado summary` on a POD5, which takes a
BAM, and never exercised a short last batch. This replaces it.

#!/bin/bash
# End-to-end `escpod demux` benchmark on the CRF (barcode basecalling) path,
# and an A/B across escpod builds.
#
# The counterpart to `crates/escapepod-demux/benches/crf_encoder.rs` and
# `crf_decode.rs`: those measure one read's encoder and lattice decode in
# isolation, this one measures the whole command on real input — POD5 decode,
# boundary detection, windowing, encoder, decode, barcode match, sidecar
# write — in the exact mode the aa-tRNA-seq pipeline runs it (`--annotate
# --classifications --ref-scores`). It exists because that path had no
# harness: `benchmark_demux.sh` measures the SVM/DTW path against WarpDemuX,
# wall only, one binary (rnabioco/escapepod-rs#331).
#
# Usage:
#   benchmarks/benchmark_demux_crf.sh --model [NAME=]DIR [--model ...] \
#       [--pod5 FILE|DIR] [--truth CSV|none] [--bin PATH]... \
#       [--device cpu|gpu] [--threads N] [--reps N] [--out DIR] \
#       [--no-ref-scores] [-- ESCPOD FLAGS...]
#
# --pod5 and --truth default to the labelled 20k dual-axis set under
# data/bench/dual_axis_20k/ (provenance in benchmarks/README.md); --model has
# no default because the bundles live outside this repository. Two models
# named `ldx=`/`fdx=` is the production dual-axis pass, one unnamed model a
# single-axis pass. Anything after `--` goes to `escpod demux` verbatim
# (e.g. `-- --min-crf-margin 2`). `--device gpu` needs the caller to supply
# the onnxruntime environment (`pixi run -e gpu benchmarks/...`).
#
# Each --bin adds an ARM. Give it several to A/B released binaries against a
# local build:
#
#   benchmarks/benchmark_demux_crf.sh --model ldx=ldx32/ --model fdx=fdx4/ \
#       --bin /path/to/escpod-0.20.0 --bin ./target/release/escpod
#
# With no --bin it benchmarks ./target/release/escpod alone.
#
# Three rules this script enforces, the first two inherited from
# benchmark_charging.sh where they were learned the hard way:
#
#   1. ARMS ARE INTERLEAVED (A B C A B C), never grouped (A A B B C C). The
#      first touch of a POD5 pages it in off shared storage and lands on
#      whichever arm runs first; grouped arms hand that whole effect to one
#      version.
#
#   2. BOTH wall and CPU are reported. Wall alone cannot tell a slower binary
#      from a busier node or a colder cache; CPU alone hides the I/O that
#      dominates a cold run. When the two disagree, the run was I/O-bound and
#      the wall number belongs to the filesystem, not to escpod.
#
#   3. THE INPUT IS NEVER TOUCHED. `--annotate` writes a `.p5s` sidecar beside
#      its input, and `sidecar_path` appends to the path *as given*, so every
#      arm runs against a symlink to the POD5 in node-local scratch and its
#      sidecar lands beside the link. A `.p5s` beside the canonical file would
#      be picked up by the next run of anything, so its absence is asserted
#      after every arm. The POD5 is never copied.
#
# Rep 1 is reported but should be READ AS A WARM-UP, not as a measurement: the
# cold-cache effect lands entirely on it. Compare arms within rep 2 and later.
#
# After the reps, the classification CSVs of the LAST rep are scored by
# benchmarks/evaluate_demux_crf.py: accuracy against the truth per axis, and
# every arm against arm 0 read for read, with disagreements attributed by
# `crf_margin` — which is why --ref-scores is on by default. That needs
# pandas: point EVAL_PYTHON at an interpreter that has it (the `python-test`
# pixi env does), or run the command the script prints.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
POD5="$REPO/data/bench/dual_axis_20k/test20k.pod5"
TRUTH="$REPO/data/bench/dual_axis_20k/truth.csv"
MODELS=()
BINS=()
DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-32}"
REPS="${REPS:-2}"
OUT="${OUT:-/tmp/escapepod_demux_crf_benchmark}"
REF_SCORES=1
EXTRA=()
EVAL_PYTHON="${EVAL_PYTHON:-python3}"

usage() {
    awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
    exit "${1:-1}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --pod5)          POD5="$2"; shift 2 ;;
        --truth)         TRUTH="$2"; shift 2 ;;
        --model)         MODELS+=("$2"); shift 2 ;;
        --bin)           BINS+=("$2"); shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --threads)       THREADS="$2"; shift 2 ;;
        --reps)          REPS="$2"; shift 2 ;;
        --out)           OUT="$2"; shift 2 ;;
        --no-ref-scores) REF_SCORES=0; shift ;;
        --)              shift; EXTRA=("$@"); break ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: at least one --model is required" >&2
    usage
fi
if [ ! -e "$POD5" ]; then
    echo "Error: no such POD5 file or directory: $POD5" >&2
    exit 1
fi
if [ "$TRUTH" != none ] && [ ! -f "$TRUTH" ]; then
    echo "Error: no such truth table: $TRUTH (pass --truth none to skip scoring)" >&2
    exit 1
fi
MODEL_ARGS=()
for m in "${MODELS[@]}"; do
    dir="${m#*=}"
    if [ ! -e "$dir" ]; then
        echo "Error: no such model: $dir" >&2
        exit 1
    fi
    MODEL_ARGS+=(--model "$m")
done

if [ ${#BINS[@]} -eq 0 ]; then
    BINS=("./target/release/escpod")
fi
for bin in "${BINS[@]}"; do
    if [ ! -x "$bin" ]; then
        echo "Error: not an executable: $bin" >&2
        exit 1
    fi
done

# The sidecar a *direct* run would write beside the input. Its presence
# before the run is somebody else's doing and only warned about; its
# appearance during the run is this script's bug and fatal.
CANON="$(realpath "$POD5")"
CANON_SIDECAR="${CANON%/}.p5s"
if [ -e "$CANON_SIDECAR" ]; then
    echo "WARNING: $CANON_SIDECAR already exists beside the input (a direct run" >&2
    echo "         wrote it); this script will not touch it, but other tools read it." >&2
    HAD_SIDECAR=1
else
    HAD_SIDECAR=0
fi

# Node-local scratch: the symlinked input and its sidecars go here, never
# beside the canonical file. The classification CSVs are small and are kept
# under $OUT for the scoring step.
SCRATCH="${TMPDIR:-/tmp}/escpod_demux_crf_bench.$$"
mkdir -p "$OUT" "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "node:      $(hostname)"
echo "device:    $DEVICE"
echo "threads:   $THREADS"
echo "reps:      $REPS"
echo "pod5:      $POD5"
echo "truth:     $TRUTH"
for m in "${MODELS[@]}"; do
    echo "model:     $m"
done
echo "ref-scores: $REF_SCORES"
if [ ${#EXTRA[@]} -gt 0 ]; then
    echo "extra:     ${EXTRA[*]}"
fi
echo "arms:      ${#BINS[@]}"
# Label each arm by index AND version, never by basename alone: every released
# tarball unpacks to a binary called `escpod`.
VERSIONS=()
for i in "${!BINS[@]}"; do
    v="$("${BINS[$i]}" --version 2>&1 | head -1)"
    VERSIONS+=("$v")
    echo "  arm$i  $v  <- ${BINS[$i]}"
done
echo

for rep in $(seq 1 "$REPS"); do
    for i in "${!BINS[@]}"; do
        bin="${BINS[$i]}"
        tag="arm${i}_rep${rep}"
        log="$OUT/$tag.log"
        tm="$OUT/$tag.time"
        csv="$OUT/$tag.csv"

        # A fresh symlink per run, so each run also starts without a sidecar.
        run_dir="$SCRATCH/$tag"
        mkdir -p "$run_dir"
        input="$run_dir/$(basename "$CANON")"
        ln -s "$CANON" "$input"

        cmd=("$bin" demux "$input" "${MODEL_ARGS[@]}"
             --annotate --classifications "$csv"
             --device "$DEVICE" --threads "$THREADS")
        if [ "$REF_SCORES" = 1 ]; then
            cmd+=(--ref-scores)
        fi
        if [ ${#EXTRA[@]} -gt 0 ]; then
            cmd+=("${EXTRA[@]}")
        fi

        /usr/bin/time -v -o "$tm" "${cmd[@]}" >"$log" 2>&1
        rc=$?

        if [ "$HAD_SIDECAR" = 0 ] && [ -e "$CANON_SIDECAR" ]; then
            echo "ERROR: $CANON_SIDECAR appeared beside the canonical input during $tag;" >&2
            echo "       the symlink trick did not hold. Remove it and fix this script." >&2
            exit 1
        fi

        wall=$(awk -F': ' '/Elapsed \(wall/ {print $NF}' "$tm")
        user=$(awk -F': ' '/User time/ {print $NF}' "$tm")
        sys=$(awk -F': ' '/System time/ {print $NF}' "$tm")
        rss=$(awk -F': ' '/Maximum resident set size/ {printf "%.1f", $NF/1048576}' "$tm")
        # The assignment summary is the parity check at a glance; the scoring
        # step below is the real one. (The log line goes on to list the
        # sidecar's columns; that part is the same for every arm and dropped.)
        summary=$(grep -oE '[0-9]+ of [0-9]+ reads assigned.*' "$log" | head -1 | sed 's/, plus .*//')

        printf 'rep%-2s arm%-2s %-22s %-26s rc=%s wall=%s cpu=%.1fs rss=%sG  %s\n' \
            "$rep" "$i" "${VERSIONS[$i]}" "$(basename "$bin")" "$rc" "$wall" \
            "$(echo "$user + $sys" | bc)" "$rss" "${summary:-NO SUMMARY PARSED}"

        rm -rf "$run_dir"
    done
done

echo
echo "logs, /usr/bin/time reports and classification CSVs: $OUT"
echo "NOTE: rep 1 is a warm-up (cold page cache). Compare arms within rep 2+."

# Score the last rep: every arm against the truth, and against arm 0.
EVAL_ARGS=()
for i in "${!BINS[@]}"; do
    EVAL_ARGS+=("arm$i=$OUT/arm${i}_rep${REPS}.csv")
done
EVAL_CMD=("$EVAL_PYTHON" "$REPO/benchmarks/evaluate_demux_crf.py")
if [ "$TRUTH" != none ]; then
    EVAL_CMD+=(--truth "$TRUTH")
fi
EVAL_CMD+=("${EVAL_ARGS[@]}")
echo
if "$EVAL_PYTHON" -c 'import pandas' >/dev/null 2>&1; then
    "${EVAL_CMD[@]}"
else
    echo "pandas is not importable by $EVAL_PYTHON; score the last rep with:"
    printf '  %q' "${EVAL_CMD[@]}"
    echo
fi

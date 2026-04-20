#!/bin/bash
# Benchmark `escpod demux` against WarpDemuX.
#
# Usage:
#   ./benchmarks/benchmark_demux.sh [--gpu] [--no-srun] [pod5_file]
#
# SLURM: by default the script auto-dispatches itself onto a compute node
#   via `srun` if not already inside a SLURM job (login node has 2 cores
#   and would make the numbers noise). Partition/account:
#     CPU  -> srun -p rna -A rbi     -c 16
#     GPU  -> srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1
#   Pass `--no-srun` to skip the re-exec (useful on workstations or when
#   already inside an interactive `srun`/`salloc` session).
#
# Prerequisites:
#   - Cloned ext/WarpDemuX and ext/ADAPTed (see docs/cli/demux.md)
#   - `pixi install -e warpdemux-bench && pixi run -e warpdemux-bench install-warpdemux`
#   - `cargo build --release -p escapepod-cli --features "demux train"` (CPU)
#   - For --gpu: build with `--features "demux train gpu"` on a node with
#     CUDA driver + libnvrtc (use `pixi run -e gpu cargo build ...`).
#
# Runs three comparisons on the same POD5:
#
#   Bench 1  Adapter detection: `escpod demux detect` vs `adapted detect --llr`
#            (hyperfine, 3 runs each, one warm-up).
#   Bench 2  End-to-end pipeline wall-clock:
#              escpod:   detect -> fingerprint --warpdemux-compat -> classify --svm-model
#              WarpDemuX: `warpdemux demux -m WDX4_rna004_v1_0`
#            Reports total wall-clock; with --gpu, adds a third escpod variant
#            that passes `--gpu` to classify.
#   Bench 3  Classification agreement:
#            Reuses Bench 2's outputs; runs `scripts/compare_demux_results.py`
#            on escpod vs WarpDemuX predictions to report per-barcode F1,
#            confusion matrix, and agreement by confidence bin.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

USE_GPU=0
NO_SRUN=0
POD5_FILE=""
for arg in "$@"; do
    case "$arg" in
        --gpu)      USE_GPU=1 ;;
        --no-srun)  NO_SRUN=1 ;;
        *)          POD5_FILE="$arg" ;;
    esac
done

: "${POD5_FILE:=$PROJECT_ROOT/ext/WarpDemuX/test_data/demux/4000_rna004.pod5}"

# Auto-dispatch onto a compute node unless already in SLURM or opted out.
# srun replaces our PID, so this block either exec's away or falls through.
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$NO_SRUN" -eq 0 ]; then
    if ! command -v srun >/dev/null 2>&1; then
        echo "note: srun not found; continuing on the current host." >&2
    else
        if [ "$USE_GPU" -eq 1 ]; then
            SRUN_ARGS=(-p gpu -A gpu_rbi -c 16 --gres=gpu:1)
        else
            SRUN_ARGS=(-p rna -A rbi -c 16)
        fi
        echo ">>> Re-dispatching under srun ${SRUN_ARGS[*]}"
        # Preserve explicit arg set, add --no-srun so the child doesn't recurse.
        child_args=("${@}")
        child_args+=(--no-srun)
        exec srun "${SRUN_ARGS[@]}" "$0" "${child_args[@]}"
    fi
fi

# Thread count for in-process parallelism. Honour SLURM's allocation when
# present, else assume interactive.
THREADS="${SLURM_CPUS_PER_TASK:-${THREADS:-4}}"

# Binaries / wrappers
ESCAPEPOD_BIN="$PROJECT_ROOT/target/release/escpod"
WDX_MODEL_JOBLIB="$PROJECT_ROOT/ext/WarpDemuX/warpdemux/models/model_files/WDX4_rna004_v1_0.joblib"
WDX_MODEL_JSON="$PROJECT_ROOT/benchmarks/.wdx4_rna004.json"
PIXI_WDX="pixi run -e warpdemux-bench --manifest-path $PROJECT_ROOT/pixi.toml"

OUTPUT_DIR="/tmp/demux_benchmark"
WARMUP=1
RUNS=3

echo "========================================"
echo "escpod demux vs WarpDemuX Benchmark"
echo "========================================"
echo "Input:    $POD5_FILE"
echo "Size:     $(du -h "$POD5_FILE" | cut -f1)"
echo "GPU path: $([[ $USE_GPU -eq 1 ]] && echo enabled || echo disabled)"
echo "Threads:  $THREADS (SLURM_JOB_ID=${SLURM_JOB_ID:-none})"
echo ""

# Sanity checks
if [ ! -f "$ESCAPEPOD_BIN" ]; then
    echo "error: $ESCAPEPOD_BIN not found." >&2
    echo "       cargo build --release -p escapepod-cli --features 'demux train'$([[ $USE_GPU -eq 1 ]] && echo ' gpu')" >&2
    exit 1
fi
if [ ! -f "$POD5_FILE" ]; then
    echo "error: POD5 not found: $POD5_FILE" >&2
    exit 1
fi
if [ ! -f "$WDX_MODEL_JOBLIB" ]; then
    echo "error: WarpDemuX bundled model not found: $WDX_MODEL_JOBLIB" >&2
    echo "       did you git clone https://github.com/KleistLab/WarpDemuX ext/WarpDemuX ?" >&2
    exit 1
fi
if ! command -v hyperfine >/dev/null 2>&1 && ! $PIXI_WDX which hyperfine >/dev/null 2>&1; then
    echo "error: hyperfine not found on PATH nor in pixi warpdemux-bench env" >&2
    exit 1
fi

# Resolve hyperfine via pixi env (it's installed there per pixi.toml) so we
# don't require a system install.
HYPERFINE="$($PIXI_WDX which hyperfine 2>/dev/null || command -v hyperfine)"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{escpod,wdx}

# One-time model export (cheap, cache between runs).
if [ ! -f "$WDX_MODEL_JSON" ]; then
    echo ">>> Exporting WarpDemuX WDX4 model -> escpod SVM JSON"
    $PIXI_WDX python "$PROJECT_ROOT/scripts/convert_warpdemux_model.py" \
        "$WDX_MODEL_JOBLIB" "$WDX_MODEL_JSON" > /dev/null
fi

# ----------------------------------------------------------------------
# Bench 1: Adapter detection (hyperfine)
# ----------------------------------------------------------------------
echo ""
echo "=== Bench 1: Adapter Detection ==="
DETECT_JSON="$OUTPUT_DIR/detect.hyperfine.json"

"$HYPERFINE" \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-json "$DETECT_JSON" \
    --command-name "escpod detect" \
    "$ESCAPEPOD_BIN demux detect $POD5_FILE -o $OUTPUT_DIR/escpod/boundaries.csv -j $THREADS" \
    --command-name "adapted detect (LLR)" \
    "$PIXI_WDX adapted detect -i $POD5_FILE -o $OUTPUT_DIR/wdx/adapted --chemistry RNA004 -j $THREADS" \
    2>&1

# ----------------------------------------------------------------------
# Bench 2: End-to-end pipeline (wall-clock, not hyperfine — long-running)
# ----------------------------------------------------------------------
echo ""
echo "=== Bench 2: End-to-End Demux ==="

# Fresh output dirs per variant.
mkdir -p "$OUTPUT_DIR/escpod_cpu" "$OUTPUT_DIR/wdx_out"
[ $USE_GPU -eq 1 ] && mkdir -p "$OUTPUT_DIR/escpod_gpu"

run_escpod_pipeline() {
    local tag="$1"
    local outdir="$2"
    local extra="$3"

    echo ">>> escpod ($tag): detect | fingerprint | classify --svm-model"
    local t0
    t0=$(date +%s.%N)
    "$ESCAPEPOD_BIN" demux detect "$POD5_FILE" \
        -o "$outdir/boundaries.csv" -j $THREADS -q
    "$ESCAPEPOD_BIN" demux fingerprint "$POD5_FILE" \
        --boundaries "$outdir/boundaries.csv" \
        --warpdemux-compat \
        -o "$outdir/fingerprints.csv" -j $THREADS -q
    "$ESCAPEPOD_BIN" demux classify "$outdir/fingerprints.csv" \
        --svm-model "$WDX_MODEL_JSON" \
        --probabilities \
        $extra \
        -o "$outdir/classifications.csv"
    local t1
    t1=$(date +%s.%N)
    echo "$(echo "$t1 - $t0" | bc)"
}

ESCPOD_CPU_TIME=$(run_escpod_pipeline "CPU" "$OUTPUT_DIR/escpod_cpu" "" | tail -1)
echo "    escpod CPU total:  ${ESCPOD_CPU_TIME}s"

if [ $USE_GPU -eq 1 ]; then
    # cudarc dlopen's libnvrtc; the conda-forge cuda-nvrtc package lives
    # under the pixi `gpu` env's lib dir. Export it for the GPU run only.
    GPU_ENV_LIB="$PROJECT_ROOT/.pixi/envs/gpu/lib"
    if [ ! -f "$GPU_ENV_LIB/libnvrtc.so.12" ]; then
        echo "error: libnvrtc not found under $GPU_ENV_LIB. Run \`pixi install -e gpu\` first." >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="$GPU_ENV_LIB:${LD_LIBRARY_PATH:-}"
    ESCPOD_GPU_TIME=$(run_escpod_pipeline "GPU" "$OUTPUT_DIR/escpod_gpu" "--gpu" | tail -1)
    echo "    escpod GPU total:  ${ESCPOD_GPU_TIME}s"
fi

echo ">>> WarpDemuX: warpdemux demux -m WDX4_rna004_v1_0"
t0=$(date +%s.%N)
$PIXI_WDX warpdemux demux \
    -i "$POD5_FILE" \
    -o "$OUTPUT_DIR/wdx_out" \
    -m WDX4_rna004_v1_0 \
    --ncores "$THREADS" \
    --save_boundaries true \
    2>&1 | tail -5
t1=$(date +%s.%N)
WDX_TIME=$(echo "$t1 - $t0" | bc)
echo "    WarpDemuX total:   ${WDX_TIME}s"

# Locate WarpDemuX's timestamped output dir.
WDX_RUN_DIR=$(find "$OUTPUT_DIR/wdx_out" -maxdepth 1 -type d -name "warpdemux_*" | head -1)

# ----------------------------------------------------------------------
# Bench 3: Classification agreement
# ----------------------------------------------------------------------
echo ""
echo "=== Bench 3: Classification Agreement ==="
if [ -n "$WDX_RUN_DIR" ] && [ -d "$WDX_RUN_DIR/predictions" ]; then
    $PIXI_WDX python "$PROJECT_ROOT/scripts/compare_demux_results.py" \
        --escapepod-b "$OUTPUT_DIR/escpod_cpu/classifications.csv" \
        --warpdemux "$WDX_RUN_DIR/predictions" \
        --boundaries-escapepod "$OUTPUT_DIR/escpod_cpu/boundaries.csv" \
        --boundaries-warpdemux "$WDX_RUN_DIR/boundaries"
else
    echo "    skip: WarpDemuX predictions dir not found ($WDX_RUN_DIR)"
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Input:      $POD5_FILE"
echo ""
echo "Detection (hyperfine means):"
if [ -f "$DETECT_JSON" ]; then
    python3 -c "
import json
d = json.load(open('$DETECT_JSON'))
for r in d['results']:
    print(f\"    {r['command']:<30s} {r['mean']:.3f}s ± {r['stddev']:.3f}s\")
"
fi
echo ""
echo "End-to-end pipeline:"
echo "    escpod CPU:        ${ESCPOD_CPU_TIME}s"
[ $USE_GPU -eq 1 ] && echo "    escpod GPU:        ${ESCPOD_GPU_TIME}s"
echo "    WarpDemuX:         ${WDX_TIME}s"
if [ -n "$ESCPOD_CPU_TIME" ] && [ -n "$WDX_TIME" ]; then
    speedup=$(python3 -c "print(f'{float($WDX_TIME) / float($ESCPOD_CPU_TIME):.2f}')")
    echo "    Speedup (CPU):     ${speedup}x"
    if [ $USE_GPU -eq 1 ] && [ -n "$ESCPOD_GPU_TIME" ]; then
        gpu_speedup=$(python3 -c "print(f'{float($WDX_TIME) / float($ESCPOD_GPU_TIME):.2f}')")
        echo "    Speedup (GPU):     ${gpu_speedup}x"
    fi
fi
echo ""
echo "Raw outputs: $OUTPUT_DIR"

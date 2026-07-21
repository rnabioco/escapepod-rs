#!/bin/bash
# Stage-isolation parity ladder: escpod demux vs WarpDemuX, same model+data.
#
# A classification disagreement can enter at three stages (detect, fingerprint,
# classify). This script runs four layers that swap escpod stages in one at a
# time, so the agreement drop between adjacent layers attributes the gap to a
# specific stage:
#
#   Layer A        WDX boundaries + WDX fingerprints  -> escpod classify
#                  (isolates DTW + RBF + Platt + OvO; this is the ceiling)
#   Layer B-bounds WDX boundaries + escpod fingerprint -> escpod classify
#                  (adds escpod fingerprint extraction)
#   Layer B-cnn    escpod CNN detect  + escpod fingerprint -> escpod classify
#   Layer B-llr    escpod LLR detect  + escpod fingerprint -> escpod classify
#                  (the full default path)
#
# All layers classify with the SAME converted WarpDemuX SVM model, and are
# compared against WarpDemuX's own predictions as ground truth.
#
# Usage:
#   ./benchmarks/benchmark_demux_parity.sh [--no-srun] \
#       [--model NAME] [--out-dir DIR] [--cnn-model ONNX] \
#       [--layers "A bounds cnn llr"] [--dump-mismatches] [pod5_file]
#
# Prereqs: same as benchmark_demux.sh, plus for Layer B-cnn an exported ADAPTed
# ONNX model (scripts/export_adapter_cnn_to_onnx.py -> benchmarks/adapter_cnn_rna004.onnx)
# and an escpod binary built with `--features cnn-detect`.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

NO_SRUN=0
MODEL_NAME="WDX4_rna004_v1_0"
POD5_FILE=""
OUT_DIR="/tmp/demux_parity"
CNN_MODEL="$PROJECT_ROOT/benchmarks/adapter_cnn_rna004.onnx"
LAYERS="A bounds cnn llr"
DUMP=0
ORIG_ARGS=("$@")

while [ $# -gt 0 ]; do
    case "$1" in
        --no-srun)          NO_SRUN=1 ;;
        --model)            MODEL_NAME="$2"; shift ;;
        --model=*)          MODEL_NAME="${1#*=}" ;;
        --out-dir)          OUT_DIR="$2"; shift ;;
        --out-dir=*)        OUT_DIR="${1#*=}" ;;
        --cnn-model)        CNN_MODEL="$2"; shift ;;
        --cnn-model=*)      CNN_MODEL="${1#*=}" ;;
        --layers)           LAYERS="$2"; shift ;;
        --layers=*)         LAYERS="${1#*=}" ;;
        --dump-mismatches)  DUMP=1 ;;
        *)                  POD5_FILE="$1" ;;
    esac
    shift
done

: "${POD5_FILE:=$PROJECT_ROOT/ext/WarpDemuX/test_data/demux/4000_rna004.pod5}"

# Auto-dispatch onto a CPU compute node (parity work is CPU-only).
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$NO_SRUN" -eq 0 ] && command -v srun >/dev/null 2>&1; then
    echo ">>> Re-dispatching under srun -p rna -A rbi -c 16"
    exec srun -p rna -A rbi -c 16 "$0" "${ORIG_ARGS[@]}" --no-srun
fi

THREADS="${SLURM_CPUS_PER_TASK:-${THREADS:-4}}"

ESCPOD="$PROJECT_ROOT/target/release/escpod"
WDX_MODEL_JOBLIB="$PROJECT_ROOT/ext/WarpDemuX/warpdemux/models/model_files/${MODEL_NAME}.joblib"
WDX_MODEL_JSON="$PROJECT_ROOT/benchmarks/.${MODEL_NAME}.json"
PIXI_WDX="pixi run -e warpdemux-bench --manifest-path $PROJECT_ROOT/pixi.toml"
CONVERT_NPZ="$PROJECT_ROOT/scripts/convert_warpdemux_npz_to_csv.py"
COMPARE="$PROJECT_ROOT/scripts/compare_demux_results.py"

echo "=========================================="
echo "Demux parity ladder"
echo "=========================================="
echo "Input:  $POD5_FILE"
echo "Model:  $MODEL_NAME"
echo "Layers: $LAYERS"
echo "Out:    $OUT_DIR"
echo ""

[ -x "$ESCPOD" ] || { echo "error: escpod binary not found: $ESCPOD" >&2; exit 1; }
[ -f "$WDX_MODEL_JOBLIB" ] || { echo "error: model joblib missing: $WDX_MODEL_JOBLIB" >&2; exit 1; }

rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"

# One-time model export.
if [ ! -f "$WDX_MODEL_JSON" ]; then
    echo ">>> Converting $MODEL_NAME -> escpod SVM JSON"
    $PIXI_WDX python "$PROJECT_ROOT/scripts/convert_warpdemux_model.py" \
        "$WDX_MODEL_JOBLIB" "$WDX_MODEL_JSON" > /dev/null
fi

# ----------------------------------------------------------------------
# Run WarpDemuX once: ground-truth predictions + boundaries + fingerprints.
# ----------------------------------------------------------------------
WDX_OUT="$OUT_DIR/wdx_out"
mkdir -p "$WDX_OUT"
echo ">>> WarpDemuX: demux -m $MODEL_NAME --save_fpts True --save_boundaries True"
$PIXI_WDX warpdemux demux \
    -i "$POD5_FILE" -o "$WDX_OUT" -m "$MODEL_NAME" \
    --ncores "$THREADS" --save_fpts True --save_boundaries True 2>&1 | tail -4

WDX_RUN_DIR=$(find "$WDX_OUT" -maxdepth 1 -type d -name "warpdemux_*" | head -1)
[ -n "$WDX_RUN_DIR" ] || { echo "error: WarpDemuX output dir not found" >&2; exit 1; }
WDX_PRED="$WDX_RUN_DIR/predictions"
WDX_BOUND_DIR="$WDX_RUN_DIR/boundaries"
# Fingerprints land under the run dir; locate the npz shards.
WDX_FPTS_DIR=$(dirname "$(find "$WDX_RUN_DIR" -name 'barcode_fpts_*.npz' | head -1)" 2>/dev/null || true)

# Concatenated WDX boundaries CSV (parser strips the first '#' header, skips the
# rest as comments — so a plain zcat of all shards is safe).
WDX_BOUND_CSV="$OUT_DIR/wdx_boundaries.csv"
if ls "$WDX_BOUND_DIR"/detected_boundaries_*.csv.gz >/dev/null 2>&1; then
    zcat "$WDX_BOUND_DIR"/detected_boundaries_*.csv.gz > "$WDX_BOUND_CSV"
fi

# Common boundary-comparison args.
BCMP=()
[ -f "$WDX_BOUND_CSV" ] && BCMP=(--boundaries-warpdemux "$WDX_BOUND_DIR")

# classify helper: <fingerprints_csv> <out_csv>
classify() {
    "$ESCPOD" demux classify "$1" --model "$WDX_MODEL_JSON" --probabilities -o "$2"
}

# fingerprint helper: <boundaries_csv> <out_csv>
fingerprint() {
    "$ESCPOD" demux fingerprint "$POD5_FILE" --boundaries "$1" \
        --warpdemux-compat -o "$2" -j "$THREADS" -q
}

# compare helper: <pred_csv> <label> [boundaries_escapepod_csv]
compare() {
    local pred="$1" label="$2" bnd="${3:-}"
    local args=(--escapepod-b "$pred" --warpdemux "$WDX_PRED"
                --summary-json "$OUT_DIR/${label}.summary.json")
    [ -n "$bnd" ] && [ -f "$WDX_BOUND_CSV" ] && \
        args+=(--boundaries-escapepod "$bnd" "${BCMP[@]}")
    [ "$DUMP" -eq 1 ] && args+=(--dump-mismatches "$OUT_DIR/${label}.mismatches.csv")
    echo ">>> compare: $label"
    $PIXI_WDX python "$COMPARE" "${args[@]}" | sed -n '/Overall agreement/p;/conf/p' | head -8
}

for layer in $LAYERS; do
    echo ""
    echo "----- Layer $layer -----"
    case "$layer" in
        A)
            if [ -z "$WDX_FPTS_DIR" ]; then
                echo "skip: no WDX fingerprint npz found (need --save_fpts True)"; continue
            fi
            $PIXI_WDX python "$CONVERT_NPZ" "$WDX_FPTS_DIR" "$OUT_DIR/A.fpts.csv"
            classify "$OUT_DIR/A.fpts.csv" "$OUT_DIR/A.pred.csv"
            compare "$OUT_DIR/A.pred.csv" "A"
            ;;
        bounds)
            [ -f "$WDX_BOUND_CSV" ] || { echo "skip: no WDX boundaries"; continue; }
            fingerprint "$WDX_BOUND_CSV" "$OUT_DIR/bounds.fpts.csv"
            classify "$OUT_DIR/bounds.fpts.csv" "$OUT_DIR/bounds.pred.csv"
            compare "$OUT_DIR/bounds.pred.csv" "bounds" "$WDX_BOUND_CSV"
            ;;
        cnn)
            if [ ! -f "$CNN_MODEL" ]; then
                echo "skip: CNN ONNX model missing ($CNN_MODEL); run scripts/export_adapter_cnn_to_onnx.py"; continue
            fi
            "$ESCPOD" demux detect "$POD5_FILE" --method cnn --cnn-model "$CNN_MODEL" \
                --downscale 10 -o "$OUT_DIR/cnn.boundaries.csv" -j "$THREADS" -q
            fingerprint "$OUT_DIR/cnn.boundaries.csv" "$OUT_DIR/cnn.fpts.csv"
            classify "$OUT_DIR/cnn.fpts.csv" "$OUT_DIR/cnn.pred.csv"
            compare "$OUT_DIR/cnn.pred.csv" "cnn" "$OUT_DIR/cnn.boundaries.csv"
            ;;
        llr)
            "$ESCPOD" demux detect "$POD5_FILE" \
                -o "$OUT_DIR/llr.boundaries.csv" -j "$THREADS" -q
            fingerprint "$OUT_DIR/llr.boundaries.csv" "$OUT_DIR/llr.fpts.csv"
            classify "$OUT_DIR/llr.fpts.csv" "$OUT_DIR/llr.pred.csv"
            compare "$OUT_DIR/llr.pred.csv" "llr" "$OUT_DIR/llr.boundaries.csv"
            ;;
        *) echo "unknown layer: $layer" >&2 ;;
    esac
done

# ----------------------------------------------------------------------
# Ladder summary.
# ----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Agreement ladder ($MODEL_NAME)"
echo "=========================================="
python3 - "$OUT_DIR" $LAYERS <<'PY'
import json, sys, os
out = sys.argv[1]; layers = sys.argv[2:]
names = {"A": "A  (WDX bounds + WDX fpts)",
         "bounds": "B-bounds (WDX bounds)",
         "cnn": "B-cnn (escpod CNN)",
         "llr": "B-llr (escpod LLR, default)"}
print(f"  {'Layer':<28s} {'overall':>9s} {'conf>=0.5':>10s}")
for L in layers:
    p = os.path.join(out, f"{L}.summary.json")
    if not os.path.exists(p):
        print(f"  {names.get(L,L):<28s} {'(skipped)':>9s}")
        continue
    s = json.load(open(p))
    print(f"  {names.get(L,L):<28s} {s['agreement_pct']:>8.2f}% {s['agreement_conf_ge_0.5_pct']:>9.2f}%")
PY
echo ""
echo "Raw outputs + per-layer summaries/mismatches: $OUT_DIR"

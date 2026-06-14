#!/bin/bash
# Sweep `escpod demux` vs WarpDemuX across {models} x {dataset tiers} x {devices}
# and collect speed + classification-agreement into one table.
#
# Cluster use: every (model x tier x device) cell is submitted as its OWN srun
# allocation and they run CONCURRENTLY across the cluster — SLURM packs them
# onto whatever nodes/GPUs are free. A one-time prep step builds the size-tier
# inputs and converts all models up front so the parallel cells never race on
# them. Each cell writes its own row file; the controller concatenates them.
#
# Usage:
#   ./benchmarks/benchmark_demux_matrix.sh \
#       [--models "WDX4_rna004_v1_0 WDX6_rna004_v1_0 WDX10_rna004_v1_0"] \
#       [--tiers "4000 25000 100000"] \
#       [--devices "cpu gpu"] \
#       [--src REAL_RUN_POD5] \
#       [--cpu-cores N] [--gpu-cores N] \
#       [--out-dir SHARED_DIR] [--no-srun]
#
# --out-dir MUST be on a shared filesystem (cells run on different nodes and
# the controller renders from the login node). Default is a shared, gitignored
# dir under the repo.
#
# Build the binary with the matching features first:
#   cpu cells: --features "demux train cnn-detect"
#   gpu cells: --features "demux train gpu cnn-detect" (on a CUDA node)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ESCPOD_BIN="$PROJECT_ROOT/target/release/escpod"
PIXI_WDX="pixi run -e warpdemux-bench --manifest-path $PROJECT_ROOT/pixi.toml"

MODELS="WDX4_rna004_v1_0 WDX6_rna004_v1_0 WDX10_rna004_v1_0"
TIERS="4000 25000 100000"
DEVICES="cpu gpu"
SRC=""
CPU_CORES=24
GPU_CORES=16
# MUST be shared FS — see header.
OUT_DIR="$PROJECT_ROOT/.demux_matrix_out"
NO_SRUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --models)     MODELS="$2"; shift ;;
        --models=*)   MODELS="${1#*=}" ;;
        --tiers)      TIERS="$2"; shift ;;
        --tiers=*)    TIERS="${1#*=}" ;;
        --devices)    DEVICES="$2"; shift ;;
        --devices=*)  DEVICES="${1#*=}" ;;
        --src)        SRC="$2"; shift ;;
        --src=*)      SRC="${1#*=}" ;;
        --cpu-cores)  CPU_CORES="$2"; shift ;;
        --cpu-cores=*) CPU_CORES="${1#*=}" ;;
        --gpu-cores)  GPU_CORES="$2"; shift ;;
        --gpu-cores=*) GPU_CORES="${1#*=}" ;;
        --out-dir)    OUT_DIR="$2"; shift ;;
        --out-dir=*)  OUT_DIR="${1#*=}" ;;
        --no-srun)    NO_SRUN=1 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

INPUTS_DIR="$OUT_DIR/inputs"
ROWS_DIR="$OUT_DIR/rows"
LOGS_DIR="$OUT_DIR/logs"
MANIFEST="$INPUTS_DIR/inputs.manifest"
MATRIX_TSV="$OUT_DIR/matrix.tsv"
MATRIX_MD="$OUT_DIR/matrix.md"
mkdir -p "$ROWS_DIR" "$LOGS_DIR"

[ -x "$ESCPOD_BIN" ] || { echo "error: escpod binary not found: $ESCPOD_BIN" >&2; exit 1; }

render() {
    # Concatenate per-cell row files (each is a one-row TSV with header) into
    # matrix.tsv, then render matrix.md.
    {
        printf 'model\tpod5\tn_reads\tdevice\tescpod_s\twdx_s\tspeedup\tagree_pct\tagree_conf_pct\n'
        for f in "$ROWS_DIR"/*.tsv; do
            [ -f "$f" ] || continue
            tail -n +2 "$f"
        done
    } > "$MATRIX_TSV"
    python3 - "$MATRIX_TSV" "$MATRIX_MD" <<'PY'
import sys
tsv, md = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
head, data = rows[0], rows[1:]
seen = {}
for r in data:                         # dedup by (model,pod5,device), keep last
    seen[(r[0], r[1], r[3])] = r
data = sorted(seen.values(), key=lambda r: (r[0], int(r[2] or 0), r[3]))
with open(md, "w") as f:
    f.write("| " + " | ".join(head) + " |\n")
    f.write("|" + "|".join(["---"] * len(head)) + "|\n")
    for r in data:
        f.write("| " + " | ".join(r) + " |\n")
print(open(md).read())
PY
}

# ----------------------------------------------------------------------
# Inline mode (--no-srun / already in SLURM): run cells sequentially. Used
# as the fallback and by the per-cell srun jobs is NOT this path — cells call
# benchmark_demux.sh directly. This block only runs the whole sweep inline.
# ----------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ] || [ "$NO_SRUN" -eq 1 ] || ! command -v srun >/dev/null 2>&1; then
    echo ">>> inline (sequential) sweep — no parallel fan-out"
    "$SCRIPT_DIR/make_demux_inputs.sh" ${SRC:+--src "$SRC"} \
        --out-dir "$INPUTS_DIR" --tiers "$TIERS" --escpod "$ESCPOD_BIN"
    [ -s "$MANIFEST" ] || { echo "error: no manifest" >&2; exit 1; }
    while IFS=$'\t' read -r tier n pod5; do
        [ -z "$pod5" ] && continue
        for model in $MODELS; do
            for dev in $DEVICES; do
                gf=""; [ "$dev" = gpu ] && gf="--gpu"
                "$SCRIPT_DIR/benchmark_demux.sh" --no-srun $gf --model "$model" \
                    --out-dir "$OUT_DIR/cell_${model}_${tier}_${dev}" \
                    --emit-tsv "$ROWS_DIR/${model}_${tier}_${dev}.tsv" "$pod5" \
                    || echo "warn: cell ${model}/${tier}/${dev} failed" >&2
            done
        done
    done < "$MANIFEST"
    render
    exit 0
fi

# ----------------------------------------------------------------------
# Phase A — PREP (one rna allocation): build size tiers + convert all models.
# Done once, before the parallel cells, so they never race on these.
# ----------------------------------------------------------------------
echo "========================================"
echo "Matrix sweep — parallel fan-out"
echo "  models:  $MODELS"
echo "  tiers:   $TIERS"
echo "  devices: $DEVICES   (cpu -c $CPU_CORES, gpu -c $GPU_CORES)"
echo "  out:     $OUT_DIR"
echo "========================================"
echo ">>> Phase A: build inputs + convert models (srun -p rna)"
srun -p rna -A rbi -c 16 --mem=48G --job-name demux_prep bash -c "
    set -e
    '$SCRIPT_DIR/make_demux_inputs.sh' ${SRC:+--src '$SRC'} \
        --out-dir '$INPUTS_DIR' --tiers '$TIERS' --escpod '$ESCPOD_BIN'
    for m in $MODELS; do
        json='$PROJECT_ROOT/benchmarks/.'\$m'.json'
        if [ ! -f \"\$json\" ]; then
            echo \">>> convert \$m\"
            $PIXI_WDX python '$PROJECT_ROOT/scripts/convert_warpdemux_model.py' \
                '$PROJECT_ROOT/ext/WarpDemuX/warpdemux/models/model_files/'\$m'.joblib' \"\$json\" >/dev/null
        fi
    done
"
[ -s "$MANIFEST" ] || { echo "error: prep produced no manifest ($MANIFEST)" >&2; exit 1; }

# ----------------------------------------------------------------------
# Phase B — FAN OUT: one backgrounded srun per cell; SLURM schedules them
# concurrently across the cluster.
# ----------------------------------------------------------------------
echo ">>> Phase B: submitting cell jobs (parallel)"
pids=()
ncell=0
while IFS=$'\t' read -r tier n_reads pod5; do
    [ -z "$pod5" ] && continue
    for model in $MODELS; do
        for dev in $DEVICES; do
            case "$dev" in
                cpu) SRUN=(-p rna -A rbi -c "$CPU_CORES" --mem=64G); GF="" ;;
                gpu) SRUN=(-p gpu -A gpu_rbi -c "$GPU_CORES" --gres=gpu:1 --mem=64G); GF="--gpu" ;;
                *)   echo "unknown device: $dev" >&2; continue ;;
            esac
            cell="${model}_${tier}_${dev}"
            ncell=$((ncell + 1))
            echo "    submit cell: $cell  (${n_reads} reads)"
            srun "${SRUN[@]}" --job-name "dx_$cell" \
                "$SCRIPT_DIR/benchmark_demux.sh" --no-srun $GF \
                --model "$model" --out-dir "$OUT_DIR/cell_$cell" \
                --emit-tsv "$ROWS_DIR/$cell.tsv" "$pod5" \
                > "$LOGS_DIR/$cell.log" 2>&1 &
            pids+=("$!")
        done
    done
done < "$MANIFEST"

echo ">>> $ncell cell jobs submitted; waiting for completion..."
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=$((fail + 1))
done
[ "$fail" -gt 0 ] && echo "warning: $fail cell job(s) exited non-zero (see $LOGS_DIR)" >&2

# ----------------------------------------------------------------------
# Phase C — render.
# ----------------------------------------------------------------------
echo ""
echo "========================================"
echo "Matrix complete ($ncell cells, $fail failed) -> $MATRIX_MD"
echo "========================================"
render

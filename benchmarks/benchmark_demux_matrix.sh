#!/bin/bash
# Sweep `escpod demux` vs WarpDemuX across {models} x {dataset tiers} x {devices}
# and collect speed + classification-agreement into one table.
#
# Each cell delegates to benchmark_demux.sh (which runs the full
# detect->fingerprint->classify pipeline for both tools and compares them),
# passing --emit-tsv so every cell appends a machine-readable row. Results land
# in <out-dir>/matrix.tsv and are rendered to <out-dir>/matrix.md.
#
# Usage:
#   ./benchmarks/benchmark_demux_matrix.sh \
#       [--models "WDX4_rna004_v1_0 WDX6_rna004_v1_0 WDX10_rna004_v1_0"] \
#       [--tiers "4000 25000 100000"] \
#       [--devices "cpu gpu"] \
#       [--src REAL_RUN_POD5] \
#       [--out-dir DIR] [--no-srun]
#
# SLURM: not inside an allocation, the script self-dispatches ONE srun per
# device onto the right partition (cpu -> rna, gpu -> gpu) and runs every cell
# inside it with --no-srun. Build the binary with the matching features first:
#   cpu cells: --features "demux train cnn-detect"
#   gpu cells: --features "demux train gpu cnn-detect" (on a CUDA node)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ESCPOD_BIN="$PROJECT_ROOT/target/release/escpod"

MODELS="WDX4_rna004_v1_0 WDX6_rna004_v1_0 WDX10_rna004_v1_0"
TIERS="4000 25000 100000"
DEVICES="cpu gpu"
SRC=""
# MUST be on a shared filesystem: the per-device cells run in separate srun
# allocations (cpu on rna, gpu on gpu) and both the size-tier inputs and the
# appended matrix.tsv have to be visible across nodes and to the outer render
# step. Node-local /tmp would break cross-device aggregation. Override with
# --out-dir to any shared-FS path.
OUT_DIR="$PROJECT_ROOT/.demux_matrix_out"
NO_SRUN=0
ORIG_ARGS=("$@")

while [ $# -gt 0 ]; do
    case "$1" in
        --models)    MODELS="$2"; shift ;;
        --models=*)  MODELS="${1#*=}" ;;
        --tiers)     TIERS="$2"; shift ;;
        --tiers=*)   TIERS="${1#*=}" ;;
        --devices)   DEVICES="$2"; shift ;;
        --devices=*) DEVICES="${1#*=}" ;;
        --src)       SRC="$2"; shift ;;
        --src=*)     SRC="${1#*=}" ;;
        --out-dir)   OUT_DIR="$2"; shift ;;
        --out-dir=*) OUT_DIR="${1#*=}" ;;
        --no-srun)   NO_SRUN=1 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

INPUTS_DIR="$OUT_DIR/inputs"
MATRIX_TSV="$OUT_DIR/matrix.tsv"
MATRIX_MD="$OUT_DIR/matrix.md"
mkdir -p "$OUT_DIR"

render_markdown() {
    [ -f "$MATRIX_TSV" ] || return 0
    python3 - "$MATRIX_TSV" "$MATRIX_MD" <<'PY'
import sys
tsv, md = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
if not rows:
    sys.exit(0)
head, data = rows[0], rows[1:]
# de-dup by (model,pod5,device) keeping the last run
seen = {}
for r in data:
    seen[(r[0], r[1], r[3])] = r
data = sorted(seen.values(), key=lambda r: (r[0], int(r[2] or 0), r[3]))
cols = ["model", "pod5", "n_reads", "device", "escpod_s", "wdx_s", "speedup", "agree_pct", "agree_conf_pct"]
with open(md, "w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for r in data:
        f.write("| " + " | ".join(r) + " |\n")
print(open(md).read())
PY
}

# ----------------------------------------------------------------------
# Outer stage: self-dispatch one srun per device onto its partition.
# ----------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$NO_SRUN" -eq 0 ] && command -v srun >/dev/null 2>&1; then
    for dev in $DEVICES; do
        case "$dev" in
            cpu) SRUN_ARGS=(-p rna -A rbi -c 48 --mem=64G) ;;
            gpu) SRUN_ARGS=(-p gpu -A gpu_rbi -c 16 --gres=gpu:1) ;;
            *)   echo "unknown device: $dev" >&2; exit 2 ;;
        esac
        echo ">>> [$dev] dispatching: srun ${SRUN_ARGS[*]}"
        srun "${SRUN_ARGS[@]}" "$0" \
            --models "$MODELS" --tiers "$TIERS" --devices "$dev" \
            ${SRC:+--src "$SRC"} --out-dir "$OUT_DIR" --no-srun \
            || echo "warning: [$dev] cell run exited non-zero" >&2
    done
    echo ""
    echo "========================================"
    echo "Matrix complete -> $MATRIX_MD"
    echo "========================================"
    render_markdown
    exit 0
fi

# ----------------------------------------------------------------------
# Inner stage (inside an allocation, or --no-srun): run every cell.
# ----------------------------------------------------------------------
if [ ! -x "$ESCPOD_BIN" ]; then
    echo "error: escpod binary not found: $ESCPOD_BIN" >&2
    exit 1
fi

# Ensure size-tier inputs exist (idempotent).
echo ">>> ensuring inputs in $INPUTS_DIR"
"$SCRIPT_DIR/make_demux_inputs.sh" \
    ${SRC:+--src "$SRC"} --out-dir "$INPUTS_DIR" --tiers "$TIERS" --escpod "$ESCPOD_BIN"

MANIFEST="$INPUTS_DIR/inputs.manifest"
if [ ! -s "$MANIFEST" ]; then
    echo "error: no inputs manifest produced ($MANIFEST)" >&2
    exit 1
fi

for dev in $DEVICES; do
    GPU_FLAG=""
    [ "$dev" = "gpu" ] && GPU_FLAG="--gpu"
    for model in $MODELS; do
        while IFS=$'\t' read -r tier_name n_reads pod5_path; do
            [ -z "$pod5_path" ] && continue
            cell="$OUT_DIR/cell_${model}_${tier_name}_${dev}"
            echo ""
            echo "######## CELL: model=$model tier=$tier_name ($n_reads reads) dev=$dev ########"
            "$SCRIPT_DIR/benchmark_demux.sh" --no-srun $GPU_FLAG \
                --model "$model" --out-dir "$cell" --emit-tsv "$MATRIX_TSV" \
                "$pod5_path" \
                || echo "warning: cell failed (model=$model tier=$tier_name dev=$dev)" >&2
        done < "$MANIFEST"
    done
done

render_markdown

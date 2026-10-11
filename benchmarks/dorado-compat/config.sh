# shellcheck shell=bash
# Shared parameters for the dorado-compat check. Everything is an environment
# variable; nothing here names a site path. Source this, do not execute it.
#
# REQUIRED (unless CONFIG_NO_INPUT=1):
#   INPUT_DIR     directory (searched recursively) holding real POD5 files with
#                 more reads than COUNTS asks for. Read-only: files are symlinked,
#                 never copied, and no .p5s is written beside them.
#
# Optional (defaults in parentheses):
#   SCRATCH_DIR   ($HOME/scratch)             parent of the work area
#   DC            ($SCRATCH_DIR/dorado-check) work area: tools/ models/ results/ target/ work/
#   WORK          ($DC/work)                  bulky and regenerable; removed after a run
#   REPO          (the checkout holding these scripts)  supplies pixi python (pod5, pysam)
#   SRC           ($REPO)                     checkout escpod is built from
#   ESCPOD        ($CARGO_TARGET_DIR/release/escpod)
#   DORADO_VERSION (pixi.toml's [activation] DORADO_VERSION, else 2.1.2)
#   DORADO_BIN    explicit dorado binary; else $PIXI_PROJECT_ROOT/.pixi/tools/dorado-$VER-linux-x64/bin/dorado,
#                 else $DC/tools/... (installed there on demand by run_check.sh)
#   MODEL_NAME    (rna004_fast@v6.0.0)        MODELS_DIR ($DC/models)
#   RAW_INPUTS    space-separated globs relative to INPUT_DIR; default = first N *.pod5 found, sorted
#   COUNTS        ("1865 2431 1207 1733")     reads taken from each input; none may be a multiple of 1000
#   CPU_PARTITION (rna)  BUILD_PARTITION (=CPU_PARTITION)  GPU_PARTITION (gpu)  GPU_ACCOUNT (gpu_rbi)
#   NO_SRUN=1     run every stage directly (no Slurm)
_cfg_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
export SCRATCH_DIR=${SCRATCH_DIR:-$HOME/scratch}
export DC=${DC:-$SCRATCH_DIR/dorado-check}
export WORK=${WORK:-$DC/work}
export REPO=${REPO:-$(cd "$_cfg_dir/../.." && pwd)}
export SRC=${SRC:-$REPO}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$DC/target}
export ESCPOD=${ESCPOD:-$CARGO_TARGET_DIR/release/escpod}

if [ -z "${DORADO_VERSION:-}" ]; then
  DORADO_VERSION=$(sed -n 's/.*DORADO_VERSION *= *"\([^"]*\)".*/\1/p' "$REPO/pixi.toml" 2>/dev/null | head -1)
fi
export DORADO_VERSION=${DORADO_VERSION:-2.1.2}
export DORADO_URL=${DORADO_URL:-https://cdn.oxfordnanoportal.com/software/analysis/dorado-$DORADO_VERSION-linux-x64.tar.gz}
_dorado_rel=dorado-$DORADO_VERSION-linux-x64/bin/dorado
if [ -z "${DORADO_BIN:-}" ]; then
  for _c in "${PIXI_PROJECT_ROOT:-$REPO}/.pixi/tools/$_dorado_rel" "$DC/tools/$_dorado_rel"; do
    if [ -x "$_c" ]; then DORADO_BIN=$_c; break; fi
  done
fi
export DORADO_BIN=${DORADO_BIN:-$DC/tools/$_dorado_rel}   # not found -> run_check.sh installs here
export DORADO=$DORADO_BIN

# 2.x renamed the RNA004 models; rna004_fast@v6.0.0 is the current non-outdated fast model.
export MODEL_NAME=${MODEL_NAME:-rna004_fast@v6.0.0}
export MODELS_DIR=${MODELS_DIR:-$DC/models}
export MODEL=$MODELS_DIR/$MODEL_NAME

if [ -z "${INPUT_DIR:-}" ] && [ "${CONFIG_NO_INPUT:-0}" != 1 ]; then
  echo "INPUT_DIR is required: a directory of real POD5 files (set the env var or pass --input-dir to run_check.sh)" >&2
  return 2 2>/dev/null || exit 2
fi
export INPUT_DIR=${INPUT_DIR:-}
export RAW_INPUTS=${RAW_INPUTS:-}
# Counts that are not multiples of 1000 leave a short last signal batch in each input.
export COUNTS=${COUNTS:-"1865 2431 1207 1733"}

CPU_PARTITION=${CPU_PARTITION:-rna}
BUILD_PARTITION=${BUILD_PARTITION:-$CPU_PARTITION}
GPU_PARTITION=${GPU_PARTITION:-gpu}
GPU_ACCOUNT=${GPU_ACCOUNT:-gpu_rbi}
if [ "${NO_SRUN:-0}" = 1 ]; then
  export CPU_SRUN="" GPU_SRUN="" BUILD_SRUN=""
else
  export CPU_SRUN=${CPU_SRUN:-"srun -p $CPU_PARTITION -c 8 --mem=16G -J dorado-check-cpu --comment=dorado-check-cpu --"}
  export GPU_SRUN=${GPU_SRUN:-"srun -p $GPU_PARTITION -A $GPU_ACCOUNT -c 16 --gres=gpu:1 -J dorado-check-gpu --comment=dorado-check-gpu --"}
  export BUILD_SRUN=${BUILD_SRUN:-"srun -p $BUILD_PARTITION -c 32 --mem=32G -J escpod-build --comment=escpod-build --"}
fi
export PIXI="pixi run --manifest-path $REPO/pixi.toml"

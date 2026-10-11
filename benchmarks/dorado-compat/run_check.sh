#!/usr/bin/env bash
# One entry point for the dorado-vs-escpod POD5 compatibility check.
# usage: run_check.sh --input-dir DIR [--sha REF | --src DIR] [--rebuild] [--keep-work]
#   --input-dir DIR  directory of real POD5 files (same as env INPUT_DIR; required)
#   --sha REF        build escpod from a detached `git worktree add $DC/src REF`
#   --src DIR        build escpod from this checkout as-is
# Run from the login node: it srun's the heavy stages itself; do not background it.
# Every other knob is an environment variable; see config.sh.
set -euo pipefail
D=$(dirname "$(readlink -f "$0")")
SHA=""; REBUILD=0; KEEP=0
while [ $# -gt 0 ]; do
  case $1 in
    --input-dir) INPUT_DIR=$2; export INPUT_DIR; shift 2;;
    --sha) SHA=$2; shift 2;;
    --src) SRC=$2; export SRC; shift 2;;
    --rebuild) REBUILD=1; shift;;
    --keep-work) KEEP=1; shift;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "bad arg $1" >&2; exit 2;;
  esac
done
# shellcheck source=config.sh
. "$D/config.sh"
mkdir -p "$DC/results" "$DC/tools" "$MODELS_DIR"
# --- dorado: install the pinned version if absent (needs network); verify, never fall back
if [ ! -x "$DORADO" ]; then
  t=$(mktemp -d "$DC/tools.XXXX")
  curl -sSfL -o "$t/d.tgz" "$DORADO_URL" && tar xzf "$t/d.tgz" -C "$DC/tools"; rm -rf "$t"
fi
[ -x "$DORADO" ] || { echo "dorado $DORADO_VERSION not installed at $DORADO"; exit 3; }
DV=$("$DORADO" --version 2>&1 | tail -1)
case "$DV" in "$DORADO_VERSION"*) ;; *) echo "dorado version mismatch: wanted $DORADO_VERSION got $DV"; exit 3;; esac
[ -d "$MODEL" ] || "$DORADO" download --model "$MODEL_NAME" --models-directory "$MODELS_DIR"
# --- source: must be origin/main and clean, else a detached worktree at origin/main (or --sha REF)
git -C "$REPO" fetch -q origin
if [ -z "$SHA" ] && [ "$SRC" = "$REPO" ]; then
  if [ "$(git -C "$REPO" rev-parse HEAD)" != "$(git -C "$REPO" rev-parse origin/main)" ] || [ -n "$(git -C "$REPO" status --porcelain)" ]; then
    echo "REPO checkout is not clean origin/main -> using detached worktree at origin/main"; SHA=origin/main
  fi
fi
if [ -n "$SHA" ]; then
  SHA=$(git -C "$REPO" rev-parse --verify "$SHA^{commit}")
  git -C "$REPO" worktree remove --force "$DC/src" 2>/dev/null || true; rm -rf "$DC/src"; git -C "$REPO" worktree prune
  git -C "$REPO" worktree add --detach "$DC/src" "$SHA"; SRC=$DC/src
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$SHA" ] || { echo "worktree sha mismatch"; exit 4; }
  REBUILD=1
fi
if [ $REBUILD = 1 ] || [ ! -x "$ESCPOD" ]; then
  # $PIXI / $BUILD_SRUN are intentionally word-split command prefixes
  # shellcheck disable=SC2086
  (cd "$SRC" && CARGO_TARGET_DIR=$CARGO_TARGET_DIR $BUILD_SRUN $PIXI cargo build --release -p escapepod-cli)
fi
echo "escpod: $("$ESCPOD" --version)  dorado: $DV  model: $MODEL_NAME  src sha: $(git -C "$SRC" rev-parse --short HEAD)" | tee "$DC/results/last_versions.txt"
before=$(find -L "$INPUT_DIR" -name '*.p5s' | sort | md5sum)
# shellcheck disable=SC2086
{
  $CPU_SRUN "$D/build_inputs.sh"
  $CPU_SRUN "$D/make_outputs.sh"
  $GPU_SRUN "$D/run_dorado.sh"
  $CPU_SRUN "$D/report.sh" | tee "$DC/results/last_report.txt"
}
after=$(find -L "$INPUT_DIR" -name '*.p5s' | sort | md5sum)
[ "$before" = "$after" ] && echo "p5s in INPUT_DIR unchanged" || echo "WARNING: .p5s set in INPUT_DIR changed"
if [ -n "$SHA" ]; then git -C "$REPO" worktree remove --force "$DC/src"; git -C "$REPO" worktree prune; fi
[ $KEEP = 1 ] || rm -rf "$WORK"

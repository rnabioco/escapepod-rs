#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2086,SC2012  # $PIXI is a word-split command prefix; config.sh is sourced by relative path
# Negative control: prove the check can fail. Takes $WORK/in/in1.pod5 (left by build_inputs.sh, so run
# run_check.sh --keep-work first) and re-batches its SIGNAL table to 500,1000,<rest> rows with
# mk_nonuniform_raw.py (pyarrow + footer patch; escpod is not involved). Then runs every probe on it.
# usage: negative_control.sh [prep|dorado|report|all]   (default all; `all` srun's each stage)
#   prep    (CPU)  build $WORK/neg/nonuniform.pod5
#   dorado  (GPU)  basecall it; record rc and stderr
#   report  (CPU)  escpod inspect, python pod5, dorado rc/errors, BAM parents vs POD5
# Expected: escpod inspect says NOT PORTABLE, python pod5 FAIL, dorado rc=0 with hundreds of
# 'Row outside batch bounds' errors and a large share of parents missing from the BAM.
set -uo pipefail
. "$(dirname "$0")/config.sh"; S=$(dirname "$(readlink -f "$0")")
N=$WORK/neg
stage=${1:-all}
# $PIXI is an intentionally word-split command prefix
# shellcheck disable=SC2086
prep() {
  set -e; mkdir -p "$N"
  rows=$($PIXI python "$S/total_rows.py" "$WORK/in/in1.pod5")
  [ "$rows" -gt 1500 ] || { echo "in1.pod5 has only $rows signal rows; need > 1500" >&2; exit 2; }
  $PIXI python "$S/mk_nonuniform_raw.py" "$WORK/in/in1.pod5" "$N/nonuniform.pod5" "500,1000,$((rows-1500))"
}
run_dorado() {
  cd "$N" || exit 1
  "$DORADO" basecaller "$MODEL" nonuniform.pod5 > nonuniform.bam 2> nonuniform.err; echo $? > nonuniform.rc
  echo "dorado rc=$(cat nonuniform.rc)"; grep -viE 'optimal batch size' nonuniform.err | tail -15 | cut -c1-300
}
# shellcheck disable=SC2086
report() {
  cd "$N" || exit 1
  echo "== escpod inspect =="; "$ESCPOD" inspect summary nonuniform.pod5 2>&1 | grep -i 'not portable' || echo "(no NOT PORTABLE line)"
  echo "== signal batches =="; $PIXI python "$S/batches.py" nonuniform.pod5
  echo "== python pod5 =="; $PIXI python "$S/verify_pod5.py" nonuniform.pod5
  echo "== dorado =="; echo "rc=$(cat nonuniform.rc)  error lines: $(grep -c 'Failed to get read signal' nonuniform.err || true)"
  $PIXI python "$S/pod5_ids.py" pod5.ids nonuniform.pod5 > /dev/null
  $PIXI python "$S/bam_parents.py" nonuniform.bam bam.ids
  echo "parents missing from BAM: $(comm -23 pod5.ids bam.ids | wc -l) of $(wc -l < pod5.ids)  (rc alone would have said fine)"
}
case $stage in
  prep) prep;;
  dorado) run_dorado;;
  report) report;;
  all)
    # shellcheck disable=SC2086
    $CPU_SRUN "$0" prep && $GPU_SRUN "$0" dorado; $CPU_SRUN "$0" report;;
  *) echo "usage: $0 [prep|dorado|report|all]" >&2; exit 2;;
esac

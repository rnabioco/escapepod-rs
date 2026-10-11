#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2086,SC2012  # $PIXI is a word-split command prefix; config.sh is sourced by relative path
# Runs on a GPU node (called under srun). For each name<TAB>path in $WORK/list.tsv: dorado basecaller -> bam, stderr, rc.
set -uo pipefail; . "$(dirname "$0")/config.sh"
mkdir -p "$WORK/bam"; cd "$WORK" || exit 1
while IFS=$'\t' read -r name path; do
  $DORADO basecaller $MODEL "$path" > bam/$name.bam 2> bam/$name.err; echo $? > bam/$name.rc
  echo "$name rc=$(cat bam/$name.rc)"
done < list.tsv

#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2086,SC2012  # $PIXI is a word-split command prefix; config.sh is sourced by relative path
# Build short-last-batch inputs: symlink raw files, take first N reads of each via `escpod filter -i`.
set -euo pipefail
. "$(dirname "$0")/config.sh"; S=$(dirname "$(readlink -f "$0")")
mkdir -p "$WORK"/{raw,in,ids,tmp}
read -r -a counts <<< "$COUNTS"
if [ -n "$RAW_INPUTS" ]; then
  files=()
  for pat in $RAW_INPUTS; do
    # shellcheck disable=SC2086
    f=$(ls $INPUT_DIR/$pat 2>/dev/null | head -1); [ -n "$f" ] || { echo "no input matches $INPUT_DIR/$pat" >&2; exit 2; }
    files+=("$f")
  done
else
  mapfile -t files < <(find -L "$INPUT_DIR" -name '*.pod5' | sort | head -n "${#counts[@]}")
fi
[ "${#files[@]}" -ge "${#counts[@]}" ] || { echo "need ${#counts[@]} POD5 files under $INPUT_DIR, found ${#files[@]}" >&2; exit 2; }
i=0
for f in "${files[@]:0:${#counts[@]}}"; do
  n=${counts[$i]}; i=$((i+1))
  ln -sf "$f" "$WORK/raw/raw$i.pod5"
  # shellcheck disable=SC2086
  $PIXI python "$S/mkids.py" "$WORK/raw/raw$i.pod5" "$n" "$WORK/tmp/ids$i.txt"
  "$ESCPOD" filter -q -f -i "$WORK/tmp/ids$i.txt" -o "$WORK/in/in$i.pod5" "$WORK/raw/raw$i.pod5"
done

#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2086,SC2012  # $PIXI is a word-split command prefix; config.sh is sourced by relative path
# Per-file table: batches | reads | python-pod5 | NOT PORTABLE | dorado rc/errors | BAM parents | missing.
set -euo pipefail; . "$(dirname "$0")/config.sh"; S=$(dirname "$(readlink -f "$0")")
cat $DC/results/last_versions.txt 2>/dev/null; echo "dorado: $($DORADO --version 2>&1|tail -1)"
cd $WORK; mkdir -p ids
while IFS=$'\t' read -r name path; do
  $PIXI python $S/pod5_ids.py ids/$name.pod5.ids "$path" >/dev/null
  $PIXI python $S/bam_parents.py bam/$name.bam ids/$name.bam.ids > bam/$name.parents
  $ESCPOD inspect summary "$path" > bam/$name.summary 2>&1 || true
done < list.tsv
echo "== signal batch row counts (RLE rows x count) =="; cut -f2 list.tsv | xargs $PIXI python $S/batches.py
echo "== python pod5 full-signal read =="; cut -f2 list.tsv | xargs $PIXI python $S/verify_pod5.py
echo "== inspect summary NOT PORTABLE =="; grep -il 'not portable' bam/*.summary || echo none
echo "== dorado rc / stderr errors =="
for r in bam/*.rc; do n=$(basename $r .rc); echo -e "$n\trc=$(cat $r)\terr_lines=$(grep -iE 'error|fail|invalid' bam/$n.err | grep -vc 'Failed to find optimal batch size' || true)"; done
echo "== bam parents vs pod5 =="; $PIXI python $S/compare.py $WORK

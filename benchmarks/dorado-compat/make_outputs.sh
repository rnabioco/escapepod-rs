#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2086,SC2012  # $PIXI is a word-split command prefix; config.sh is sourced by relative path
# Run merge / filter / subset / demux split over the inputs in $WORK/in ; write $WORK/list.tsv (name<TAB>path).
set -euo pipefail; . "$(dirname "$0")/config.sh"; S=$(dirname "$(readlink -f "$0")")
cd $WORK; mkdir -p out; rm -rf out/*; : > list.tsv
ins=(in/in*.pod5)
for f in "${ins[@]}"; do printf 'ctl_%s\t%s\n' "$(basename $f .pod5)" "$WORK/$f" >> list.tsv; done
$ESCPOD merge -q -f -o out/merged.pod5 "${ins[@]}"
printf 'merged\t%s\n' $WORK/out/merged.pod5 >> list.tsv
# filter over all inputs in one invocation: every 3rd read of the union
$PIXI python $S/pod5_ids.py tmp/all_sorted.txt "${ins[@]}" >/dev/null
awk 'NR%3==0' tmp/all_sorted.txt > tmp/every3rd.txt
$ESCPOD filter -q -f -i tmp/every3rd.txt -o out/filtered.pod5 "${ins[@]}"
printf 'filtered\t%s\n' $WORK/out/filtered.pod5 >> list.tsv
# subset over the multi-file set (directory of the inputs): two groups
awk 'BEGIN{print "read_id,output"} {print $1 "," (NR%2? "grpA.pod5":"grpB.pod5")}' tmp/all_sorted.txt > tmp/subset.csv
mkdir -p out/subset; $ESCPOD subset -q -f --csv tmp/subset.csv -o out/subset $WORK/in
for f in out/subset/*.pod5; do printf 'subset_%s\t%s\n' "$(basename $f .pod5)" $WORK/$f >> list.tsv; done
# demux split from read_id,barcode CSV
$PIXI python $S/mkbarcodes.py tmp/barcodes.csv "${ins[@]}"
mkdir -p out/split; $ESCPOD demux split -q --force --classifications tmp/barcodes.csv -d out/split "${ins[@]}"
for f in out/split/*.pod5; do printf 'split_%s\t%s\n' "$(basename $f .pod5)" $WORK/$f >> list.tsv; done

#!/usr/bin/env bash
# Benchmark: escapepod resquiggle vs fishnet align
#
# Usage: bash scripts/bench_resquiggle.sh [1k|10k|100k] [threads]
#   Default: 10k reads, 1 thread
#
# Requires: hyperfine, fishnet (built), escapepod (built)
# POD5 files must be pre-subsetted (e.g. no_aaRS_ctrl_10k.pod5)

set -euo pipefail

READS="${1:-10k}"
THREADS="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ESCPOD="$PROJECT_DIR/target/release/escpod"
FISHNET="/beevol/home/jhessel/devel/rnabioco/escapepod-rs/ext/fishnet/target/release/fishnet"

# Use subsetted POD5 matching the BAM read count (not the full 44GB file)
POD5="$PROJECT_DIR/data/bench/no_aaRS_ctrl_${READS}.pod5"
BAM="$PROJECT_DIR/data/bench/no_aaRS_ctrl_${READS}.bam"
KMER_GZ="$PROJECT_DIR/data/kmer_models/rna004_9mer_levels_v1.txt.gz"
KMER_TXT="/tmp/rna004_9mer_levels_v1.txt"

# fishnet needs uncompressed kmer table
if [ ! -f "$KMER_TXT" ]; then
    zcat "$KMER_GZ" > "$KMER_TXT"
fi

# Verify inputs exist
for f in "$ESCPOD" "$FISHNET" "$POD5" "$BAM" "$KMER_GZ" "$KMER_TXT"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: $f not found" >&2
        exit 1
    fi
done

OUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUT_DIR"' EXIT

NREADS=$(pixi run samtools view -c "$BAM" 2>/dev/null || echo "?")
echo "=== Resquiggle Benchmark ==="
echo "Reads:   $NREADS ($READS subset)"
echo "Threads: $THREADS"
echo "POD5:    $POD5 ($(du -h "$POD5" | cut -f1))"
echo "BAM:     $BAM"
echo ""

# --- Thread argument handling ---
# fishnet needs >=4 threads for multithreading; <4 falls back to single-threaded
FISHNET_THREADS="$THREADS"

echo "--- Warmup runs (verifying output) ---"

"$ESCPOD" resquiggle "$POD5" \
    --bam "$BAM" \
    --kmer-table "$KMER_GZ" \
    --output "$OUT_DIR/escpod.bam" \
    --threads "$THREADS" 2>&1

"$FISHNET" align \
    --pod5 "$POD5" \
    --bam "$BAM" \
    --kmer-table "$KMER_TXT" \
    --out "$OUT_DIR/fishnet.parquet" \
    --rna --alignment-type query --output-level 1 \
    --threads "$FISHNET_THREADS" 2>&1

echo ""
echo "--- Benchmark ($READS reads, $THREADS threads) ---"
echo ""

# fishnet won't overwrite existing output; rm before each run
hyperfine \
    --warmup 1 \
    --min-runs 3 \
    --export-markdown "$PROJECT_DIR/data/bench/results_${READS}_${THREADS}t.md" \
    --command-name "escapepod ($READS, ${THREADS}t)" \
    "$ESCPOD resquiggle $POD5 --bam $BAM --kmer-table $KMER_GZ --output $OUT_DIR/escpod.bam --threads $THREADS" \
    --command-name "fishnet ($READS, ${THREADS}t)" \
    --prepare "rm -f $OUT_DIR/fishnet.parquet" \
    "$FISHNET align --pod5 $POD5 --bam $BAM --kmer-table $KMER_TXT --out $OUT_DIR/fishnet.parquet --rna --alignment-type query --output-level 1 --threads $FISHNET_THREADS"

echo ""
echo "Results saved to: data/bench/results_${READS}_${THREADS}t.md"
cat "$PROJECT_DIR/data/bench/results_${READS}_${THREADS}t.md"

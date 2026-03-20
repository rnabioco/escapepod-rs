#!/bin/bash
# Comprehensive benchmark comparing escapepod-rs vs official pod5
#
# Usage: ./benchmarks/benchmark.sh <pod5_dir>
#
# The script expects a directory containing POD5 files.

set -e

POD5_DIR="${1:-.}"
ESCAPEPOD_BIN="./target/release/escpod"
POD5_BIN="$(cd /beevol/home/jhessel/devel/rnabioco/escapepod-rs && pixi run which pod5)"
OUTPUT_DIR="/tmp/escapepod_benchmark"
WARMUP=1
RUNS=3

# Check dependencies
if ! command -v hyperfine &> /dev/null; then
    echo "Error: hyperfine not found. Install with: brew install hyperfine"
    exit 1
fi

if [ ! -f "$ESCAPEPOD_BIN" ]; then
    echo "Error: escapepod-rs binary not found. Run: cargo build --release"
    exit 1
fi

if [ ! -f "$POD5_BIN" ]; then
    echo "Error: pod5 not found. Install with: uv pip install pod5"
    exit 1
fi

# Find POD5 files
POD5_FILES=$(find -L "$POD5_DIR" -name "*.pod5" -type f | sort)
FILE_COUNT=$(echo "$POD5_FILES" | wc -l | tr -d ' ')

if [ "$FILE_COUNT" -lt 1 ]; then
    echo "Error: No POD5 files found in $POD5_DIR"
    exit 1
fi

# Get first file for single-file benchmarks
FIRST_FILE=$(echo "$POD5_FILES" | head -1)
FILE_LIST=$(echo "$POD5_FILES" | tr '\n' ' ')

echo "========================================"
echo "POD5 Benchmark Suite"
echo "========================================"
echo ""
echo "Input directory: $POD5_DIR"
echo "Files found: $FILE_COUNT"
echo ""
echo "Input files:"
du -sh $FILE_LIST 2>/dev/null | head -10
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ========================================
# Benchmark: inspect/summary
# ========================================
echo ""
echo "========================================"
echo "Benchmark: inspect summary (single file)"
echo "========================================"

hyperfine \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --export-json "$OUTPUT_DIR/inspect_summary.json" \
    --command-name "escapepod-rs" \
    "$ESCAPEPOD_BIN inspect summary $FIRST_FILE" \
    --command-name "pod5 (Python)" \
    "$POD5_BIN inspect summary $FIRST_FILE"

# ========================================
# Benchmark: view
# ========================================
echo ""
echo "========================================"
echo "Benchmark: view (single file)"
echo "========================================"

hyperfine \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --export-json "$OUTPUT_DIR/view.json" \
    --command-name "escapepod-rs" \
    "$ESCAPEPOD_BIN view $FIRST_FILE > /dev/null" \
    --command-name "pod5 (Python)" \
    "$POD5_BIN view $FIRST_FILE > /dev/null"

# ========================================
# Benchmark: merge (if multiple files)
# ========================================
if [ "$FILE_COUNT" -ge 2 ]; then
    echo ""
    echo "========================================"
    echo "Benchmark: merge ($FILE_COUNT files)"
    echo "========================================"

    for THREADS in 1 4; do
        echo ""
        echo "--- merge with $THREADS thread(s) ---"
        hyperfine \
            --warmup "$WARMUP" \
            --runs "$RUNS" \
            --prepare "rm -f $OUTPUT_DIR/merged_escapepod.pod5 $OUTPUT_DIR/merged_pod5.pod5 || true" \
            --export-json "$OUTPUT_DIR/merge_${THREADS}t.json" \
            --command-name "escapepod-rs (${THREADS}t)" \
            "$ESCAPEPOD_BIN merge $FILE_LIST -o $OUTPUT_DIR/merged_escapepod.pod5 -t $THREADS" \
            --command-name "pod5 (Python, ${THREADS}t)" \
            "$POD5_BIN merge $FILE_LIST -o $OUTPUT_DIR/merged_pod5.pod5 -t $THREADS"
    done

    # Verify merge outputs
    echo ""
    echo "Verifying merge outputs..."
    echo "escapepod-rs: $($ESCAPEPOD_BIN inspect summary $OUTPUT_DIR/merged_escapepod.pod5 2>/dev/null | grep 'Reads:' | head -1)"
    echo "pod5:       $($POD5_BIN inspect summary $OUTPUT_DIR/merged_pod5.pod5 2>/dev/null | grep -i 'read' | head -1)"
fi

# ========================================
# Benchmark: filter (need to create ID list first)
# ========================================
echo ""
echo "========================================"
echo "Benchmark: filter (10% of reads)"
echo "========================================"

# Extract 10% of read IDs for filtering
$ESCAPEPOD_BIN view $FIRST_FILE --include read_id 2>/dev/null | tail -n +2 | awk 'NR % 10 == 1' > "$OUTPUT_DIR/filter_ids.txt"
FILTER_COUNT=$(wc -l < "$OUTPUT_DIR/filter_ids.txt" | tr -d ' ')
echo "Filtering $FILTER_COUNT reads..."

if [ "$FILTER_COUNT" -gt 0 ]; then
    hyperfine \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --prepare "rm -f $OUTPUT_DIR/filtered_escapepod.pod5 $OUTPUT_DIR/filtered_pod5.pod5 || true" \
        --export-json "$OUTPUT_DIR/filter.json" \
        --command-name "escapepod-rs" \
        "$ESCAPEPOD_BIN filter $FIRST_FILE --ids $OUTPUT_DIR/filter_ids.txt -o $OUTPUT_DIR/filtered_escapepod.pod5" \
        --command-name "pod5 (Python)" \
        "$POD5_BIN filter $FIRST_FILE --ids $OUTPUT_DIR/filter_ids.txt -o $OUTPUT_DIR/filtered_pod5.pod5 --missing-ok"
else
    echo "Skipping filter benchmark - no reads found"
fi

# ========================================
# Benchmark: subset (split into 2 groups)
# ========================================
echo ""
echo "========================================"
echo "Benchmark: subset (split into 2 groups)"
echo "========================================"

# Generate subset CSV mappings (different formats for each tool)
# escapepod format: read_id,output
echo "read_id,output" > "$OUTPUT_DIR/subset_escapepod.csv"
$ESCAPEPOD_BIN view $FIRST_FILE --include read_id 2>/dev/null | tail -n +2 | \
    awk 'NR % 2 == 0 {print $0",group_a.pod5"} NR % 2 == 1 {print $0",group_b.pod5"}' \
    >> "$OUTPUT_DIR/subset_escapepod.csv"

# pod5 format: output,read_id
$ESCAPEPOD_BIN view $FIRST_FILE --include read_id 2>/dev/null | tail -n +2 | \
    awk 'NR % 2 == 0 {print "group_a.pod5,"$0} NR % 2 == 1 {print "group_b.pod5,"$0}' \
    > "$OUTPUT_DIR/subset_pod5.csv"

SUBSET_COUNT=$(tail -n +2 "$OUTPUT_DIR/subset_escapepod.csv" | wc -l | tr -d ' ')
echo "Subsetting $SUBSET_COUNT reads into 2 groups..."

if [ "$SUBSET_COUNT" -gt 0 ]; then
    hyperfine \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --prepare "rm -rf $OUTPUT_DIR/subset_escapepod/ $OUTPUT_DIR/subset_pod5/ || true; mkdir -p $OUTPUT_DIR/subset_escapepod $OUTPUT_DIR/subset_pod5" \
        --export-json "$OUTPUT_DIR/subset.json" \
        --command-name "escapepod-rs" \
        "$ESCAPEPOD_BIN subset $FIRST_FILE --csv $OUTPUT_DIR/subset_escapepod.csv -o $OUTPUT_DIR/subset_escapepod/ -f" \
        --command-name "pod5 (Python)" \
        "$POD5_BIN subset $FIRST_FILE --csv $OUTPUT_DIR/subset_pod5.csv -o $OUTPUT_DIR/subset_pod5/ -f -M"
else
    echo "Skipping subset benchmark - no reads found"
fi

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null

# Generate summary table
echo ""
echo "========================================"
echo "Summary (mean times)"
echo "========================================"
echo ""

for json_file in "$OUTPUT_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        benchmark_name=$(basename "$json_file" .json)
        echo "=== $benchmark_name ==="
        # Extract mean times using Python (jq alternative)
        python3 -c "
import json
with open('$json_file') as f:
    data = json.load(f)
for result in data['results']:
    name = result['command']
    mean = result['mean']
    stddev = result['stddev']
    print(f\"  {name}: {mean:.3f}s ± {stddev:.3f}s\")
" 2>/dev/null || echo "  (could not parse results)"
        echo ""
    fi
done

# Cleanup
rm -rf "$OUTPUT_DIR"/*.pod5 "$OUTPUT_DIR/filter_ids.txt" "$OUTPUT_DIR/subset_escapepod/" "$OUTPUT_DIR/subset_pod5/" "$OUTPUT_DIR/subset_escapepod.csv" "$OUTPUT_DIR/subset_pod5.csv"

#!/bin/bash
# Benchmark script comparing podfive-rs merge vs official pod5 merge
#
# Usage: ./benchmarks/merge_benchmark.sh <pod5_dir>
#
# The script expects a directory containing multiple POD5 files to merge.

set -e

POD5_DIR="${1:-.}"
PODFIVE_BIN="./target/release/podfive"
POD5_BIN="${HOME}/.venv/bin/pod5"
OUTPUT_DIR="/tmp/merge_benchmark"
WARMUP=1
RUNS=5

# Check dependencies
if ! command -v hyperfine &> /dev/null; then
    echo "Error: hyperfine not found. Install with: brew install hyperfine"
    exit 1
fi

if [ ! -f "$PODFIVE_BIN" ]; then
    echo "Error: podfive-rs binary not found. Run: cargo build --release"
    exit 1
fi

if [ ! -f "$POD5_BIN" ]; then
    echo "Error: pod5 not found. Install with: uv pip install pod5"
    exit 1
fi

# Find POD5 files
POD5_FILES=$(find "$POD5_DIR" -name "*.pod5" -type f | sort)
FILE_COUNT=$(echo "$POD5_FILES" | wc -l | tr -d ' ')

if [ "$FILE_COUNT" -lt 2 ]; then
    echo "Error: Need at least 2 POD5 files in $POD5_DIR"
    exit 1
fi

echo "Found $FILE_COUNT POD5 files in $POD5_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create file list for commands
FILE_LIST=$(echo "$POD5_FILES" | tr '\n' ' ')

# Show file sizes
echo "Input files:"
du -sh $FILE_LIST 2>/dev/null | head -10
echo ""

# Cleanup function
cleanup() {
    rm -f "$OUTPUT_DIR/merged_podfive.pod5" "$OUTPUT_DIR/merged_pod5.pod5"
}

# Run benchmarks
echo "Running merge benchmarks..."
echo "  - Warmup runs: $WARMUP"
echo "  - Benchmark runs: $RUNS"
echo ""

hyperfine \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --prepare "rm -f $OUTPUT_DIR/merged_podfive.pod5 $OUTPUT_DIR/merged_pod5.pod5" \
    --export-markdown "$OUTPUT_DIR/results.md" \
    --export-json "$OUTPUT_DIR/results.json" \
    --command-name "podfive-rs" \
    "$PODFIVE_BIN merge $FILE_LIST -o $OUTPUT_DIR/merged_podfive.pod5" \
    --command-name "pod5 (Python)" \
    "$POD5_BIN merge $FILE_LIST -o $OUTPUT_DIR/merged_pod5.pod5"

echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/results.md"
echo "  - $OUTPUT_DIR/results.json"

# Verify outputs are valid
echo ""
echo "Verifying output files..."
echo "podfive-rs output:"
$PODFIVE_BIN inspect summary "$OUTPUT_DIR/merged_podfive.pod5" 2>/dev/null | grep -E "^(Reads|File):"

echo ""
echo "pod5 output:"
$POD5_BIN inspect summary "$OUTPUT_DIR/merged_pod5.pod5" 2>/dev/null | head -5

cleanup

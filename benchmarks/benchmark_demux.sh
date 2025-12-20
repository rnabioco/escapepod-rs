#!/bin/bash
# Benchmark comparing escapepod demux vs ADAPTed and WarpDemuX
#
# Usage: ./benchmarks/benchmark_demux.sh [pod5_file]
#
# Prerequisites:
#   - cargo build --release
#   - WarpDemuX venv at ext/WarpDemuX/.venv with adapted and warpdemux installed
#   - hyperfine installed (brew install hyperfine)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default test file
POD5_FILE="${1:-$PROJECT_ROOT/ext/WarpDemuX/test_data/demux/4000_rna004.pod5}"

# Tool paths
ESCAPEPOD_BIN="$PROJECT_ROOT/target/release/escapepod"
WARPDEMUX_VENV="$PROJECT_ROOT/ext/WarpDemuX/.venv"
ADAPTED_BIN="$WARPDEMUX_VENV/bin/adapted"
WARPDEMUX_BIN="$WARPDEMUX_VENV/bin/warpdemux"

OUTPUT_DIR="/tmp/demux_benchmark"
WARMUP=1
RUNS=3

echo "========================================"
echo "Demux Benchmark Suite"
echo "========================================"
echo ""
echo "Input file: $POD5_FILE"
echo "File size: $(du -h "$POD5_FILE" | cut -f1)"
echo ""

# Check dependencies
if ! command -v hyperfine &> /dev/null; then
    echo "Error: hyperfine not found. Install with: brew install hyperfine"
    exit 1
fi

if [ ! -f "$ESCAPEPOD_BIN" ]; then
    echo "Error: escapepod binary not found. Run: cargo build --release"
    exit 1
fi

if [ ! -f "$ADAPTED_BIN" ]; then
    echo "Error: adapted not found in WarpDemuX venv"
    echo "Run: cd ext/WarpDemuX && uv venv .venv --python 3.11 && uv pip install -e . -e ../ADAPTed"
    exit 1
fi

if [ ! -f "$POD5_FILE" ]; then
    echo "Error: Test file not found: $POD5_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get read count for context
echo "Getting read count..."
READ_COUNT=$("$ESCAPEPOD_BIN" view "$POD5_FILE" --include read_id 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
echo "Reads in file: $READ_COUNT"
echo ""

# ========================================
# Benchmark: Adapter Detection
# ========================================
echo "========================================"
echo "Benchmark 1: Adapter Detection"
echo "========================================"
echo ""
echo "Comparing:"
echo "  - escapepod demux detect (LLR-based)"
echo "  - adapted detect (LLR + CNN + fallback)"
echo ""

# Clean up before benchmark
rm -rf "$OUTPUT_DIR/escapepod_detect" "$OUTPUT_DIR/adapted_detect"
mkdir -p "$OUTPUT_DIR/escapepod_detect" "$OUTPUT_DIR/adapted_detect"

hyperfine \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --export-json "$OUTPUT_DIR/detect_benchmark.json" \
    --command-name "escapepod demux detect" \
    "$ESCAPEPOD_BIN demux detect $POD5_FILE -o $OUTPUT_DIR/escapepod_detect/boundaries.csv -j 4" \
    --command-name "adapted detect (LLR)" \
    "$ADAPTED_BIN detect -i $POD5_FILE -o $OUTPUT_DIR/adapted_detect --chemistry RNA004 -j 4 2>/dev/null"

echo ""
echo "Output comparison:"
echo "  escapepod: $(wc -l < "$OUTPUT_DIR/escapepod_detect/boundaries.csv") lines"
ADAPTED_OUT=$(find "$OUTPUT_DIR/adapted_detect" -name "detected_boundaries*.csv" | head -1)
if [ -f "$ADAPTED_OUT" ]; then
    echo "  adapted:   $(wc -l < "$ADAPTED_OUT") lines"
fi

# ========================================
# Benchmark: Full Demux Pipeline (WarpDemuX only)
# ========================================
echo ""
echo "========================================"
echo "Benchmark 2: Full Demux Pipeline"
echo "========================================"
echo ""
echo "Comparing WarpDemuX full pipeline (detect + fingerprint + classify)"
echo "Note: escapepod demux requires separate steps, timing total workflow"
echo ""

# Clean up
rm -rf "$OUTPUT_DIR/warpdemux_out" "$OUTPUT_DIR/escapepod_demux"
mkdir -p "$OUTPUT_DIR/warpdemux_out" "$OUTPUT_DIR/escapepod_demux"

# WarpDemuX full pipeline
echo "Running WarpDemuX demux..."
time_start=$(date +%s.%N)
"$WARPDEMUX_BIN" demux "$POD5_FILE" -o "$OUTPUT_DIR/warpdemux_out" -m WDX4 -j 4 --save_boundaries 2>/dev/null || true
time_end=$(date +%s.%N)
warpdemux_time=$(echo "$time_end - $time_start" | bc)
echo "WarpDemuX time: ${warpdemux_time}s"

# Escapepod multi-step pipeline
echo ""
echo "Running escapepod demux pipeline..."
time_start=$(date +%s.%N)

# Step 1: Detect
"$ESCAPEPOD_BIN" demux detect "$POD5_FILE" -o "$OUTPUT_DIR/escapepod_demux/boundaries.csv" -j 4

# Step 2: Fingerprint
"$ESCAPEPOD_BIN" demux fingerprint "$POD5_FILE" \
    --boundaries "$OUTPUT_DIR/escapepod_demux/boundaries.csv" \
    -o "$OUTPUT_DIR/escapepod_demux/fingerprints.csv" -j 4

time_end=$(date +%s.%N)
escapepod_time=$(echo "$time_end - $time_start" | bc)
echo "Escapepod time (detect + fingerprint): ${escapepod_time}s"

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "Benchmark Summary"
echo "========================================"
echo ""
echo "Test file: $POD5_FILE"
echo "Reads: $READ_COUNT"
echo ""

# Parse hyperfine results
if [ -f "$OUTPUT_DIR/detect_benchmark.json" ]; then
    echo "=== Adapter Detection (hyperfine) ==="
    python3 -c "
import json
with open('$OUTPUT_DIR/detect_benchmark.json') as f:
    data = json.load(f)
for result in data['results']:
    name = result['command']
    mean = result['mean']
    stddev = result['stddev']
    print(f'  {name}: {mean:.3f}s ± {stddev:.3f}s')
" 2>/dev/null || echo "  (could not parse results)"
fi

echo ""
echo "=== Full Pipeline Timing ==="
echo "  WarpDemuX (demux):              ${warpdemux_time:-N/A}s"
echo "  Escapepod (detect+fingerprint): ${escapepod_time:-N/A}s"

echo ""
echo "Results saved to: $OUTPUT_DIR/"

# Cleanup large files
rm -rf "$OUTPUT_DIR/warpdemux_out" "$OUTPUT_DIR/adapted_detect" "$OUTPUT_DIR/escapepod_detect"

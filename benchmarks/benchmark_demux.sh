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
# Benchmark: Classification Accuracy
# ========================================
echo ""
echo "========================================"
echo "Benchmark 3: Classification Accuracy"
echo "========================================"
echo ""
echo "Testing escapepod demux accuracy against gold standard"
echo "Using RNA004 test data with 5 barcodes (1000 reads total)"
echo ""

ACCURACY_DATA_DIR="$PROJECT_ROOT/ext/WarpDemuX/test_data/live_balancing"
ACCURACY_OUTPUT_DIR="$OUTPUT_DIR/accuracy_test"

# Check if accuracy test data exists
if [ -d "$ACCURACY_DATA_DIR" ] && [ -f "$ACCURACY_DATA_DIR/small_pod5_0.pod5" ]; then
    mkdir -p "$ACCURACY_OUTPUT_DIR"

    # Step 1: Create assignments CSV from gold standard
    echo "Creating assignments from gold standard..."
    export ACCURACY_DATA_DIR ACCURACY_OUTPUT_DIR
    python3 << 'PYEOF'
import os

data_dir = os.environ['ACCURACY_DATA_DIR']
output_dir = os.environ['ACCURACY_OUTPUT_DIR']

with open(f"{output_dir}/assignments.csv", 'w') as out:
    out.write("read_id,barcode,pod5_file\n")
    for i in range(5):
        barcode = f"BC0{i}"
        pod5_file = f"{data_dir}/small_pod5_{i}.pod5"
        with open(f"{data_dir}/read_ids{i}.txt") as f:
            for line in f:
                read_id = line.strip()
                if read_id:
                    out.write(f"{read_id},{barcode},{pod5_file}\n")
print("Created assignments.csv")
PYEOF

    # Step 2: Train KNN model
    echo "Training KNN model..."
    "$ESCAPEPOD_BIN" demux train \
        --assignments "$ACCURACY_OUTPUT_DIR/assignments.csv" \
        -o "$ACCURACY_OUTPUT_DIR/knn_model.json" \
        --knn -j 4 2>/dev/null

    # Step 3-5: Time the full pipeline (detect + fingerprint + classify)
    echo "Running timed classification pipeline..."
    accuracy_time_start=$(date +%s.%N)

    # Step 3: Detect adapter boundaries
    "$ESCAPEPOD_BIN" demux detect \
        "$ACCURACY_DATA_DIR/small_pod5_0.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_1.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_2.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_3.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_4.pod5" \
        -o "$ACCURACY_OUTPUT_DIR/boundaries.csv" -j 4 2>/dev/null

    # Step 4: Extract fingerprints
    "$ESCAPEPOD_BIN" demux fingerprint \
        "$ACCURACY_DATA_DIR/small_pod5_0.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_1.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_2.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_3.pod5" \
        "$ACCURACY_DATA_DIR/small_pod5_4.pod5" \
        --boundaries "$ACCURACY_OUTPUT_DIR/boundaries.csv" \
        -o "$ACCURACY_OUTPUT_DIR/fingerprints.csv" -j 4 2>/dev/null

    # Step 5: Classify
    "$ESCAPEPOD_BIN" demux classify \
        "$ACCURACY_OUTPUT_DIR/fingerprints.csv" \
        --model "$ACCURACY_OUTPUT_DIR/knn_model.json" \
        -o "$ACCURACY_OUTPUT_DIR/classifications.csv" 2>/dev/null

    accuracy_time_end=$(date +%s.%N)
    accuracy_pipeline_time=$(echo "$accuracy_time_end - $accuracy_time_start" | bc)
    accuracy_read_count=$(wc -l < "$ACCURACY_OUTPUT_DIR/classifications.csv" | tr -d ' ')
    accuracy_read_count=$((accuracy_read_count - 1))  # subtract header
    accuracy_throughput=$(echo "scale=0; $accuracy_read_count / $accuracy_pipeline_time" | bc)

    echo ""
    echo "=== Classification Timing ==="
    echo "  Pipeline time: ${accuracy_pipeline_time}s"
    echo "  Reads classified: ${accuracy_read_count}"
    echo "  Throughput: ~${accuracy_throughput} reads/sec"

    # Step 6: Calculate accuracy
    echo ""
    echo "=== Classification Results ==="
    python3 << 'PYEOF'
import os

data_dir = os.environ['ACCURACY_DATA_DIR']
output_dir = os.environ['ACCURACY_OUTPUT_DIR']

# Load ground truth
ground_truth = {}
for i in range(5):
    barcode = f"BC0{i}"
    with open(f"{data_dir}/read_ids{i}.txt") as f:
        for line in f:
            read_id = line.strip()
            if read_id:
                ground_truth[read_id] = barcode

# Load classifications
correct = 0
incorrect = 0
confident_correct = 0
confident_incorrect = 0
barcode_counts = {f"BC0{i}": {"correct": 0, "total": 0} for i in range(5)}

with open(f"{output_dir}/classifications.csv") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(',')
        read_id = parts[0]
        predicted = parts[1]
        is_confident = parts[5].lower() == 'true'

        if read_id in ground_truth:
            actual = ground_truth[read_id]
            barcode_counts[actual]["total"] += 1
            if predicted == actual:
                correct += 1
                barcode_counts[actual]["correct"] += 1
                if is_confident:
                    confident_correct += 1
            else:
                incorrect += 1
                if is_confident:
                    confident_incorrect += 1

total = correct + incorrect
accuracy = correct / total * 100 if total > 0 else 0

print(f"  Total classified: {total}")
print(f"  Correct: {correct}")
print(f"  Incorrect: {incorrect}")
print(f"  Overall accuracy: {accuracy:.1f}%")
print()
print("  Per-barcode accuracy:")
for bc in sorted(barcode_counts.keys()):
    counts = barcode_counts[bc]
    if counts["total"] > 0:
        acc = counts["correct"] / counts["total"] * 100
        print(f"    {bc}: {counts['correct']}/{counts['total']} = {acc:.1f}%")
PYEOF

else
    echo "  Skipping accuracy test (test data not found at $ACCURACY_DATA_DIR)"
fi

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

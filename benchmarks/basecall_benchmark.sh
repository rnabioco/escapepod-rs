#!/bin/bash
# Benchmark the impact of signal downsampling on basecalling accuracy.
#
# Usage: ./basecall_benchmark.sh <pod5_file> <reference.fa> <output_dir> [factors]
#
# Arguments:
#   pod5_file    - Input POD5 file to benchmark
#   reference.fa - Reference genome for alignment (FASTA format)
#   output_dir   - Directory for results (will be created if needed)
#   factors      - Downsample factors, space-separated (default: "2 4")
#
# Requirements:
#   - dorado (in PATH)
#   - podfive (in PATH or ./target/release/)
#   - samtools (in PATH)
#   - Python 3 with pysam, pandas
#
# Example:
#   ./benchmarks/basecall_benchmark.sh data/test.pod5 ref/genome.fa results/
#   ./benchmarks/basecall_benchmark.sh data/test.pod5 ref/genome.fa results/ "2 4 8"

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

usage() {
    echo "Usage: $0 <pod5_file> <reference.fa> <output_dir> [factors]"
    echo ""
    echo "Benchmark signal downsampling impact on basecalling accuracy."
    echo ""
    echo "Arguments:"
    echo "  pod5_file    - Input POD5 file to benchmark"
    echo "  reference.fa - Reference genome for alignment"
    echo "  output_dir   - Directory for results"
    echo "  factors      - Downsample factors (default: \"2 4\")"
    echo ""
    echo "Example:"
    echo "  $0 data/test.pod5 ref/genome.fa results/"
    exit 1
}

# Check arguments
if [[ $# -lt 3 ]]; then
    usage
fi

POD5_FILE="$1"
REFERENCE="$2"
OUTPUT_DIR="$3"
FACTORS="${4:-2 4}"
MODELS="fast hac sup"

# Validate inputs
if [[ ! -f "$POD5_FILE" ]]; then
    log_error "POD5 file not found: $POD5_FILE"
    exit 1
fi

if [[ ! -f "$REFERENCE" ]]; then
    log_error "Reference file not found: $REFERENCE"
    exit 1
fi

# Find podfive binary
PODFIVE=""
if command -v podfive &> /dev/null; then
    PODFIVE="podfive"
elif [[ -x "./target/release/podfive" ]]; then
    PODFIVE="./target/release/podfive"
else
    log_error "podfive not found. Build with 'cargo build --release' or add to PATH."
    exit 1
fi

# Check other dependencies
for cmd in dorado samtools python3; do
    if ! command -v "$cmd" &> /dev/null; then
        log_error "$cmd not found in PATH"
        exit 1
    fi
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_info "Benchmark Configuration:"
log_info "  POD5 file: $POD5_FILE"
log_info "  Reference: $REFERENCE"
log_info "  Output dir: $OUTPUT_DIR"
log_info "  Downsample factors: $FACTORS"
log_info "  Models: $MODELS"
log_info "  podfive: $PODFIVE"
echo ""

# Step 1: Create archived (downsampled) POD5 files
log_info "Creating downsampled POD5 files..."
for factor in $FACTORS; do
    archived="$OUTPUT_DIR/archived_${factor}x.pod5"
    if [[ -f "$archived" ]]; then
        log_warn "Skipping existing: $archived"
    else
        log_info "  Creating ${factor}x downsampled file..."
        "$PODFIVE" archive "$POD5_FILE" -o "$archived" --factor "$factor"
    fi
done
echo ""

# Step 2: Basecall all variants
log_info "Running basecalling (this may take a while)..."

for model in $MODELS; do
    log_info "=== Testing $model model ==="

    # Basecall original
    original_bam="$OUTPUT_DIR/original_${model}.bam"
    if [[ -f "$original_bam" ]]; then
        log_warn "Skipping existing: $original_bam"
    else
        log_info "  Basecalling original with $model..."
        dorado basecaller "$model" "$POD5_FILE" --reference "$REFERENCE" \
            > "$original_bam" 2>"$OUTPUT_DIR/original_${model}.log"
        samtools index "$original_bam"
    fi

    # Basecall downsampled versions
    for factor in $FACTORS; do
        archived="$OUTPUT_DIR/archived_${factor}x.pod5"
        archived_bam="$OUTPUT_DIR/archived_${factor}x_${model}.bam"

        if [[ -f "$archived_bam" ]]; then
            log_warn "Skipping existing: $archived_bam"
        else
            log_info "  Basecalling ${factor}x downsampled with $model..."
            dorado basecaller "$model" "$archived" --reference "$REFERENCE" \
                > "$archived_bam" 2>"$OUTPUT_DIR/archived_${factor}x_${model}.log"
            samtools index "$archived_bam"
        fi
    done
done
echo ""

# Step 3: Run analysis
log_info "Analyzing basecalling quality..."
SCRIPT_DIR="$(dirname "$0")"
python3 "$SCRIPT_DIR/analyze_basecall_quality.py" "$OUTPUT_DIR" "$FACTORS" "$MODELS"

log_success "Benchmark complete! Results in: $OUTPUT_DIR"
log_info "  Summary: $OUTPUT_DIR/quality_summary.tsv"
log_info "  Details: $OUTPUT_DIR/quality_metrics.json"

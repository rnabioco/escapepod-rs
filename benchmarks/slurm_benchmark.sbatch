#!/bin/bash
#SBATCH --job-name=escapepod-basecall-bench
#SBATCH --partition=aa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# Basecall benchmark for escapepod archive command on SLURM cluster
#
# Usage:
#   sbatch slurm_benchmark.sbatch <pod5_file> <reference.fa> <output_dir> [factors]
#
# Example:
#   sbatch slurm_benchmark.sbatch /path/to/data.pod5 /path/to/ref.fa /scratch/$USER/results "2 4 8"
#
# Environment:
#   Expects pixi environment with dorado, samtools, and python (pysam, pandas)
#   Adjust PIXI_ENV below or comment out if tools are already in PATH

set -euo pipefail

# ==============================================================================
# Configuration - adjust these for your environment
# ==============================================================================

# Pixi environment name (set to empty string if not using pixi)
PIXI_ENV="${PIXI_ENV:-}"

# Path to escapepod binary (will auto-detect if in PATH or built locally)
escapepod_BIN="${escapepod_BIN:-}"

# ==============================================================================
# Parse arguments
# ==============================================================================

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 <pod5_file> <reference.fa> <output_dir> [factors]"
  echo ""
  echo "Arguments:"
  echo "  pod5_file    - Input POD5 file to benchmark"
  echo "  reference.fa - Reference genome for alignment"
  echo "  output_dir   - Directory for results"
  echo "  factors      - Downsample factors (default: \"2 4\")"
  exit 1
fi

POD5_FILE="$1"
REFERENCE="$2"
OUTPUT_DIR="$3"
FACTORS="${4:-2 4}"

# ==============================================================================
# Setup environment
# ==============================================================================

echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Start Time:    $(date)"
echo "=============================================="
echo ""

# Activate pixi environment if specified
if [[ -n "$PIXI_ENV" ]]; then
  echo "Activating pixi environment: $PIXI_ENV"
  eval "$(pixi shell-hook -s bash -e "$PIXI_ENV")"
fi

# Find escapepod binary
if [[ -n "$escapepod_BIN" ]]; then
  escapepod="$escapepod_BIN"
elif command -v escapepod &>/dev/null; then
  escapepod="escapepod"
elif [[ -x "./target/release/escapepod" ]]; then
  escapepod="./target/release/escapepod"
else
  echo "ERROR: escapepod not found. Set escapepod_BIN or add to PATH."
  exit 1
fi

# Verify dependencies
echo "Checking dependencies..."
for cmd in dorado samtools python3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: $cmd not found in PATH"
    exit 1
  fi
done

echo "  escapepod:  $escapepod"
echo "  dorado:   $(command -v dorado)"
echo "  samtools: $(command -v samtools)"
echo "  python3:  $(command -v python3)"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "  (nvidia-smi not available)"
echo ""

# ==============================================================================
# Run benchmark
# ==============================================================================

echo "=============================================="
echo "Benchmark Configuration"
echo "=============================================="
echo "POD5 file:     $POD5_FILE"
echo "Reference:     $REFERENCE"
echo "Output dir:    $OUTPUT_DIR"
echo "Factors:       $FACTORS"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the benchmark script
exec bash "$SCRIPT_DIR/basecall_benchmark.sh" "$POD5_FILE" "$REFERENCE" "$OUTPUT_DIR" "$FACTORS"

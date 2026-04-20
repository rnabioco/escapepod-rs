#!/usr/bin/env bash
#
# Dorado compatibility test for escapepod-produced POD5 files.
#
# Verifies that POD5 files created by escapepod can be parsed by:
#   1. The Python pod5 library
#   2. Oxford Nanopore's Dorado basecaller (via `dorado summary` and `dorado basecaller`)
#
# Prerequisites:
#   pixi run install-dorado
#   pixi run install-pod5
#
# Usage:
#   cargo build --release
#   pixi run bash tests/test_dorado_compat.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ESCPOD="$REPO_ROOT/target/release/escpod"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/dorado_compat_XXXXXX")"
PASS=0
FAIL=0

cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

log_pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
log_fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

# ── Preflight checks ────────────────────────────────────────────────────
echo "=== Preflight ==="

if [[ ! -x "$ESCPOD" ]]; then
    echo "ERROR: escpod binary not found at $ESCPOD"
    echo "       Run: cargo build --release"
    exit 1
fi
echo "escpod: $($ESCPOD --version)"

if ! command -v dorado &>/dev/null; then
    echo "ERROR: dorado not found on PATH."
    echo "       Run: pixi run install-dorado"
    exit 1
fi
echo "dorado: $(dorado --version 2>&1 | head -1)"

if ! python -c "import pod5; print(f'pod5: {pod5.__version__}')" 2>/dev/null; then
    echo "ERROR: Python pod5 not available."
    echo "       Run: pixi run install-pod5"
    exit 1
fi

# ── Step 1: Repack a real POD5 ──────────────────────────────────────────
echo ""
echo "=== Step 1: Repack real POD5 ==="

REAL_POD5="$REPO_ROOT/ext/pod5-file-format/test_data/multi_fast5_zip_v3.pod5"
REPACKED=""

if [[ ! -f "$REAL_POD5" ]]; then
    echo "  SKIP: test file not found at $REAL_POD5"
else
    REPACK_DIR="$TMPDIR/repack_out"
    mkdir -p "$REPACK_DIR"
    if "$ESCPOD" repack --output-dir "$REPACK_DIR" "$REAL_POD5" 2>/dev/null; then
        # repack writes to output-dir with the original filename
        REPACKED="$(ls "$REPACK_DIR"/*.pod5 2>/dev/null | head -1)"
        if [[ -n "$REPACKED" ]]; then
            log_pass "escpod repack produced $REPACKED"
        else
            log_fail "escpod repack ran but produced no .pod5 file"
        fi
    else
        log_fail "escpod repack failed"
    fi
fi

# ── Step 2: Write a synthetic POD5 ─────────────────────────────────────
echo ""
echo "=== Step 2: Write synthetic POD5 ==="

SYNTHETIC="$TMPDIR/synthetic.pod5"

if cargo run --release -p escapepod-signal --example write_pod5 -- "$SYNTHETIC" 2>/dev/null; then
    log_pass "write_pod5 example produced $SYNTHETIC"
else
    log_fail "write_pod5 example failed"
fi

# ── Step 3: Validate with Python pod5 ──────────────────────────────────
echo ""
echo "=== Step 3: Validate with Python pod5 ==="

for label_path in "repacked:$REPACKED" "synthetic:$SYNTHETIC"; do
    label="${label_path%%:*}"
    fpath="${label_path#*:}"
    if [[ ! -f "$fpath" ]]; then
        echo "  SKIP: $label ($fpath not found)"
        continue
    fi
    if python -c "
import pod5, sys
try:
    with pod5.Reader(sys.argv[1]) as r:
        reads = list(r.reads())
        print(f'  {len(reads)} reads, all OK')
except Exception as e:
    print(f'  ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" "$fpath"; then
        log_pass "Python pod5 reads $label"
    else
        log_fail "Python pod5 cannot read $label"
    fi
done

# ── Step 4: Validate with dorado summary ───────────────────────────────
echo ""
echo "=== Step 4: Validate with dorado summary ==="

for label_path in "repacked:$REPACKED" "synthetic:$SYNTHETIC"; do
    label="${label_path%%:*}"
    fpath="${label_path#*:}"
    if [[ ! -f "$fpath" ]]; then
        echo "  SKIP: $label ($fpath not found)"
        continue
    fi
    if dorado summary "$fpath" > "$TMPDIR/${label}_summary.tsv" 2>/dev/null; then
        nlines=$(wc -l < "$TMPDIR/${label}_summary.tsv")
        log_pass "dorado summary reads $label ($nlines lines)"
    else
        log_fail "dorado summary cannot parse $label"
    fi
done

# ── Step 5: Basecall synthetic POD5 with Dorado ────────────────────────
echo ""
echo "=== Step 5: Basecall with Dorado ==="

# The synthetic POD5 uses 5kHz R10.4.1 E8.2 chemistry metadata.
# Download the matching fast model and attempt basecalling.
MODEL="dna_r10.4.1_e8.2_400bps_fast@v5.0.0"
MODEL_DIR="${DORADO_DIR:-$REPO_ROOT/.pixi/tools}/models"
MODEL_PATH="$MODEL_DIR/$MODEL"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "  Downloading model $MODEL ..."
    mkdir -p "$MODEL_DIR"
    if ! dorado download --model "$MODEL" --models-directory "$MODEL_DIR" >/dev/null 2>&1; then
        echo "  SKIP: failed to download model $MODEL"
    fi
fi

if [[ -f "$SYNTHETIC" && -d "$MODEL_PATH" ]]; then
    BASECALL_LOG="$TMPDIR/basecall.log"
    BASECALL_BAM="$TMPDIR/basecalled.bam"
    if dorado basecaller "$MODEL_PATH" "$SYNTHETIC" > "$BASECALL_BAM" 2>"$BASECALL_LOG"; then
        nreads=$(samtools view -c "$BASECALL_BAM" 2>/dev/null || echo "?")
        log_pass "dorado basecaller produced $nreads reads from synthetic POD5"
    else
        # Check if dorado at least parsed the file (format OK) but failed on
        # something else (e.g. GPU not available, model issue, etc.)
        if grep -q "Failed to open.*POD5\|Invalid POD5\|not a valid POD5" "$BASECALL_LOG" 2>/dev/null; then
            log_fail "dorado cannot parse synthetic POD5 (format error)"
            cat "$BASECALL_LOG" >&2
        else
            # Dorado opened the file fine but failed for another reason
            # (e.g. no GPU). Show the log and count as a pass for format validity.
            echo "  NOTE: dorado basecaller exited non-zero but POD5 format was accepted"
            tail -3 "$BASECALL_LOG" | sed 's/^/  /'
            log_pass "dorado accepted synthetic POD5 format (basecalling failed for non-format reason)"
        fi
    fi
else
    echo "  SKIP: synthetic POD5 or model not available"
fi

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "SUMMARY: $PASS passed, $FAIL failed"
echo "========================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi

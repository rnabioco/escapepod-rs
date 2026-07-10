#!/bin/bash
# Build reproducible size-tiered POD5 inputs for the demux benchmark matrix.
#
# Both escpod and WarpDemuX must run on byte-identical inputs, and the real
# runs under the aars-in-vitro project are 40-66 GB each (millions of reads) —
# far too big to feed WarpDemuX whole. This script carves fixed-size tiers
# (default 4k / 25k / 100k reads) out of one real run by taking the first N
# read IDs (deterministic) and `escpod filter`-ing them into a small POD5.
#
# The bundled 4000-read file ships with the repo and is always included as the
# smallest, runs-anywhere tier.
#
# Usage:
#   ./benchmarks/make_demux_inputs.sh [--src POD5] [--out-dir DIR] \
#       [--tiers "4000 25000 100000"] [--escpod BIN]
#
# Writes <out-dir>/inputs.manifest with `name<TAB>n_reads<TAB>path` lines that
# benchmark_demux_matrix.sh consumes.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SRC=""
OUT_DIR="/tmp/demux_inputs"
TIERS="4000 25000 100000"
ESCPOD_BIN="$PROJECT_ROOT/target/release/escpod"
# A real multi-barcode run (large). Override with --src. Resolved lazily so the
# default path only matters when tiers beyond the bundled one are requested.
DEFAULT_SRC="$HOME/devel/rnabioco/2026-aars-in-vitro/results/demux/pod5/AlaRS_all20_b4"
BUNDLED="$PROJECT_ROOT/ext/WarpDemuX/test_data/demux/4000_rna004.pod5"

while [ $# -gt 0 ]; do
    case "$1" in
        --src)       SRC="$2"; shift ;;
        --src=*)     SRC="${1#*=}" ;;
        --out-dir)   OUT_DIR="$2"; shift ;;
        --out-dir=*) OUT_DIR="${1#*=}" ;;
        --tiers)     TIERS="$2"; shift ;;
        --tiers=*)   TIERS="${1#*=}" ;;
        --escpod)    ESCPOD_BIN="$2"; shift ;;
        --escpod=*)  ESCPOD_BIN="${1#*=}" ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ ! -x "$ESCPOD_BIN" ]; then
    echo "error: escpod binary not found/executable: $ESCPOD_BIN" >&2
    echo "       build with: cargo build --release -p escapepod-cli --features 'demux train cnn-detect'" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/inputs.manifest"
: > "$MANIFEST"

# count_reads <pod5>  -> number of data rows in `escpod view` (minus header)
count_reads() {
    "$ESCPOD_BIN" view "$1" 2>/dev/null | tail -n +2 | wc -l | tr -d ' '
}

# Always include the bundled tier.
if [ -f "$BUNDLED" ]; then
    n=$(count_reads "$BUNDLED")
    printf 'bundled_4k\t%s\t%s\n' "$n" "$BUNDLED" >> "$MANIFEST"
    echo ">>> bundled_4k: $n reads  ($BUNDLED)"
fi

# Build the larger tiers from a real run, if requested.
NEED_SRC=0
for N in $TIERS; do
    # The bundled file already covers ~4000; skip a redundant tier near it.
    [ "$N" -le 5000 ] && continue
    NEED_SRC=1
done

if [ "$NEED_SRC" -eq 1 ]; then
    : "${SRC:=$DEFAULT_SRC}"
    # Resolve a single source POD5 (the dirs hold one big .pod5 each).
    if [ -d "$SRC" ]; then
        SRC_POD5="$(find -L "$SRC" -name '*.pod5' -type f | sort | head -1)"
    else
        SRC_POD5="$SRC"
    fi
    if [ -z "$SRC_POD5" ] || [ ! -f "$SRC_POD5" ]; then
        echo "error: no source POD5 found for larger tiers (--src '$SRC')." >&2
        echo "       point --src at a real run dir or .pod5 file." >&2
        exit 1
    fi
    echo ">>> source for large tiers: $SRC_POD5"

    # Build an ordered read-ID list once, then head -N per tier. Only the
    # largest requested tier's worth of IDs are needed, so cap the listing:
    # `head -n MAXN` closes the pipe and `escpod view` streams per-read and
    # stops early on SIGPIPE — no full scan of a 50 GB source file.
    MAXN=0
    for N in $TIERS; do [ "$N" -gt "$MAXN" ] && MAXN="$N"; done
    ALL_IDS="$OUT_DIR/.src_read_ids.txt"
    if [ "$(wc -l < "$ALL_IDS" 2>/dev/null || echo 0)" -lt "$MAXN" ]; then
        echo ">>> listing first $MAXN source read IDs (one-time)..."
        "$ESCPOD_BIN" view "$SRC_POD5" | tail -n +2 | head -n "$MAXN" | cut -f1 > "$ALL_IDS" || true
    fi
    TOTAL=$(wc -l < "$ALL_IDS" | tr -d ' ')
    echo "    listed $TOTAL source reads (cap $MAXN)"

    for N in $TIERS; do
        [ "$N" -le 5000 ] && continue
        tier="tier_$((N/1000))k"
        out="$OUT_DIR/${tier}.pod5"
        if [ "$N" -gt "$TOTAL" ]; then
            echo "    skip $tier: source only has $TOTAL reads (< $N)"
            continue
        fi
        if [ ! -f "$out" ]; then
            ids="$OUT_DIR/${tier}.ids.txt"
            head -n "$N" "$ALL_IDS" > "$ids"
            echo ">>> building $tier ($N reads) -> $out"
            "$ESCPOD_BIN" -q filter "$SRC_POD5" -i "$ids" -o "$out" -f
        fi
        n=$(count_reads "$out")
        printf '%s\t%s\t%s\n' "$tier" "$n" "$out" >> "$MANIFEST"
        echo "    $tier: $n reads"
    done
fi

echo ""
echo "Manifest -> $MANIFEST"
cat "$MANIFEST"

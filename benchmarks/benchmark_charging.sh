#!/bin/bash
# End-to-end `escpod classify` benchmark, and an A/B across escpod builds.
#
# The counterpart to `crates/escapepod-classify/benches/charging.rs`: that one
# measures the per-read feature path in isolation, this one measures the whole
# command on real input — BAM scan, POD5 signal lookup, features, scoring, BAM
# write. Between them they answer "did the classifier get slower", which in
# September 2026 could only be answered by building this harness from scratch.
#
# Usage:
#   benchmarks/benchmark_charging.sh --pod5 DIR --bam FILE --reference FA \
#       --model DIR [--bin PATH]... [--threads N] [--reps N] [--out DIR]
#
# Each --bin adds an ARM. Give it several to A/B released binaries against a
# local build:
#
#   benchmarks/benchmark_charging.sh --pod5 run/pod5 --bam s.bam \
#       --reference ref.fa --model bundle/ \
#       --bin /path/to/escpod-0.20.0 --bin ./target/release/escpod
#
# With no --bin it benchmarks ./target/release/escpod alone.
#
# Two rules this script exists to enforce, both learned the hard way:
#
#   1. ARMS ARE INTERLEAVED (A B C A B C), never grouped (A A B B C C). The
#      first touch of a POD5 set pages it in off shared storage; on a 12 GB set
#      that made arm 1 take 212 s against 57-68 s for every later arm, and CPU
#      time read 107 s against 211-270 s for identical work. Grouped arms hand
#      that entire effect to whichever version happens to run first.
#
#   2. BOTH wall and CPU are reported. Wall alone cannot tell a slower binary
#      from a busier node or a colder cache; CPU alone hides the I/O that
#      dominates a real run. When the two disagree, the run was I/O-bound and
#      the wall number belongs to the filesystem, not to escpod.
#
# Rep 1 is reported but should be READ AS A WARM-UP, not as a measurement: the
# cold-cache effect lands entirely on it. Compare arms within rep 2 and later.

set -uo pipefail

POD5=""
BAM=""
REFERENCE=""
MODEL=""
THREADS="${THREADS:-8}"
REPS="${REPS:-2}"
MIN_MAPQ="${MIN_MAPQ:-0}"
OUT="${OUT:-/tmp/escapepod_charging_benchmark}"
BINS=()

usage() {
    sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --pod5)      POD5="$2"; shift 2 ;;
        --bam)       BAM="$2"; shift 2 ;;
        --reference) REFERENCE="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --bin)       BINS+=("$2"); shift 2 ;;
        --threads)   THREADS="$2"; shift 2 ;;
        --reps)      REPS="$2"; shift 2 ;;
        --min-mapq)  MIN_MAPQ="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

for req in POD5 BAM REFERENCE MODEL; do
    if [ -z "${!req}" ]; then
        echo "Error: --${req,,} is required" >&2
        usage
    fi
done

if [ ${#BINS[@]} -eq 0 ]; then
    BINS=("./target/release/escpod")
fi

for bin in "${BINS[@]}"; do
    if [ ! -x "$bin" ]; then
        echo "Error: not an executable: $bin" >&2
        exit 1
    fi
done

# Node-local scratch for the output BAM: it is the size of the input and is
# discarded, so it must not land on shared storage beside the inputs.
SCRATCH="${TMPDIR:-/tmp}/escpod_charging_bench.$$"
mkdir -p "$OUT" "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "node:      $(hostname)"
echo "threads:   $THREADS"
echo "reps:      $REPS"
echo "pod5:      $POD5"
echo "bam:       $BAM"
echo "model:     $MODEL"
echo "arms:      ${#BINS[@]}"
# Label each arm by index AND version, never by basename: every released
# tarball unpacks to a binary called `escpod`, so a basename label makes the
# rows of an A/B indistinguishable — which is the one thing this script is for.
VERSIONS=()
for i in "${!BINS[@]}"; do
    v="$("${BINS[$i]}" --version 2>&1 | head -1)"
    VERSIONS+=("$v")
    echo "  arm$i  $v  <- ${BINS[$i]}"
done
echo

for rep in $(seq 1 "$REPS"); do
    for i in "${!BINS[@]}"; do
        bin="${BINS[$i]}"
        tag="arm${i}_rep${rep}"
        log="$OUT/$tag.log"
        tm="$OUT/$tag.time"

        /usr/bin/time -v -o "$tm" \
            "$bin" classify "$POD5" \
                --bam "$BAM" \
                --reference "$REFERENCE" \
                --model "$MODEL" \
                --output "$SCRATCH/$tag.bam" \
                --tsv "$SCRATCH/$tag.tsv" \
                --min-mapq "$MIN_MAPQ" \
                --threads "$THREADS" >"$log" 2>&1
        rc=$?

        wall=$(awk -F': ' '/Elapsed \(wall/ {print $NF}' "$tm")
        user=$(awk -F': ' '/User time/ {print $NF}' "$tm")
        sys=$(awk -F': ' '/System time/ {print $NF}' "$tm")
        rss=$(awk -F': ' '/Maximum resident set size/ {printf "%.1f", $NF/1048576}' "$tm")
        # The call count and the median probability together are the parity
        # check: two arms that disagree on either were not doing the same work,
        # and their times do not compare however clean the timing looks.
        calls=$(grep -oE '[0-9]+ reads classified; median P\(charged\) = [0-9.]+' "$log" | head -1)

        # Version AND basename: two builds of one commit (an allocator
        # feature, a wrapper setting an env var) report the same version.
        printf 'rep%-2s arm%-2s %-22s %-26s rc=%s wall=%s cpu=%.1fs rss=%sG  %s\n' \
            "$rep" "$i" "${VERSIONS[$i]}" "$(basename "$bin")" "$rc" "$wall" \
            "$(echo "$user + $sys" | bc)" "$rss" "${calls:-NO CALLS PARSED}"

        rm -f "$SCRATCH/$tag.bam" "$SCRATCH/$tag.tsv"
    done
done

echo
echo "logs and /usr/bin/time reports: $OUT"
echo "NOTE: rep 1 is a warm-up (cold page cache). Compare arms within rep 2+."

#!/bin/bash
# End-to-end `escpod align` benchmark across the kernel backends and devices.
#
# Runs the same alignment once per arm — a CPU backend (`ESCAPEPOD_ALIGN_BACKEND`
# caps the dispatch) crossed with a `--device` — interleaved and repeated, and
# reports wall, CPU, MaxRSS and reads/s per run. The DP's cost is data-independent — reads x references x
# lengths — so any real sample against any panel measures the right thing.
#
# Usage:
#   benchmarks/benchmark_align.sh --reads FILE --reference FA \
#       [--backend NAME]... [--device auto|cpu|gpu]... [--reps N] [--threads N] \
#       [--bin PATH] [-- EXTRA...]
#
# Default backends: avx512 avx2 scalar. Default device: none (the flag is not
# passed, i.e. `auto`). With `--device cpu --device gpu` every backend runs on
# both, so CPU vs GPU scoring is one call; under `gpu` the backend still does
# the tracebacks and any read the GPU leaves to the CPU, and the run is checked
# to have actually placed scoring on the GPU. A GPU arm needs a `gpu` build and
# a GPU allocation (`srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1`). Arguments
# after `--` go to escpod align.
#
# The same rules as benchmark_charging.sh: arms are INTERLEAVED (A B C A B C),
# the input is read once before the first arm so no arm pays to page it in,
# and CPU is reported beside wall. The output BAM goes to node-local scratch
# and is deleted after each run — it is the command's cost, not a result.

set -uo pipefail

reads="" reference="" reps=1 threads="" bin="./target/release/escpod"
backends=()
devices=()
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reads) reads="$2"; shift 2 ;;
        --reference) reference="$2"; shift 2 ;;
        --backend) backends+=("$2"); shift 2 ;;
        --device) devices+=("$2"); shift 2 ;;
        --reps) reps="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --bin) bin="$2"; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$reads" && -n "$reference" ]] || { echo "--reads and --reference are required" >&2; exit 2; }
[[ ${#backends[@]} -gt 0 ]] || backends=(avx512 avx2 scalar)
[[ ${#devices[@]} -gt 0 ]] || devices=("-")
threads="${threads:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/escpod-align-bench.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

echo "# $(hostname) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')" >&2
echo "# $("$bin" --version) threads=$threads reads=$reads" >&2
cat "$reads" > /dev/null  # page the input in once, before any arm

printf 'rep\tbackend\tdevice\treads\twall_s\tcpu_s\tmax_rss_mib\treads_per_s\n'
for rep in $(seq 1 "$reps"); do
    for be in "${backends[@]}"; do
    for dev in "${devices[@]}"; do
        log="$scratch/$be.$dev.$rep.log"
        dev_args=()
        [[ "$dev" == "-" ]] || dev_args=(--device "$dev")
        ESCAPEPOD_ALIGN_BACKEND="$be" /usr/bin/time -v \
            "$bin" align "$reads" -r "$reference" -o "$scratch/out.bam" -t "$threads" \
            "${dev_args[@]}" "${extra[@]}" \
            2> "$log" || { cat "$log" >&2; exit 1; }
        grep -q "kernel $be " "$log" || { echo "backend $be did not run:" >&2; grep kernel "$log" >&2; }
        if [[ "$dev" == "gpu" ]]; then
            grep -q "on GPU" "$log" || { echo "--device gpu did not score on the GPU:" >&2; cat "$log" >&2; exit 1; }
            grep "GPU scored" "$log" >&2
        fi
        n=$(grep -oP '[0-9]+(?= reads: )' "$log" | tail -1)
        wall=$(awk -F': ' '/Elapsed \(wall clock\)/ {n=split($2,a,":"); s=0; for(i=1;i<=n;i++) s=s*60+a[i]; print s}' "$log")
        cpu=$(awk -F': ' '/User time/ {u=$2} /System time/ {s=$2} END {print u+s}' "$log")
        rss=$(awk -F': ' '/Maximum resident/ {printf "%.0f", $2/1024}' "$log")
        printf '%s\t%s\t%s\t%s\t%.1f\t%.1f\t%s\t%.0f\n' "$rep" "$be" "$dev" "$n" "$wall" "$cpu" "$rss" "$(echo "$n / $wall" | bc -l)"
        rm -f "$scratch/out.bam"
    done
    done
done

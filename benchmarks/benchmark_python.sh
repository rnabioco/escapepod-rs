#!/bin/bash
# Benchmark the escapepod Python library against the official pod5 library.
#
# Wraps benchmark_python.py in the `python-test` pixi env (which has both the
# freshly-built `escapepod` extension and `pod5` installed). Unlike benchmark.sh
# — which times the escpod vs pod5 *CLIs* with hyperfine — this measures the
# two *Python libraries* in-process.
#
# Usage:
#   ./benchmarks/benchmark_python.sh <pod5_file_or_dir> [extra python args]
#
# On a large input, run under SLURM so it isn't on the 2-core login node:
#   srun -p rna -c 32 --mem=32G ./benchmarks/benchmark_python.sh big.pod5 --limit 20000
#
# IMPORTANT: rebuild the extension first if you changed the Rust bindings:
#   pixi run -e python-test maturin develop --release \
#       --manifest-path crates/escapepod-python/Cargo.toml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/benchmarks/benchmark_python.py"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <pod5_file_or_dir> [--runs N] [--warmup N] [--limit N] [--json out.json]" >&2
    exit 1
fi

cd "$REPO_ROOT"

# Confirm the installed escapepod matches the workspace version; a stale wheel
# would benchmark an old API.
WS_VER="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
INST_VER="$(pixi run -e python-test python -c 'import escapepod; print(escapepod.__version__)' 2>/dev/null | tail -1)"
if [ "$WS_VER" != "$INST_VER" ]; then
    echo "WARNING: installed escapepod ($INST_VER) != workspace ($WS_VER)." >&2
    echo "         Rebuild with: pixi run -e python-test maturin develop --release \\" >&2
    echo "                       --manifest-path crates/escapepod-python/Cargo.toml" >&2
fi

exec pixi run -e python-test python "$SCRIPT" "$@"

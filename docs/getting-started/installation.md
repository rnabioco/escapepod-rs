# Installation

## Prerequisites

Only to build from source:

- Rust 1.95 or later
- Cargo (comes with Rust)

## Installing the CLI

### Prebuilt binaries

Every tagged version publishes `escpod` binaries on the
[releases page](https://github.com/rnabioco/escapepod-rs/releases):

| Artifact | Linkage | GPU paths |
|----------|---------|---------|
| `escpod-<ver>-x86_64-unknown-linux-musl.tar.gz` | static (musl) | no |
| `escpod-<ver>-aarch64-unknown-linux-musl.tar.gz` | static (musl) | no |
| `escpod-<ver>-x86_64-unknown-linux-gnu-gpu.tar.gz` | dynamic, glibc ≥ 2.28 | **yes** |
| `escpod-<ver>-x86_64-apple-darwin.tar.gz` | dynamic | no |
| `escpod-<ver>-aarch64-apple-darwin.tar.gz` | dynamic | no |

```bash
VER=v0.16.1
curl -L "https://github.com/rnabioco/escapepod-rs/releases/download/$VER/escpod-$VER-x86_64-unknown-linux-musl.tar.gz" | tar xz
install -m755 escpod ~/.local/bin/
escpod --version
```

`SHA256SUMS.txt` on the same page covers every archive.

The **musl** builds are the portable default: static, so they run on any
Linux with no library requirements whatsoever, and the right thing for an
unattended installer to fetch.

#### The GPU artifact

`--features gpu` cannot be static — every GPU path `dlopen`s its runtime (the
CUDA driver and `libnvrtc` for DTW, a CUDA-enabled `libonnxruntime` for CNN
adapter detection and the CTC-CRF encoder) — so it ships in the one
dynamically linked artifact, `x86_64-unknown-linux-gnu-gpu`. It is built
against glibc 2.28, which covers RHEL/Rocky/Alma 8+ and Ubuntu 20.04+.

Actually reaching the device with that binary needs a **CUDA 12** runtime and
cuDNN 9 at run time, with an NVIDIA driver ≥ 535 (the DTW kernels target the
CUDA 12.2 driver API). Don't assemble that by hand — [GPU
acceleration](../cli/demux.md#gpu-acceleration) covers the pixi
environment that supplies it and how to confirm the CUDA execution provider
actually loaded.

Placement is a run-time choice: `--device auto` (the default) uses the device
for the stages where it wins and falls back silently otherwise, `--device gpu`
demands it and errors instead of falling back, and `--device cpu` forces CPU.
So on a CPU-only box this artifact needs none of the above and behaves exactly
like the musl one.

### From Source

Clone the repository and build:

```bash
git clone https://github.com/rnabioco/escapepod-rs.git
cd escapepod-rs
cargo build --release
```

The binary will be at `target/release/escpod`. You can copy it to a directory in your PATH:

```bash
cp target/release/escpod ~/.local/bin/
# or
sudo cp target/release/escpod /usr/local/bin/
```

### Optional features

The default build ships the stable CLI surface (summary, view, inspect,
merge, filter, bam-filter, subset, `index`) **plus the full demux tree** —
fused `demux`, `detect`, `fingerprint`, `classify`, `basecall`, `split`,
`models`, `train` — with CNN adapter detection, CRF basecalling, and model
fetching included, **and `classify`**, the read-level model runner
(tRNA charging), with both bundle scorers. Extra commands and accelerators
live behind Cargo features:

| Feature | Commands unlocked |
|---------|-------------------|
| `experimental` | `repack`, `resquiggle`, `annotate` |
| `train` | adds `demux train-svm` (SVM training via linfa) |
| `gpu` | every GPU path, and the only GPU feature — CNN adapter detection, the CTC-CRF encoder, DTW classify; selected at run time with `--device` (default `auto`), CUDA libraries at run time only |
| `models-download` | `resquiggle models fetch` (k-mer table prefetch) |

Note the sidecar asymmetry: `escpod index` builds caches that can always be
rebuilt from the POD5, so it is in the default build; `escpod annotate` writes
data products that exist nowhere else, so it needs `--features experimental`.
*Consuming* a sidecar (`demux --annotate`, `demux split --sidecar`,
`filter --annotation`, `view`, `inspect`) is default-build throughout.

`gpu` needs CUDA libraries at run time only; the repository's pixi `gpu`
environment provides all of them — see
[GPU acceleration](../cli/demux.md#gpu-acceleration). Building for
`gpu` is optional: the `x86_64-unknown-linux-gnu-gpu` release artifact above
already carries it.

Combine as needed:

```bash
cargo build --release --features experimental
cargo install --git https://github.com/rnabioco/escapepod-rs --features experimental
```

See the [Experimental](../experimental/index.md) section for per-command
details.

### Verify Installation

```bash
escpod --version
escpod --help
```

## Installing the Python package

The `escapepod` Python package provides a `pod5`-compatible API. Install it
from PyPI:

```bash
uv pip install escapepod
```

Wheels are published for CPython 3.9+ (abi3) on Linux (x86_64/aarch64,
manylinux + musllinux) and macOS (x86_64/arm64). To build from a checkout
instead (or on an unsupported platform), use
[maturin](https://www.maturin.rs/):

```bash
uv pip install maturin
maturin develop --release --manifest-path crates/escapepod-python/Cargo.toml
```

This installs `escapepod` into the active environment. See the
[Python API](../python/index.md) for usage.

## Using the Rust Library

The workspace splits the library layer in two: `escapepod-pod5` for format
I/O and `escapepod-signal` for signal-processing algorithms. `escapepod-signal`
re-exports the full `escapepod-pod5` surface, so most users only need to
depend on the signal crate:

```toml
[dependencies]
escapepod-signal = { git = "https://github.com/rnabioco/escapepod-rs.git" }
```

If you only need POD5 file I/O without the signal algorithms:

```toml
[dependencies]
escapepod-pod5 = { git = "https://github.com/rnabioco/escapepod-rs.git" }
```

Barcode demultiplexing lives in its own crate, `escapepod-demux`, which the
CLI links by default; library consumers opt in with `--features demux`.

## Building Documentation

To build the API documentation locally:

```bash
cargo doc --open --no-deps
```

## Development Setup

For contributing to escapepod-rs:

```bash
# Clone the repository
git clone https://github.com/rnabioco/escapepod-rs.git
cd escapepod-rs

# Run tests (cargo-nextest)
cargo nextest run

# Run doctests separately — nextest does not execute them
cargo test --doc --workspace

# Run clippy lints
cargo clippy

# Format code
cargo fmt
```

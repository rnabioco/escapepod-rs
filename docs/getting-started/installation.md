# Installation

## Prerequisites

- Rust 1.95 or later
- Cargo (comes with Rust)

## Installing the CLI

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
merge, filter, bam-filter, subset) **plus the full demux tree** — fused
`demux`, `detect`, `fingerprint`, `classify`, `basecall`, `split`,
`models`, `train` — with CNN adapter detection, CRF basecalling, and model
fetching included. Extra commands and accelerators live behind Cargo
features:

| Feature | Commands unlocked |
|---------|-------------------|
| `experimental` | `repack`, `resquiggle`, `index`, `annotate` |
| `train` | adds `demux train-svm` (SVM training via linfa) |
| `gpu` | every `--gpu` path — CNN adapter detection, the CTC-CRF encoder, DTW classify (CUDA libraries at run time only) |
| `models-download` | `resquiggle models fetch` (k-mer table prefetch) |

Note the sidecar asymmetry: *writing* `.p5s` sidecars (`index`, `annotate`)
needs `--features experimental`, but consuming them (`demux --annotate`,
`demux split --sidecar`, `filter --annotation`, `view`, `inspect`) works in
the default build.

The GPU features need CUDA libraries at run time only; the repository's pixi
`gpu` environment provides all of them — see
[GPU acceleration](../experimental/demux.md#gpu-acceleration).

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

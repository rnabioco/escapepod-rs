# Installation

## Prerequisites

- Rust 1.88 or later
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
merge, filter, bam-filter, subset). Extra commands live behind Cargo
features:

| Feature | Commands unlocked |
|---------|-------------------|
| `experimental` | `repack`, `resquiggle`, `index` |
| `demux` | `demux detect`, `fingerprint`, `classify`, `split`, `train` |
| `train` | implies `demux`; adds `demux train-svm` |
| `gpu` | implies `demux`; batched GPU DTW for `classify` / `train-svm` (CUDA driver + libnvrtc required at runtime) |
| `cnn-detect` | implies `demux`; ADAPTed-style CNN adapter detection (bring-your-own ONNX model) |

Combine as needed:

```bash
cargo build --release --features experimental,demux
cargo install --git https://github.com/rnabioco/escapepod-rs --features experimental
```

See the [Experimental](../experimental/index.md) section for per-command
details.

### Verify Installation

```bash
escpod --version
escpod --help
```

## Using the Library

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

Barcode demultiplexing lives in its own crate, `escapepod-demux`, which
mirrors the CLI's `demux` feature gate.

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

# Run tests
cargo test

# Run clippy lints
cargo clippy

# Format code
cargo fmt
```

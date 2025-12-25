# Installation

## Prerequisites

- Rust 1.85 or later
- Cargo (comes with Rust)

## Installing the CLI

### From Source

Clone the repository and build:

```bash
git clone https://github.com/rnabioco/escapepod-rs.git
cd escapepod-rs
cargo build --release
```

The binary will be at `target/release/escapepod`. You can copy it to a directory in your PATH:

```bash
cp target/release/escapepod ~/.local/bin/
# or
sudo cp target/release/escapepod /usr/local/bin/
```

### Verify Installation

```bash
escapepod --version
escapepod --help
```

## Using the Library

Add escapepod to your `Cargo.toml`:

```toml
[dependencies]
escapepod = { git = "https://github.com/rnabioco/escapepod-rs.git" }
```

Or if published to crates.io:

```toml
[dependencies]
escapepod = "0.1"
```

## Building Documentation

To build the API documentation locally:

```bash
cargo doc --open
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

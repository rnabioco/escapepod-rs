# Installation

## Prerequisites

- Rust 1.70 or later
- Cargo (comes with Rust)

## Installing the CLI

### From Source

Clone the repository and build:

```bash
git clone https://github.com/rnabioco/podfive-rs.git
cd podfive-rs
cargo build --release
```

The binary will be at `target/release/podfive`. You can copy it to a directory in your PATH:

```bash
cp target/release/podfive ~/.local/bin/
# or
sudo cp target/release/podfive /usr/local/bin/
```

### Verify Installation

```bash
podfive --version
podfive --help
```

## Using the Library

Add podfive-core to your `Cargo.toml`:

```toml
[dependencies]
podfive-core = { git = "https://github.com/rnabioco/podfive-rs.git" }
```

Or if published to crates.io:

```toml
[dependencies]
podfive-core = "0.1"
```

## Building Documentation

To build the API documentation locally:

```bash
cargo doc --open
```

## Development Setup

For contributing to podfive-rs:

```bash
# Clone the repository
git clone https://github.com/rnabioco/podfive-rs.git
cd podfive-rs

# Run tests
cargo test

# Run clippy lints
cargo clippy

# Format code
cargo fmt
```

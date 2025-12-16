# podfive-rs

A Rust CLI for reading and writing Oxford Nanopore POD5 files.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![podfive-rs demo](demo.gif)

## Overview

podfive-rs is a pure Rust implementation for working with POD5 files, the native file format for Oxford Nanopore sequencing data.

## Features

- Read POD5 files with memory-mapped I/O for efficiency
- Write POD5 files with VBZ signal compression
- Full support for reads, signal, and run info tables
- CLI tools for viewing, inspecting, merging, and filtering POD5 files

## Installation

### From source

```bash
git clone https://github.com/jayhesselberth/podfive-rs.git
cd podfive-rs
cargo build --release
```

The binary will be available at `target/release/podfive`.

## CLI Usage

### View reads

Display reads as a table:

```bash
podfive view input.pod5
```

### Inspect file metadata

```bash
# Summary information
podfive inspect summary input.pod5

# List all reads
podfive inspect reads input.pod5

# Inspect a specific read
podfive inspect read input.pod5 <read-id>
```

### Merge files

Combine multiple POD5 files into one:

```bash
podfive merge -o output.pod5 input1.pod5 input2.pod5 input3.pod5
```

### Filter reads

Extract specific reads by ID:

```bash
podfive filter -i read_ids.txt -o filtered.pod5 input.pod5
```

The `read_ids.txt` file should contain one UUID per line.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project is a Rust reimplementation inspired by the original [pod5-file-format](https://github.com/nanoporetech/pod5-file-format) library by Oxford Nanopore Technologies.

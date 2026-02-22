# escapepod-rs

A Rust library and CLI for reading and writing Oxford Nanopore POD5 files.

## What is POD5?

POD5 is the native file format for Oxford Nanopore sequencing data. It stores:

- **Raw signal data** - The electrical current measurements from the nanopore
- **Read metadata** - Information about each read (channel, timing, calibration)
- **Run information** - Experimental metadata (flow cell, protocol, sample)

## Why escapepod-rs?

escapepod-rs provides a fast, memory-efficient Rust implementation for working with POD5 files:

- **Performance** - Memory-mapped I/O and efficient compression handling
- **Safety** - Rust's type system prevents common errors
- **Simplicity** - Clean API for both library and CLI usage
- **Compatibility** - Reads and writes files compatible with ONT tools

## Quick Example

### CLI

```bash
# View reads in a POD5 file
escpod view experiment.pod5

# Merge multiple files
escpod merge -o combined.pod5 run1.pod5 run2.pod5

# Filter by read IDs
escpod filter -i interesting_reads.txt -o subset.pod5 experiment.pod5
```

### Library

```rust
use escapepod::Reader;

let reader = Reader::open("experiment.pod5")?;

for read in reader.reads()? {
    let read = read?;
    println!("{}: {} samples", read.read_id, read.num_samples);

    // Get the raw signal
    let signal = reader.get_signal(&read.signal_rows)?;
}
```

## Getting Started

- [Installation](getting-started/installation.md) - How to install escapepod-rs
- [Quick Start](getting-started/quickstart.md) - Get up and running quickly

## Documentation

- [CLI Reference](cli/index.md) - Command-line tool documentation
- [Library Guide](library/index.md) - Using escapepod in your Rust projects
- [File Format](format/index.md) - Technical details of the POD5 format

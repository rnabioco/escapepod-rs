# podfive-rs

A Rust library and CLI for reading and writing Oxford Nanopore POD5 files.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

podfive-rs is a pure Rust implementation for working with POD5 files, the native file format for Oxford Nanopore sequencing data. It provides:

- **podfive-core**: A library for reading and writing POD5 files
- **podfive-cli**: A command-line tool for common POD5 operations

## Features

- Read POD5 files with memory-mapped I/O for efficiency
- Write POD5 files with VBZ signal compression
- Full support for reads, signal, and run info tables
- CLI tools for viewing, inspecting, merging, and filtering POD5 files

## Installation

### From source

```bash
git clone https://github.com/rnabioco/podfive-rs.git
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

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
podfive-core = { git = "https://github.com/rnabioco/podfive-rs.git" }
```

### Reading POD5 files

```rust
use podfive_core::Reader;

fn main() -> podfive_core::Result<()> {
    let reader = Reader::open("example.pod5")?;

    // Get file metadata
    println!("Read count: {}", reader.read_count()?);

    // Iterate over reads
    for read_result in reader.reads()? {
        let read = read_result?;
        println!("Read ID: {}", read.read_id);
        println!("Channel: {}", read.channel);
        println!("Samples: {}", read.num_samples);

        // Get signal data
        let signal = reader.get_signal(&read.signal_rows)?;
        println!("Signal length: {}", signal.len());
    }

    // Access run info
    for run_info in reader.run_infos() {
        println!("Acquisition ID: {}", run_info.acquisition_id);
        println!("Sample rate: {} Hz", run_info.sample_rate);
    }

    Ok(())
}
```

### Writing POD5 files

```rust
use podfive_core::{Writer, WriterOptions, ReadData, RunInfoData, EndReason};
use std::collections::HashMap;

fn main() -> podfive_core::Result<()> {
    let options = WriterOptions::default();
    let mut writer = Writer::create("output.pod5", options)?;

    // Add run info
    let run_info = RunInfoData {
        acquisition_id: "my_run".to_string(),
        sample_rate: 4000,
        // ... other fields
        ..Default::default()
    };
    let run_info_idx = writer.add_run_info(run_info)?;

    // Add reads with signal
    let read = ReadData {
        read_id: uuid::Uuid::new_v4(),
        read_number: 1,
        channel: 1,
        well: 1,
        run_info_index: run_info_idx,
        // ... other fields
        ..Default::default()
    };

    let signal: Vec<i16> = vec![100, 105, 98, 102, /* ... */];
    writer.add_read(read, &signal)?;

    // Finalize the file
    writer.finish()?;

    Ok(())
}
```

## File Format

POD5 files use a container format with:

- **Signature**: PNG-inspired magic bytes for file identification
- **Section markers**: UUID-based markers separating data sections
- **Arrow IPC tables**: Reads, signal, and run info stored as Apache Arrow tables
- **FlatBuffer footer**: Metadata describing embedded file locations

Signal data is compressed using VBZ compression:
1. Delta encoding between consecutive samples
2. Zigzag encoding to handle signed values
3. SVB16 (StreamVByte 16-bit) variable-length encoding
4. ZSTD compression (level 1)

## Project Structure

```
podfive-rs/
├── podfive-core/          # Core library
│   └── src/
│       ├── compression/   # VBZ compression (SVB16 + ZSTD)
│       ├── reader/        # POD5 file reader
│       ├── writer/        # POD5 file writer
│       ├── schema/        # Arrow schema definitions
│       ├── footer.rs      # FlatBuffer footer parsing
│       ├── types.rs       # Core data types
│       └── error.rs       # Error handling
├── podfive-cli/           # CLI application
│   └── src/
│       └── commands/      # CLI commands
└── docs/                  # Documentation
```

## Dependencies

- [arrow-rs](https://github.com/apache/arrow-rs) - Apache Arrow implementation
- [zstd](https://github.com/gyscos/zstd-rs) - ZSTD compression
- [memmap2](https://github.com/RazrFalcon/memmap2-rs) - Memory-mapped file I/O
- [uuid](https://github.com/uuid-rs/uuid) - UUID handling
- [clap](https://github.com/clap-rs/clap) - CLI argument parsing

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project is a Rust reimplementation inspired by the original [pod5-file-format](https://github.com/nanoporetech/pod5-file-format) library by Oxford Nanopore Technologies.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

podfive-rs is a pure Rust implementation for reading and writing POD5 files, the native file format for Oxford Nanopore sequencing data. The project provides both a library crate (`podfive-core`) and a CLI tool (`podfive-cli`).

## Build Commands

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run a specific test
cargo test test_round_trip_single_read

# Run clippy lints
cargo clippy

# Run the CLI (after building)
./target/release/podfive <command>
```

## Architecture

### Workspace Structure

- **podfive-core**: Core library for reading/writing POD5 files
- **podfive-cli**: CLI binary that uses podfive-core

### POD5 File Format

POD5 is a container format wrapping Apache Arrow IPC (Feather v2) tables:

```
<POD5 signature>
<section marker>
<Signal table (Arrow IPC)><section marker>
<Run Info table (Arrow IPC)><section marker>
<Reads table (Arrow IPC)><section marker>
<FOOTER magic>
<FlatBuffer footer>
<footer length>
<section marker>
<POD5 signature>
```

### Core Library (podfive-core)

- **reader/file_reader.rs**: Memory-mapped file reader using `memmap2`. Opens POD5 files, parses the FlatBuffer footer, and provides iterators over reads and signal data.
- **writer/file_writer.rs**: Buffered writer that constructs POD5 files. Handles signal chunking, VBZ compression, and batching of Arrow record batches.
- **compression/**: VBZ signal compression (SVB16 + ZSTD pipeline)
  - `svb16.rs`: StreamVByte encoding with delta and zigzag transforms
  - `vbz.rs`: Full VBZ pipeline combining SVB16 with ZSTD compression
- **footer.rs**: Manual FlatBuffer parsing for the POD5 footer (locates embedded Arrow tables)
- **schema/**: Arrow schema definitions for reads, signal, and run_info tables
- **types.rs**: Core data types (`ReadData`, `RunInfoData`, `EndReason`, etc.)

### CLI Commands

- `view`: Display reads as TSV with configurable columns
- `inspect`: Show file metadata (summary, reads list, specific read)
- `summary`: Comprehensive summary with statistics
- `merge`: Combine multiple POD5 files (parallel reading with rayon)
- `filter`: Extract reads by ID list
- `repack`: Repack files for optimized storage
- `subset`: Split reads into multiple files based on CSV mapping

### Key Patterns

**Block-level copying**: For merge/filter operations, signal data is kept compressed (`CompressedSignalChunk` with `Arc<[u8]>`) to avoid decompression/recompression overhead. Use `add_read_with_compressed_signal()` instead of `add_read()` when copying between files.

**Dictionary tracking**: The writer maintains O(1) lookup for pore types and end reasons using HashMap indexes alongside Vec storage for Arrow dictionary encoding.

**Run info deduplication**: When merging files, run infos are deduplicated by `acquisition_id` to avoid redundant entries.

## Dependencies

- **arrow**: Arrow IPC file reading/writing
- **memmap2**: Memory-mapped file I/O
- **zstd**: ZSTD compression for VBZ
- **rayon**: Parallel iteration for merge operations
- **clap**: CLI argument parsing
- **tabled**: Table formatting for CLI output

## Test Data

Test POD5 files from Oxford Nanopore are in `ext/nanopore-dna-data/pod5/`. The `ext/pod5-file-format/` directory contains the official POD5 specification.

# API Reference

Full API documentation is generated from source code using `cargo doc`.

## Generating Documentation

```bash
cd escapepod-rs
cargo doc --open --no-deps
```

This opens the documentation in your browser.

## Crate Structure

### escapepod

The core library providing POD5 read/write functionality.

#### Main Types

| Type | Description |
|------|-------------|
| `Reader` | Read POD5 files |
| `Writer` | Create POD5 files |
| `WriterOptions` | Writer configuration |
| `ReadData` | Single read record |
| `RunInfoData` | Run metadata |
| `EndReason` | Read end reason enum |
| `Error` | Error type |

#### Modules

| Module | Description |
|--------|-------------|
| `compression` | VBZ compression codec |
| `footer` | FlatBuffer footer parsing |
| `reader` | File reading implementation |
| `writer` | File writing implementation |
| `schema` | Arrow schema definitions |
| `types` | Core data types |

### escapepod-cli

The command-line interface binary.

#### Commands

| Command | Description |
|---------|-------------|
| `view` | Display reads as table |
| `inspect` | Inspect file contents |
| `merge` | Combine multiple files |
| `filter` | Extract reads by ID |

## Quick Reference

### Opening Files

```rust
use escapepod::Reader;

let reader = Reader::open("file.pod5")?;
```

### Creating Files

```rust
use escapepod::{Writer, WriterOptions};

let writer = Writer::create("output.pod5", WriterOptions::default())?;
```

### Read Iteration

```rust
for read in reader.reads()? {
    println!("{}: {} samples", read.read_id, read.num_samples);
}
```

### Signal Access

```rust
let signal: Vec<i16> = reader.get_signal(&read)?;
```

### Run Info

```rust
let run_info = reader.get_run_info(read.run_info_index)?;
println!("Sample rate: {} Hz", run_info.sample_rate);
```

### Writing Reads

```rust
writer.add_run_info(run_info)?;
writer.add_read(read_data, &signal)?;
writer.finish()?;
```

## Error Handling

```rust
use escapepod::Error;

match result {
    Ok(value) => { /* success */ }
    Err(Error::Io(e)) => eprintln!("I/O: {}", e),
    Err(Error::InvalidSignature) => eprintln!("Invalid file"),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `train` | Enables SVM model training with `linfa` and `linfa-svm` |

Core functionality is included by default. Use `--features train` to enable model training capabilities.

## Dependencies

### escapepod

| Crate | Purpose |
|-------|---------|
| `arrow` | Arrow IPC format |
| `flatbuffers` | Footer serialization |
| `zstd` | ZSTD compression |
| `uuid` | UUID handling |
| `memmap2` | Memory-mapped files |
| `thiserror` | Error derive |

### escapepod-cli

| Crate | Purpose |
|-------|---------|
| `clap` | Argument parsing |
| `rayon` | Parallel processing |

## Minimum Supported Rust Version

Rust 1.85 or later is required.

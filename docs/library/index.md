# Library Overview

The `escapepod` library provides Rust APIs for reading and writing POD5 files.

## Features

- **Read POD5 files** - Memory-mapped, efficient reading of reads and signal data
- **Write POD5 files** - Create new POD5 files with automatic compression
- **Signal compression** - VBZ codec (SVB16 + ZSTD) for signal data
- **Full metadata support** - Read/write run info, calibration, and pore data

## Quick Example

```rust
use escapepod_signal::{Reader, Writer, WriterOptions, ReadData, RunInfoData};

// Read a POD5 file
let reader = Reader::open("input.pod5")?;
println!("File contains {} reads", reader.read_count());

for read in reader.reads() {
    println!("Read {} has {} samples", read.read_id, read.num_samples);
}

// Write a new POD5 file
let mut writer = Writer::create("output.pod5", WriterOptions::default())?;
writer.add_run_info(run_info)?;
writer.add_read(read_data, &signal)?;
writer.finish()?;
```

## Modules

| Module | Description |
|--------|-------------|
| [Reading](reading.md) | Reading POD5 files |
| [Writing](writing.md) | Creating POD5 files |
| [Types](types.md) | Data structures and types |

## Error Handling

All operations return `Result<T, escapepod_signal::Error>`. The error type provides detailed information about what went wrong:

```rust
use escapepod_signal::{Reader, Error};

match Reader::open("file.pod5") {
    Ok(reader) => { /* use reader */ }
    Err(Error::Io(e)) => eprintln!("I/O error: {}", e),
    Err(Error::InvalidSignature) => eprintln!("Not a valid POD5 file"),
    Err(e) => eprintln!("Error: {}", e),
}
```

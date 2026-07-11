# escapepod

A fast, memory-efficient toolkit for reading and writing Oxford Nanopore
**POD5** files — available as a command-line tool, a Python package, and a Rust
library, all backed by the same pure-Rust engine.

## What is POD5?

POD5 is the native file format for Oxford Nanopore sequencing data. It stores:

- **Raw signal data** — the electrical current measurements from the nanopore
- **Read metadata** — information about each read (channel, timing, calibration)
- **Run information** — experimental metadata (flow cell, protocol, sample)

## Why escapepod?

- **Fast** — memory-mapped I/O and efficient VBZ compression; up to ~9× faster
  than the official Python `pod5` tools on large-file operations
- **Compatible** — reads and writes files interchangeable with ONT tools, and
  the Python API mirrors the official `pod5` package
- **One engine, three surfaces** — the CLI, Python package, and Rust crates all
  share the same implementation
- **Safe** — Rust's type system prevents whole classes of errors

## Ways to use escapepod

Pick the surface that fits how you work:

| I want to… | Use | Guide |
|------------|-----|-------|
| Explore, filter, merge, and convert files from a shell | the **`escpod`** CLI | [CLI Reference](cli/index.md) |
| Read/write POD5 in a Python script or notebook | the **`escapepod`** package | [Python API](python/index.md) |
| Embed POD5 I/O in a Rust program | the **`escapepod-signal`** crate | [Rust Library](library/index.md) |
| Understand or implement the on-disk format | — | [File Format](format/index.md) |

### Command line — `escpod`

```bash
# View reads in a POD5 file
escpod view experiment.pod5

# Merge multiple files, then filter by read ID
escpod merge -o combined.pod5 run1.pod5 run2.pod5
escpod filter -i interesting_reads.txt -o subset.pod5 combined.pod5
```

See the [CLI Reference](cli/index.md) for every command.

### Python — `escapepod`

A `pod5`-compatible package backed by the same Rust engine.

```python linenums="1"
import escapepod

with escapepod.Reader("experiment.pod5") as reader:
    print(f"{reader.read_count} reads")
    for read in reader:
        signal = reader.get_signal(read)   # numpy int16 (raw ADC)
        print(f"{read.read_id}: {read.num_samples} samples")
```

See the [Python API](python/index.md) for reading, writing, and signal helpers.

### Rust

```rust linenums="1"
use escapepod_signal::Reader;

let reader = Reader::open("experiment.pod5")?;

for read in reader.reads()? {
    let read = read?;
    println!("{}: {} samples", read.read_id, read.num_samples);

    // Get the raw signal
    let signal = reader.get_signal(&read.signal_rows)?;
}
```

See the [Rust Library](library/index.md) for the full crate API.

## Getting Started

- [Installation](getting-started/installation.md) — install the CLI, Python package, or Rust crates
- [Quick Start](getting-started/quickstart.md) — up and running in a few minutes

## Documentation

- [CLI Reference](cli/index.md) — every `escpod` command
- [Python API](python/index.md) — reading and writing POD5 from Python
- [Rust Library](library/index.md) — using escapepod in your Rust projects
- [File Format](format/index.md) — technical details of the POD5 container

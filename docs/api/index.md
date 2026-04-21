# API Reference

Full API documentation is generated from source code using `cargo doc`.

## Generating Documentation

```bash
cd escapepod-rs
cargo doc --open --no-deps
```

This opens the documentation in your browser.

## Crate Structure

The workspace is split into five crates:

| Crate | Role |
|-------|------|
| `escapepod-pod5` | POD5 format I/O (reader, writer, VBZ, footer, block-level merge/filter/subset) |
| `escapepod-signal` | Signal algorithms (DTW, resquiggle, segmentation); **re-exports the full `escapepod-pod5` surface** |
| `escapepod-demux` | WarpDemuX-compatible barcode demultiplexing (DTW + SVM classifier, optional CNN adapter detection and GPU acceleration) |
| `escapepod-cli` | The `escpod` binary |
| `escapepod-python` | pyo3 bindings |

### escapepod-pod5

Format I/O.

**Main types:** `Reader`, `Writer`, `WriterOptions`, `ReadData`, `RunInfoData`, `EndReason`, `Error`.

**Modules:** `reader`, `writer`, `compression` (VBZ / SVB16 / ZSTD), `footer` (FlatBuffer), `schema` (Arrow schemas), `types`, `merge`, `operations::{filter, repack, subset}`.

### escapepod-signal

Signal-processing algorithms, layered on top of `escapepod-pod5` (which it re-exports).

**Modules:** `dtw` (distance, fingerprint, kernel, optional `cuda`), `segmentation` (LLR, t-test, normalize), `resquiggle` (banded DP).

### escapepod-demux

Barcode demultiplexing. Separate crate; the CLI pulls it in only when built with `--features demux`.

**Modules:** `model` (JSON loaders), `classify` (per-read and batched GPU), `svm` (RBF kernel + Platt scaling), `probability`, `train` (feature `train`), `adapter_cnn` (feature `cnn-detect`).

### escapepod-cli

The `escpod` binary. Stable commands (always built): `summary`, `view`, `inspect`, `merge`, `filter`, `bam-filter`, `subset`. Experimental commands live behind Cargo features — see below.

## Quick Reference

### Opening Files

```rust
use escapepod_signal::Reader;

let reader = Reader::open("file.pod5")?;
```

### Creating Files

```rust
use escapepod_signal::{Writer, WriterOptions};

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
use escapepod_signal::Error;

match result {
    Ok(value) => { /* success */ }
    Err(Error::Io(e)) => eprintln!("I/O: {}", e),
    Err(Error::InvalidSignature) => eprintln!("Invalid file"),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Feature Flags

### `escapepod-cli`

| Feature | Effect |
|---------|--------|
| `experimental` | Unlocks `repack`, `resquiggle`, `index` |
| `demux` | Unlocks the `demux` subcommand tree (detect / fingerprint / classify / split / train) |
| `train` | Implies `demux`; adds `demux train-svm` (linfa-svm) |
| `gpu` | Implies `demux`; batched GPU DTW for classify / train-svm (CUDA driver + libnvrtc at runtime) |
| `cnn-detect` | Implies `demux`; ADAPTed-style CNN adapter detection (bring-your-own ONNX model; weights are CC BY-NC 4.0 and not bundled) |

### `escapepod-demux`

| Feature | Effect |
|---------|--------|
| `train` | `DtwSvmModel` training via `linfa-svm` |
| `gpu` | Routes to `escapepod-signal`'s CUDA DTW kernel |
| `cnn-detect` | ADAPTed-style CNN adapter detection via `tract-onnx` |

The CLI features forward to the matching demux features, so building the
CLI with `--features gpu` transitively enables demux's `gpu` feature.

## Dependencies

### escapepod-pod5

| Crate | Purpose |
|-------|---------|
| `arrow` | Arrow IPC format |
| `flatbuffers` | Footer serialization |
| `zstd` | ZSTD compression |
| `memmap2` | Memory-mapped files |
| `uuid` | UUID handling |
| `thiserror` | Error derive |

### escapepod-signal

| Crate | Purpose |
|-------|---------|
| `escapepod-pod5` | Re-exported as `pod5` |
| `ndarray` | Array operations |
| `rand`, `flate2` | Resquiggle internals |

### escapepod-demux

| Crate | Purpose |
|-------|---------|
| `escapepod-pod5`, `escapepod-signal` | Format I/O + DTW |
| `ndarray` | Feature vectors |
| `serde`, `serde_json` | Model JSON |
| `linfa`, `linfa-svm` | SVM training (feature `train`) |
| `tract-onnx` | CNN adapter detection (feature `cnn-detect`) |

### escapepod-cli

| Crate | Purpose |
|-------|---------|
| `clap` | Argument parsing |
| `rayon` | Parallel processing |
| `noodles-bam`, `noodles-sam` | BAM integration |
| `tabled` | Table formatting |

## Minimum Supported Rust Version

Rust 1.88 or later is required (tracked in `[workspace.package].rust-version`).

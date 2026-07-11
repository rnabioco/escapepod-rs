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
| `escapepod-cli` | The `escpod` CLI binary (default `cli` feature) plus an optional umbrella library (imported as `escapepod_cli`) re-exporting the layers below |
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

### escapepod

The `escpod` binary, built by the default `cli` feature — so `cargo install --git https://github.com/rnabioco/escapepod-rs` ships the tool. The same crate doubles as an umbrella library: `default-features = false` plus `pod5` / `signal` / `demux` re-exports the corresponding layer (e.g. `escapepod_cli::signal`) without the CLI's dependency tree. Stable commands (built with `cli`): `summary`, `view`, `inspect`, `merge`, `filter`, `bam-filter`, `subset`. Experimental commands live behind Cargo features — see below.

## Quick Reference

### Opening Files

```rust linenums="1"
use escapepod_signal::Reader;

let reader = Reader::open("file.pod5")?;
```

### Creating Files

```rust linenums="1"
use escapepod_signal::{Writer, WriterOptions};

let writer = Writer::create("output.pod5", WriterOptions::default())?;
```

### Read Iteration

```rust linenums="1"
for read in reader.reads()? {
    println!("{}: {} samples", read.read_id, read.num_samples);
}
```

### Signal Access

```rust linenums="1"
let signal: Vec<i16> = reader.get_signal(&read)?;
```

### Run Info

```rust linenums="1"
let run_info = reader.get_run_info(read.run_info_index)?;
println!("Sample rate: {} Hz", run_info.sample_rate);
```

### Writing Reads

```rust linenums="1"
writer.add_run_info(run_info)?;
writer.add_read(read_data, &signal)?;
writer.finish()?;
```

## Error Handling

```rust linenums="1"
use escapepod_signal::Error;

match result {
    Ok(value) => { /* success */ }
    Err(Error::Io(e)) => eprintln!("I/O: {}", e),
    Err(Error::InvalidSignature) => eprintln!("Invalid file"),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Feature Flags

### `escapepod`

| Feature | Effect |
|---------|--------|
| `cli` *(default)* | Builds the `escpod` binary and its CLI dependencies; implies `signal` |
| `pod5` / `signal` / `demux` | Library re-exports of each layer (for `default-features = false` consumers) |
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

### escapepod (`cli` feature)

| Crate | Purpose |
|-------|---------|
| `clap` | Argument parsing |
| `rayon` | Parallel processing |
| `noodles-bam`, `noodles-sam` | BAM integration |
| `tabled` | Table formatting |

## Minimum Supported Rust Version

Rust 1.95 or later is required (tracked in `[workspace.package].rust-version`).

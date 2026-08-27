# API Reference

Full API documentation is generated from source code using `cargo doc`.

## Generating Documentation

```bash
cd escapepod-rs
cargo doc --open --no-deps
```

This opens the documentation in your browser.

## Crate Structure

The workspace is split into six crates:

| Crate | Role |
|-------|------|
| `escapepod-pod5` | POD5 format I/O (reader, writer, VBZ, footer, block-level merge/filter/subset) |
| `escapepod-signal` | Signal algorithms (DTW, resquiggle, segmentation, k-mer primitives); **re-exports the full `escapepod-pod5` surface** |
| `escapepod-demux` | WarpDemuX-compatible barcode demultiplexing (DTW + SVM classifier, CTC-CRF basecalling, optional CNN adapter detection and GPU acceleration) |
| `escapepod-classify` | Read-level classification against model bundles — the tRNA charging classifier behind `escpod signal classify` |
| `escapepod-cli` | The `escpod` CLI binary (default `cli` feature) plus an optional umbrella library (imported as `escapepod_cli`) re-exporting the layers below |
| `escapepod-python` | pyo3 bindings |

### escapepod-pod5

Format I/O.

**Main types:** `Reader`, `Writer`, `WriterOptions`, `ReadData`, `RunInfoData`, `EndReason`, `Error`.

**Modules:** `reader`, `writer`, `compression` (VBZ / SVB16 / ZSTD), `footer` (FlatBuffer), `schema` (Arrow schemas), `types`, `merge`, `sidecar` (the `.p5s` companion file), `operations::{filter, repack, subset, annotate}` (including `read_annotation` / `write_annotation` / `read_design`).

### escapepod-signal

Signal-processing algorithms, layered on top of `escapepod-pod5` (which it re-exports).

**Modules:** `dtw` (distance, fingerprint, kernel, optional `cuda`), `segmentation` (LLR, t-test, normalize), `resquiggle` (banded DP), `seq_encoding` (signal-level k-mer encoding and the k-mer context window), `mapping` (move-table and CIGAR coordinate mapping), `stats` (span statistics), `features`.

### escapepod-demux

Barcode demultiplexing. Separate crate; included in the default CLI build (it adds no third-party dependencies), and available to library consumers via `--features demux`.

**Modules:** `model` (JSON loaders), `classify` (per-read and batched GPU), `svm` (RBF kernel + Platt scaling), `probability`, `crf` (CTC-CRF lattice decode; `encoder`/`barcode` behind `crf-decode`), `train` (feature `train`), `adapter_cnn` (feature `cnn-detect`).

### escapepod-classify

Read-level classification against model bundles. Included in the default CLI build (it adds no third-party crates beyond the CLI graph), and available to library consumers via `--features classify`.

The bundle carries the whole feature recipe — feature order and offsets, the k-mer table pinned by sha256, the operating point — under a **closed** metadata schema, so a rule this runtime does not implement is refused at load rather than silently dropped. Every definition of the model's input lives here rather than in a caller, so a corpus builder and the inference path cannot diverge.

**Modules:** `bundle` (loading + the closed metadata schema), `recipe` (the feature space as a borrowed view), `features`, `window` (raw-signal windowing, junction anchoring, common-arm mask), `anchor` / `geometry` (CIGAR and move-table coordinate mapping), `pipeline`, `bam_tags`, `fnn` (the ONNX feature-network scorer, feature `fnn-onnx`).

**Main types:** `ChargingBundle`, `ChargingScorer` (`gbm` | `feature_model`), `FeatureRecipe`, `FeatureNet`.

### escapepod-cli

The `escpod` binary, built by the default `cli` feature — so `cargo install --git https://github.com/rnabioco/escapepod-rs` ships the tool. The same crate doubles as an umbrella library: `default-features = false` plus `pod5` / `signal` / `demux` re-exports the corresponding layer (e.g. `escapepod_cli::signal`) without the CLI's dependency tree. Commands built with `cli`: `summary`, `view`, `inspect`, `merge`, `filter`, `bam-filter`, `subset`, `index`, the `demux` tree, and `signal classify`. `repack`, `resquiggle`, and `annotate` live behind the `experimental` feature — see below.

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

### `escapepod-cli`

| Feature | Effect |
|---------|--------|
| `cli` *(default)* | Builds the `escpod` binary and its CLI dependencies; implies `signal`, `demux`, `classify`, `cnn-detect`, `crf-decode`, `demux-models`, `model-fetch` |
| `pod5` / `signal` / `demux` | Library re-exports of each layer (for `default-features = false` consumers) |
| `experimental` | Implies `classify`; unlocks `repack`, `resquiggle`, `annotate` |
| `demux` | The `demux` subcommand tree (fused pipeline, detect / fingerprint / classify / basecall / split / models / train) — *implied by `cli`* |
| `classify` | `escpod signal classify` (tRNA charging) via `escapepod-classify`, with `fnn-onnx` on — *implied by `cli`* |
| `crf-decode` | CTC-CRF barcode basecalling (`demux basecall`) — *implied by `cli`* |
| `demux-models` / `model-fetch` | Model-bundle registry and `demux models fetch` — *implied by `cli`* |
| `train` | Implies `demux`; adds `demux train-svm` (linfa-svm) |
| `gpu` | Every GPU path reachable from `--device gpu`, in one atomic flag: CNN adapter detection + CRF encoder (onnxruntime CUDA) and DTW classify (cudarc). There is no way to build half a GPU binary |
| `cnn-detect` | Part of `cli`; implies `demux`. CNN/TCN adapter detection through `tract-onnx` (bring-your-own ONNX model — no weights are bundled) |
| `models-download` | Implies `experimental`; `resquiggle models fetch` (k-mer tables) |

### `escapepod-demux`

| Feature | Effect |
|---------|--------|
| `train` | `DtwSvmModel` training via `linfa-svm` |
| `gpu` | Every GPU path in one flag: `escapepod-signal`'s CUDA DTW kernel plus the onnxruntime CUDA CNN detector and CRF encoder (implies `cnn-detect` + `crf-decode`) |
| `cnn-detect` | ADAPTed-style CNN adapter detection via `tract-onnx` |
| `crf-decode` | CTC-CRF encoder via `tract-onnx` + barcode matching |

### `escapepod-classify`

| Feature | Effect |
|---------|--------|
| `fnn-onnx` | The ONNX feature-network scorer via `tract-onnx` — *enabled by the CLI's `classify` feature* |

A bundle holding a `feature_model` is unusable without `fnn-onnx`, and that is
the arm escapepod-models ships, so the CLI turns it on unconditionally: tract is
already in the binary via `cnn-detect`, making it free there.

The CLI features forward to the matching demux features, so building the
CLI with `--features gpu` transitively enables demux's `gpu` feature.

`escapepod-signal`'s own `gpu` feature is narrower on purpose: it is the cudarc
DTW kernel and nothing else, with no onnxruntime in its graph. That crate has no
git dependency blocking publication, which is why it keeps a separate flag while
`escapepod-demux` does not.

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
| `fqxv-align` | WFA barcode matching (feature `crf-decode`) |

### escapepod-classify

| Crate | Purpose |
|-------|---------|
| `escapepod-signal` | k-mer primitives (`extract_levels`, `load_kmer_table`) + POD5 |
| `escapepod-demux` | The GBM runtime |
| `noodles-bam`, `noodles-bgzf`, `noodles-sam` | BAM reading + record types (writing the `cl`-tagged output stays in the CLI) |
| `serde`, `serde_json` | Bundle `metadata.json` |
| `sha2` | Pinning the bundle's k-mer table |
| `tract-onnx` | Feature-network scorer (feature `fnn-onnx`) |

### escapepod-cli (`cli` feature)

| Crate | Purpose |
|-------|---------|
| `clap` | Argument parsing |
| `rayon` | Parallel processing |
| `noodles-bam`, `noodles-sam` | BAM integration |
| `tabled` | Table formatting |
| `ureq`, `sha2`, `zip` | Model prefetch (feature `model-fetch`) |

## Minimum Supported Rust Version

Rust 1.95 or later is required (tracked in `[workspace.package].rust-version`).

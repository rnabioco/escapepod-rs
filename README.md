# escapepod

🚧 **escapepod is under active development.** Caveat emptor. 🚧

A Rust library and CLI for reading and writing Oxford Nanopore POD5 files.

[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![escapepod demo](docs/images/readme.gif)

## Highlights

- **Fast** - Up to 47x faster than Python pod5 tools
- **Memory efficient** - Memory-mapped I/O for large files
- **Full featured** - View, inspect, merge, filter, subset, and repack
- **Barcode demultiplexing** - DTW-based classification with SVM support
- **BAM integration** - Filter reads by alignment status

## Performance

| Command | escapepod | pod5 | Speedup |
|---------|-----------|------|---------|
| inspect | 36 ms | 1.7 s | **47x** |
| view | 238 ms | 4.5 s | **19x** |
| filter | 513 ms | 4.7 s | **9x** |
| subset | 2.8 s | 8.3 s | **3x** |
| merge | 3.0 s | 4.1 s | **1.4x** |

## Barcode Demultiplexing

escapepod includes a complete barcode demultiplexing workflow using Dynamic Time Warping (DTW) distance-based classification, compatible with WarpDemuX models.

```bash
# Detect adapter boundaries
escpod demux detect *.pod5 -o boundaries.csv

# Extract signal fingerprints
escpod demux fingerprint *.pod5 --boundaries boundaries.csv -o fingerprints.csv

# Classify reads by barcode
escpod demux classify fingerprints.csv --model model.json -o classifications.csv

# Split reads into per-barcode files
escpod demux split *.pod5 --classifications classifications.csv -d demuxed/
```

See the [demux documentation](https://rnabioco.github.io/escapepod-rs/cli/demux/) for details on training custom models and SVM-based classification.

## Install

```bash
cargo install --git https://github.com/rnabioco/escapepod-rs
```

## License

MIT

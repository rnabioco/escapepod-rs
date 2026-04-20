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
- **BAM integration** - Filter reads by alignment status

Experimental features (barcode demultiplexing, resquiggling) live behind
Cargo feature flags and ship separately — see the [docs](https://rnabioco.github.io/escapepod-rs/) for status and build instructions.

GPU-accelerated DTW for demux classify is available via `--features gpu`
(opt-in, experimental). Uses NVRTC to compile a CUDA kernel at runtime —
no `nvcc` needed at build time, only the CUDA driver + libnvrtc at run.

## Performance

| Command | escapepod | pod5 | Speedup |
|---------|-----------|------|---------|
| inspect | 36 ms | 1.7 s | **47x** |
| view | 238 ms | 4.5 s | **19x** |
| filter | 513 ms | 4.7 s | **9x** |
| subset | 2.8 s | 8.3 s | **3x** |
| merge | 3.0 s | 4.1 s | **1.4x** |

## Install

```bash
cargo install --git https://github.com/rnabioco/escapepod-rs
```

## License

MIT

# escapepod

A fast Rust CLI for Oxford Nanopore POD5 files.

[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![escapepod demo](docs/images/readme.gif)

## Highlights

- **Fast** - Up to 43x faster than Python pod5 tools
- **Memory efficient** - Memory-mapped I/O for large files
- **Full featured** - View, inspect, merge, filter, subset, and repack
- **BAM integration** - Filter reads by alignment status

## Performance

| Command | escapepod | pod5 | Speedup |
|---------|-----------|------|---------|
| inspect | 5 ms | 225 ms | **43x** |
| view | 18 ms | 458 ms | **25x** |
| merge | 1.3 s | 4.6 s | **3.6x** |
| filter | 66 ms | 539 ms | **8x** |

## Install

```bash
cargo install --git https://github.com/rnabioco/escapepod-rs
```

## License

MIT

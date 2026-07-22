# escapepod

Fast Python reader/writer for Oxford Nanopore **POD5** files, backed by a pure
Rust engine (the same engine as the [`escpod`](https://github.com/rnabioco/escapepod-rs)
CLI). It mirrors the API of Oxford Nanopore's official
[`pod5`](https://github.com/nanoporetech/pod5-file-format) package closely
enough to be a mostly drop-in replacement, while reading and writing
considerably faster.

> **Alpha quality and under active development.** APIs and output formats may
> change without notice. Verify results against the official ONT `pod5` tools
> before relying on it for anything important.

## Install

```bash
uv pip install escapepod
```

Wheels are published for CPython 3.9+ (abi3) on Linux (x86_64/aarch64,
manylinux + musllinux) and macOS (x86_64/arm64).

## Usage

```python
import escapepod

with escapepod.Reader("experiment.pod5") as reader:
    print(f"{reader.read_count} reads")
    for read in reader:
        signal = reader.get_signal(read)          # raw ADC (int16)
        print(read.read_id, read.num_samples, signal.mean())
```

See the [documentation](https://rnabioco.github.io/escapepod-rs/python/) for the
full API.

## License

MIT

# Python API

`escapepod` is a Python package for reading and writing POD5 files, backed by
the same Rust engine as the `escpod` CLI. It mirrors the API of Oxford
Nanopore's official [`pod5`](https://github.com/nanoporetech/pod5-file-format)
package closely enough to be a mostly drop-in replacement, while reading and
writing considerably faster.

```python
import escapepod

with escapepod.Reader("experiment.pod5") as reader:
    print(f"{reader.read_count} reads")
    for read in reader:
        signal = reader.get_signal(read)          # raw ADC (int16)
        print(read.read_id, read.num_samples, signal.mean())
```

## Installation

```bash
pip install escapepod
# or, with uv:
uv pip install escapepod
```

Wheels are published for CPython 3.9+ (abi3) on Linux (x86_64/aarch64,
manylinux + musllinux) and macOS (x86_64/arm64).

### Building from source

To build against a checkout of the repository (or on a platform without a
prebuilt wheel), use [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin develop --release --manifest-path crates/escapepod-python/Cargo.toml
```

`maturin develop` compiles the extension and installs it as `escapepod` in the
active virtualenv/conda env. Drop `--release` for a faster (unoptimized) build
while iterating.

You can also install straight from Git once you have a Rust toolchain and
maturin available:

```bash
pip install "git+https://github.com/rnabioco/escapepod-rs.git#subdirectory=crates/escapepod-python"
```

Verify the install:

```python
import escapepod
print(escapepod.__version__)
```

## What's in the package

| Object | Purpose |
|--------|---------|
| [`Reader`](reading.md) | Read reads, metadata, and signal from a single POD5 file |
| [`DatasetReader`](reading.md#reading-many-files-datasetreader) | Read a directory or list of files as one logical dataset |
| [`Writer`](writing.md) | Create new POD5 files |
| [`ReadData`](reading.md#the-readdata-object) | A single read's metadata |
| [`RunInfo`](writing.md#run-info) / `create_run_info` | Acquisition/run metadata |
| [Signal processing](signal.md) | `normalize_signal`, `mad_normalize`, `refine_signal_map`, `KmerTable` |
| `Pod5Error` | Raised on malformed/invalid POD5 data |

## Coming from `pod5`

The API is intentionally close to the official package. The main differences:

| Task | `pod5` | `escapepod` |
|------|--------|-------------|
| Open a file | `pod5.Reader(path)` | `escapepod.Reader(path)` |
| Open a directory | `pod5.DatasetReader(path)` | `escapepod.DatasetReader(path)` |
| Iterate reads | `for read in reader.reads():` | `for read in reader:` (or `reader.reads()`) |
| Raw signal | `read.signal` | `reader.get_signal(read)` |
| Signal in pA | `read.signal_pa` | `reader.get_signal_pa(read)` |
| Reads → DataFrame | *(manual)* | `reader.to_pandas()` / `reader.to_polars()` |

The most important structural difference: signal is **not** attached to the
`ReadData` object. You request it from the reader with `get_signal(read)` (raw
ADC `int16`) or `get_signal_pa(read)` (calibrated picoamps `float32`). Keeping
metadata and signal separate lets you scan every read's metadata without paying
to decode signal you may not need.

See [Reading Files](reading.md) and [Writing Files](writing.md) for the full
surface, and [Signal Processing](signal.md) for the analysis helpers.

## Error handling

Invalid or corrupt POD5 data raises `escapepod.Pod5Error`; ordinary I/O
problems (missing file, permissions) raise the usual `OSError`/`IOError`:

```python
import escapepod

try:
    reader = escapepod.Reader("experiment.pod5")
except escapepod.Pod5Error as e:
    print(f"Not a valid POD5 file: {e}")
except OSError as e:
    print(f"Could not open file: {e}")
```

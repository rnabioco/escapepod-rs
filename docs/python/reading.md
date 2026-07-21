# Reading POD5 Files

`escapepod.Reader` gives efficient, memory-mapped access to a single POD5
file. `escapepod.DatasetReader` presents many files as one stream.

## Opening a file

```python linenums="1"
import escapepod

reader = escapepod.Reader("experiment.pod5")
```

`Reader` accepts a `str` or any `os.PathLike` (e.g. `pathlib.Path`). Use it as a
context manager so the underlying file is released promptly:

```python linenums="1"
from pathlib import Path

with escapepod.Reader(Path("experiment.pod5")) as reader:
    print(reader.read_count)
```

## File information

Reader exposes file-level metadata as properties:

```python linenums="1"
reader.read_count        # number of reads
reader.read_batch_count  # number of internal read batches
reader.signal_row_count  # number of signal rows
reader.file_identifier   # file UUID
reader.software          # writer software string
reader.pod5_version      # POD5 format version
reader.run_infos         # list[RunInfo]
reader.has_index         # whether a read-id index is built
len(reader)              # same as reader.read_count
```

## Iterating over reads

Iterating the reader yields [`ReadData`](#the-readdata-object) objects — one per
read, metadata only:

```python linenums="1"
for read in reader:
    print(read.read_id, read.channel, read.num_samples, read.end_reason)
```

`reader.reads()` returns the same reads as a `list`. Pass a `selection` of read
IDs to restrict it:

```python linenums="1"
reads = reader.reads()                              # every read
subset = reader.reads(selection=["<uuid-1>", "<uuid-2>"])
subset = reader.reads(selection=ids, missing_ok=True)  # skip IDs not present
```

By default a requested ID that isn't in the file raises `KeyError`; pass
`missing_ok=True` to silently skip it.

`reader.read_ids()` returns just the IDs as a `list[str]`.

## Looking up specific reads

```python linenums="1"
read = reader.get_read("<uuid>")                    # one read, KeyError if absent
reads = reader.get_reads(["<uuid-1>", "<uuid-2>"])  # many reads
reads = reader.get_reads(ids, missing_ok=True)      # skip absent IDs
```

Repeated lookups build and reuse an in-memory index. Call
`reader.build_index()` up front to pay that cost once (it returns the number of
reads indexed); `reader.has_index` reports whether it's built.

## Accessing signal data

Signal is stored separately from metadata and is fetched on demand from the
reader — it is not an attribute of `ReadData`. Request it per read:

```python linenums="1"
read = reader.reads()[0]

signal = reader.get_signal(read)      # numpy int16, raw ADC values
signal_pa = reader.get_signal_pa(read)  # numpy float32, picoamps (calibrated)
```

`get_signal` returns raw ADC counts as `int16`. `get_signal_pa` applies the
read's calibration (`(adc + offset) * scale`) and returns `float32` picoamps.

For many reads at once, the bulk variants decode in parallel and return
`(read_id, signal)` tuples:

```python linenums="1"
reads = reader.reads()
for read_id, signal in reader.get_signals(reads):       # int16 ADC
    ...
for read_id, signal_pa in reader.get_signals_pa(reads):  # float32 pA
    ...
```

`reader.prefetch_signal()` is an optional hint that warms the signal region of
the file; `reader.byte_count(read)` reports the compressed on-disk size of a
read's signal.

## Reads as a DataFrame

Pull every read's metadata into a table in one call:

```python linenums="1"
d = reader.to_dict()        # dict[str, list] — column name -> values
df = reader.to_pandas()     # pandas.DataFrame  (requires pandas)
df = reader.to_polars()     # polars.DataFrame  (requires polars)
```

All three accept the same `selection` / `missing_ok` arguments as `reads()`.
Columns include `read_id`, `channel`, `well`, `num_samples`,
`calibration_offset`, `calibration_scale`, `end_reason`, and the rest of the
read metadata fields.

```python linenums="1"
df = reader.to_pandas()
long_reads = df[df["num_samples"] > 100_000]
print(long_reads[["read_id", "channel", "num_samples"]])
```

## The `ReadData` object

Each read is a `ReadData` with the POD5 read fields as read-only properties:

```python linenums="1"
read.read_id             # str (UUID)
read.read_number
read.start_sample
read.channel
read.well
read.pore_type
read.calibration_offset
read.calibration_scale
read.median_before
read.end_reason          # e.g. "signal_positive", "mux_change", "unknown"
read.end_reason_forced
read.run_info_index      # index into reader.run_infos
read.num_samples
read.signal_rows
# plus scaling/mux fields: tracked_scaling_*, predicted_scaling_*,
# num_reads_since_mux_change, time_since_mux_change, open_pore_level, ...
```

If you already hold a raw ADC array, `read.calibrate_signal_array(adc)` converts
it to picoamps using that read's calibration:

```python linenums="1"
adc = reader.get_signal(read)
pa = read.calibrate_signal_array(adc)   # numpy float32
```

## Reading many files (`DatasetReader`)

`DatasetReader` reads a single file, a directory (scanned for `*.pod5`), or a
list mixing both, and presents every read across all files as one stream — the
escapepod analogue of `pod5.DatasetReader`.

```python linenums="1"
# A directory (recurses by default)
with escapepod.DatasetReader("run_dir/") as ds:
    print(ds.file_count, "files,", ds.read_count, "reads")
    for read in ds:
        signal = ds.get_signal(read)

# An explicit list of files and/or directories
ds = escapepod.DatasetReader(["a.pod5", "b.pod5", "more_reads/"])
```

Control the directory scan with keyword arguments:

```python linenums="1"
ds = escapepod.DatasetReader("run_dir/", recursive=False, pattern="*.pod5")
```

`DatasetReader` offers the same reading surface as `Reader` —
`reads()`/`read_ids()`, `to_dict()`/`to_pandas()`/`to_polars()`,
`get_signal()`/`get_signal_pa()` and their bulk `get_signals*` forms,
`byte_count()`, iteration, and `len()` — plus `paths`, `file_count`,
`read_count`, and `run_infos` properties.

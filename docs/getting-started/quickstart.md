# Quick Start

This guide will help you get started with escapepod-rs in just a few minutes.

## CLI Quick Start

### Viewing POD5 Files

The simplest way to explore a POD5 file is with the `view` command:

```bash
escpod view experiment.pod5
```

This displays a table of reads with key information like read ID, channel, and sample count.

### Inspecting File Details

Get a summary of the file:

```bash
escpod inspect summary experiment.pod5
```

Output:
```
File: experiment.pod5
Reads: 10,000
Run info entries: 1
File size: 1.2 GB
```

List all reads:

```bash
escpod inspect reads experiment.pod5
```

### Merging Files

Combine multiple POD5 files from a sequencing run:

```bash
escpod merge -o combined.pod5 file1.pod5 file2.pod5 file3.pod5
```

### Filtering Reads

Create a file with read IDs you want to extract (one UUID per line):

```
a1b2c3d4-e5f6-7890-abcd-ef1234567890
b2c3d4e5-f6a7-8901-bcde-f12345678901
```

Then filter:

```bash
escpod filter -i read_ids.txt -o filtered.pod5 experiment.pod5
```

## Python Quick Start

The `escapepod` Python package offers a `pod5`-compatible API. See
[Installation](installation.md#installing-the-python-package) to build it, then:

### Reading a POD5 File

```python linenums="1"
import escapepod

with escapepod.Reader("experiment.pod5") as reader:
    print(f"Total reads: {reader.read_count}")

    for read in reader:
        signal = reader.get_signal(read)   # raw ADC values (numpy int16)
        print(read.read_id)
        print(f"  Channel: {read.channel}")
        print(f"  Samples: {read.num_samples}")
        print(f"  Mean signal: {signal.mean():.2f}")
```

Pull every read's metadata into a DataFrame in one call:

```python linenums="1"
with escapepod.Reader("experiment.pod5") as reader:
    df = reader.to_pandas()      # or reader.to_polars()

print(df[["read_id", "channel", "num_samples"]])
```

### Accessing Run Information

```python linenums="1"
with escapepod.Reader("experiment.pod5") as reader:
    for run_info in reader.run_infos:
        print(f"Acquisition ID: {run_info.acquisition_id}")
        print(f"Sample rate: {run_info.sample_rate} Hz")
        print(f"Flow cell: {run_info.flow_cell_id}")
        print(f"Protocol: {run_info.protocol_name}")

        # Context tags contain experiment metadata
        for key, value in run_info.context_tags.items():
            print(f"  {key}: {value}")
```

### Writing a POD5 File

```python linenums="1"
import uuid
import numpy as np
import escapepod

run_info = escapepod.create_run_info(
    acquisition_id="my_experiment",
    sample_rate=4000,
    flow_cell_id="FAK12345",
)

with escapepod.Writer("output.pod5") as writer:
    run_idx = writer.add_run_info(run_info)

    for i in range(10):
        signal = (np.sin(np.arange(1000) * 0.1) * 100).astype(np.int16)
        writer.add_read(
            read_id=str(uuid.uuid4()),
            read_number=i + 1,
            start_sample=0,
            channel=1,
            well=1,
            pore_type="not_set",
            calibration_offset=0.0,
            calibration_scale=1.0,
            median_before=200.0,
            end_reason="signal_positive",
            end_reason_forced=False,
            run_info_index=run_idx,
            num_minknow_events=0,
            signal=signal,
        )
```

Prefer Rust? The same operations are available in the
[Rust Library](../library/index.md).

## Next Steps

- [Python API](../python/index.md) - Full Python reading/writing/signal reference
- [CLI Reference](../cli/index.md) - Full documentation of all commands
- [File Format](../format/index.md) - Understanding the POD5 format

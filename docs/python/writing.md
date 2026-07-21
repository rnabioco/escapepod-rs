# Writing POD5 Files

`escapepod.Writer` creates new POD5 files. Signal is compressed with the VBZ
codec automatically. Use it as a context manager so the footer is written and
the file is finalized on exit.

## A minimal round trip

```python linenums="1"
import numpy as np
import escapepod

# 1. Describe the acquisition/run
run_info = escapepod.create_run_info(
    acquisition_id="acq-001",
    sample_rate=4000,
    experiment_name="demo",
)

# 2. Write a file
signal = np.arange(1000, dtype=np.int16)   # raw ADC values

with escapepod.Writer("out.pod5") as writer:
    ri_idx = writer.add_run_info(run_info)
    writer.add_read(
        read_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        read_number=1,
        start_sample=0,
        channel=42,
        well=1,
        pore_type="not_set",
        calibration_offset=0.0,
        calibration_scale=1.0,
        median_before=200.0,
        end_reason="signal_positive",
        end_reason_forced=False,
        run_info_index=ri_idx,
        num_minknow_events=100,
        signal=signal,
    )

# 3. Read it back
with escapepod.Reader("out.pod5") as reader:
    read = reader.reads()[0]
    assert reader.get_signal(read).tolist() == signal.tolist()
```

## Run info

Every read references a run info by index. Create one with `create_run_info`
(or construct a [`RunInfo`](#the-runinfo-object) directly) and register it with
`writer.add_run_info`, which returns the index to pass as `run_info_index`:

```python linenums="1"
run_info = escapepod.create_run_info(
    acquisition_id="acq-001",     # required
    sample_rate=4000,
    experiment_name="demo",
    flow_cell_id="FA01234",
    sequencing_kit="SQK-RNA004",
    context_tags={"experiment_type": "rna"},
    tracking_id={"device_id": "MN00001"},
)

with escapepod.Writer("out.pod5") as writer:
    ri_idx = writer.add_run_info(run_info)
    # ... add reads referencing ri_idx
```

Only `acquisition_id` is required; every other field has a sensible default
(see the [`RunInfo`](#the-runinfo-object) constructor for the full list).

## Adding reads

`add_read` takes the read fields as keyword arguments plus a numpy `int16`
signal array. The required fields are shown in the round trip above; the
optional scaling/mux fields (`tracked_scaling_scale`, `predicted_scaling_shift`,
`open_pore_level`, …) default to zero/one and can be omitted.

Signal must be `numpy.int16` (raw ADC). `num_samples` is inferred from the
array length unless you pass it explicitly.

### From an existing `ReadData`

If you already have a [`ReadData`](reading.md#the-readdata-object) — for example
when copying or transforming reads from another file — write it directly with
`add_read_data`:

```python linenums="1"
with escapepod.Reader("in.pod5") as reader, escapepod.Writer("out.pod5") as writer:
    src_ri = reader.run_infos[0]
    ri_idx = writer.add_run_info(escapepod.create_run_info(
        acquisition_id=src_ri.acquisition_id,
        sample_rate=src_ri.sample_rate,
    ))
    for read in reader:
        signal = reader.get_signal(read)
        writer.add_read_data(read, signal)
```

For many reads at once, `add_reads(reads, signals)` takes a list of `ReadData`
and a matching list of `int16` signal arrays.

## Writer options

Tuning options are keyword arguments on the constructor; all are optional:

```python linenums="1"
writer = escapepod.Writer(
    "out.pod5",
    compress_signal=True,        # VBZ-compress signal (default on)
    signal_batch_size=None,      # reads per signal record batch
    read_batch_size=None,        # reads per reads record batch
    max_signal_chunk_size=None,  # max samples per signal chunk
    software="my-tool 1.0",      # writer software string
)
```

## Closing

Exiting the `with` block finalizes the file. If you don't use a context
manager, call `writer.close()` explicitly — the file is not valid until it is
closed.

```python linenums="1"
writer = escapepod.Writer("out.pod5")
# ... add_run_info / add_read ...
writer.close()
```

## The `RunInfo` object

`create_run_info(...)` returns a `RunInfo`; you can also construct one directly
with `escapepod.RunInfo(acquisition_id, ...)`. Its fields are exposed as
read-only properties: `acquisition_id`, `acquisition_start_time`, `adc_min`,
`adc_max`, `experiment_name`, `flow_cell_id`, `flow_cell_product_code`,
`protocol_name`, `protocol_run_id`, `protocol_start_time`, `sample_id`,
`sample_rate`, `sequencing_kit`, `sequencer_position`,
`sequencer_position_type`, `software`, `system_name`, `system_type`,
`context_tags`, and `tracking_id`.

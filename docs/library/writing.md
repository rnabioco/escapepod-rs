# Writing POD5 Files

The `Writer` struct creates new POD5 files with automatic signal compression.

## Creating a Writer

```rust
use podfive_core::{Writer, WriterOptions};

// Default options
let writer = Writer::create("output.pod5", WriterOptions::default())?;

// Custom options
let options = WriterOptions {
    signal_compression: true,
    signal_chunk_size: 102_400,
    ..Default::default()
};
let writer = Writer::create("output.pod5", options)?;
```

## Adding Run Info

Every read references a run info entry. Add run info before adding reads:

```rust
use podfive_core::RunInfoData;
use std::collections::HashMap;

let run_info = RunInfoData {
    acquisition_id: "abc123-def456".to_string(),
    acquisition_start_time_ms: 1705320600000,
    adc_max: 2047,
    adc_min: -2048,
    sample_rate: 4000,
    context_tags: HashMap::new(),
    tracking_id: {
        let mut m = HashMap::new();
        m.insert("flow_cell_id".to_string(), "FAK12345".to_string());
        m
    },
    ..Default::default()
};

let run_info_index = writer.add_run_info(run_info)?;
```

## Adding Reads

Add reads with their signal data:

```rust
use podfive_core::{ReadData, EndReason};
use uuid::Uuid;

let read = ReadData {
    read_id: Uuid::new_v4(),
    read_number: 1,
    start_sample: 0,
    channel: 1,
    well: 1,
    pore_type: "not_set".to_string(),
    calibration_offset: -240.0,
    calibration_scale: 0.145,
    median_before: 210.5,
    end_reason: EndReason::SignalPositive,
    end_reason_forced: false,
    run_info_index,
    num_minknow_events: 100,
    num_samples: 50000,
    ..Default::default()
};

// Signal as raw ADC values
let signal: Vec<i16> = vec![/* ... */];

writer.add_read(read, &signal)?;
```

## Finishing the File

Always call `finish()` to write the footer and finalize the file:

```rust
writer.finish()?;
```

The file is not valid until `finish()` completes successfully.

## Complete Example

```rust
use podfive_core::{Writer, WriterOptions, ReadData, RunInfoData, EndReason};
use uuid::Uuid;
use std::collections::HashMap;

fn write_pod5_file() -> Result<(), podfive_core::Error> {
    let mut writer = Writer::create("output.pod5", WriterOptions::default())?;

    // Add run info
    let run_info = RunInfoData {
        acquisition_id: "run-001".to_string(),
        sample_rate: 4000,
        adc_max: 2047,
        adc_min: -2048,
        ..Default::default()
    };
    let run_idx = writer.add_run_info(run_info)?;

    // Add some reads
    for i in 0..10 {
        let read = ReadData {
            read_id: Uuid::new_v4(),
            read_number: i + 1,
            channel: ((i % 512) + 1) as u16,
            well: ((i % 4) + 1) as u8,
            run_info_index: run_idx,
            num_samples: 10000,
            end_reason: EndReason::SignalPositive,
            ..Default::default()
        };

        // Generate dummy signal
        let signal: Vec<i16> = (0..10000).map(|j| ((j % 500) as i16) - 250).collect();

        writer.add_read(read, &signal)?;
    }

    writer.finish()?;
    Ok(())
}
```

## WriterOptions

| Option | Default | Description |
|--------|---------|-------------|
| `signal_compression` | `true` | Enable VBZ compression for signal |
| `signal_chunk_size` | `102400` | Max samples per signal chunk |

## Signal Compression

By default, signal data is compressed using the VBZ codec:

1. **Delta encoding** - Store differences between consecutive samples
2. **Zigzag encoding** - Map signed integers to unsigned
3. **SVB16** - Variable-length encoding (1-2 bytes per sample)
4. **ZSTD** - Final compression pass

This typically achieves 60-80% compression on nanopore signal data.

## Copying Reads Between Files

```rust
use podfive_core::{Reader, Writer, WriterOptions};

let reader = Reader::open("input.pod5")?;
let mut writer = Writer::create("output.pod5", WriterOptions::default())?;

// Copy run infos
for run_info in reader.run_infos() {
    writer.add_run_info(run_info.clone())?;
}

// Copy specific reads
for read in reader.reads() {
    if should_include(&read) {
        let signal = reader.get_signal(&read)?;
        writer.add_read(read, &signal)?;
    }
}

writer.finish()?;
```

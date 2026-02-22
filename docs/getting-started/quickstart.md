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

## Library Quick Start

### Reading a POD5 File

```rust
use escapepod::{Reader, Result};

fn main() -> Result<()> {
    // Open a POD5 file
    let reader = Reader::open("experiment.pod5")?;

    // Print basic info
    println!("Total reads: {}", reader.read_count()?);

    // Iterate over reads
    for read_result in reader.reads()? {
        let read = read_result?;

        println!("Read: {}", read.read_id);
        println!("  Channel: {}", read.channel);
        println!("  Samples: {}", read.num_samples);

        // Get the raw signal data
        let signal = reader.get_signal(&read.signal_rows)?;

        // Calculate mean signal
        let mean: f64 = signal.iter()
            .map(|&s| s as f64)
            .sum::<f64>() / signal.len() as f64;
        println!("  Mean signal: {:.2}", mean);
    }

    Ok(())
}
```

### Accessing Run Information

```rust
use escapepod::Reader;

let reader = Reader::open("experiment.pod5")?;

for run_info in reader.run_infos() {
    println!("Acquisition ID: {}", run_info.acquisition_id);
    println!("Sample rate: {} Hz", run_info.sample_rate);
    println!("Flow cell: {}", run_info.flow_cell_id);
    println!("Protocol: {}", run_info.protocol_name);

    // Context tags contain experiment metadata
    for (key, value) in &run_info.context_tags {
        println!("  {}: {}", key, value);
    }
}
```

### Writing a POD5 File

```rust
use escapepod::{Writer, WriterOptions, ReadData, RunInfoData};

fn main() -> escapepod::Result<()> {
    // Create a new POD5 file
    let mut writer = Writer::create("output.pod5", WriterOptions::default())?;

    // Add run information
    let run_info = RunInfoData {
        acquisition_id: "my_experiment".to_string(),
        sample_rate: 4000,
        flow_cell_id: "FAK12345".to_string(),
        ..Default::default()
    };
    let run_idx = writer.add_run_info(run_info)?;

    // Add reads with signal data
    for i in 0..10 {
        let read = ReadData {
            read_id: uuid::Uuid::new_v4(),
            read_number: i + 1,
            channel: 1,
            run_info_index: run_idx,
            num_samples: 1000,
            ..Default::default()
        };

        // Generate or provide signal data
        let signal: Vec<i16> = (0..1000)
            .map(|j| ((j as f64 * 0.1).sin() * 100.0) as i16)
            .collect();

        writer.add_read(read, &signal)?;
    }

    // Finalize the file
    writer.finish()?;

    Ok(())
}
```

## Next Steps

- [CLI Reference](../cli/index.md) - Full documentation of all commands
- [Library Guide](../library/index.md) - Detailed library usage
- [File Format](../format/index.md) - Understanding the POD5 format

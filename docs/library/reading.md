# Reading POD5 Files

The `Reader` struct provides efficient access to POD5 file contents.

## Opening a File

```rust
use escapepod::Reader;

let reader = Reader::open("experiment.pod5")?;
```

The file is memory-mapped for efficient access. The reader validates the file signature and parses the footer on open.

## File Information

```rust
// Number of reads in the file
let count = reader.read_count();

// Number of batches (internal structure)
let batches = reader.batch_count();

// Access run info
for (i, run_info) in reader.run_infos().iter().enumerate() {
    println!("Run {}: {}", i, run_info.acquisition_id);
}
```

## Iterating Over Reads

```rust
// Iterate over all reads
for read in reader.reads() {
    println!("Read: {}", read.read_id);
    println!("  Channel: {}", read.channel);
    println!("  Samples: {}", read.num_samples);
    println!("  End reason: {:?}", read.end_reason);
}
```

## Accessing Signal Data

Signal data is stored separately and must be explicitly requested:

```rust
// Get signal for a specific read
let signal: Vec<i16> = reader.get_signal(&read)?;

// Signal is raw ADC values
println!("Signal length: {}", signal.len());
println!("First 10 samples: {:?}", &signal[..10.min(signal.len())]);
```

## Converting to Physical Units

The signal is stored as raw ADC values. Convert to picoamps (pA) using calibration:

```rust
fn to_picoamps(signal: &[i16], offset: f32, scale: f32) -> Vec<f32> {
    signal.iter()
        .map(|&s| (s as f32 + offset) * scale)
        .collect()
}

let pa_signal = to_picoamps(&signal, read.calibration_offset, read.calibration_scale);
```

## Finding Specific Reads

Find reads by their UUIDs:

```rust
use uuid::Uuid;

let target_ids = vec![
    Uuid::parse_str("a1b2c3d4-e5f6-7890-abcd-ef1234567890")?,
];

let found_reads = reader.find_reads(&target_ids)?;
```

## Batch Access

For advanced use cases, access raw Arrow batches:

```rust
for batch_idx in 0..reader.batch_count() {
    let batch = reader.read_batch(batch_idx)?;
    println!("Batch {} has {} rows", batch_idx, batch.num_rows());
}
```

## Run Info Access

Access run information by index:

```rust
// Get run info for a read
let run_info = reader.get_run_info(read.run_info_index)?;

println!("Acquisition ID: {}", run_info.acquisition_id);
println!("Sample rate: {} Hz", run_info.sample_rate);
println!("Flow cell: {:?}", run_info.tracking_id.get("flow_cell_id"));
```

## Memory Efficiency

The reader uses memory-mapping, so:

- Only accessed data is loaded into memory
- Large files don't consume proportional RAM
- Multiple readers can share the same memory-mapped data

```rust
// Safe to open very large files
let reader = Reader::open("large_file.pod5")?;  // Doesn't load entire file
```

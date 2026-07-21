# Types Reference

Core data structures used throughout the library.

## ReadData

Represents a single nanopore read.

```rust linenums="1"
pub struct ReadData {
    /// Unique identifier for the read
    pub read_id: Uuid,

    /// Sequential read number within the channel
    pub read_number: u32,

    /// Sample position where read started
    pub start_sample: u64,

    /// Channel number (1-512 typically)
    pub channel: u16,

    /// Well number (1-4)
    pub well: u8,

    /// Pore type string
    pub pore_type: String,

    /// Calibration offset for converting ADC to pA
    pub calibration_offset: f32,

    /// Calibration scale for converting ADC to pA
    pub calibration_scale: f32,

    /// Median current level before read started
    pub median_before: f32,

    /// Why the read ended
    pub end_reason: EndReason,

    /// Whether end reason was forced by software
    pub end_reason_forced: bool,

    /// Index into run info table
    pub run_info_index: u32,

    /// Number of MinKNOW events
    pub num_minknow_events: u64,

    /// Total number of signal samples
    pub num_samples: u64,

    /// Estimated open pore current level
    pub open_pore_level: f32,

    /// Signal row indices (internal use)
    pub signal_rows: Vec<u64>,
}
```

### Converting Signal to Physical Units

```rust linenums="1"
fn adc_to_picoamps(adc: i16, read: &ReadData) -> f32 {
    (adc as f32 + read.calibration_offset) * read.calibration_scale
}
```

## RunInfoData

Metadata about a sequencing run.

```rust linenums="1"
pub struct RunInfoData {
    /// Unique acquisition identifier
    pub acquisition_id: String,

    /// Start time in milliseconds since epoch
    pub acquisition_start_time: i64,

    /// Maximum ADC value
    pub adc_max: i16,

    /// Minimum ADC value
    pub adc_min: i16,

    /// Context tags (key-value metadata)
    pub context_tags: HashMap<String, String>,

    /// Experiment name
    pub experiment_name: String,

    /// Flow cell ID
    pub flow_cell_id: String,

    /// Flow cell product code
    pub flow_cell_product_code: String,

    /// Protocol name
    pub protocol_name: String,

    /// Protocol run ID
    pub protocol_run_id: String,

    /// Protocol start time in milliseconds since epoch
    pub protocol_start_time: i64,

    /// Sample ID
    pub sample_id: String,

    /// Sampling rate in Hz
    pub sample_rate: u16,

    /// Sequencing kit name
    pub sequencing_kit: String,

    /// Sequencer position identifier
    pub sequencer_position: String,

    /// Sequencer position type
    pub sequencer_position_type: String,

    /// Software that produced the data
    pub software: String,

    /// System name
    pub system_name: String,

    /// System type
    pub system_type: String,

    /// Tracking ID metadata (key-value pairs)
    pub tracking_id: HashMap<String, String>,
}
```

### Common Tracking ID Fields

| Key | Description |
|-----|-------------|
| `flow_cell_id` | Flow cell identifier |
| `device_id` | Sequencer device ID |
| `sample_id` | User-provided sample name |
| `experiment_id` | Experiment identifier |
| `protocol_group_id` | Protocol group |

## EndReason

Why a read ended.

```rust linenums="1"
pub enum EndReason {
    Unknown,
    MuxChange,
    UnblockMuxChange,
    DataServiceUnblockMuxChange,
    SignalPositive,
    SignalNegative,
}
```

| Variant | Description |
|---------|-------------|
| `Unknown` | Reason not recorded |
| `MuxChange` | Mux changed to different well |
| `UnblockMuxChange` | Unblock triggered mux change |
| `DataServiceUnblockMuxChange` | Data service triggered unblock |
| `SignalPositive` | Normal end, positive signal |
| `SignalNegative` | Normal end, negative signal |

## Error

Error types returned by library operations.

```rust linenums="1"
pub enum Error {
    /// I/O error
    Io(std::io::Error),

    /// Invalid POD5 file signature
    InvalidSignature,

    /// Invalid footer structure
    InvalidFooter(String),

    /// Arrow error during IPC operations
    Arrow(arrow::error::ArrowError),

    /// Compression/decompression error
    Compression(String),

    /// Missing required field
    MissingField(String),

    /// Read not found
    ReadNotFound(Uuid),
}
```

### Error Handling Example

```rust linenums="1"
use escapepod_signal::{Reader, Error};

fn process_file(path: &str) -> Result<(), String> {
    let reader = Reader::open(path).map_err(|e| match e {
        Error::Io(io_err) => format!("Cannot open file: {}", io_err),
        Error::InvalidSignature => "Not a valid POD5 file".to_string(),
        Error::InvalidFooter(msg) => format!("Corrupt file: {}", msg),
        _ => format!("Error: {}", e),
    })?;

    // Process file...
    Ok(())
}
```

## WriterOptions

Configuration for file writing.

```rust linenums="1"
pub struct WriterOptions {
    /// Enable VBZ signal compression (default: true)
    pub signal_compression: bool,

    /// Maximum samples per signal chunk (default: 102400)
    pub signal_chunk_size: u32,
}
```

## UUID Handling

Read IDs are UUIDs stored as 16-byte fixed-size binary:

```rust linenums="1"
use uuid::Uuid;

// Parse from string
let id = Uuid::parse_str("a1b2c3d4-e5f6-7890-abcd-ef1234567890")?;

// Generate new
let new_id = Uuid::new_v4();

// Access bytes
let bytes: &[u8; 16] = id.as_bytes();
```

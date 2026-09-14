# Types Reference

Core data structures used throughout the library.

## ReadData

Represents a single nanopore read.

```rust linenums="1"
pub struct ReadData {
    /// Unique identifier for the read
    pub read_id: Uuid,

    /// Sequential read number within the run
    pub read_number: u32,

    /// Sample position where read started
    pub start_sample: u64,

    /// Channel number (1-indexed). Widened to `u32` to hold either physical
    /// width: POD5 V6 stores this as `uint32`, V3-V5 as `uint16`. escapepod
    /// still *writes* the V5 narrow column, so a value above `u16::MAX`
    /// is refused rather than truncated.
    pub channel: u32,

    /// Well number (typically 1-4)
    pub well: u8,

    /// Chemistry/pore-type identifier (e.g. `"dna_r10.4.1"`, `"rna_004"`).
    /// A thin `Arc<str>` newtype, not a closed enum — POD5 leaves this
    /// column open-vocabulary. `.as_str()` / `AsRef<str>` / `Display` borrow
    /// the string; cloning is a refcount bump.
    pub pore_type: PoreType,

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

    /// Number of MinKNOW events in this read
    pub num_minknow_events: u64,

    /// Tracked scaling scale/shift (post-hoc rescaling MinKNOW recorded)
    pub tracked_scaling_scale: f32,
    pub tracked_scaling_shift: f32,

    /// Predicted scaling scale/shift (MinKNOW's live estimate)
    pub predicted_scaling_scale: f32,
    pub predicted_scaling_shift: f32,

    /// Number of reads since the channel's last mux change
    pub num_reads_since_mux_change: u32,

    /// Time since the last mux change, in seconds
    pub time_since_mux_change: f32,

    /// Total number of signal samples
    pub num_samples: u64,

    /// Estimated open pore current level
    pub open_pore_level: f32,

    /// Expected open pore current level for this read (POD5 V5+)
    pub expected_open_pore_level: f32,

    /// Selected pore level for this read (POD5 V5+)
    pub selected_read_level: f32,

    /// Signal row indices into the signal table (internal use)
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
    ApiRequest,
    DeviceDataError,
    AnalysisConfigChange,
    Paused,
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
| `ApiRequest` | Ended by an explicit API request |
| `DeviceDataError` | Ended due to a device data error |
| `AnalysisConfigChange` | Ended by an analysis configuration change |
| `Paused` | Acquisition was paused |

## Error

Error types returned by library operations.

```rust linenums="1"
pub enum Error {
    /// I/O error
    Io(std::io::Error),

    /// Invalid POD5 file signature
    InvalidSignature,

    /// File signature mismatch between start and end (truncated/corrupt)
    SignatureMismatch,

    /// Invalid or corrupted footer
    InvalidFooter(String),

    /// FlatBuffer parsing error
    FlatBuffer(String),

    /// Arrow error during IPC operations
    Arrow(arrow::error::ArrowError),

    /// Signal compression/decompression error
    Compression(String),
    Decompression(String),

    /// Invalid UUID format
    InvalidUuid(String),

    /// Unsupported POD5 schema version
    UnsupportedVersion(String),

    /// Missing required field, or invalid data in a field
    MissingField(String),
    InvalidField { field: String, message: String },

    /// Read ID not found
    ReadNotFound(Uuid),

    /// Batch index out of bounds
    BatchIndexOutOfBounds { index: usize, max: usize },

    // ...plus `.p5s`-sidecar-specific variants (`SidecarIndexMismatch`,
    // `SidecarRowOutOfBounds`, `DictionaryValueNotFound`, `WriterFinalized`,
    // `InvalidArrowIpc`, `InvalidSectionMarker`, `InvalidState`, `Parse`,
    // `Zstd`) for failure modes specific to those subsystems.
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
    /// Maximum number of samples per signal chunk (default: 102400)
    pub max_signal_chunk_size: u32,

    /// Number of signal chunks per Arrow batch (default: 100)
    pub signal_batch_size: u32,

    /// Number of reads per Arrow batch (default: 1000)
    pub read_batch_size: u32,

    /// Whether to compress signal data using VBZ (default: true)
    pub compress_signal: bool,

    /// Software name recorded in the footer (default: `"escapepod-rs <ver>"`)
    pub software: String,

    /// Predefined dictionary values for multi-batch consistency (default: `None`)
    pub predefined_dictionaries: Option<PredefinedDictionaries>,

    /// How hard to push bytes to stable storage before the staged file is
    /// renamed into place (default: [`Durability::None`] — rename only)
    pub durability: Durability,
}
```

`Durability` trades write cost for crash safety: `None` (default, rename
only), `File` (`fsync` the staging file before rename), or `FileAndDir`
(also `fsync` the parent directory, so the rename record itself is durable).

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

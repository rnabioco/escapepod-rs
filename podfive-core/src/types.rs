//! Core data types for POD5 files.

use std::collections::HashMap;

pub use uuid::Uuid;

/// POD5 file signature (PNG-inspired format).
/// This appears at both the start and end of valid POD5 files.
pub const POD5_SIGNATURE: [u8; 8] = [0x8B, b'P', b'O', b'D', b'\r', b'\n', 0x1A, b'\n'];

/// Footer magic bytes that precede the FlatBuffer footer.
pub const FOOTER_MAGIC: [u8; 8] = *b"FOOTER\0\0";

/// Current POD5 specification version.
pub const POD5_VERSION: &str = "1.0.0";

/// Length of section marker UUIDs in bytes.
pub const SECTION_MARKER_LENGTH: usize = 16;

/// Reason why a read ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EndReason {
    Unknown = 0,
    MuxChange = 1,
    UnblockMuxChange = 2,
    DataServiceUnblockMuxChange = 3,
    SignalPositive = 4,
    SignalNegative = 5,
    ApiRequest = 6,
    DeviceDataError = 7,
    AnalysisConfigChange = 8,
    Paused = 9,
}

impl EndReason {
    /// Convert from string representation.
    pub fn from_str(s: &str) -> Self {
        match s {
            "unknown" => Self::Unknown,
            "mux_change" => Self::MuxChange,
            "unblock_mux_change" => Self::UnblockMuxChange,
            "data_service_unblock_mux_change" => Self::DataServiceUnblockMuxChange,
            "signal_positive" => Self::SignalPositive,
            "signal_negative" => Self::SignalNegative,
            "api_request" => Self::ApiRequest,
            "device_data_error" => Self::DeviceDataError,
            "analysis_config_change" => Self::AnalysisConfigChange,
            "paused" => Self::Paused,
            _ => Self::Unknown,
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::MuxChange => "mux_change",
            Self::UnblockMuxChange => "unblock_mux_change",
            Self::DataServiceUnblockMuxChange => "data_service_unblock_mux_change",
            Self::SignalPositive => "signal_positive",
            Self::SignalNegative => "signal_negative",
            Self::ApiRequest => "api_request",
            Self::DeviceDataError => "device_data_error",
            Self::AnalysisConfigChange => "analysis_config_change",
            Self::Paused => "paused",
        }
    }
}

impl Default for EndReason {
    fn default() -> Self {
        Self::Unknown
    }
}

impl std::fmt::Display for EndReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Data for a single read.
#[derive(Debug, Clone)]
pub struct ReadData {
    /// Unique identifier for the read.
    pub read_id: Uuid,
    /// Sequential read number within the run.
    pub read_number: u32,
    /// Start sample number (absolute position in acquisition).
    pub start_sample: u64,
    /// Channel number (1-indexed).
    pub channel: u16,
    /// Well number (typically 1-4).
    pub well: u8,
    /// Pore type string.
    pub pore_type: String,
    /// Calibration offset for converting ADC to pA.
    pub calibration_offset: f32,
    /// Calibration scale for converting ADC to pA.
    pub calibration_scale: f32,
    /// Median current before the read started.
    pub median_before: f32,
    /// Reason the read ended.
    pub end_reason: EndReason,
    /// Whether the end reason was forced.
    pub end_reason_forced: bool,
    /// Index into the run info table.
    pub run_info_index: u32,
    /// Number of MinKNOW events in this read.
    pub num_minknow_events: u64,
    /// Total number of samples in this read's signal.
    pub num_samples: u64,
    /// Estimated open pore current level.
    pub open_pore_level: f32,
    /// Signal row indices into the signal table.
    pub signal_rows: Vec<u64>,
}

impl Default for ReadData {
    fn default() -> Self {
        Self {
            read_id: Uuid::nil(),
            read_number: 0,
            start_sample: 0,
            channel: 0,
            well: 0,
            pore_type: String::new(),
            calibration_offset: 0.0,
            calibration_scale: 1.0,
            median_before: 0.0,
            end_reason: EndReason::Unknown,
            end_reason_forced: false,
            run_info_index: 0,
            num_minknow_events: 0,
            num_samples: 0,
            open_pore_level: 0.0,
            signal_rows: Vec::new(),
        }
    }
}

/// Run information metadata.
#[derive(Debug, Clone)]
pub struct RunInfoData {
    /// Unique acquisition identifier.
    pub acquisition_id: String,
    /// Acquisition start time (milliseconds since epoch).
    pub acquisition_start_time: i64,
    /// Maximum ADC value.
    pub adc_max: i16,
    /// Minimum ADC value.
    pub adc_min: i16,
    /// Context tags (key-value metadata).
    pub context_tags: HashMap<String, String>,
    /// Experiment name.
    pub experiment_name: String,
    /// Flow cell ID.
    pub flow_cell_id: String,
    /// Flow cell product code.
    pub flow_cell_product_code: String,
    /// Protocol name.
    pub protocol_name: String,
    /// Protocol run ID.
    pub protocol_run_id: String,
    /// Protocol start time (milliseconds since epoch).
    pub protocol_start_time: i64,
    /// Sample ID.
    pub sample_id: String,
    /// Sample rate in Hz.
    pub sample_rate: u16,
    /// Sequencing kit name.
    pub sequencing_kit: String,
    /// Sequencer position identifier.
    pub sequencer_position: String,
    /// Sequencer position type.
    pub sequencer_position_type: String,
    /// Software that produced the data.
    pub software: String,
    /// System name.
    pub system_name: String,
    /// System type.
    pub system_type: String,
    /// Tracking ID tags (key-value metadata).
    pub tracking_id: HashMap<String, String>,
}

impl Default for RunInfoData {
    fn default() -> Self {
        Self {
            acquisition_id: String::new(),
            acquisition_start_time: 0,
            adc_max: 0,
            adc_min: 0,
            context_tags: HashMap::new(),
            experiment_name: String::new(),
            flow_cell_id: String::new(),
            flow_cell_product_code: String::new(),
            protocol_name: String::new(),
            protocol_run_id: String::new(),
            protocol_start_time: 0,
            sample_id: String::new(),
            sample_rate: 0,
            sequencing_kit: String::new(),
            sequencer_position: String::new(),
            sequencer_position_type: String::new(),
            software: String::new(),
            system_name: String::new(),
            system_type: String::new(),
            tracking_id: HashMap::new(),
        }
    }
}

/// Signal data type in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Uncompressed int16 samples.
    Uncompressed,
    /// VBZ compressed (SVB16 + ZSTD).
    Vbz,
}

impl Default for SignalType {
    fn default() -> Self {
        Self::Vbz
    }
}

/// A chunk of signal data from the signal table.
#[derive(Debug, Clone)]
pub struct SignalChunk {
    /// Read ID this chunk belongs to.
    pub read_id: Uuid,
    /// Number of samples in this chunk.
    pub samples: u32,
    /// Raw signal data (compressed or uncompressed).
    pub data: Vec<u8>,
    /// Whether the data is VBZ compressed.
    pub is_compressed: bool,
}

/// Index into the signal table.
#[derive(Debug, Clone, Copy)]
pub struct SignalRowIndex {
    /// Batch index in the signal table.
    pub batch: u32,
    /// Row index within the batch.
    pub row: u32,
}

//! Core data types for POD5 files.

use std::collections::HashMap;

pub use uuid::Uuid;

/// POD5 file signature (PNG-inspired format).
/// This appears at both the start and end of valid POD5 files.
pub const POD5_SIGNATURE: [u8; 8] = [0x8B, b'P', b'O', b'D', b'\r', b'\n', 0x1A, b'\n'];

/// Footer magic bytes that precede the FlatBuffer footer.
pub const FOOTER_MAGIC: [u8; 8] = *b"FOOTER\0\0";

/// Current POD5 specification version.
/// See: https://pod5-file-format.readthedocs.io/en/latest/SPECIFICATION.html
pub const POD5_VERSION: &str = "0.3.10";

/// Length of section marker UUIDs in bytes.
pub const SECTION_MARKER_LENGTH: usize = 16;

/// Reason why a read ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum EndReason {
    #[default]
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

impl std::str::FromStr for EndReason {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
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
        })
    }
}

impl std::fmt::Display for EndReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for EndReason {
    fn from(s: &str) -> Self {
        s.parse().unwrap()
    }
}

impl From<u8> for EndReason {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Unknown,
            1 => Self::MuxChange,
            2 => Self::UnblockMuxChange,
            3 => Self::DataServiceUnblockMuxChange,
            4 => Self::SignalPositive,
            5 => Self::SignalNegative,
            6 => Self::ApiRequest,
            7 => Self::DeviceDataError,
            8 => Self::AnalysisConfigChange,
            9 => Self::Paused,
            _ => Self::Unknown,
        }
    }
}

impl From<EndReason> for u8 {
    fn from(reason: EndReason) -> Self {
        reason as u8
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
    /// Tracked scaling scale.
    pub tracked_scaling_scale: f32,
    /// Tracked scaling shift.
    pub tracked_scaling_shift: f32,
    /// Predicted scaling scale.
    pub predicted_scaling_scale: f32,
    /// Predicted scaling shift.
    pub predicted_scaling_shift: f32,
    /// Number of reads since last mux change.
    pub num_reads_since_mux_change: u32,
    /// Time since last mux change (seconds).
    pub time_since_mux_change: f32,
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
            tracked_scaling_scale: 1.0,
            tracked_scaling_shift: 0.0,
            predicted_scaling_scale: 1.0,
            predicted_scaling_shift: 0.0,
            num_reads_since_mux_change: 0,
            time_since_mux_change: 0.0,
            num_samples: 0,
            open_pore_level: 0.0,
            signal_rows: Vec::new(),
        }
    }
}

impl ReadData {
    /// Create a copy of this ReadData suitable for writing to a new file.
    ///
    /// The signal_rows are cleared as they will be populated by the writer.
    /// Use this when copying reads between files with a different run_info mapping.
    pub fn for_writing(&self, new_run_info_index: u32) -> Self {
        Self {
            read_id: self.read_id,
            read_number: self.read_number,
            start_sample: self.start_sample,
            channel: self.channel,
            well: self.well,
            pore_type: self.pore_type.clone(),
            calibration_offset: self.calibration_offset,
            calibration_scale: self.calibration_scale,
            median_before: self.median_before,
            end_reason: self.end_reason,
            end_reason_forced: self.end_reason_forced,
            run_info_index: new_run_info_index,
            num_minknow_events: self.num_minknow_events,
            tracked_scaling_scale: self.tracked_scaling_scale,
            tracked_scaling_shift: self.tracked_scaling_shift,
            predicted_scaling_scale: self.predicted_scaling_scale,
            predicted_scaling_shift: self.predicted_scaling_shift,
            num_reads_since_mux_change: self.num_reads_since_mux_change,
            time_since_mux_change: self.time_since_mux_change,
            num_samples: self.num_samples,
            open_pore_level: self.open_pore_level,
            signal_rows: Vec::new(),
        }
    }

    /// Create a copy preserving the original run_info_index.
    ///
    /// Equivalent to `for_writing(self.run_info_index)`.
    pub fn for_writing_same_run(&self) -> Self {
        self.for_writing(self.run_info_index)
    }
}

/// Run information metadata.
#[derive(Debug, Clone, Default)]
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

impl std::fmt::Display for ReadData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Read {} (ch:{}, well:{}, samples:{}, end:{})",
            self.read_id, self.channel, self.well, self.num_samples, self.end_reason
        )
    }
}

impl std::fmt::Display for RunInfoData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RunInfo {} (flow_cell:{}, sample:{}, rate:{}Hz)",
            self.acquisition_id, self.flow_cell_id, self.sample_id, self.sample_rate
        )
    }
}

/// Signal data type in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignalType {
    /// Uncompressed int16 samples.
    Uncompressed,
    /// VBZ compressed (SVB16 + ZSTD).
    #[default]
    Vbz,
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

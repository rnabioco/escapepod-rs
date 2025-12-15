//! Error types for POD5 file operations.

use thiserror::Error;

/// Result type alias for POD5 operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when working with POD5 files.
#[derive(Debug, Error)]
pub enum Error {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid file signature - not a POD5 file.
    #[error("Invalid POD5 signature")]
    InvalidSignature,

    /// File signature mismatch between start and end.
    #[error("File signature mismatch - file may be truncated or corrupted")]
    SignatureMismatch,

    /// Invalid or corrupted footer.
    #[error("Invalid footer: {0}")]
    InvalidFooter(String),

    /// FlatBuffer parsing error.
    #[error("FlatBuffer error: {0}")]
    FlatBuffer(String),

    /// Arrow IPC error.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// Signal decompression error.
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// Signal compression error.
    #[error("Compression error: {0}")]
    Compression(String),

    /// Invalid UUID format.
    #[error("Invalid UUID: {0}")]
    InvalidUuid(String),

    /// Schema version not supported.
    #[error("Unsupported schema version: {0}")]
    UnsupportedVersion(String),

    /// Missing required field in record.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid data in a field.
    #[error("Invalid field data: {field}: {message}")]
    InvalidField { field: String, message: String },

    /// Read ID not found.
    #[error("Read ID not found: {0}")]
    ReadNotFound(uuid::Uuid),

    /// Batch index out of bounds.
    #[error("Batch index {index} out of bounds (max: {max})")]
    BatchIndexOutOfBounds { index: usize, max: usize },

    /// Writer has already been finalized.
    #[error("Writer has already been finalized")]
    WriterFinalized,

    /// Section marker validation failed.
    #[error("Invalid section marker at offset {offset}")]
    InvalidSectionMarker { offset: u64 },

    /// ZSTD compression/decompression error.
    #[error("ZSTD error: {0}")]
    Zstd(std::io::Error),
}

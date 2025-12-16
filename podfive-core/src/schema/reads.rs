//! Reads table Arrow schema definition.

use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Extension type name for MinKNOW UUIDs.
pub const UUID_EXTENSION_NAME: &str = "minknow.uuid";

/// Create the Arrow schema for the reads table.
pub fn reads_schema() -> Schema {
    Schema::new(vec![
        // read_id: UUID stored as FixedSizeBinary(16) with extension metadata
        Field::new("read_id", DataType::FixedSizeBinary(16), false).with_metadata(
            [(
                "ARROW:extension:name".to_string(),
                UUID_EXTENSION_NAME.to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        // signal: List of indices into signal table
        Field::new(
            "signal",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            false,
        ),
        // channel: 1-indexed channel number
        Field::new("channel", DataType::UInt16, false),
        // well: well number (typically 1-4)
        Field::new("well", DataType::UInt8, false),
        // pore_type: dictionary-encoded pore type string
        Field::new(
            "pore_type",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        ),
        // calibration_offset: offset for ADC to pA conversion
        Field::new("calibration_offset", DataType::Float32, false),
        // calibration_scale: scale for ADC to pA conversion
        Field::new("calibration_scale", DataType::Float32, false),
        // read_number: sequential read number
        Field::new("read_number", DataType::UInt32, false),
        // start: absolute start sample position
        Field::new("start", DataType::UInt64, false),
        // median_before: median current before read
        Field::new("median_before", DataType::Float32, false),
        // end_reason: dictionary-encoded end reason
        Field::new(
            "end_reason",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        ),
        // end_reason_forced: whether end was forced
        Field::new("end_reason_forced", DataType::Boolean, false),
        // run_info: dictionary index into run info table
        Field::new(
            "run_info",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        ),
        // num_minknow_events: number of events
        Field::new("num_minknow_events", DataType::UInt64, false),
        // num_samples: total signal samples
        Field::new("num_samples", DataType::UInt64, false),
        // open_pore_level: estimated open pore current (V4+)
        Field::new("open_pore_level", DataType::Float32, false),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reads_schema_has_expected_fields() {
        let schema = reads_schema();
        assert!(schema.field_with_name("read_id").is_ok());
        assert!(schema.field_with_name("signal").is_ok());
        assert!(schema.field_with_name("channel").is_ok());
        assert!(schema.field_with_name("num_samples").is_ok());
    }
}

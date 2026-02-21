//! Reads table Arrow schema definition.

use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Extension type name for MinKNOW UUIDs.
pub const UUID_EXTENSION_NAME: &str = "minknow.uuid";

/// Create the Arrow schema for the reads table.
///
/// Field order follows the C++ POD5 write order (V0 through V4):
/// V0: read_id, signal, read_number, start, median_before
/// V1: num_minknow_events, tracked_scaling_scale, tracked_scaling_shift,
///     predicted_scaling_scale, predicted_scaling_shift,
///     num_reads_since_mux_change, time_since_mux_change
/// V2: num_samples
/// V3: channel, well, pore_type, calibration_offset, calibration_scale,
///     end_reason, end_reason_forced, run_info
/// V4: open_pore_level
pub fn reads_schema() -> Schema {
    Schema::new(vec![
        // V0 fields
        Field::new("read_id", DataType::FixedSizeBinary(16), false).with_metadata(
            [(
                "ARROW:extension:name".to_string(),
                UUID_EXTENSION_NAME.to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new(
            "signal",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            false,
        ),
        Field::new("read_number", DataType::UInt32, false),
        Field::new("start", DataType::UInt64, false),
        Field::new("median_before", DataType::Float32, false),
        // V1 fields
        Field::new("num_minknow_events", DataType::UInt64, false),
        Field::new("tracked_scaling_scale", DataType::Float32, false),
        Field::new("tracked_scaling_shift", DataType::Float32, false),
        Field::new("predicted_scaling_scale", DataType::Float32, false),
        Field::new("predicted_scaling_shift", DataType::Float32, false),
        Field::new("num_reads_since_mux_change", DataType::UInt32, false),
        Field::new("time_since_mux_change", DataType::Float32, false),
        // V2 fields
        Field::new("num_samples", DataType::UInt64, false),
        // V3 fields
        Field::new("channel", DataType::UInt16, false),
        Field::new("well", DataType::UInt8, false),
        Field::new(
            "pore_type",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("calibration_offset", DataType::Float32, false),
        Field::new("calibration_scale", DataType::Float32, false),
        Field::new(
            "end_reason",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("end_reason_forced", DataType::Boolean, false),
        Field::new(
            "run_info",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        ),
        // V4 fields
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
        assert!(schema.field_with_name("tracked_scaling_scale").is_ok());
    }
}

//! Reads table Arrow schema definition.

use crate::error::{Error, Result};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Extension type name for MinKNOW UUIDs.
pub const UUID_EXTENSION_NAME: &str = "minknow.uuid";

/// Narrow a channel number onto the `uint16` column this writer emits.
///
/// POD5 V6 (upstream 0.3.46) retypes the *existing* `channel` column from
/// `uint16` to `uint32`. That is not an additive change the way V4 and V5
/// were, so a V6 file is not merely missing fields to an older reader — it is
/// rejected outright ("Schema field 'channel' is incorrect type: 'uint32'"),
/// and the newest `pod5` on PyPI is still 0.3.44. Emitting V6 today would
/// therefore make every file escpod writes unreadable by every installable
/// reader, in exchange for channel numbers no flow cell produces (PromethION
/// tops out at 3000). [`crate::ReadData::channel`] is a `u32` so both widths
/// *read* losslessly; only the write side stays narrow.
///
/// The one input that would actually lose data is refused rather than
/// truncated — a silently wrong channel is worse than a failed write.
pub(crate) fn narrow_channel(channel: u32) -> Result<u16> {
    u16::try_from(channel).map_err(|_| Error::InvalidField {
        field: "channel".to_string(),
        message: format!(
            "{channel} does not fit the uint16 reads-table column of POD5 V5; \
             writing it needs the V6 (uint32) schema, which no released pod5 reader accepts yet"
        ),
    })
}

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
/// V5: expected_open_pore_level, selected_read_level
///
/// This is the V5 schema. V6 retypes `channel` to uint32 in place; we read
/// that width but do not emit it — see [`narrow_channel`].
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
        // V3 fields (V6 retypes this to UInt32; see narrow_channel)
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
        // V5 fields
        Field::new("expected_open_pore_level", DataType::Float32, false),
        Field::new("selected_read_level", DataType::Float32, false),
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

    /// The width we write and the version we stamp have to move together:
    /// POD5 V6 *is* the `uint32` `channel` and nothing else, so emitting one
    /// without the other mislabels every file. Widening the column is a
    /// deliberate, breaking act — this pins it so it cannot happen by drift.
    #[test]
    fn emitted_channel_width_matches_the_stamped_version() {
        let schema = reads_schema();
        assert_eq!(
            schema.field_with_name("channel").unwrap().data_type(),
            &DataType::UInt16,
            "emitting uint32 makes the file V6; bump POD5_VERSION to 0.3.46 with it"
        );
        assert_eq!(crate::types::POD5_VERSION, "0.3.44");
    }

    /// A channel that does not fit the emitted column fails the write; it
    /// must never arrive on disk as `channel % 65536`.
    #[test]
    fn oversized_channel_is_refused_not_truncated() {
        assert_eq!(narrow_channel(3000).unwrap(), 3000);
        assert_eq!(narrow_channel(u16::MAX as u32).unwrap(), u16::MAX);
        let err = narrow_channel(70_000).unwrap_err().to_string();
        assert!(err.contains("70000"), "{err}");
        assert!(err.contains("V6"), "{err}");
    }
}

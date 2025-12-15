//! Signal table Arrow schema definition.

use arrow::datatypes::{DataType, Field, Schema};

/// Extension type name for VBZ-compressed signal data.
pub const VBZ_EXTENSION_NAME: &str = "minknow.vbz";

/// Create the Arrow schema for the signal table.
pub fn signal_schema() -> Schema {
    Schema::new(vec![
        // read_id: UUID for consistency checking
        Field::new("read_id", DataType::FixedSizeBinary(16), false).with_metadata(
            [(
                "ARROW:extension:name".to_string(),
                "minknow.uuid".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        // signal: VBZ-compressed or uncompressed signal data
        // When compressed: LargeBinary with minknow.vbz extension
        // When uncompressed: LargeList<Int16>
        Field::new("signal", DataType::LargeBinary, false).with_metadata(
            [(
                "ARROW:extension:name".to_string(),
                VBZ_EXTENSION_NAME.to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        // samples: number of samples in this chunk
        Field::new("samples", DataType::UInt32, false),
    ])
}

/// Create the Arrow schema for uncompressed signal data.
pub fn signal_schema_uncompressed() -> Schema {
    use std::sync::Arc;

    Schema::new(vec![
        Field::new("read_id", DataType::FixedSizeBinary(16), false).with_metadata(
            [(
                "ARROW:extension:name".to_string(),
                "minknow.uuid".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new(
            "signal",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int16, false))),
            false,
        ),
        Field::new("samples", DataType::UInt32, false),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_schema_has_expected_fields() {
        let schema = signal_schema();
        assert!(schema.field_with_name("read_id").is_ok());
        assert!(schema.field_with_name("signal").is_ok());
        assert!(schema.field_with_name("samples").is_ok());
    }
}

//! Run info table Arrow schema definition.

use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Create the Arrow schema for the run info table.
pub fn run_info_schema() -> Schema {
    Schema::new(vec![
        // acquisition_id: unique run identifier
        Field::new("acquisition_id", DataType::Utf8, false),
        // acquisition_start_time: milliseconds since epoch
        Field::new(
            "acquisition_start_time",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
            false,
        ),
        // adc_max: maximum ADC value
        Field::new("adc_max", DataType::Int16, false),
        // adc_min: minimum ADC value
        Field::new("adc_min", DataType::Int16, false),
        // context_tags: key-value metadata
        Field::new(
            "context_tags",
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Utf8, true),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
            false,
        ),
        // experiment_name
        Field::new("experiment_name", DataType::Utf8, false),
        // flow_cell_id
        Field::new("flow_cell_id", DataType::Utf8, false),
        // flow_cell_product_code
        Field::new("flow_cell_product_code", DataType::Utf8, false),
        // protocol_name
        Field::new("protocol_name", DataType::Utf8, false),
        // protocol_run_id
        Field::new("protocol_run_id", DataType::Utf8, false),
        // protocol_start_time: milliseconds since epoch
        Field::new(
            "protocol_start_time",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
            false,
        ),
        // sample_id
        Field::new("sample_id", DataType::Utf8, false),
        // sample_rate: Hz
        Field::new("sample_rate", DataType::UInt16, false),
        // sequencing_kit
        Field::new("sequencing_kit", DataType::Utf8, false),
        // sequencer_position
        Field::new("sequencer_position", DataType::Utf8, false),
        // sequencer_position_type
        Field::new("sequencer_position_type", DataType::Utf8, false),
        // software
        Field::new("software", DataType::Utf8, false),
        // system_name
        Field::new("system_name", DataType::Utf8, false),
        // system_type
        Field::new("system_type", DataType::Utf8, false),
        // tracking_id: key-value metadata
        Field::new(
            "tracking_id",
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Utf8, true),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
            false,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_info_schema_has_expected_fields() {
        let schema = run_info_schema();
        assert!(schema.field_with_name("acquisition_id").is_ok());
        assert!(schema.field_with_name("sample_rate").is_ok());
        assert!(schema.field_with_name("context_tags").is_ok());
        assert!(schema.field_with_name("tracking_id").is_ok());
    }
}

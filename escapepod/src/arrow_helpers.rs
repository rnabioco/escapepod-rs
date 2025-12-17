//! Shared Arrow field extraction helpers.
//!
//! This module provides utility functions for extracting typed values from
//! Arrow RecordBatches, reducing code duplication across the reader module.

use crate::error::{Error, Result};
use crate::types::Uuid;
use arrow::array::{
    Array, BooleanArray, DictionaryArray, FixedSizeBinaryArray, Float32Array, Int16Array,
    ListArray, StringArray, TimestampMillisecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::Int16Type;
use arrow::record_batch::RecordBatch;

/// Helper for extracting typed values from Arrow RecordBatches.
///
/// This struct provides convenient methods for extracting values from Arrow
/// columns with proper error handling and type checking.
pub struct BatchFieldExtractor<'a> {
    batch: &'a RecordBatch,
    row: usize,
}

impl<'a> BatchFieldExtractor<'a> {
    /// Create a new extractor for the given batch and row.
    pub fn new(batch: &'a RecordBatch, row: usize) -> Self {
        Self { batch, row }
    }

    /// Get a UUID from a FixedSizeBinary column.
    pub fn get_uuid(&self, name: &str) -> Result<Uuid> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr = col
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| Error::InvalidField {
                field: name.to_string(),
                message: "Expected FixedSizeBinaryArray".to_string(),
            })?;
        let bytes = arr.value(self.row);
        Uuid::from_slice(bytes).map_err(|e| Error::InvalidUuid(e.to_string()))
    }

    /// Get a u8 value.
    pub fn get_u8(&self, name: &str) -> Result<u8> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr = col
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| Error::InvalidField {
                field: name.to_string(),
                message: "Expected UInt8Array".to_string(),
            })?;
        Ok(arr.value(self.row))
    }

    /// Get a u16 value.
    pub fn get_u16(&self, name: &str) -> Result<u16> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr =
            col.as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected UInt16Array".to_string(),
                })?;
        Ok(arr.value(self.row))
    }

    /// Get a u32 value.
    pub fn get_u32(&self, name: &str) -> Result<u32> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr =
            col.as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected UInt32Array".to_string(),
                })?;
        Ok(arr.value(self.row))
    }

    /// Get a u64 value.
    pub fn get_u64(&self, name: &str) -> Result<u64> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr =
            col.as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected UInt64Array".to_string(),
                })?;
        Ok(arr.value(self.row))
    }

    /// Get an i16 value.
    pub fn get_i16(&self, name: &str) -> Result<i16> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr = col
            .as_any()
            .downcast_ref::<Int16Array>()
            .ok_or_else(|| Error::InvalidField {
                field: name.to_string(),
                message: "Expected Int16Array".to_string(),
            })?;
        Ok(arr.value(self.row))
    }

    /// Get an f32 value.
    pub fn get_f32(&self, name: &str) -> Result<f32> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr =
            col.as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected Float32Array".to_string(),
                })?;
        Ok(arr.value(self.row))
    }

    /// Get a bool value.
    pub fn get_bool(&self, name: &str) -> Result<bool> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr =
            col.as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected BooleanArray".to_string(),
                })?;
        Ok(arr.value(self.row))
    }

    /// Get a string value.
    pub fn get_string(&self, name: &str) -> Result<String> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr =
            col.as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected StringArray".to_string(),
                })?;
        Ok(arr.value(self.row).to_string())
    }

    /// Get a timestamp value (milliseconds since epoch).
    pub fn get_timestamp(&self, name: &str) -> Result<i64> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr = col
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .ok_or_else(|| Error::InvalidField {
                field: name.to_string(),
                message: "Expected TimestampMillisecondArray".to_string(),
            })?;
        Ok(arr.value(self.row))
    }

    /// Get a dictionary-encoded string value (Int16 keys).
    pub fn get_dict_string(&self, name: &str) -> Result<String> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;

        if let Some(dict) = col.as_any().downcast_ref::<DictionaryArray<Int16Type>>() {
            let keys = dict.keys();
            let values = dict.values();
            let values = values
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected String dictionary values".to_string(),
                })?;
            let key = keys.value(self.row);
            return Ok(values.value(key as usize).to_string());
        }

        Err(Error::InvalidField {
            field: name.to_string(),
            message: "Expected DictionaryArray<Int16Type>".to_string(),
        })
    }

    /// Get the dictionary key index from an Int16 dictionary column.
    pub fn get_dict_index(&self, name: &str) -> Result<i16> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;

        if let Some(dict) = col.as_any().downcast_ref::<DictionaryArray<Int16Type>>() {
            let keys = dict.keys();
            return Ok(keys.value(self.row));
        }

        Err(Error::InvalidField {
            field: name.to_string(),
            message: "Expected DictionaryArray<Int16Type>".to_string(),
        })
    }

    /// Get signal row indices from a list column.
    pub fn get_signal_rows(&self) -> Result<Vec<u64>> {
        let col = self
            .batch
            .column_by_name("signal")
            .ok_or_else(|| Error::MissingField("signal".to_string()))?;
        let list_arr =
            col.as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: "signal".to_string(),
                    message: "Expected ListArray".to_string(),
                })?;
        let values = list_arr.value(self.row);
        let u64_arr = values
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::InvalidField {
                field: "signal".to_string(),
                message: "Expected UInt64Array values".to_string(),
            })?;
        Ok(u64_arr.values().to_vec())
    }
}

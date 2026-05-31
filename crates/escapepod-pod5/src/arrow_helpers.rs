//! Shared Arrow field extraction helpers.
//!
//! This module provides utility functions for extracting typed values from
//! Arrow RecordBatches, reducing code duplication across the reader module.

use crate::error::{Error, Result};
use crate::types::{PoreType, ReadData, Uuid};
use arrow::array::{
    Array, AsArray, BooleanArray, DictionaryArray, FixedSizeBinaryArray, Float32Array, Int16Array,
    ListArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{
    Float32Type, Int16Type, TimestampMillisecondType, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
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
            .as_fixed_size_binary_opt()
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
            .as_primitive_opt::<UInt8Type>()
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
            col.as_primitive_opt::<UInt16Type>()
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
            col.as_primitive_opt::<UInt32Type>()
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
            col.as_primitive_opt::<UInt64Type>()
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
            .as_primitive_opt::<Int16Type>()
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
            col.as_primitive_opt::<Float32Type>()
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
            col.as_boolean_opt()
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
            col.as_string_opt::<i32>()
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
            .as_primitive_opt::<TimestampMillisecondType>()
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

        if let Some(dict) = col.as_dictionary_opt::<Int16Type>() {
            let keys = dict.keys();
            let values = dict.values();
            let values = values
                .as_string_opt::<i32>()
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

        if let Some(dict) = col.as_dictionary_opt::<Int16Type>() {
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
            col.as_list_opt::<i32>()
                .ok_or_else(|| Error::InvalidField {
                    field: "signal".to_string(),
                    message: "Expected ListArray".to_string(),
                })?;
        let values = list_arr.value(self.row);
        let u64_arr = values
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| Error::InvalidField {
                field: "signal".to_string(),
                message: "Expected UInt64Array values".to_string(),
            })?;
        Ok(u64_arr.values().to_vec())
    }
}

// ---- ReadsBatchView ---------------------------------------------------------
//
// Pre-resolves every column lookup in a reads-table RecordBatch once at
// construction. Per-row extraction is then a direct array index — no
// `column_by_name` linear scan, no `as_any().downcast_ref::<…>()` per call.
// The `reads()` iterator and the by-id read paths build one view per batch
// and reuse it across all rows of that batch, which dominates merge's
// metadata-load phase and filter's non-UUID path.

fn require_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    batch
        .column_by_name(name)
        .map(|c| c.as_ref())
        .ok_or_else(|| Error::MissingField(name.to_string()))
}

fn downcast<'a, T: Array + 'static>(col: &'a dyn Array, name: &str, ty: &str) -> Result<&'a T> {
    col.as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| Error::InvalidField {
            field: name.to_string(),
            message: format!("Expected {}", ty),
        })
}

fn require_typed<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    name: &str,
    ty: &str,
) -> Result<&'a T> {
    let col = require_col(batch, name)?;
    downcast::<T>(col, name, ty)
}

fn optional_typed<'a, T: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Option<&'a T> {
    batch.column_by_name(name)?.as_any().downcast_ref::<T>()
}

/// Resolved typed columns for a reads-table `RecordBatch`.
///
/// Construct once per batch with `ReadsBatchView::new`, then call `read(row)`
/// to extract a `ReadData` without re-doing column lookups or downcasts.
pub struct ReadsBatchView<'a> {
    // V0
    read_id: &'a FixedSizeBinaryArray,
    signal: &'a ListArray,
    read_number: &'a UInt32Array,
    start: &'a UInt64Array,
    median_before: &'a Float32Array,
    // V1
    num_minknow_events: &'a UInt64Array,
    tracked_scaling_scale: Option<&'a Float32Array>,
    tracked_scaling_shift: Option<&'a Float32Array>,
    predicted_scaling_scale: Option<&'a Float32Array>,
    predicted_scaling_shift: Option<&'a Float32Array>,
    num_reads_since_mux_change: Option<&'a UInt32Array>,
    time_since_mux_change: Option<&'a Float32Array>,
    // V2
    num_samples: &'a UInt64Array,
    // V3
    channel: &'a UInt16Array,
    well: &'a UInt8Array,
    pore_type_keys: &'a Int16Array,
    /// Pre-built `PoreType` per unique pore-type dictionary value, indexed
    /// by the dictionary key. Per-row `read()` calls clone a `PoreType`
    /// out of here — refcount-only on the underlying `Arc<str>`, no
    /// allocation per read.
    pore_type_values: Vec<PoreType>,
    calibration_offset: &'a Float32Array,
    calibration_scale: &'a Float32Array,
    end_reason_keys: &'a Int16Array,
    end_reason_values: &'a StringArray,
    end_reason_forced: &'a BooleanArray,
    run_info_keys: &'a Int16Array,
    // V4
    open_pore_level: Option<&'a Float32Array>,
}

impl<'a> ReadsBatchView<'a> {
    /// Resolve every column once. `try_alternate_field_names` is used to
    /// accept older POD5 files that name `start_sample`/`open_pore_level`
    /// fields differently — the actual column is resolved here, so per-row
    /// extraction never needs to retry.
    pub fn new(batch: &'a RecordBatch, try_alternate_field_names: bool) -> Result<Self> {
        let start = if try_alternate_field_names {
            optional_typed::<UInt64Array>(batch, "start_sample")
                .or_else(|| optional_typed::<UInt64Array>(batch, "start"))
                .ok_or_else(|| Error::MissingField("start_sample/start".to_string()))?
        } else {
            require_typed::<UInt64Array>(batch, "start", "UInt64Array")?
        };

        let open_pore_level = if try_alternate_field_names {
            optional_typed::<Float32Array>(batch, "predicted_scaling_open_pore_level")
                .or_else(|| optional_typed::<Float32Array>(batch, "open_pore_level"))
        } else {
            optional_typed::<Float32Array>(batch, "open_pore_level")
        };

        let pore_type_dict = require_typed::<DictionaryArray<Int16Type>>(
            batch,
            "pore_type",
            "DictionaryArray<Int16>",
        )?;
        let pore_type_dict_values = downcast::<StringArray>(
            pore_type_dict.values().as_ref(),
            "pore_type",
            "String dictionary values",
        )?;
        let pore_type_values: Vec<PoreType> = (0..pore_type_dict_values.len())
            .map(|i| PoreType::from(pore_type_dict_values.value(i)))
            .collect();

        let end_reason_dict = require_typed::<DictionaryArray<Int16Type>>(
            batch,
            "end_reason",
            "DictionaryArray<Int16>",
        )?;
        let end_reason_values = downcast::<StringArray>(
            end_reason_dict.values().as_ref(),
            "end_reason",
            "String dictionary values",
        )?;

        let run_info_dict = require_typed::<DictionaryArray<Int16Type>>(
            batch,
            "run_info",
            "DictionaryArray<Int16>",
        )?;

        Ok(ReadsBatchView {
            read_id: require_typed::<FixedSizeBinaryArray>(
                batch,
                "read_id",
                "FixedSizeBinaryArray",
            )?,
            signal: require_typed::<ListArray>(batch, "signal", "ListArray")?,
            read_number: require_typed::<UInt32Array>(batch, "read_number", "UInt32Array")?,
            start,
            median_before: require_typed::<Float32Array>(batch, "median_before", "Float32Array")?,
            num_minknow_events: require_typed::<UInt64Array>(
                batch,
                "num_minknow_events",
                "UInt64Array",
            )?,
            tracked_scaling_scale: optional_typed::<Float32Array>(batch, "tracked_scaling_scale"),
            tracked_scaling_shift: optional_typed::<Float32Array>(batch, "tracked_scaling_shift"),
            predicted_scaling_scale: optional_typed::<Float32Array>(
                batch,
                "predicted_scaling_scale",
            ),
            predicted_scaling_shift: optional_typed::<Float32Array>(
                batch,
                "predicted_scaling_shift",
            ),
            num_reads_since_mux_change: optional_typed::<UInt32Array>(
                batch,
                "num_reads_since_mux_change",
            ),
            time_since_mux_change: optional_typed::<Float32Array>(batch, "time_since_mux_change"),
            num_samples: require_typed::<UInt64Array>(batch, "num_samples", "UInt64Array")?,
            channel: require_typed::<UInt16Array>(batch, "channel", "UInt16Array")?,
            well: require_typed::<UInt8Array>(batch, "well", "UInt8Array")?,
            pore_type_keys: pore_type_dict.keys(),
            pore_type_values,
            calibration_offset: require_typed::<Float32Array>(
                batch,
                "calibration_offset",
                "Float32Array",
            )?,
            calibration_scale: require_typed::<Float32Array>(
                batch,
                "calibration_scale",
                "Float32Array",
            )?,
            end_reason_keys: end_reason_dict.keys(),
            end_reason_values,
            end_reason_forced: require_typed::<BooleanArray>(
                batch,
                "end_reason_forced",
                "BooleanArray",
            )?,
            run_info_keys: run_info_dict.keys(),
            open_pore_level,
        })
    }

    /// Row count of the underlying batch.
    pub fn num_rows(&self) -> usize {
        self.read_id.len()
    }

    /// Read ID of a single row (for fast UUID scans without building a full ReadData).
    pub fn read_id(&self, row: usize) -> Result<Uuid> {
        Uuid::from_slice(self.read_id.value(row)).map_err(|e| Error::InvalidUuid(e.to_string()))
    }

    /// Build a `ReadData` for one row from the resolved columns.
    pub fn read(&self, row: usize) -> Result<ReadData> {
        let pore_type = {
            let key = self.pore_type_keys.value(row);
            self.pore_type_values
                .get(key as usize)
                .cloned()
                .unwrap_or_default()
        };

        let end_reason = {
            let key = self.end_reason_keys.value(row);
            self.end_reason_values
                .value(key as usize)
                .parse()
                .unwrap_or_default()
        };

        let run_info_index = self.run_info_keys.value(row) as u32;

        // Signal rows
        let signal_rows = {
            let values = self.signal.value(row);
            let u64_arr = values
                .as_primitive_opt::<UInt64Type>()
                .ok_or_else(|| Error::InvalidField {
                    field: "signal".to_string(),
                    message: "Expected UInt64Array values".to_string(),
                })?;
            u64_arr.values().to_vec()
        };

        Ok(ReadData {
            read_id: Uuid::from_slice(self.read_id.value(row))
                .map_err(|e| Error::InvalidUuid(e.to_string()))?,
            read_number: self.read_number.value(row),
            start_sample: self.start.value(row),
            channel: self.channel.value(row),
            well: self.well.value(row),
            pore_type,
            calibration_offset: self.calibration_offset.value(row),
            calibration_scale: self.calibration_scale.value(row),
            median_before: self.median_before.value(row),
            end_reason,
            end_reason_forced: self.end_reason_forced.value(row),
            run_info_index,
            num_minknow_events: self.num_minknow_events.value(row),
            tracked_scaling_scale: self
                .tracked_scaling_scale
                .map(|a| a.value(row))
                .unwrap_or(1.0),
            tracked_scaling_shift: self
                .tracked_scaling_shift
                .map(|a| a.value(row))
                .unwrap_or(0.0),
            predicted_scaling_scale: self
                .predicted_scaling_scale
                .map(|a| a.value(row))
                .unwrap_or(1.0),
            predicted_scaling_shift: self
                .predicted_scaling_shift
                .map(|a| a.value(row))
                .unwrap_or(0.0),
            num_reads_since_mux_change: self
                .num_reads_since_mux_change
                .map(|a| a.value(row))
                .unwrap_or(0),
            time_since_mux_change: self
                .time_since_mux_change
                .map(|a| a.value(row))
                .unwrap_or(0.0),
            num_samples: self.num_samples.value(row),
            open_pore_level: self.open_pore_level.map(|a| a.value(row)).unwrap_or(0.0),
            signal_rows,
        })
    }
}

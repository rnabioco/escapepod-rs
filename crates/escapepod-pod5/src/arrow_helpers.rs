//! Shared Arrow field extraction helpers.
//!
//! This module provides utility functions for extracting typed values from
//! Arrow RecordBatches, reducing code duplication across the reader module.

use crate::error::{Error, Result};
use crate::types::{EndReason, PoreType, ReadData, Uuid};
use arrow::array::{
    Array, AsArray, BooleanArray, DictionaryArray, FixedSizeBinaryArray, Float32Array, Int16Array,
    ListArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{
    Float32Type, Int16Type, TimestampMillisecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
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
        let arr = col
            .as_primitive_opt::<UInt16Type>()
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
        let arr = col
            .as_primitive_opt::<UInt32Type>()
            .ok_or_else(|| Error::InvalidField {
                field: name.to_string(),
                message: "Expected UInt32Array".to_string(),
            })?;
        Ok(arr.value(self.row))
    }

    /// Get a u32 value from a column that may be physically `uint16`.
    ///
    /// POD5 V6 (upstream 0.3.46) widened the reads-table `channel` column from
    /// `uint16` to `uint32` under the same name; V3–V5 files still carry the
    /// narrow type. Both are accepted here and surfaced as `u32`.
    pub fn get_u32_or_u16(&self, name: &str) -> Result<u32> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        if let Some(arr) = col.as_primitive_opt::<UInt32Type>() {
            return Ok(arr.value(self.row));
        }
        let arr = col
            .as_primitive_opt::<UInt16Type>()
            .ok_or_else(|| Error::InvalidField {
                field: name.to_string(),
                message: "Expected UInt32Array (POD5 V6) or UInt16Array (V3–V5)".to_string(),
            })?;
        Ok(u32::from(arr.value(self.row)))
    }

    /// Get a u64 value.
    pub fn get_u64(&self, name: &str) -> Result<u64> {
        let col = self
            .batch
            .column_by_name(name)
            .ok_or_else(|| Error::MissingField(name.to_string()))?;
        let arr = col
            .as_primitive_opt::<UInt64Type>()
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
        let arr = col
            .as_primitive_opt::<Float32Type>()
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
        let arr = col.as_boolean_opt().ok_or_else(|| Error::InvalidField {
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
        let arr = col
            .as_string_opt::<i32>()
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
        let list_arr = col
            .as_list_opt::<i32>()
            .ok_or_else(|| Error::InvalidField {
                field: "signal".to_string(),
                message: "Expected ListArray".to_string(),
            })?;
        let values = list_arr.value(self.row);
        let u64_arr =
            values
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

/// Resolve a reads-table batch's `read_id` column.
///
/// By name rather than by position, so it works on the column-projected
/// readers the signal paths use as well as on a full batch.
pub fn read_id_column(batch: &RecordBatch) -> Result<&FixedSizeBinaryArray> {
    require_typed::<FixedSizeBinaryArray>(batch, "read_id", "FixedSizeBinaryArray")
}

/// Confirm a `.p5s` read-index locator points at the read it claims to.
///
/// The identity guard in `sidecar::read_sidecar_file` proves the sidecar
/// belongs to *this* POD5; it says nothing about whether the `(batch_idx,
/// row_idx)` locators inside it are right. Following one blind returns a
/// **different real read, correctly self-labelled** — which is worse than
/// garbage, because nothing downstream can tell. So every indexed lookup
/// confirms the row before using it.
///
/// The cost is a bounds check and a 16-byte compare against the cost of
/// decoding a read; the column is already resolved by the caller, and the
/// signal paths already project it.
pub fn verify_index_row(
    read_ids: &FixedSizeBinaryArray,
    requested: Uuid,
    batch_idx: usize,
    row: usize,
) -> Result<()> {
    // Ahead of the compare, not folded into it: `FixedSizeBinaryArray::value`
    // panics out of range rather than erroring.
    if row >= read_ids.len() {
        return Err(Error::SidecarRowOutOfBounds {
            requested,
            batch: batch_idx,
            row,
            rows: read_ids.len(),
        });
    }
    let found =
        Uuid::from_slice(read_ids.value(row)).map_err(|e| Error::InvalidUuid(e.to_string()))?;
    if found != requested {
        return Err(Error::SidecarIndexMismatch {
            requested,
            found,
            batch: batch_idx,
            row,
        });
    }
    Ok(())
}

/// Append an optional f32 column, filling `default` × `n` when the column is
/// absent — matching `ReadsBatchView::read`'s `.map(..).unwrap_or(default)`.
fn extend_opt_f32(dst: &mut Vec<f32>, arr: Option<&Float32Array>, n: usize, default: f32) {
    match arr {
        Some(a) => dst.extend_from_slice(a.values()),
        None => dst.resize(dst.len() + n, default),
    }
}

/// u32 counterpart to [`extend_opt_f32`].
fn extend_opt_u32(dst: &mut Vec<u32>, arr: Option<&UInt32Array>, n: usize, default: u32) {
    match arr {
        Some(a) => dst.extend_from_slice(a.values()),
        None => dst.resize(dst.len() + n, default),
    }
}

/// Read metadata in **struct-of-arrays** form — one `Vec` per field, every
/// read's value at the same index.
///
/// This is the columnar counterpart to a `Vec<ReadData>`: it omits `signal_rows`
/// (the per-read row-index list, unused by metadata consumers) and lets the
/// numeric columns be filled by a bulk slice copy straight from the Arrow buffers
/// instead of one `ReadData` struct — with three heap strings and a `Vec` — per
/// read. Populate it with [`Reader::read_columns`](crate::Reader::read_columns);
/// the Python bindings hand each `Vec` to numpy zero-copy.
#[derive(Debug, Default, Clone)]
pub struct ReadColumns {
    pub read_id: Vec<Uuid>,
    pub read_number: Vec<u32>,
    pub start_sample: Vec<u64>,
    pub channel: Vec<u32>,
    pub well: Vec<u8>,
    pub pore_type: Vec<PoreType>,
    pub calibration_offset: Vec<f32>,
    pub calibration_scale: Vec<f32>,
    pub median_before: Vec<f32>,
    pub end_reason: Vec<EndReason>,
    pub end_reason_forced: Vec<bool>,
    pub run_info_index: Vec<u32>,
    pub num_minknow_events: Vec<u64>,
    pub num_samples: Vec<u64>,
    pub tracked_scaling_scale: Vec<f32>,
    pub tracked_scaling_shift: Vec<f32>,
    pub predicted_scaling_scale: Vec<f32>,
    pub predicted_scaling_shift: Vec<f32>,
    pub num_reads_since_mux_change: Vec<u32>,
    pub time_since_mux_change: Vec<f32>,
    pub open_pore_level: Vec<f32>,
    pub expected_open_pore_level: Vec<f32>,
    pub selected_read_level: Vec<f32>,
}

impl ReadColumns {
    /// Number of reads accumulated (all columns share this length).
    pub fn len(&self) -> usize {
        self.read_id.len()
    }

    /// Whether any reads have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.read_id.is_empty()
    }

    /// Reserve capacity across every column.
    pub(crate) fn reserve(&mut self, n: usize) {
        self.read_id.reserve(n);
        self.read_number.reserve(n);
        self.start_sample.reserve(n);
        self.channel.reserve(n);
        self.well.reserve(n);
        self.pore_type.reserve(n);
        self.calibration_offset.reserve(n);
        self.calibration_scale.reserve(n);
        self.median_before.reserve(n);
        self.end_reason.reserve(n);
        self.end_reason_forced.reserve(n);
        self.run_info_index.reserve(n);
        self.num_minknow_events.reserve(n);
        self.num_samples.reserve(n);
        self.tracked_scaling_scale.reserve(n);
        self.tracked_scaling_shift.reserve(n);
        self.predicted_scaling_scale.reserve(n);
        self.predicted_scaling_shift.reserve(n);
        self.num_reads_since_mux_change.reserve(n);
        self.time_since_mux_change.reserve(n);
        self.open_pore_level.reserve(n);
        self.expected_open_pore_level.reserve(n);
        self.selected_read_level.reserve(n);
    }
}

/// The reads-table `channel` column, whichever physical width the file uses.
///
/// POD5 V6 (upstream 0.3.46) widened `channel` from `uint16` to `uint32` in
/// place — same field name, same position — so a reader that pins the narrow
/// type simply fails on every file written by pod5 0.3.46 and later. Resolving
/// to this enum keeps both widths readable and hands callers a `u32` either
/// way, which is what upstream's C++ `ReadData` now uses too.
enum ChannelColumn<'a> {
    /// V3-V5 files.
    U16(&'a UInt16Array),
    /// V6 and later.
    U32(&'a UInt32Array),
}

impl<'a> ChannelColumn<'a> {
    fn resolve(batch: &'a RecordBatch) -> Result<Self> {
        let col = require_col(batch, "channel")?;
        if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
            return Ok(Self::U32(arr));
        }
        col.as_any()
            .downcast_ref::<UInt16Array>()
            .map(Self::U16)
            .ok_or_else(|| Error::InvalidField {
                field: "channel".to_string(),
                message: "Expected UInt32Array (POD5 V6) or UInt16Array (V3-V5)".to_string(),
            })
    }

    #[inline]
    fn value(&self, row: usize) -> u32 {
        match self {
            Self::U16(arr) => u32::from(arr.value(row)),
            Self::U32(arr) => arr.value(row),
        }
    }

    /// Bulk-append every row to a `u32` column. V6 files copy the buffer
    /// wholesale; older ones widen element by element.
    fn extend_into(&self, out: &mut Vec<u32>) {
        match self {
            Self::U16(arr) => out.extend(arr.values().iter().copied().map(u32::from)),
            Self::U32(arr) => out.extend_from_slice(arr.values()),
        }
    }
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
    // V3 (widened to uint32 in V6)
    channel: ChannelColumn<'a>,
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
    // V5
    expected_open_pore_level: Option<&'a Float32Array>,
    selected_read_level: Option<&'a Float32Array>,
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
            channel: ChannelColumn::resolve(batch)?,
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
            expected_open_pore_level: optional_typed::<Float32Array>(
                batch,
                "expected_open_pore_level",
            ),
            selected_read_level: optional_typed::<Float32Array>(batch, "selected_read_level"),
        })
    }

    /// Row count of the underlying batch.
    pub fn num_rows(&self) -> usize {
        self.read_id.len()
    }

    /// The distinct `pore_type` dictionary labels for this batch (O(dict), not
    /// O(rows)).
    pub fn pore_type_dict(&self) -> Vec<String> {
        self.pore_type_values
            .iter()
            .map(|p| p.as_str().to_string())
            .collect()
    }

    /// The distinct `end_reason` dictionary labels for this batch.
    pub fn end_reason_dict(&self) -> Vec<String> {
        (0..self.end_reason_values.len())
            .map(|i| self.end_reason_values.value(i).to_string())
            .collect()
    }

    /// Read ID of a single row (for fast UUID scans without building a full ReadData).
    pub fn read_id(&self, row: usize) -> Result<Uuid> {
        Uuid::from_slice(self.read_id.value(row)).map_err(|e| Error::InvalidUuid(e.to_string()))
    }

    /// Confirm a `.p5s` locator points at the read it claims to, before this
    /// row is used. See [`verify_index_row`].
    pub fn verify_row(&self, requested: Uuid, batch_idx: usize, row: usize) -> Result<()> {
        verify_index_row(self.read_id, requested, batch_idx, row)
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
            let u64_arr =
                values
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
            expected_open_pore_level: self
                .expected_open_pore_level
                .map(|a| a.value(row))
                .unwrap_or(0.0),
            selected_read_level: self
                .selected_read_level
                .map(|a| a.value(row))
                .unwrap_or(0.0),
            signal_rows,
        })
    }

    /// Append every row of this batch to a [`ReadColumns`] struct-of-arrays.
    ///
    /// Numeric columns are filled by a bulk slice copy from the Arrow buffers;
    /// only `read_id` and the two dictionary-encoded columns (`pore_type`,
    /// `end_reason`) need a per-row step. Value-for-value identical to calling
    /// [`Self::read`] on each row and reading its fields — the optional-column
    /// defaults and dictionary lookups match `read()` exactly — but without
    /// allocating a `ReadData` (or its `signal_rows` `Vec`) per read.
    pub fn append_columns(&self, cols: &mut ReadColumns) -> Result<()> {
        let n = self.num_rows();
        cols.reserve(n);

        for row in 0..n {
            cols.read_id.push(
                Uuid::from_slice(self.read_id.value(row))
                    .map_err(|e| Error::InvalidUuid(e.to_string()))?,
            );
            let pkey = self.pore_type_keys.value(row);
            cols.pore_type.push(
                self.pore_type_values
                    .get(pkey as usize)
                    .cloned()
                    .unwrap_or_default(),
            );
            let ekey = self.end_reason_keys.value(row);
            cols.end_reason.push(
                self.end_reason_values
                    .value(ekey as usize)
                    .parse()
                    .unwrap_or_default(),
            );
            cols.end_reason_forced
                .push(self.end_reason_forced.value(row));
        }

        cols.read_number
            .extend_from_slice(self.read_number.values());
        cols.start_sample.extend_from_slice(self.start.values());
        self.channel.extend_into(&mut cols.channel);
        cols.well.extend_from_slice(self.well.values());
        cols.calibration_offset
            .extend_from_slice(self.calibration_offset.values());
        cols.calibration_scale
            .extend_from_slice(self.calibration_scale.values());
        cols.median_before
            .extend_from_slice(self.median_before.values());
        cols.num_minknow_events
            .extend_from_slice(self.num_minknow_events.values());
        cols.num_samples
            .extend_from_slice(self.num_samples.values());
        cols.run_info_index
            .extend(self.run_info_keys.values().iter().map(|&k| k as u32));

        extend_opt_f32(
            &mut cols.tracked_scaling_scale,
            self.tracked_scaling_scale,
            n,
            1.0,
        );
        extend_opt_f32(
            &mut cols.tracked_scaling_shift,
            self.tracked_scaling_shift,
            n,
            0.0,
        );
        extend_opt_f32(
            &mut cols.predicted_scaling_scale,
            self.predicted_scaling_scale,
            n,
            1.0,
        );
        extend_opt_f32(
            &mut cols.predicted_scaling_shift,
            self.predicted_scaling_shift,
            n,
            0.0,
        );
        extend_opt_u32(
            &mut cols.num_reads_since_mux_change,
            self.num_reads_since_mux_change,
            n,
            0,
        );
        extend_opt_f32(
            &mut cols.time_since_mux_change,
            self.time_since_mux_change,
            n,
            0.0,
        );
        extend_opt_f32(&mut cols.open_pore_level, self.open_pore_level, n, 0.0);
        extend_opt_f32(
            &mut cols.expected_open_pore_level,
            self.expected_open_pore_level,
            n,
            0.0,
        );
        extend_opt_f32(
            &mut cols.selected_read_level,
            self.selected_read_level,
            n,
            0.0,
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, BooleanBuilder, FixedSizeBinaryBuilder, Float32Builder, ListBuilder,
        StringDictionaryBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a reads-table batch carrying `channels`, with the `channel`
    /// column at the V6 width (`uint32`) or the V3-V5 one (`uint16`).
    ///
    /// Only the columns `ReadsBatchView::new` requires are built — the
    /// optional V1/V4/V5 ones are left out on purpose, so this doubles as a
    /// check that they stay optional.
    fn reads_batch(channels: &[u32], narrow: bool) -> RecordBatch {
        let n = channels.len();

        let mut read_id = FixedSizeBinaryBuilder::new(16);
        let mut signal = ListBuilder::new(UInt64Builder::new());
        let mut read_number = UInt32Builder::new();
        let mut start = UInt64Builder::new();
        let mut median_before = Float32Builder::new();
        let mut num_minknow_events = UInt64Builder::new();
        let mut num_samples = UInt64Builder::new();
        let mut well = UInt8Builder::new();
        let mut pore_type = StringDictionaryBuilder::<Int16Type>::new();
        let mut calibration_offset = Float32Builder::new();
        let mut calibration_scale = Float32Builder::new();
        let mut end_reason = StringDictionaryBuilder::<Int16Type>::new();
        let mut end_reason_forced = BooleanBuilder::new();
        let mut run_info = StringDictionaryBuilder::<Int16Type>::new();

        for i in 0..n {
            read_id
                .append_value(Uuid::from_u128(i as u128).as_bytes())
                .unwrap();
            signal.values().append_value(i as u64);
            signal.append(true);
            read_number.append_value(i as u32);
            start.append_value(i as u64 * 1000);
            median_before.append_value(200.0);
            num_minknow_events.append_value(0);
            num_samples.append_value(100);
            well.append_value(1);
            pore_type.append_value("not_set");
            calibration_offset.append_value(-220.0);
            calibration_scale.append_value(0.19);
            end_reason.append_value("signal_positive");
            end_reason_forced.append_value(false);
            run_info.append_value("acq");
        }

        let channel: ArrayRef = if narrow {
            let mut b = UInt16Builder::new();
            for &c in channels {
                b.append_value(u16::try_from(c).expect("narrow fixture value must fit u16"));
            }
            Arc::new(b.finish())
        } else {
            let mut b = UInt32Builder::new();
            for &c in channels {
                b.append_value(c);
            }
            Arc::new(b.finish())
        };

        let columns: Vec<(&str, ArrayRef)> = vec![
            ("read_id", Arc::new(read_id.finish())),
            ("signal", Arc::new(signal.finish())),
            ("read_number", Arc::new(read_number.finish())),
            ("start", Arc::new(start.finish())),
            ("median_before", Arc::new(median_before.finish())),
            ("num_minknow_events", Arc::new(num_minknow_events.finish())),
            ("num_samples", Arc::new(num_samples.finish())),
            ("channel", channel),
            ("well", Arc::new(well.finish())),
            ("pore_type", Arc::new(pore_type.finish())),
            ("calibration_offset", Arc::new(calibration_offset.finish())),
            ("calibration_scale", Arc::new(calibration_scale.finish())),
            ("end_reason", Arc::new(end_reason.finish())),
            ("end_reason_forced", Arc::new(end_reason_forced.finish())),
            ("run_info", Arc::new(run_info.finish())),
        ];

        // Derive the schema from the arrays so the two widths stay in step
        // with whatever Arrow types the builders produced.
        let schema = Schema::new(
            columns
                .iter()
                .map(|(name, arr)| Field::new(*name, arr.data_type().clone(), false))
                .collect::<Vec<_>>(),
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            columns.into_iter().map(|(_, a)| a).collect(),
        )
        .unwrap();
        assert_eq!(batch.num_rows(), n);
        batch
    }

    /// Every channel value, read three ways: the per-row `ReadsBatchView`, its
    /// bulk `append_columns` counterpart, and the extractor `read_iter` uses.
    fn channels_via_all_paths(batch: &RecordBatch) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let view = ReadsBatchView::new(batch, false).unwrap();

        let per_row: Vec<u32> = (0..batch.num_rows())
            .map(|row| view.read(row).unwrap().channel)
            .collect();

        let mut cols = ReadColumns::default();
        view.append_columns(&mut cols).unwrap();

        let extracted: Vec<u32> = (0..batch.num_rows())
            .map(|row| {
                BatchFieldExtractor::new(batch, row)
                    .get_u32_or_u16("channel")
                    .unwrap()
            })
            .collect();

        (per_row, cols.channel, extracted)
    }

    /// POD5 V6 widened `channel` to `uint32`; values above `u16::MAX` must
    /// survive every read path intact.
    #[test]
    fn v6_uint32_channel_round_trips_above_u16_max() {
        let want = [1u32, 512, 65_535, 70_000, 1_000_000];
        let batch = reads_batch(&want, false);
        let (per_row, bulk, extracted) = channels_via_all_paths(&batch);
        assert_eq!(per_row, want);
        assert_eq!(bulk, want);
        assert_eq!(extracted, want);
    }

    /// V3-V5 files still carry `channel` as `uint16`. Pinning the reader to
    /// one width breaks half the corpus whichever width it picks, so the
    /// narrow column has to widen to the same `u32`.
    #[test]
    fn v5_uint16_channel_widens_to_u32() {
        let want = [1u32, 512, 2675, 65_535];
        let batch = reads_batch(&want, true);
        assert_eq!(
            batch.column_by_name("channel").unwrap().data_type(),
            &DataType::UInt16,
            "fixture must actually carry the narrow column"
        );
        let (per_row, bulk, extracted) = channels_via_all_paths(&batch);
        assert_eq!(per_row, want);
        assert_eq!(bulk, want);
        assert_eq!(extracted, want);
    }

    /// A `channel` column of neither width is an error, not a silent zero.
    #[test]
    fn wrong_width_channel_is_rejected() {
        let base = reads_batch(&[1, 2], false);
        let schema = Schema::new(
            base.schema()
                .fields()
                .iter()
                .map(|f| {
                    if f.name() == "channel" {
                        Arc::new(Field::new("channel", DataType::UInt64, false))
                    } else {
                        f.clone()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let mut columns = base.columns().to_vec();
        let idx = base.schema().index_of("channel").unwrap();
        let mut b = UInt64Builder::new();
        b.append_value(1);
        b.append_value(2);
        columns[idx] = Arc::new(b.finish());
        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        assert!(ReadsBatchView::new(&batch, false).is_err());
        assert!(
            BatchFieldExtractor::new(&batch, 0)
                .get_u32_or_u16("channel")
                .is_err()
        );
    }
}

//! Main POD5 file reader.

use crate::compression;
use crate::error::{Error, Result};
use crate::footer::{self, Footer};
use crate::types::{EndReason, ReadData, RunInfoData, Uuid, POD5_SIGNATURE};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

/// A reader for POD5 files.
pub struct Reader {
    /// Memory-mapped file data.
    mmap: Mmap,
    /// Parsed file footer.
    footer: Footer,
    /// Cached run info data.
    run_info_cache: Vec<RunInfoData>,
}

impl Reader {
    /// Open a POD5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Verify signature at start
        if mmap.len() < 8 || mmap[..8] != POD5_SIGNATURE {
            return Err(Error::InvalidSignature);
        }

        // Parse footer
        let footer = footer::parse_footer(&mmap)?;

        // Load run info eagerly (it's usually small)
        let run_info_cache = Self::load_run_info(&mmap, &footer)?;

        Ok(Self {
            mmap,
            footer,
            run_info_cache,
        })
    }

    /// Get the file identifier (UUID).
    pub fn file_identifier(&self) -> &str {
        &self.footer.file_identifier
    }

    /// Get the software that wrote this file.
    pub fn software(&self) -> &str {
        &self.footer.software
    }

    /// Get the POD5 version.
    pub fn pod5_version(&self) -> &str {
        &self.footer.pod5_version
    }

    /// Get the number of run info entries.
    pub fn run_info_count(&self) -> usize {
        self.run_info_cache.len()
    }

    /// Get run info by index.
    pub fn get_run_info(&self, index: usize) -> Option<&RunInfoData> {
        self.run_info_cache.get(index)
    }

    /// Get all run info entries.
    pub fn run_infos(&self) -> &[RunInfoData] {
        &self.run_info_cache
    }

    /// Get the number of read batches.
    pub fn read_batch_count(&self) -> Result<usize> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;
        Ok(reader.num_batches())
    }

    /// Get a specific read batch.
    pub fn read_batch(&self, index: usize) -> Result<RecordBatch> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let mut reader = self.create_arrow_reader(embedded)?;

        if index >= reader.num_batches() {
            return Err(Error::BatchIndexOutOfBounds {
                index,
                max: reader.num_batches(),
            });
        }

        // Skip to the desired batch
        for _ in 0..index {
            reader.next();
        }

        reader
            .next()
            .ok_or_else(|| Error::BatchIndexOutOfBounds {
                index,
                max: reader.num_batches(),
            })?
            .map_err(Error::from)
    }

    /// Iterate over all reads in the file.
    pub fn reads(&self) -> Result<ReadIterator<'_>> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;

        Ok(ReadIterator {
            pod5_reader: self,
            arrow_reader: reader,
            current_batch: None,
            batch_row: 0,
        })
    }

    /// Get the total number of reads (requires scanning all batches).
    pub fn read_count(&self) -> Result<usize> {
        let embedded = self
            .footer
            .reads_table()
            .ok_or_else(|| Error::MissingField("reads table".to_string()))?;

        let reader = self.create_arrow_reader(embedded)?;

        let mut count = 0;
        for batch_result in reader {
            let batch = batch_result?;
            count += batch.num_rows();
        }

        Ok(count)
    }

    /// Get signal data for a read.
    ///
    /// The `signal_rows` parameter should be the signal row indices from the read record.
    pub fn get_signal(&self, signal_rows: &[u64]) -> Result<Vec<i16>> {
        let embedded = self
            .footer
            .signal_table()
            .ok_or_else(|| Error::MissingField("signal table".to_string()))?;

        let mut reader = self.create_arrow_reader(embedded)?;
        let mut all_samples = Vec::new();

        // For now, we'll do a simple implementation that reads all batches
        // A more efficient implementation would use batch indices
        let mut signal_batches: Vec<RecordBatch> = Vec::new();
        for batch_result in reader {
            signal_batches.push(batch_result?);
        }

        for &row_idx in signal_rows {
            // Find which batch contains this row
            let mut cumulative_rows = 0u64;
            for batch in &signal_batches {
                let batch_rows = batch.num_rows() as u64;
                if row_idx < cumulative_rows + batch_rows {
                    let local_row = (row_idx - cumulative_rows) as usize;
                    let samples = self.extract_signal_from_batch(batch, local_row)?;
                    all_samples.extend(samples);
                    break;
                }
                cumulative_rows += batch_rows;
            }
        }

        Ok(all_samples)
    }

    /// Extract signal samples from a signal table batch row.
    fn extract_signal_from_batch(&self, batch: &RecordBatch, row: usize) -> Result<Vec<i16>> {
        use arrow::array::{Array, BinaryArray, LargeBinaryArray, UInt32Array};

        // Get signal column (LargeBinary with VBZ data)
        let signal_col = batch
            .column_by_name("signal")
            .ok_or_else(|| Error::MissingField("signal column".to_string()))?;

        // Get samples column for count
        let samples_col = batch
            .column_by_name("samples")
            .ok_or_else(|| Error::MissingField("samples column".to_string()))?;

        let samples_array = samples_col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::InvalidField {
                field: "samples".to_string(),
                message: "Expected UInt32Array".to_string(),
            })?;

        let sample_count = samples_array.value(row) as usize;

        // Handle signal data (could be LargeBinary for VBZ)
        let signal_array = signal_col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| Error::InvalidField {
                field: "signal".to_string(),
                message: "Expected LargeBinaryArray".to_string(),
            })?;

        let compressed_data = signal_array.value(row);

        // Decompress VBZ data
        compression::decompress_signal(compressed_data, sample_count)
    }

    /// Create an Arrow IPC file reader for an embedded file.
    fn create_arrow_reader(
        &self,
        embedded: &crate::footer::EmbeddedFile,
    ) -> Result<ArrowFileReader<Cursor<&[u8]>>> {
        let start = embedded.offset as usize;
        let end = start + embedded.length as usize;

        if end > self.mmap.len() {
            return Err(Error::InvalidFooter(format!(
                "Embedded file extends beyond file end: {} + {} > {}",
                start,
                embedded.length,
                self.mmap.len()
            )));
        }

        let slice = &self.mmap[start..end];
        let cursor = Cursor::new(slice);
        ArrowFileReader::try_new(cursor, None).map_err(Error::from)
    }

    /// Load run info from the run info table.
    fn load_run_info(mmap: &Mmap, footer: &Footer) -> Result<Vec<RunInfoData>> {
        let embedded = match footer.run_info_table() {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let start = embedded.offset as usize;
        let end = start + embedded.length as usize;

        if end > mmap.len() {
            return Err(Error::InvalidFooter(
                "Run info table extends beyond file".to_string(),
            ));
        }

        let slice = &mmap[start..end];
        let cursor = Cursor::new(slice);
        let reader = ArrowFileReader::try_new(cursor, None)?;

        let mut run_infos = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            for row in 0..batch.num_rows() {
                run_infos.push(Self::run_info_from_batch(&batch, row)?);
            }
        }

        Ok(run_infos)
    }

    /// Extract RunInfoData from a batch row.
    fn run_info_from_batch(batch: &RecordBatch, row: usize) -> Result<RunInfoData> {
        use arrow::array::{
            Array, Int16Array, MapArray, StringArray, TimestampMillisecondArray, UInt16Array,
        };

        let get_string = |name: &str| -> Result<String> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected StringArray".to_string(),
                    })?;
            Ok(arr.value(row).to_string())
        };

        let get_i16 = |name: &str| -> Result<i16> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<Int16Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected Int16Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u16 = |name: &str| -> Result<u16> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt16Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_timestamp = |name: &str| -> Result<i64> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr = col
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected TimestampMillisecondArray".to_string(),
                })?;
            Ok(arr.value(row))
        };

        Ok(RunInfoData {
            acquisition_id: get_string("acquisition_id")?,
            acquisition_start_time: get_timestamp("acquisition_start_time")?,
            adc_max: get_i16("adc_max")?,
            adc_min: get_i16("adc_min")?,
            context_tags: HashMap::new(), // TODO: parse map
            experiment_name: get_string("experiment_name").unwrap_or_default(),
            flow_cell_id: get_string("flow_cell_id").unwrap_or_default(),
            flow_cell_product_code: get_string("flow_cell_product_code").unwrap_or_default(),
            protocol_name: get_string("protocol_name").unwrap_or_default(),
            protocol_run_id: get_string("protocol_run_id").unwrap_or_default(),
            protocol_start_time: get_timestamp("protocol_start_time").unwrap_or(0),
            sample_id: get_string("sample_id").unwrap_or_default(),
            sample_rate: get_u16("sample_rate")?,
            sequencing_kit: get_string("sequencing_kit").unwrap_or_default(),
            sequencer_position: get_string("sequencer_position").unwrap_or_default(),
            sequencer_position_type: get_string("sequencer_position_type").unwrap_or_default(),
            software: get_string("software").unwrap_or_default(),
            system_name: get_string("system_name").unwrap_or_default(),
            system_type: get_string("system_type").unwrap_or_default(),
            tracking_id: HashMap::new(), // TODO: parse map
        })
    }
}

/// Iterator over reads in a POD5 file.
pub struct ReadIterator<'a> {
    pod5_reader: &'a Reader,
    arrow_reader: ArrowFileReader<Cursor<&'a [u8]>>,
    current_batch: Option<RecordBatch>,
    batch_row: usize,
}

impl<'a> Iterator for ReadIterator<'a> {
    type Item = Result<ReadData>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we need a new batch
            let need_new_batch = match &self.current_batch {
                None => true,
                Some(batch) => self.batch_row >= batch.num_rows(),
            };

            if need_new_batch {
                match self.arrow_reader.next() {
                    Some(Ok(batch)) => {
                        self.current_batch = Some(batch);
                        self.batch_row = 0;
                    }
                    Some(Err(e)) => return Some(Err(Error::from(e))),
                    None => return None,
                }
            }

            // Extract read from current batch
            if let Some(batch) = &self.current_batch {
                let row = self.batch_row;
                self.batch_row += 1;
                return Some(Self::read_from_batch(batch, row));
            }
        }
    }
}

impl<'a> ReadIterator<'a> {
    fn read_from_batch(batch: &RecordBatch, row: usize) -> Result<ReadData> {
        use arrow::array::{
            Array, BooleanArray, DictionaryArray, FixedSizeBinaryArray, Float32Array, ListArray,
            UInt16Array, UInt32Array, UInt64Array, UInt8Array,
        };

        // Helper functions
        let get_uuid = |name: &str| -> Result<Uuid> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr = col
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| Error::InvalidField {
                    field: name.to_string(),
                    message: "Expected FixedSizeBinaryArray".to_string(),
                })?;
            let bytes = arr.value(row);
            Uuid::from_slice(bytes).map_err(|e| Error::InvalidUuid(e.to_string()))
        };

        let get_u8 = |name: &str| -> Result<u8> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt8Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u16 = |name: &str| -> Result<u16> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt16Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u32 = |name: &str| -> Result<u32> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt32Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_u64 = |name: &str| -> Result<u64> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected UInt64Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_f32 = |name: &str| -> Result<f32> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected Float32Array".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        let get_bool = |name: &str| -> Result<bool> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;
            let arr =
                col.as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected BooleanArray".to_string(),
                    })?;
            Ok(arr.value(row))
        };

        // Get dictionary-encoded string value
        let get_dict_string = |name: &str| -> Result<String> {
            let col = batch
                .column_by_name(name)
                .ok_or_else(|| Error::MissingField(name.to_string()))?;

            // Try Int16 dictionary first
            if let Some(dict) = col
                .as_any()
                .downcast_ref::<DictionaryArray<arrow::datatypes::Int16Type>>()
            {
                let keys = dict.keys();
                let values = dict.values();
                let values = values
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: name.to_string(),
                        message: "Expected String dictionary values".to_string(),
                    })?;
                let key = keys.value(row);
                return Ok(values.value(key as usize).to_string());
            }

            Err(Error::InvalidField {
                field: name.to_string(),
                message: "Expected DictionaryArray".to_string(),
            })
        };

        // Extract signal row indices from list
        let signal_rows = {
            let col = batch
                .column_by_name("signal")
                .ok_or_else(|| Error::MissingField("signal".to_string()))?;
            let list_arr =
                col.as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| Error::InvalidField {
                        field: "signal".to_string(),
                        message: "Expected ListArray".to_string(),
                    })?;
            let values = list_arr.value(row);
            let u64_arr = values
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::InvalidField {
                    field: "signal".to_string(),
                    message: "Expected UInt64Array values".to_string(),
                })?;
            u64_arr.values().to_vec()
        };

        Ok(ReadData {
            read_id: get_uuid("read_id")?,
            read_number: get_u32("read_number")?,
            start_sample: get_u64("start")?,
            channel: get_u16("channel")?,
            well: get_u8("well")?,
            pore_type: get_dict_string("pore_type").unwrap_or_default(),
            calibration_offset: get_f32("calibration_offset")?,
            calibration_scale: get_f32("calibration_scale")?,
            median_before: get_f32("median_before")?,
            end_reason: EndReason::from_str(&get_dict_string("end_reason").unwrap_or_default()),
            end_reason_forced: get_bool("end_reason_forced")?,
            run_info_index: 0, // TODO: parse from dictionary
            num_minknow_events: get_u64("num_minknow_events")?,
            num_samples: get_u64("num_samples")?,
            open_pore_level: get_f32("open_pore_level").unwrap_or(0.0),
            signal_rows,
        })
    }
}

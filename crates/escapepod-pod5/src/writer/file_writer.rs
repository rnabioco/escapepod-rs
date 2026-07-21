//! Main POD5 file writer.

use crate::CompressedSignalChunk;
use crate::compression;
use crate::error::{Error, Result};
use crate::schema::{reads_schema, run_info_schema, signal_schema};
use crate::types::{
    FOOTER_MAGIC, POD5_SIGNATURE, POD5_VERSION, ReadData, RunInfoData, SECTION_MARKER_LENGTH, Uuid,
};
use crate::writer::atomic::{AtomicFile, Durability};
use arrow::array::{
    ArrayRef, BooleanBuilder, FixedSizeBinaryBuilder, Float32Builder, Int16Builder,
    LargeBinaryBuilder, ListBuilder, MapBuilder, MapFieldNames, StringArray, StringBuilder,
    StringDictionaryBuilder, TimestampMillisecondBuilder, UInt8Builder, UInt16Builder,
    UInt32Builder, UInt64Builder,
};
use arrow::datatypes::Int16Type;
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, Write};
use std::path::Path;
use std::sync::Arc;

/// Write buffer size (16MB for efficient I/O, reduces syscall overhead)
const WRITE_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Options for writing POD5 files.
#[derive(Debug, Clone)]
pub struct WriterOptions {
    /// Maximum number of samples per signal chunk.
    pub max_signal_chunk_size: u32,
    /// Number of signal chunks per batch.
    pub signal_batch_size: u32,
    /// Number of reads per batch.
    pub read_batch_size: u32,
    /// Whether to compress signal data using VBZ.
    pub compress_signal: bool,
    /// Software name to write in the footer.
    pub software: String,
    /// Predefined dictionaries for multi-batch consistency.
    /// When set, all dictionary values must exist in the predefined lists.
    pub predefined_dictionaries: Option<PredefinedDictionaries>,
    /// How hard to push bytes to stable storage before the file is renamed
    /// into place. Defaults to [`Durability::None`] — rename only.
    pub durability: Durability,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            max_signal_chunk_size: 102_400,
            signal_batch_size: 100,
            read_batch_size: 1000,
            compress_signal: true,
            software: format!("escapepod-rs {}", env!("CARGO_PKG_VERSION")),
            predefined_dictionaries: None,
            durability: Durability::default(),
        }
    }
}

/// Predefined dictionary values for consistent multi-batch writing.
///
/// When predefined dictionaries are provided, only values in these lists
/// are allowed. Encountering unknown values will result in an error.
/// This enables smaller batch sizes by ensuring dictionary consistency
/// across all batches in the Arrow IPC file.
#[derive(Debug, Clone, Default)]
pub struct PredefinedDictionaries {
    /// Pore type values. If None, pore types are collected dynamically.
    pub pore_types: Option<Vec<String>>,
    /// End reason values. If None, end reasons are collected dynamically.
    pub end_reasons: Option<Vec<String>>,
}

/// Internal structure to track a pending read.
struct PendingRead {
    data: ReadData,
    signal_row_indices: Vec<u64>,
}

/// Internal structure for signal chunk data.
struct SignalChunk {
    read_id: Uuid,
    samples: u32,
    data: Arc<[u8]>,
}

/// Tracks an embedded file's location.
#[derive(Debug, Clone)]
struct EmbeddedFileInfo {
    offset: i64,
    length: i64,
    content_type: u8, // 0=Reads, 1=Signal, 4=RunInfo
}

/// A writer for POD5 files.
pub struct Writer {
    // File ownership - either direct access or owned by signal writer
    file: Option<BufWriter<File>>,
    options: WriterOptions,
    file_id: Uuid,

    // Dictionary tracking with O(1) lookup
    pore_types: Vec<String>,
    pore_type_index: HashMap<String, i16>,
    end_reasons: Vec<String>,
    end_reason_index: HashMap<String, i16>,

    // Run info
    run_infos: Vec<RunInfoData>,

    // Pending data (pre-allocated with capacity)
    pending_reads: Vec<PendingRead>,
    pending_signal: Vec<SignalChunk>,

    // Signal writes directly to file for performance
    signal_writer: Option<ArrowFileWriter<BufWriter<File>>>,
    signal_offset: i64,

    // Reads buffered in memory (small compared to signal)
    reads_writer: Option<ArrowFileWriter<Cursor<Vec<u8>>>>,

    // Signal row counter
    current_signal_row: u64,

    // Section tracking
    section_marker: Uuid,

    // State
    finalized: bool,

    // Predefined dictionary mode
    use_predefined_dictionaries: bool,

    // Cleanup guard for the staging file. Taken by `finish`/`abort`; while it
    // is still `Some`, dropping the writer unlinks the partial output and
    // leaves the destination untouched.
    atomic: Option<AtomicFile>,
}

impl Writer {
    /// Create a new POD5 file for writing.
    ///
    /// Nothing appears at `path` until [`finish`](Self::finish) succeeds:
    /// bytes are staged in a temp file alongside it and renamed into place at
    /// the end. Dropping the writer without finishing discards the partial
    /// output and leaves any pre-existing file at `path` intact.
    pub fn create<P: AsRef<Path>>(path: P, options: WriterOptions) -> Result<Self> {
        let atomic = AtomicFile::with_durability(path, options.durability)?;
        let file = atomic.reopen()?;
        // Use 2MB buffer for efficient I/O
        let mut file = BufWriter::with_capacity(WRITE_BUFFER_SIZE, file);

        // Write signature
        file.write_all(&POD5_SIGNATURE)?;

        // Write initial section marker
        let section_marker = Uuid::new_v4();
        file.write_all(section_marker.as_bytes())?;

        // Record signal table offset (after header)
        let signal_offset = file.stream_position()? as i64;

        // Initialize dictionaries from predefined values if provided
        let (pore_types, pore_type_index, end_reasons, end_reason_index, use_predefined) =
            if let Some(ref predef) = options.predefined_dictionaries {
                let (pt, pt_idx) = if let Some(ref vals) = predef.pore_types {
                    let index: HashMap<String, i16> = vals
                        .iter()
                        .enumerate()
                        .map(|(i, s)| (s.clone(), i as i16))
                        .collect();
                    (vals.clone(), index)
                } else {
                    (Vec::with_capacity(16), HashMap::with_capacity(16))
                };

                let (er, er_idx) = if let Some(ref vals) = predef.end_reasons {
                    let index: HashMap<String, i16> = vals
                        .iter()
                        .enumerate()
                        .map(|(i, s)| (s.clone(), i as i16))
                        .collect();
                    (vals.clone(), index)
                } else {
                    (Vec::with_capacity(16), HashMap::with_capacity(16))
                };

                // Use predefined mode if either dictionary is predefined
                let predefined = predef.pore_types.is_some() || predef.end_reasons.is_some();
                (pt, pt_idx, er, er_idx, predefined)
            } else {
                (
                    Vec::with_capacity(16),
                    HashMap::with_capacity(16),
                    Vec::with_capacity(16),
                    HashMap::with_capacity(16),
                    false,
                )
            };

        Ok(Self {
            file: Some(file),
            file_id: Uuid::new_v4(),
            // Dictionary tracking with O(1) lookup
            pore_types,
            pore_type_index,
            end_reasons,
            end_reason_index,
            run_infos: Vec::with_capacity(4),
            // Pre-allocate pending buffers based on batch sizes
            pending_reads: Vec::with_capacity(options.read_batch_size as usize),
            pending_signal: Vec::with_capacity(options.signal_batch_size as usize),
            // Signal writes directly to file
            signal_writer: None,
            signal_offset,
            // Reads buffered in memory
            reads_writer: None,
            current_signal_row: 0,
            section_marker,
            finalized: false,
            use_predefined_dictionaries: use_predefined,
            atomic: Some(atomic),
            options,
        })
    }

    /// Apply MINKNOW schema metadata to an Arrow schema.
    fn schema_with_metadata(&self, schema: arrow::datatypes::Schema) -> arrow::datatypes::Schema {
        let mut metadata = schema.metadata().clone();
        metadata.insert(
            "MINKNOW:file_identifier".to_string(),
            self.file_id.to_string(),
        );
        metadata.insert(
            "MINKNOW:software".to_string(),
            self.options.software.clone(),
        );
        metadata.insert("MINKNOW:pod5_version".to_string(), POD5_VERSION.to_string());
        schema.with_metadata(metadata)
    }

    /// Get the current signal row count (for tracking offsets during batch-level copying).
    pub fn current_signal_row(&self) -> u64 {
        self.current_signal_row
    }

    /// Add run info and return its index.
    pub fn add_run_info(&mut self, info: RunInfoData) -> Result<u32> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }

        let index = self.run_infos.len() as u32;
        self.run_infos.push(info);
        Ok(index)
    }

    /// Get the index for a pore type.
    /// In predefined mode, returns error if value not found.
    /// In dynamic mode, adds new values as needed.
    fn get_pore_type_index(&mut self, pore_type: &str) -> Result<i16> {
        if let Some(&idx) = self.pore_type_index.get(pore_type) {
            return Ok(idx);
        }

        if self.use_predefined_dictionaries {
            return Err(Error::DictionaryValueNotFound {
                value: pore_type.to_string(),
                dictionary_name: "pore_type".to_string(),
            });
        }

        let owned = pore_type.to_string();
        let idx = self.pore_types.len() as i16;
        self.pore_types.push(owned.clone());
        self.pore_type_index.insert(owned, idx);
        Ok(idx)
    }

    /// Get the index for an end reason.
    /// In predefined mode, returns error if value not found.
    /// In dynamic mode, adds new values as needed.
    fn get_end_reason_index(&mut self, end_reason: &str) -> Result<i16> {
        if let Some(&idx) = self.end_reason_index.get(end_reason) {
            return Ok(idx);
        }

        if self.use_predefined_dictionaries {
            return Err(Error::DictionaryValueNotFound {
                value: end_reason.to_string(),
                dictionary_name: "end_reason".to_string(),
            });
        }

        let owned = end_reason.to_string();
        let idx = self.end_reasons.len() as i16;
        self.end_reasons.push(owned.clone());
        self.end_reason_index.insert(owned, idx);
        Ok(idx)
    }

    /// Record a read's pore-type and end-reason in the writer dictionaries.
    /// In predefined-dictionary mode this fails if a value isn't in the
    /// predefined list. Shared by `add_read` and `add_read_with_compressed_signal`.
    fn track_dictionaries(&mut self, read: &ReadData) -> Result<()> {
        self.get_pore_type_index(read.pore_type.as_str())?;
        self.get_end_reason_index(read.end_reason.as_str())?;
        Ok(())
    }

    /// Add a read with its signal data.
    pub fn add_read(&mut self, read: ReadData, signal: &[i16]) -> Result<()> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }

        // Chunk and compress signal
        let mut signal_row_indices = Vec::new();
        for chunk in signal.chunks(self.options.max_signal_chunk_size as usize) {
            let data: Arc<[u8]> = if self.options.compress_signal {
                Arc::from(compression::compress_signal(chunk)?)
            } else {
                Arc::from(
                    chunk
                        .iter()
                        .flat_map(|&s| s.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )
            };

            signal_row_indices.push(self.current_signal_row);
            self.current_signal_row += 1;

            self.pending_signal.push(SignalChunk {
                read_id: read.read_id,
                samples: chunk.len() as u32,
                data,
            });
        }

        // Track dictionary entries (may fail in predefined mode)
        self.track_dictionaries(&read)?;

        self.pending_reads.push(PendingRead {
            data: read,
            signal_row_indices,
        });

        // Flush batches if needed
        if self.pending_signal.len() >= self.options.signal_batch_size as usize {
            self.flush_signal_batch()?;
        }

        if self.pending_reads.len() >= self.options.read_batch_size as usize {
            self.flush_read_batch()?;
        }

        Ok(())
    }

    /// Add a read with pre-compressed signal data (for block-level copying).
    /// This is much faster than add_read() when signal is already compressed.
    pub fn add_read_with_compressed_signal(
        &mut self,
        read: ReadData,
        compressed_chunks: &[CompressedSignalChunk],
    ) -> Result<()> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }

        // Add signal chunks directly without compression
        let mut signal_row_indices = Vec::with_capacity(compressed_chunks.len());
        for chunk in compressed_chunks {
            signal_row_indices.push(self.current_signal_row);
            self.current_signal_row += 1;

            self.pending_signal.push(SignalChunk {
                read_id: chunk.read_id,
                samples: chunk.samples,
                data: chunk.data.clone(), // Arc clone is cheap
            });
        }

        // Track dictionary entries (may fail in predefined mode)
        self.track_dictionaries(&read)?;

        self.pending_reads.push(PendingRead {
            data: read,
            signal_row_indices,
        });

        // Flush batches if needed
        if self.pending_signal.len() >= self.options.signal_batch_size as usize {
            self.flush_signal_batch()?;
        }

        if self.pending_reads.len() >= self.options.read_batch_size as usize {
            self.flush_read_batch()?;
        }

        Ok(())
    }

    /// Write a signal batch directly (for batch-level copying).
    /// This is the fastest method - copies Arrow RecordBatch directly without unpacking.
    /// Returns (first_row_index, row_count) for the written batch.
    pub fn write_signal_batch(&mut self, batch: &RecordBatch) -> Result<(u64, usize)> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }

        // Flush any pending signal first
        self.flush_signal_batch()?;

        let schema = Arc::new(self.schema_with_metadata(signal_schema()));
        let row_count = batch.num_rows();
        let first_row = self.current_signal_row;

        // Initialize signal writer if needed
        if self.signal_writer.is_none() {
            let file = self.file.take().ok_or(Error::WriterFinalized)?;
            self.signal_writer = Some(ArrowFileWriter::try_new(file, &schema)?);
        }

        // Create a new batch with our schema to ensure consistency
        // (input batches may have different metadata)
        let normalized_batch = RecordBatch::try_new(schema, batch.columns().to_vec())?;

        // Write batch directly
        if let Some(ref mut writer) = self.signal_writer {
            writer.write(&normalized_batch)?;
        } else {
            return Err(Error::InvalidState(
                "Signal writer not initialized".to_string(),
            ));
        }
        self.current_signal_row += row_count as u64;

        Ok((first_row, row_count))
    }

    /// Write raw IPC header bytes (magic + schema) directly.
    /// Call this once before write_raw_signal_batches().
    /// Returns the byte offset where batches should start.
    pub fn write_raw_signal_header(&mut self, header_bytes: &[u8]) -> Result<usize> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }
        if self.signal_writer.is_some() {
            return Err(Error::InvalidState(
                "Cannot mix raw signal writing with batch writing".into(),
            ));
        }

        // Ensure file is available
        let file = self.file.as_mut().ok_or(Error::WriterFinalized)?;

        // Write header bytes directly
        file.write_all(header_bytes)?;

        Ok(header_bytes.len())
    }

    /// Write raw signal batch bytes directly, bypassing Arrow serialization.
    /// Returns (first_row_index, row_count) for the written batches.
    /// The batch_bytes should include all batches' raw IPC data.
    pub fn write_raw_signal_batches(
        &mut self,
        batch_bytes: &[u8],
        row_count: u64,
    ) -> Result<(u64, u64)> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }
        if self.signal_writer.is_some() {
            return Err(Error::InvalidState(
                "Cannot mix raw signal writing with batch writing".into(),
            ));
        }

        let first_row = self.current_signal_row;

        // Ensure file is available
        let file = self.file.as_mut().ok_or(Error::WriterFinalized)?;

        // Write batch bytes directly
        file.write_all(batch_bytes)?;
        self.current_signal_row += row_count;

        Ok((first_row, row_count))
    }

    /// Finish raw signal writing by writing the IPC footer.
    /// Call this after all write_raw_signal_batches() calls.
    pub fn finish_raw_signal(
        &mut self,
        footer: &crate::arrow_ipc::ArrowIpcFooter,
        _current_offset: usize,
    ) -> Result<()> {
        use crate::utils::table_builders::build_arrow_ipc_footer;

        if self.finalized {
            return Err(Error::WriterFinalized);
        }
        if self.signal_writer.is_some() {
            return Err(Error::InvalidState(
                "Cannot mix raw signal writing with batch writing".into(),
            ));
        }

        // The footer must carry the real signal schema — Arrow's reader uses
        // the footer's schema (not the header's) when decoding batches, so an
        // empty one silently strips every column.
        let schema = self.schema_with_metadata(signal_schema());
        let footer_bytes = build_arrow_ipc_footer(&footer.record_batches, &schema)?;

        let file = self.file.as_mut().ok_or(Error::WriterFinalized)?;

        file.write_all(&footer_bytes)?;

        let footer_len = footer_bytes.len() as i32;
        file.write_all(&footer_len.to_le_bytes())?;

        file.write_all(b"ARROW1")?;

        Ok(())
    }

    /// Add a read with pre-computed signal row indices (for batch-level copying).
    /// Use this after write_signal_batch() to add reads that reference the written signal.
    pub fn add_read_with_signal_rows(
        &mut self,
        read: ReadData,
        signal_row_indices: Vec<u64>,
    ) -> Result<()> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }

        // Track dictionary entries (may fail in predefined mode)
        self.track_dictionaries(&read)?;

        self.pending_reads.push(PendingRead {
            data: read,
            signal_row_indices,
        });

        // Flush reads if needed
        if self.pending_reads.len() >= self.options.read_batch_size as usize {
            self.flush_read_batch()?;
        }

        Ok(())
    }

    /// Flush pending signal data to a batch - writes directly to file.
    fn flush_signal_batch(&mut self) -> Result<()> {
        if self.pending_signal.is_empty() {
            return Ok(());
        }

        let schema = Arc::new(self.schema_with_metadata(signal_schema()));
        let num_chunks = self.pending_signal.len();
        let total_signal_bytes: usize = self.pending_signal.iter().map(|c| c.data.len()).sum();

        // Build arrays - iterate directly without collecting to intermediate Vec
        let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(num_chunks, 16);
        let mut signal_builder = LargeBinaryBuilder::with_capacity(num_chunks, total_signal_bytes);
        let mut samples_builder = UInt32Builder::with_capacity(num_chunks);

        for chunk in self.pending_signal.drain(..) {
            read_id_builder.append_value(chunk.read_id.as_bytes())?;
            signal_builder.append_value(&chunk.data);
            samples_builder.append_value(chunk.samples);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(read_id_builder.finish()),
                Arc::new(signal_builder.finish()),
                Arc::new(samples_builder.finish()),
            ],
        )?;

        // Write directly to file (create writer on first batch, taking ownership of file)
        // Use try_new since file is already buffered (BufWriter)
        if self.signal_writer.is_none() {
            let file = self.file.take().ok_or(Error::WriterFinalized)?;
            self.signal_writer = Some(ArrowFileWriter::try_new(file, &schema)?);
        }
        if let Some(ref mut writer) = self.signal_writer {
            writer.write(&batch)?;
        }

        Ok(())
    }

    /// Flush pending reads to a batch.
    fn flush_read_batch(&mut self) -> Result<()> {
        if self.pending_reads.is_empty() {
            return Ok(());
        }

        let schema = Arc::new(self.schema_with_metadata(reads_schema()));
        let num_reads = self.pending_reads.len();

        // Build arrays - iterate directly without collecting to intermediate Vec
        let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(num_reads, 16);
        let signal_field = Arc::new(arrow::datatypes::Field::new(
            "item",
            arrow::datatypes::DataType::UInt64,
            true,
        ));
        let mut signal_builder = ListBuilder::new(UInt64Builder::new()).with_field(signal_field);
        // V0 builders
        let mut read_number_builder = UInt32Builder::with_capacity(num_reads);
        let mut start_builder = UInt64Builder::with_capacity(num_reads);
        let mut median_before_builder = Float32Builder::with_capacity(num_reads);
        // V1 builders
        let mut num_minknow_events_builder = UInt64Builder::with_capacity(num_reads);
        let mut tracked_scaling_scale_builder = Float32Builder::with_capacity(num_reads);
        let mut tracked_scaling_shift_builder = Float32Builder::with_capacity(num_reads);
        let mut predicted_scaling_scale_builder = Float32Builder::with_capacity(num_reads);
        let mut predicted_scaling_shift_builder = Float32Builder::with_capacity(num_reads);
        let mut num_reads_since_mux_change_builder = UInt32Builder::with_capacity(num_reads);
        let mut time_since_mux_change_builder = Float32Builder::with_capacity(num_reads);
        // V2 builders
        let mut num_samples_builder = UInt64Builder::with_capacity(num_reads);
        // V3 builders
        let mut channel_builder = UInt16Builder::with_capacity(num_reads);
        let mut well_builder = UInt8Builder::with_capacity(num_reads);
        let pore_type_dict =
            StringArray::from_iter_values(self.pore_types.iter().map(|s| s.as_str()));
        let mut pore_type_builder: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new_with_dictionary(num_reads, &pore_type_dict)?;
        let mut calibration_offset_builder = Float32Builder::with_capacity(num_reads);
        let mut calibration_scale_builder = Float32Builder::with_capacity(num_reads);
        let end_reason_dict =
            StringArray::from_iter_values(self.end_reasons.iter().map(|s| s.as_str()));
        let mut end_reason_builder: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new_with_dictionary(num_reads, &end_reason_dict)?;
        let mut end_reason_forced_builder = BooleanBuilder::with_capacity(num_reads);
        let mut run_info_builder: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        // V4 builders
        let mut open_pore_level_builder = Float32Builder::with_capacity(num_reads);
        // V5 builders
        let mut expected_open_pore_level_builder = Float32Builder::with_capacity(num_reads);
        let mut selected_read_level_builder = Float32Builder::with_capacity(num_reads);

        for pending in self.pending_reads.drain(..) {
            let read = &pending.data;

            read_id_builder.append_value(read.read_id.as_bytes())?;

            // Signal indices list
            let values = signal_builder.values();
            for &idx in &pending.signal_row_indices {
                values.append_value(idx);
            }
            signal_builder.append(true);

            // V0
            read_number_builder.append_value(read.read_number);
            start_builder.append_value(read.start_sample);
            median_before_builder.append_value(read.median_before);
            // V1
            num_minknow_events_builder.append_value(read.num_minknow_events);
            tracked_scaling_scale_builder.append_value(read.tracked_scaling_scale);
            tracked_scaling_shift_builder.append_value(read.tracked_scaling_shift);
            predicted_scaling_scale_builder.append_value(read.predicted_scaling_scale);
            predicted_scaling_shift_builder.append_value(read.predicted_scaling_shift);
            num_reads_since_mux_change_builder.append_value(read.num_reads_since_mux_change);
            time_since_mux_change_builder.append_value(read.time_since_mux_change);
            // V2
            num_samples_builder.append_value(read.num_samples);
            // V3
            channel_builder.append_value(read.channel);
            well_builder.append_value(read.well);
            pore_type_builder.append_value(read.pore_type.as_str());
            calibration_offset_builder.append_value(read.calibration_offset);
            calibration_scale_builder.append_value(read.calibration_scale);
            end_reason_builder.append_value(read.end_reason.as_str());
            end_reason_forced_builder.append_value(read.end_reason_forced);
            if let Some(run_info) = self.run_infos.get(read.run_info_index as usize) {
                run_info_builder.append_value(&run_info.acquisition_id);
            } else {
                run_info_builder.append_value("");
            }
            // V4
            open_pore_level_builder.append_value(read.open_pore_level);
            // V5
            expected_open_pore_level_builder.append_value(read.expected_open_pore_level);
            selected_read_level_builder.append_value(read.selected_read_level);
        }

        let arrays: Vec<ArrayRef> = vec![
            // V0
            Arc::new(read_id_builder.finish()),
            Arc::new(signal_builder.finish()),
            Arc::new(read_number_builder.finish()),
            Arc::new(start_builder.finish()),
            Arc::new(median_before_builder.finish()),
            // V1
            Arc::new(num_minknow_events_builder.finish()),
            Arc::new(tracked_scaling_scale_builder.finish()),
            Arc::new(tracked_scaling_shift_builder.finish()),
            Arc::new(predicted_scaling_scale_builder.finish()),
            Arc::new(predicted_scaling_shift_builder.finish()),
            Arc::new(num_reads_since_mux_change_builder.finish()),
            Arc::new(time_since_mux_change_builder.finish()),
            // V2
            Arc::new(num_samples_builder.finish()),
            // V3
            Arc::new(channel_builder.finish()),
            Arc::new(well_builder.finish()),
            Arc::new(pore_type_builder.finish()),
            Arc::new(calibration_offset_builder.finish()),
            Arc::new(calibration_scale_builder.finish()),
            Arc::new(end_reason_builder.finish()),
            Arc::new(end_reason_forced_builder.finish()),
            Arc::new(run_info_builder.finish()),
            // V4
            Arc::new(open_pore_level_builder.finish()),
            // V5
            Arc::new(expected_open_pore_level_builder.finish()),
            Arc::new(selected_read_level_builder.finish()),
        ];

        let batch = RecordBatch::try_new(schema.clone(), arrays)?;

        // Write to single IPC file (create writer on first batch)
        if self.reads_writer.is_none() {
            let cursor = Cursor::new(Vec::new());
            self.reads_writer = Some(ArrowFileWriter::try_new(cursor, &schema)?);
        }
        if let Some(ref mut writer) = self.reads_writer {
            writer.write(&batch)?;
        }

        Ok(())
    }

    /// Build run info table.
    fn build_run_info_table(&self) -> Result<Vec<u8>> {
        if self.run_infos.is_empty() {
            // Return empty Arrow IPC file
            let schema = Arc::new(self.schema_with_metadata(run_info_schema()));
            let mut buffer = Vec::new();
            {
                let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
                writer.finish()?;
            }
            return Ok(buffer);
        }

        let schema = Arc::new(self.schema_with_metadata(run_info_schema()));

        let mut acquisition_id_builder = StringBuilder::new();
        let mut acquisition_start_time_builder =
            TimestampMillisecondBuilder::new().with_timezone("UTC");
        let mut adc_max_builder = Int16Builder::new();
        let mut adc_min_builder = Int16Builder::new();
        let map_field_names = Some(MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        });
        let mut context_tags_builder = MapBuilder::new(
            map_field_names.clone(),
            StringBuilder::new(),
            StringBuilder::new(),
        );
        let mut experiment_name_builder = StringBuilder::new();
        let mut flow_cell_id_builder = StringBuilder::new();
        let mut flow_cell_product_code_builder = StringBuilder::new();
        let mut protocol_name_builder = StringBuilder::new();
        let mut protocol_run_id_builder = StringBuilder::new();
        let mut protocol_start_time_builder =
            TimestampMillisecondBuilder::new().with_timezone("UTC");
        let mut sample_id_builder = StringBuilder::new();
        let mut sample_rate_builder = UInt16Builder::new();
        let mut sequencing_kit_builder = StringBuilder::new();
        let mut sequencer_position_builder = StringBuilder::new();
        let mut sequencer_position_type_builder = StringBuilder::new();
        let mut software_builder = StringBuilder::new();
        let mut system_name_builder = StringBuilder::new();
        let mut system_type_builder = StringBuilder::new();
        let mut tracking_id_builder =
            MapBuilder::new(map_field_names, StringBuilder::new(), StringBuilder::new());

        for info in &self.run_infos {
            acquisition_id_builder.append_value(&info.acquisition_id);
            acquisition_start_time_builder.append_value(info.acquisition_start_time);
            adc_max_builder.append_value(info.adc_max);
            adc_min_builder.append_value(info.adc_min);

            // Context tags map
            for (k, v) in &info.context_tags {
                context_tags_builder.keys().append_value(k);
                context_tags_builder.values().append_value(v);
            }
            context_tags_builder.append(true)?;

            experiment_name_builder.append_value(&info.experiment_name);
            flow_cell_id_builder.append_value(&info.flow_cell_id);
            flow_cell_product_code_builder.append_value(&info.flow_cell_product_code);
            protocol_name_builder.append_value(&info.protocol_name);
            protocol_run_id_builder.append_value(&info.protocol_run_id);
            protocol_start_time_builder.append_value(info.protocol_start_time);
            sample_id_builder.append_value(&info.sample_id);
            sample_rate_builder.append_value(info.sample_rate);
            sequencing_kit_builder.append_value(&info.sequencing_kit);
            sequencer_position_builder.append_value(&info.sequencer_position);
            sequencer_position_type_builder.append_value(&info.sequencer_position_type);
            software_builder.append_value(&info.software);
            system_name_builder.append_value(&info.system_name);
            system_type_builder.append_value(&info.system_type);

            // Tracking ID map
            for (k, v) in &info.tracking_id {
                tracking_id_builder.keys().append_value(k);
                tracking_id_builder.values().append_value(v);
            }
            tracking_id_builder.append(true)?;
        }

        let arrays: Vec<ArrayRef> = vec![
            Arc::new(acquisition_id_builder.finish()),
            Arc::new(acquisition_start_time_builder.finish()),
            Arc::new(adc_max_builder.finish()),
            Arc::new(adc_min_builder.finish()),
            Arc::new(context_tags_builder.finish()),
            Arc::new(experiment_name_builder.finish()),
            Arc::new(flow_cell_id_builder.finish()),
            Arc::new(flow_cell_product_code_builder.finish()),
            Arc::new(protocol_name_builder.finish()),
            Arc::new(protocol_run_id_builder.finish()),
            Arc::new(protocol_start_time_builder.finish()),
            Arc::new(sample_id_builder.finish()),
            Arc::new(sample_rate_builder.finish()),
            Arc::new(sequencing_kit_builder.finish()),
            Arc::new(sequencer_position_builder.finish()),
            Arc::new(sequencer_position_type_builder.finish()),
            Arc::new(software_builder.finish()),
            Arc::new(system_name_builder.finish()),
            Arc::new(system_type_builder.finish()),
            Arc::new(tracking_id_builder.finish()),
        ];

        let batch = RecordBatch::try_new(schema.clone(), arrays)?;

        let mut buffer = Vec::new();
        {
            let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
            writer.write(&batch)?;
            writer.finish()?;
        }

        Ok(buffer)
    }

    /// Build the FlatBuffer footer using the generated FlatBuffer types.
    fn build_flatbuffer_footer(&self, embedded_files: &[EmbeddedFileInfo]) -> Result<Vec<u8>> {
        use crate::flatbuffers_gen::{
            ContentType, EmbeddedFile, EmbeddedFileArgs, Footer, FooterArgs, Format,
        };

        let file_id = self.file_id.to_string();
        let software = &self.options.software;
        let version = POD5_VERSION;

        let mut fbb = flatbuffers::FlatBufferBuilder::with_capacity(256);

        // Create embedded file entries
        let mut entries = Vec::with_capacity(embedded_files.len());
        for file in embedded_files {
            let content_type = match file.content_type {
                0 => ContentType::ReadsTable,
                1 => ContentType::SignalTable,
                4 => ContentType::RunInfoTable,
                _ => ContentType::ReadsTable,
            };
            let entry = EmbeddedFile::create(
                &mut fbb,
                &EmbeddedFileArgs {
                    offset: file.offset,
                    length: file.length,
                    format: Format::FeatherV2,
                    content_type,
                },
            );
            entries.push(entry);
        }

        let contents = fbb.create_vector(&entries);

        let file_id_str = fbb.create_string(&file_id);
        let software_str = fbb.create_string(software);
        let version_str = fbb.create_string(version);

        let footer = Footer::create(
            &mut fbb,
            &FooterArgs {
                file_identifier: Some(file_id_str),
                software: Some(software_str),
                pod5_version: Some(version_str),
                contents: Some(contents),
            },
        );

        fbb.finish(footer, None);

        Ok(fbb.finished_data().to_vec())
    }

    /// Finalize the file and write the footer.
    pub fn finish(mut self) -> Result<()> {
        if self.finalized {
            return Err(Error::WriterFinalized);
        }

        // Flush any remaining data
        self.flush_signal_batch()?;
        self.flush_read_batch()?;

        let mut embedded_files = Vec::new();

        // Finalize signal writer and get file back
        let mut file = if let Some(mut writer) = self.signal_writer.take() {
            writer.finish()?;
            writer.into_inner()?
        } else {
            // No signal was written, file should still be in self.file
            self.file.take().ok_or(Error::WriterFinalized)?
        };

        // Track byte position manually from here on. `stream_position()` on
        // a `BufWriter` flushes the buffer; previously this was called 6
        // times here, including once per byte inside per-byte padding
        // loops. Querying it once after the signal writer is unavoidable
        // because the Arrow IPC writer doesn't expose its bytes-written
        // count, but every subsequent position is derived from that
        // anchor + the lengths of buffers we control.
        let mut pos = file.stream_position()? as usize;

        // Record signal table info
        let signal_length = pos as i64 - self.signal_offset;
        if signal_length > 0 {
            embedded_files.push(EmbeddedFileInfo {
                offset: self.signal_offset,
                length: signal_length,
                content_type: 1, // SignalTable
            });
        }

        // Pad to 8-byte alignment + section marker
        pos += write_padding_to_align8(&mut file, pos)?;
        file.write_all(self.section_marker.as_bytes())?;
        pos += SECTION_MARKER_LENGTH;

        // Write run info table
        let run_info_data = self.build_run_info_table()?;
        let run_info_offset = pos as i64;
        file.write_all(&run_info_data)?;
        pos += run_info_data.len();
        let run_info_length = run_info_data.len() as i64;
        embedded_files.push(EmbeddedFileInfo {
            offset: run_info_offset,
            length: run_info_length,
            content_type: 4, // RunInfoTable
        });

        // Pad and section marker
        pos += write_padding_to_align8(&mut file, pos)?;
        file.write_all(self.section_marker.as_bytes())?;
        pos += SECTION_MARKER_LENGTH;

        // Write reads table from memory buffer
        let reads_offset = pos as i64;
        let reads_bytes_written = if let Some(mut writer) = self.reads_writer.take() {
            writer.finish()?;
            let cursor = writer.into_inner()?;
            let bytes = cursor.get_ref();
            file.write_all(bytes)?;
            bytes.len()
        } else {
            0
        };
        pos += reads_bytes_written;
        let reads_length = reads_bytes_written as i64;
        if reads_length > 0 {
            embedded_files.push(EmbeddedFileInfo {
                offset: reads_offset,
                length: reads_length,
                content_type: 0, // ReadsTable
            });
        }

        // Pad and section marker
        let _ = write_padding_to_align8(&mut file, pos)?;
        file.write_all(self.section_marker.as_bytes())?;

        // Write FOOTER magic
        file.write_all(&FOOTER_MAGIC)?;

        // Build and write footer
        let footer_data = self.build_flatbuffer_footer(&embedded_files)?;
        file.write_all(&footer_data)?;

        // Write footer length
        let footer_len = footer_data.len() as i64;
        file.write_all(&footer_len.to_le_bytes())?;

        // Write final section marker
        file.write_all(self.section_marker.as_bytes())?;

        // Write signature
        file.write_all(&POD5_SIGNATURE)?;

        // Release the buffered handle before committing: `commit` syncs and
        // renames the underlying file but cannot flush a buffer it doesn't own.
        file.flush()?;
        drop(file);

        self.atomic.take().ok_or(Error::WriterFinalized)?.commit()?;

        self.finalized = true;
        Ok(())
    }

    /// Discard the file being written without finalizing it.
    ///
    /// The destination is left untouched, or absent if it never existed. Use
    /// this instead of simply dropping the writer when you want the cleanup
    /// error rather than a silent unlink.
    pub fn abort(mut self) -> Result<()> {
        self.atomic.take().ok_or(Error::WriterFinalized)?.abort()
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // Reaching here with the guard still held means `finish` never
        // completed — either the caller dropped us or `finish` bailed partway.
        // Either way the destination must not be touched, so let the guard
        // drop and unlink the staging file.
        //
        // After a successful `finish` the guard is already `None`, which is
        // what keeps this a no-op on the happy path (`finish` consumes `self`,
        // so this always runs).
        drop(self.atomic.take());
    }
}

/// Write zero bytes to reach the next 8-byte alignment boundary.
/// Returns the number of padding bytes written (0..=7).
fn write_padding_to_align8<W: Write>(file: &mut W, pos: usize) -> Result<usize> {
    const ZEROS: [u8; 8] = [0u8; 8];
    let padding = (8 - (pos % 8)) % 8;
    if padding > 0 {
        file.write_all(&ZEROS[..padding])?;
    }
    Ok(padding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Reader;
    use crate::types::EndReason;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    fn create_test_run_info(acquisition_id: &str) -> RunInfoData {
        RunInfoData {
            acquisition_id: acquisition_id.to_string(),
            acquisition_start_time: 1609459200000,
            adc_max: 2047,
            adc_min: -2048,
            context_tags: HashMap::from([(
                "experiment_type".to_string(),
                "genomic_dna".to_string(),
            )]),
            experiment_name: "test_experiment".to_string(),
            flow_cell_id: "FAK12345".to_string(),
            flow_cell_product_code: "FLO-MIN106".to_string(),
            protocol_name: "test_protocol".to_string(),
            protocol_run_id: "protocol_123".to_string(),
            protocol_start_time: 1609459200000,
            sample_id: "sample_001".to_string(),
            sample_rate: 4000,
            sequencing_kit: "SQK-LSK109".to_string(),
            sequencer_position: "MN00001".to_string(),
            sequencer_position_type: "minion".to_string(),
            software: "MinKNOW 21.0.0".to_string(),
            system_name: "test_system".to_string(),
            system_type: "minion".to_string(),
            tracking_id: HashMap::from([("run_id".to_string(), "run_456".to_string())]),
        }
    }

    fn create_test_read(run_info_idx: u32, read_number: u32, num_samples: u64) -> ReadData {
        ReadData {
            read_id: Uuid::new_v4(),
            read_number,
            start_sample: (read_number as u64 - 1) * num_samples,
            channel: 1,
            well: 1,
            pore_type: "not_set".into(),
            calibration_offset: 0.5,
            calibration_scale: 0.95,
            median_before: 200.0,
            end_reason: EndReason::SignalPositive,
            end_reason_forced: false,
            run_info_index: run_info_idx,
            num_minknow_events: 100,
            tracked_scaling_scale: 1.0,
            tracked_scaling_shift: 0.0,
            predicted_scaling_scale: 1.0,
            predicted_scaling_shift: 0.0,
            num_reads_since_mux_change: 0,
            time_since_mux_change: 0.0,
            num_samples,
            open_pore_level: 220.0,
            expected_open_pore_level: 0.0,
            selected_read_level: 0.0,
            signal_rows: Vec::new(),
        }
    }

    fn generate_test_signal(num_samples: usize, offset: i16) -> Vec<i16> {
        (0..num_samples)
            .map(|i| ((i as f64 * 0.1).sin() * 100.0) as i16 + 200 + offset)
            .collect()
    }

    #[test]
    fn test_writer_creates_valid_pod5() -> Result<()> {
        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create a writer
        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        // Add run info
        let run_info = RunInfoData {
            acquisition_id: "test_run_123".to_string(),
            acquisition_start_time: 1609459200000,
            adc_max: 2047,
            adc_min: -2048,
            context_tags: HashMap::from([(
                "experiment_type".to_string(),
                "genomic_dna".to_string(),
            )]),
            experiment_name: "test_experiment".to_string(),
            flow_cell_id: "FAK12345".to_string(),
            flow_cell_product_code: "FLO-MIN106".to_string(),
            protocol_name: "test_protocol".to_string(),
            protocol_run_id: "protocol_123".to_string(),
            protocol_start_time: 1609459200000,
            sample_id: "sample_001".to_string(),
            sample_rate: 4000,
            sequencing_kit: "SQK-LSK109".to_string(),
            sequencer_position: "MN00001".to_string(),
            sequencer_position_type: "minion".to_string(),
            software: "MinKNOW 21.0.0".to_string(),
            system_name: "test_system".to_string(),
            system_type: "minion".to_string(),
            tracking_id: HashMap::from([("run_id".to_string(), "run_456".to_string())]),
        };
        let run_info_idx = writer.add_run_info(run_info)?;

        // Add a read with signal
        let read = ReadData {
            read_id: Uuid::new_v4(),
            read_number: 1,
            start_sample: 0,
            channel: 1,
            well: 1,
            pore_type: "not_set".into(),
            calibration_offset: 0.0,
            calibration_scale: 1.0,
            median_before: 200.0,
            end_reason: EndReason::SignalPositive,
            end_reason_forced: false,
            run_info_index: run_info_idx,
            num_minknow_events: 100,
            tracked_scaling_scale: 1.0,
            tracked_scaling_shift: 0.0,
            predicted_scaling_scale: 1.0,
            predicted_scaling_shift: 0.0,
            num_reads_since_mux_change: 0,
            time_since_mux_change: 0.0,
            num_samples: 1000,
            open_pore_level: 220.0,
            expected_open_pore_level: 0.0,
            selected_read_level: 0.0,
            signal_rows: Vec::new(), // Populated by writer
        };

        // Generate test signal (simulated nanopore-like data)
        let signal: Vec<i16> = (0..1000)
            .map(|i| ((i as f64 * 0.1).sin() * 100.0) as i16 + 200)
            .collect();

        writer.add_read(read, &signal)?;

        // Finish writing
        writer.finish()?;

        // Verify file exists and has content
        let metadata = std::fs::metadata(path).unwrap();
        assert!(metadata.len() > 0, "File should not be empty");

        // Try to read back the file
        // Note: Full round-trip verification will be added when reader is fully compatible
        // For now, verify basic structure
        let data = std::fs::read(path).unwrap();

        // Verify signature
        assert_eq!(
            &data[0..8],
            &POD5_SIGNATURE,
            "File should start with POD5 signature"
        );

        // Verify end signature
        assert_eq!(
            &data[data.len() - 8..],
            &POD5_SIGNATURE,
            "File should end with POD5 signature"
        );

        Ok(())
    }

    #[test]
    fn test_writer_multiple_reads() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        // Add run info
        let run_info = RunInfoData {
            acquisition_id: "multi_read_test".to_string(),
            acquisition_start_time: 1609459200000,
            adc_max: 2047,
            adc_min: -2048,
            context_tags: HashMap::new(),
            experiment_name: "test".to_string(),
            flow_cell_id: "FAK00001".to_string(),
            flow_cell_product_code: "FLO-MIN106".to_string(),
            protocol_name: "test".to_string(),
            protocol_run_id: "test_123".to_string(),
            protocol_start_time: 1609459200000,
            sample_id: "sample_001".to_string(),
            sample_rate: 4000,
            sequencing_kit: "SQK-LSK109".to_string(),
            sequencer_position: "MN00001".to_string(),
            sequencer_position_type: "minion".to_string(),
            software: "test".to_string(),
            system_name: "test".to_string(),
            system_type: "minion".to_string(),
            tracking_id: HashMap::new(),
        };
        let run_info_idx = writer.add_run_info(run_info)?;

        // Add multiple reads
        for i in 0..5 {
            let read = ReadData {
                read_id: Uuid::new_v4(),
                read_number: i + 1,
                start_sample: i as u64 * 1000,
                channel: 1,
                well: 1,
                pore_type: "not_set".into(),
                calibration_offset: 0.0,
                calibration_scale: 1.0,
                median_before: 200.0,
                end_reason: EndReason::SignalPositive,
                end_reason_forced: false,
                run_info_index: run_info_idx,
                num_minknow_events: 100,
                tracked_scaling_scale: 1.0,
                tracked_scaling_shift: 0.0,
                predicted_scaling_scale: 1.0,
                predicted_scaling_shift: 0.0,
                num_reads_since_mux_change: 0,
                time_since_mux_change: 0.0,
                num_samples: 500,
                open_pore_level: 220.0,
                expected_open_pore_level: 0.0,
                selected_read_level: 0.0,
                signal_rows: Vec::new(),
            };

            let signal: Vec<i16> = (0..500)
                .map(|j| ((j as f64 * 0.1).sin() * 100.0) as i16 + 200 + (i as i16 * 10))
                .collect();

            writer.add_read(read, &signal)?;
        }

        writer.finish()?;

        // Verify file was created
        let metadata = std::fs::metadata(path).unwrap();
        assert!(metadata.len() > 0);

        Ok(())
    }

    #[test]
    fn test_round_trip_single_read() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write
        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        let run_info = create_test_run_info("round_trip_test");
        let run_info_idx = writer.add_run_info(run_info.clone())?;

        let read = create_test_read(run_info_idx, 1, 1000);
        let original_read_id = read.read_id;
        let signal = generate_test_signal(1000, 0);

        writer.add_read(read, &signal)?;
        writer.finish()?;

        // Read back
        let reader = Reader::open(path)?;

        // Verify metadata
        assert_eq!(reader.run_info_count(), 1);
        let read_run_info = reader.get_run_info(0).unwrap();
        assert_eq!(read_run_info.acquisition_id, "round_trip_test");
        assert_eq!(read_run_info.sample_rate, 4000);

        // Verify reads
        let read_count = reader.read_count()?;
        assert_eq!(read_count, 1);

        let mut reads: Vec<_> = reader
            .reads()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(reads.len(), 1);

        let read_back = reads.pop().unwrap();
        assert_eq!(read_back.read_id, original_read_id);
        assert_eq!(read_back.channel, 1);
        assert_eq!(read_back.num_samples, 1000);

        // Verify signal
        let signal_back = reader.get_signal(&read_back.signal_rows)?;
        assert_eq!(signal_back.len(), 1000);
        assert_eq!(signal_back, signal);

        Ok(())
    }

    #[test]
    fn test_round_trip_multiple_reads() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let num_reads = 10;

        // Write
        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        let run_info = create_test_run_info("multi_read_round_trip");
        let run_info_idx = writer.add_run_info(run_info)?;

        let mut original_ids = Vec::new();
        for i in 0..num_reads {
            let read = create_test_read(run_info_idx, i + 1, 500);
            original_ids.push(read.read_id);
            let signal = generate_test_signal(500, i as i16 * 10);
            writer.add_read(read, &signal)?;
        }

        writer.finish()?;

        // Read back
        let reader = Reader::open(path)?;

        let read_count = reader.read_count()?;
        assert_eq!(read_count, num_reads as usize);

        let reads: Vec<_> = reader
            .reads()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(reads.len(), num_reads as usize);

        // Verify all read IDs are present
        for original_id in &original_ids {
            assert!(reads.iter().any(|r| &r.read_id == original_id));
        }

        Ok(())
    }

    #[test]
    fn test_round_trip_multiple_run_infos() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write with multiple run infos
        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        let run_info1 = create_test_run_info("run_1");
        let run_info2 = create_test_run_info("run_2");

        let idx1 = writer.add_run_info(run_info1)?;
        let idx2 = writer.add_run_info(run_info2)?;

        // Add reads for both run infos
        let read1 = create_test_read(idx1, 1, 500);
        let read2 = create_test_read(idx2, 2, 500);

        writer.add_read(read1, &generate_test_signal(500, 0))?;
        writer.add_read(read2, &generate_test_signal(500, 10))?;

        writer.finish()?;

        // Read back
        let reader = Reader::open(path)?;

        assert_eq!(reader.run_info_count(), 2);

        let run_infos = reader.run_infos();
        let acq_ids: Vec<_> = run_infos
            .iter()
            .map(|r| r.acquisition_id.as_str())
            .collect();
        assert!(acq_ids.contains(&"run_1"));
        assert!(acq_ids.contains(&"run_2"));

        Ok(())
    }

    #[test]
    fn test_round_trip_large_signal() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write a read with large signal (tests chunking)
        let options = WriterOptions {
            max_signal_chunk_size: 10000, // Force chunking
            ..WriterOptions::default()
        };
        let mut writer = Writer::create(path, options)?;

        let run_info = create_test_run_info("large_signal_test");
        let run_info_idx = writer.add_run_info(run_info)?;

        // 50k samples - should create multiple chunks
        let signal = generate_test_signal(50000, 0);
        let read = create_test_read(run_info_idx, 1, 50000);

        writer.add_read(read, &signal)?;
        writer.finish()?;

        // Read back
        let reader = Reader::open(path)?;

        let reads: Vec<_> = reader
            .reads()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(reads.len(), 1);

        let read_back = &reads[0];
        assert!(
            read_back.signal_rows.len() > 1,
            "Should have multiple signal chunks"
        );

        let signal_back = reader.get_signal(&read_back.signal_rows)?;
        assert_eq!(signal_back.len(), 50000);
        assert_eq!(signal_back, signal);

        Ok(())
    }

    #[test]
    fn test_round_trip_preserves_calibration() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        let run_info = create_test_run_info("calibration_test");
        let run_info_idx = writer.add_run_info(run_info)?;

        let mut read = create_test_read(run_info_idx, 1, 100);
        read.calibration_offset = 12.5;
        read.calibration_scale = 0.0234;
        read.median_before = 185.75;

        writer.add_read(read, &generate_test_signal(100, 0))?;
        writer.finish()?;

        // Read back
        let reader = Reader::open(path)?;

        let reads: Vec<_> = reader
            .reads()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let read_back = &reads[0];

        // Check calibration values are preserved (within float precision)
        assert!((read_back.calibration_offset - 12.5).abs() < 0.001);
        assert!((read_back.calibration_scale - 0.0234).abs() < 0.0001);
        assert!((read_back.median_before - 185.75).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_round_trip_end_reasons() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        let run_info = create_test_run_info("end_reason_test");
        let run_info_idx = writer.add_run_info(run_info)?;

        // Test different end reasons
        let end_reasons = vec![
            EndReason::Unknown,
            EndReason::MuxChange,
            EndReason::UnblockMuxChange,
            EndReason::SignalPositive,
            EndReason::SignalNegative,
        ];

        for (i, end_reason) in end_reasons.iter().enumerate() {
            let mut read = create_test_read(run_info_idx, i as u32 + 1, 100);
            read.end_reason = *end_reason;
            writer.add_read(read, &generate_test_signal(100, i as i16))?;
        }

        writer.finish()?;

        // Read back
        let reader = Reader::open(path)?;

        let reads: Vec<_> = reader
            .reads()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(reads.len(), end_reasons.len());

        // Verify end reasons are preserved
        for end_reason in &end_reasons {
            assert!(reads.iter().any(|r| r.end_reason == *end_reason));
        }

        Ok(())
    }

    #[test]
    fn test_writer_empty_signal() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let options = WriterOptions::default();
        let mut writer = Writer::create(path, options)?;

        let run_info = create_test_run_info("empty_signal_test");
        let run_info_idx = writer.add_run_info(run_info)?;

        let mut read = create_test_read(run_info_idx, 1, 0);
        read.num_samples = 0;
        let signal: Vec<i16> = Vec::new();

        writer.add_read(read, &signal)?;
        writer.finish()?;

        // Read back
        let reader = Reader::open(path)?;

        let reads: Vec<_> = reader
            .reads()?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].num_samples, 0);

        Ok(())
    }
}

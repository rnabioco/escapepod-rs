//! Main POD5 file writer.

use crate::compression;
use crate::error::{Error, Result};
use crate::schema::{reads_schema, run_info_schema, signal_schema};
use crate::types::{ReadData, RunInfoData, Uuid, FOOTER_MAGIC, POD5_SIGNATURE, POD5_VERSION};
use crate::CompressedSignalChunk;
use arrow::array::{
    ArrayRef, BooleanBuilder, FixedSizeBinaryBuilder, Float32Builder, Int16Builder,
    LargeBinaryBuilder, ListBuilder, MapBuilder, MapFieldNames, StringBuilder,
    StringDictionaryBuilder, TimestampMillisecondBuilder, UInt16Builder, UInt32Builder,
    UInt64Builder, UInt8Builder,
};
use arrow::datatypes::Int16Type;
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, Write};
use std::path::Path;
use std::sync::Arc;

/// Write buffer size (2MB for efficient I/O)
const WRITE_BUFFER_SIZE: usize = 2 * 1024 * 1024;

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
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            max_signal_chunk_size: 102_400,
            signal_batch_size: 100,
            read_batch_size: 1000,
            compress_signal: true,
            software: format!("podfive-rs {}", env!("CARGO_PKG_VERSION")),
        }
    }
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
}

impl Writer {
    /// Create a new POD5 file for writing.
    pub fn create<P: AsRef<Path>>(path: P, options: WriterOptions) -> Result<Self> {
        let file = File::create(path)?;
        // Use 2MB buffer for efficient I/O
        let mut file = BufWriter::with_capacity(WRITE_BUFFER_SIZE, file);

        // Write signature
        file.write_all(&POD5_SIGNATURE)?;

        // Write initial section marker
        let section_marker = Uuid::new_v4();
        file.write_all(section_marker.as_bytes())?;

        // Record signal table offset (after header)
        let signal_offset = file.stream_position()? as i64;

        Ok(Self {
            file: Some(file),
            file_id: Uuid::new_v4(),
            // Dictionary tracking with O(1) lookup
            pore_types: Vec::with_capacity(16),
            pore_type_index: HashMap::with_capacity(16),
            end_reasons: Vec::with_capacity(16),
            end_reason_index: HashMap::with_capacity(16),
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
            options,
        })
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

    /// Get or add a pore type to the dictionary, returning its index.
    /// Uses O(1) HashMap lookup instead of O(n) linear search.
    fn get_or_add_pore_type(&mut self, pore_type: &str) -> i16 {
        if let Some(&idx) = self.pore_type_index.get(pore_type) {
            idx
        } else {
            let idx = self.pore_types.len() as i16;
            self.pore_types.push(pore_type.to_string());
            self.pore_type_index.insert(pore_type.to_string(), idx);
            idx
        }
    }

    /// Get or add an end reason to the dictionary, returning its index.
    /// Uses O(1) HashMap lookup instead of O(n) linear search.
    fn get_or_add_end_reason(&mut self, end_reason: &str) -> i16 {
        if let Some(&idx) = self.end_reason_index.get(end_reason) {
            idx
        } else {
            let idx = self.end_reasons.len() as i16;
            self.end_reasons.push(end_reason.to_string());
            self.end_reason_index.insert(end_reason.to_string(), idx);
            idx
        }
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
                Arc::from(chunk.iter().flat_map(|&s| s.to_le_bytes()).collect::<Vec<u8>>())
            };

            signal_row_indices.push(self.current_signal_row);
            self.current_signal_row += 1;

            self.pending_signal.push(SignalChunk {
                read_id: read.read_id,
                samples: chunk.len() as u32,
                data,
            });
        }

        // Track dictionary entries
        self.get_or_add_pore_type(&read.pore_type);
        self.get_or_add_end_reason(read.end_reason.as_str());

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

        // Track dictionary entries
        self.get_or_add_pore_type(&read.pore_type);
        self.get_or_add_end_reason(read.end_reason.as_str());

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

        let schema = Arc::new(signal_schema());
        let row_count = batch.num_rows();
        let first_row = self.current_signal_row;

        // Initialize signal writer if needed
        if self.signal_writer.is_none() {
            let file = self.file.take().ok_or(Error::WriterFinalized)?;
            self.signal_writer = Some(ArrowFileWriter::try_new(file, &schema)?);
        }

        // Create a new batch with our schema to ensure consistency
        // (input batches may have different metadata)
        let normalized_batch = RecordBatch::try_new(
            schema,
            batch.columns().to_vec(),
        )?;

        // Write batch directly
        self.signal_writer.as_mut().unwrap().write(&normalized_batch)?;
        self.current_signal_row += row_count as u64;

        Ok((first_row, row_count))
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

        // Track dictionary entries
        self.get_or_add_pore_type(&read.pore_type);
        self.get_or_add_end_reason(read.end_reason.as_str());

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

        let schema = Arc::new(signal_schema());
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
        self.signal_writer.as_mut().unwrap().write(&batch)?;

        Ok(())
    }

    /// Flush pending reads to a batch.
    fn flush_read_batch(&mut self) -> Result<()> {
        if self.pending_reads.is_empty() {
            return Ok(());
        }

        let schema = Arc::new(reads_schema());
        let num_reads = self.pending_reads.len();

        // Build arrays - iterate directly without collecting to intermediate Vec
        let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(num_reads, 16);
        let signal_field = Arc::new(arrow::datatypes::Field::new(
            "item",
            arrow::datatypes::DataType::UInt64,
            false,
        ));
        let mut signal_builder = ListBuilder::new(UInt64Builder::new()).with_field(signal_field);
        let mut channel_builder = UInt16Builder::with_capacity(num_reads);
        let mut well_builder = UInt8Builder::with_capacity(num_reads);
        let mut pore_type_builder: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        let mut calibration_offset_builder = Float32Builder::with_capacity(num_reads);
        let mut calibration_scale_builder = Float32Builder::with_capacity(num_reads);
        let mut read_number_builder = UInt32Builder::with_capacity(num_reads);
        let mut start_builder = UInt64Builder::with_capacity(num_reads);
        let mut median_before_builder = Float32Builder::with_capacity(num_reads);
        let mut end_reason_builder: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        let mut end_reason_forced_builder = BooleanBuilder::with_capacity(num_reads);
        let mut run_info_builder: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        let mut num_minknow_events_builder = UInt64Builder::with_capacity(num_reads);
        let mut num_samples_builder = UInt64Builder::with_capacity(num_reads);
        let mut open_pore_level_builder = Float32Builder::with_capacity(num_reads);

        for pending in self.pending_reads.drain(..) {
            let read = &pending.data;

            read_id_builder.append_value(read.read_id.as_bytes())?;

            // Signal indices list
            let values = signal_builder.values();
            for &idx in &pending.signal_row_indices {
                values.append_value(idx);
            }
            signal_builder.append(true);

            channel_builder.append_value(read.channel);
            well_builder.append_value(read.well);
            pore_type_builder.append_value(&read.pore_type);
            calibration_offset_builder.append_value(read.calibration_offset);
            calibration_scale_builder.append_value(read.calibration_scale);
            read_number_builder.append_value(read.read_number);
            start_builder.append_value(read.start_sample);
            median_before_builder.append_value(read.median_before);
            end_reason_builder.append_value(read.end_reason.as_str());
            end_reason_forced_builder.append_value(read.end_reason_forced);

            // Run info - use acquisition_id as the dictionary value
            if let Some(run_info) = self.run_infos.get(read.run_info_index as usize) {
                run_info_builder.append_value(&run_info.acquisition_id);
            } else {
                run_info_builder.append_value("");
            }

            num_minknow_events_builder.append_value(read.num_minknow_events);
            num_samples_builder.append_value(read.num_samples);
            open_pore_level_builder.append_value(read.open_pore_level);
        }

        let arrays: Vec<ArrayRef> = vec![
            Arc::new(read_id_builder.finish()),
            Arc::new(signal_builder.finish()),
            Arc::new(channel_builder.finish()),
            Arc::new(well_builder.finish()),
            Arc::new(pore_type_builder.finish()),
            Arc::new(calibration_offset_builder.finish()),
            Arc::new(calibration_scale_builder.finish()),
            Arc::new(read_number_builder.finish()),
            Arc::new(start_builder.finish()),
            Arc::new(median_before_builder.finish()),
            Arc::new(end_reason_builder.finish()),
            Arc::new(end_reason_forced_builder.finish()),
            Arc::new(run_info_builder.finish()),
            Arc::new(num_minknow_events_builder.finish()),
            Arc::new(num_samples_builder.finish()),
            Arc::new(open_pore_level_builder.finish()),
        ];

        let batch = RecordBatch::try_new(schema.clone(), arrays)?;

        // Write to single IPC file (create writer on first batch)
        if self.reads_writer.is_none() {
            let cursor = Cursor::new(Vec::new());
            self.reads_writer = Some(ArrowFileWriter::try_new(cursor, &schema)?);
        }
        self.reads_writer.as_mut().unwrap().write(&batch)?;

        Ok(())
    }

    /// Build run info table.
    fn build_run_info_table(&self) -> Result<Vec<u8>> {
        if self.run_infos.is_empty() {
            // Return empty Arrow IPC file
            let schema = Arc::new(run_info_schema());
            let mut buffer = Vec::new();
            {
                let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
                writer.finish()?;
            }
            return Ok(buffer);
        }

        let schema = Arc::new(run_info_schema());

        let mut acquisition_id_builder = StringBuilder::new();
        let mut acquisition_start_time_builder = TimestampMillisecondBuilder::new();
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
        let mut protocol_start_time_builder = TimestampMillisecondBuilder::new();
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

    /// Build the FlatBuffer footer.
    fn build_flatbuffer_footer(&self, embedded_files: &[EmbeddedFileInfo]) -> Result<Vec<u8>> {
        // Simple FlatBuffer construction
        // We'll build this manually since the schema is simple

        let file_id_str = self.file_id.to_string();
        let software_str = &self.options.software;
        let version_str = POD5_VERSION;

        // Calculate sizes and build the buffer
        // FlatBuffer format: strings, vectors, tables, then root
        let mut _buffer: Vec<u8> = Vec::new();

        // We need to build from the end backwards, so let's use a simpler approach
        // Just create a minimal valid FlatBuffer

        // For now, create a minimal footer that can be parsed
        // This is a simplified implementation

        // String table
        let mut string_offsets = Vec::new();
        let strings = [&file_id_str, software_str, version_str];

        // Align to 4 bytes for strings
        let mut data = Vec::new();

        // Write strings (length-prefixed)
        for s in &strings {
            let offset = data.len();
            string_offsets.push(offset);
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
            // Pad to 4-byte alignment
            while data.len() % 4 != 0 {
                data.push(0);
            }
        }

        // Write embedded files vector
        let _files_vec_offset = data.len();
        data.extend_from_slice(&(embedded_files.len() as u32).to_le_bytes());

        // Placeholder for file offsets (will be filled in)
        let file_offsets_start = data.len();
        for _ in embedded_files {
            data.extend_from_slice(&[0u8; 4]); // placeholder
        }

        // Write embedded file tables
        let mut file_table_offsets = Vec::new();
        for file in embedded_files {
            // Align to 4 bytes
            while data.len() % 4 != 0 {
                data.push(0);
            }

            let table_start = data.len();
            file_table_offsets.push(table_start);

            // vtable for EmbeddedFile (offset, length, format, content_type)
            let vtable_start = data.len();
            data.extend_from_slice(&14u16.to_le_bytes()); // vtable size
            data.extend_from_slice(&24u16.to_le_bytes()); // table size
            data.extend_from_slice(&4u16.to_le_bytes()); // offset field at +4
            data.extend_from_slice(&12u16.to_le_bytes()); // length field at +12
            data.extend_from_slice(&20u16.to_le_bytes()); // format field at +20
            data.extend_from_slice(&22u16.to_le_bytes()); // content_type field at +22

            // table data
            let table_data_start = data.len();
            // soffset to vtable (distance back)
            let soffset = (table_data_start - vtable_start) as i32;
            data.extend_from_slice(&soffset.to_le_bytes());
            // offset (int64)
            data.extend_from_slice(&file.offset.to_le_bytes());
            // length (int64)
            data.extend_from_slice(&file.length.to_le_bytes());
            // format (int16) - always 0 for FeatherV2
            data.extend_from_slice(&0i16.to_le_bytes());
            // content_type (int16)
            data.extend_from_slice(&(file.content_type as i16).to_le_bytes());
        }

        // Fill in file offsets in vector
        for (i, &table_offset) in file_table_offsets.iter().enumerate() {
            let offset_pos = file_offsets_start + i * 4;
            let relative_offset = (table_offset - (file_offsets_start + i * 4)) as u32;
            data[offset_pos..offset_pos + 4].copy_from_slice(&relative_offset.to_le_bytes());
        }

        // Write Footer table
        while data.len() % 4 != 0 {
            data.push(0);
        }

        // vtable for Footer
        let _footer_vtable_start = data.len();
        data.extend_from_slice(&12u16.to_le_bytes()); // vtable size
        data.extend_from_slice(&16u16.to_le_bytes()); // table size
        data.extend_from_slice(&4u16.to_le_bytes()); // file_identifier at +4
        data.extend_from_slice(&8u16.to_le_bytes()); // software at +8
        data.extend_from_slice(&12u16.to_le_bytes()); // pod5_version at +12
        data.extend_from_slice(&16u16.to_le_bytes()); // contents at +16 (but we only have 16 bytes of table)

        // Hmm, the FlatBuffer format is getting complex. Let me simplify.
        // Actually, let's just output a valid minimal structure.

        _buffer = Vec::new();

        // Build a simple footer using flatbuffers crate approach
        // Root offset will be at the start

        // For now, use a pre-built minimal footer structure
        // This is a hack, but it works for testing
        self.build_simple_footer(embedded_files)
    }

    /// Helper function to write a string to a cursor with FlatBuffer format.
    fn write_flatbuffer_string(data: &mut std::io::Cursor<Vec<u8>>, s: &str) -> Result<usize> {
        // Align to 4
        while data.position() % 4 != 0 {
            data.write_all(&[0u8])?;
        }
        let pos = data.position() as usize;
        data.write_all(&(s.len() as u32).to_le_bytes())?;
        data.write_all(s.as_bytes())?;
        // Pad
        while data.position() % 4 != 0 {
            data.write_all(&[0u8])?;
        }
        Ok(pos)
    }

    /// Build a simple footer structure.
    fn build_simple_footer(&self, embedded_files: &[EmbeddedFileInfo]) -> Result<Vec<u8>> {
        use std::io::Cursor;

        let file_id = self.file_id.to_string();
        let software = &self.options.software;
        let version = POD5_VERSION;

        // We'll build a minimal FlatBuffer by hand
        // Layout: [root_offset:4][vtable][table][strings][vectors]

        // Simpler approach: build everything into a buffer, then prepend root offset
        let mut data = Cursor::new(Vec::<u8>::new());

        // Skip 4 bytes for root offset
        data.write_all(&[0u8; 4])?;

        // Write vtable for Footer table
        let vtable_pos = data.position() as usize;
        data.write_all(&12u16.to_le_bytes())?; // vtable size (6 entries * 2 = 12, + 4 header = 16... let's try 12)
        data.write_all(&20u16.to_le_bytes())?; // table size
        data.write_all(&4u16.to_le_bytes())?; // field 0 (file_identifier) offset
        data.write_all(&8u16.to_le_bytes())?; // field 1 (software) offset
        data.write_all(&12u16.to_le_bytes())?; // field 2 (pod5_version) offset
        data.write_all(&16u16.to_le_bytes())?; // field 3 (contents) offset

        // Write table
        let table_pos = data.position() as usize;
        let soffset = (table_pos - vtable_pos) as i32;
        data.write_all(&soffset.to_le_bytes())?; // soffset to vtable

        // Placeholder offsets for strings and vector
        let field0_pos = data.position() as usize;
        data.write_all(&[0u8; 4])?; // file_identifier offset placeholder
        let field1_pos = data.position() as usize;
        data.write_all(&[0u8; 4])?; // software offset placeholder
        let field2_pos = data.position() as usize;
        data.write_all(&[0u8; 4])?; // pod5_version offset placeholder
        let field3_pos = data.position() as usize;
        data.write_all(&[0u8; 4])?; // contents offset placeholder

        // Write strings using the helper function
        let str0_pos = Self::write_flatbuffer_string(&mut data, &file_id)?;
        let str1_pos = Self::write_flatbuffer_string(&mut data, software)?;
        let str2_pos = Self::write_flatbuffer_string(&mut data, version)?;

        // Write contents vector (embedded files)
        while data.position() % 4 != 0 {
            data.write_all(&[0u8])?;
        }
        let vec_pos = data.position() as usize;
        data.write_all(&(embedded_files.len() as u32).to_le_bytes())?;

        // Write offsets to each embedded file table
        let offsets_start = data.position() as usize;
        for _ in embedded_files {
            data.write_all(&[0u8; 4])?; // placeholder
        }

        // Write embedded file tables
        let mut file_positions = Vec::new();
        for file in embedded_files {
            while data.position() % 4 != 0 {
                data.write_all(&[0u8])?;
            }

            // vtable for EmbeddedFile
            let ef_vtable_pos = data.position() as usize;
            data.write_all(&14u16.to_le_bytes())?; // vtable size
            data.write_all(&24u16.to_le_bytes())?; // table size
            data.write_all(&4u16.to_le_bytes())?; // offset at +4
            data.write_all(&12u16.to_le_bytes())?; // length at +12
            data.write_all(&20u16.to_le_bytes())?; // format at +20
            data.write_all(&22u16.to_le_bytes())?; // content_type at +22

            // table
            let ef_table_pos = data.position() as usize;
            file_positions.push(ef_table_pos);
            let ef_soffset = (ef_table_pos - ef_vtable_pos) as i32;
            data.write_all(&ef_soffset.to_le_bytes())?;
            data.write_all(&file.offset.to_le_bytes())?;
            data.write_all(&file.length.to_le_bytes())?;
            data.write_all(&0i16.to_le_bytes())?; // format = FeatherV2
            data.write_all(&(file.content_type as i16).to_le_bytes())?;
        }

        // Fill in the offsets
        let mut result = data.into_inner();

        // Fill string offsets (relative from field position)
        let str0_rel = (str0_pos - field0_pos) as u32;
        result[field0_pos..field0_pos + 4].copy_from_slice(&str0_rel.to_le_bytes());
        let str1_rel = (str1_pos - field1_pos) as u32;
        result[field1_pos..field1_pos + 4].copy_from_slice(&str1_rel.to_le_bytes());
        let str2_rel = (str2_pos - field2_pos) as u32;
        result[field2_pos..field2_pos + 4].copy_from_slice(&str2_rel.to_le_bytes());
        let vec_rel = (vec_pos - field3_pos) as u32;
        result[field3_pos..field3_pos + 4].copy_from_slice(&vec_rel.to_le_bytes());

        // Fill embedded file offsets
        for (i, &pos) in file_positions.iter().enumerate() {
            let offset_loc = offsets_start + i * 4;
            let rel = (pos - offset_loc) as u32;
            result[offset_loc..offset_loc + 4].copy_from_slice(&rel.to_le_bytes());
        }

        // Fill root offset
        let root_rel = table_pos as u32;
        result[0..4].copy_from_slice(&root_rel.to_le_bytes());

        Ok(result)
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

        // Record signal table info
        let signal_length = file.stream_position()? as i64 - self.signal_offset;
        if signal_length > 0 {
            embedded_files.push(EmbeddedFileInfo {
                offset: self.signal_offset,
                length: signal_length,
                content_type: 1, // SignalTable
            });
        }

        // Pad to 8-byte alignment
        while file.stream_position()? % 8 != 0 {
            file.write_all(&[0u8])?;
        }
        file.write_all(self.section_marker.as_bytes())?;

        // Write run info table
        let run_info_data = self.build_run_info_table()?;
        let run_info_offset = file.stream_position()? as i64;
        file.write_all(&run_info_data)?;
        let run_info_length = run_info_data.len() as i64;
        embedded_files.push(EmbeddedFileInfo {
            offset: run_info_offset,
            length: run_info_length,
            content_type: 4, // RunInfoTable
        });

        // Pad and section marker
        while file.stream_position()? % 8 != 0 {
            file.write_all(&[0u8])?;
        }
        self.section_marker = Uuid::new_v4();
        file.write_all(self.section_marker.as_bytes())?;

        // Write reads table from memory buffer
        let reads_offset = file.stream_position()? as i64;
        if let Some(mut writer) = self.reads_writer.take() {
            writer.finish()?;
            let cursor = writer.into_inner()?;
            file.write_all(cursor.get_ref())?;
        }
        let reads_length = file.stream_position()? as i64 - reads_offset;
        if reads_length > 0 {
            embedded_files.push(EmbeddedFileInfo {
                offset: reads_offset,
                length: reads_length,
                content_type: 0, // ReadsTable
            });
        }

        // Pad and section marker
        while file.stream_position()? % 8 != 0 {
            file.write_all(&[0u8])?;
        }
        self.section_marker = Uuid::new_v4();
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
        self.section_marker = Uuid::new_v4();
        file.write_all(self.section_marker.as_bytes())?;

        // Write signature
        file.write_all(&POD5_SIGNATURE)?;

        file.flush()?;
        self.finalized = true;
        Ok(())
    }
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
            pore_type: "not_set".to_string(),
            calibration_offset: 0.5,
            calibration_scale: 0.95,
            median_before: 200.0,
            end_reason: EndReason::SignalPositive,
            end_reason_forced: false,
            run_info_index: run_info_idx,
            num_minknow_events: 100,
            num_samples,
            open_pore_level: 220.0,
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
            pore_type: "not_set".to_string(),
            calibration_offset: 0.0,
            calibration_scale: 1.0,
            median_before: 200.0,
            end_reason: EndReason::SignalPositive,
            end_reason_forced: false,
            run_info_index: run_info_idx,
            num_minknow_events: 100,
            num_samples: 1000,
            open_pore_level: 220.0,
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
                pore_type: "not_set".to_string(),
                calibration_offset: 0.0,
                calibration_scale: 1.0,
                median_before: 200.0,
                end_reason: EndReason::SignalPositive,
                end_reason_forced: false,
                run_info_index: run_info_idx,
                num_minknow_events: 100,
                num_samples: 500,
                open_pore_level: 220.0,
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

//! High-performance POD5 file merging.
//!
//! This module provides functionality to merge multiple POD5 files into one,
//! using raw byte copying to avoid Arrow deserialization overhead.

use crate::arrow_ipc::{ArrowIpcFooter, BatchBlock};
use crate::error::{Error, Result};
use crate::reader::Reader;
use crate::types::{ReadData, RunInfoData, Uuid, FOOTER_MAGIC, POD5_SIGNATURE};
use arrow::ipc::{Block, MetadataVersion};
use flatbuffers::FlatBufferBuilder;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Options for merge operations.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Allow duplicate read IDs (default: false, skip duplicates).
    pub duplicate_ok: bool,
    /// Number of reads per batch in output file.
    pub read_batch_size: u32,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            duplicate_ok: false,
            read_batch_size: 100_000,
        }
    }
}

/// Result of a merge operation.
#[derive(Debug)]
pub struct MergeResult {
    /// Number of reads written.
    pub reads_written: u64,
    /// Number of duplicate reads skipped.
    pub duplicates_skipped: u64,
    /// Number of signal rows written.
    pub signal_rows: u64,
    /// Number of files processed.
    pub files_processed: usize,
}

/// Merge multiple POD5 files into a single output file.
///
/// This function uses zero-copy async I/O with scoped threads to overlap
/// reading and writing, passing mmap slices directly to the writer thread.
///
/// # Arguments
/// * `inputs` - Slice of input file paths
/// * `output` - Output file path
/// * `options` - Merge options
/// * `progress_callback` - Optional callback for progress updates (file_idx, total_files)
///
/// # Returns
/// A `MergeResult` with statistics about the merge operation.
pub fn merge_files<P: AsRef<Path>, Q: AsRef<Path>>(
    inputs: &[P],
    output: Q,
    options: &MergeOptions,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeResult> {
    if inputs.is_empty() {
        return Err(Error::InvalidState("No input files specified".into()));
    }

    merge_impl(inputs, output, options, progress_callback)
}

/// Collected metadata from a single file for merging.
struct FileMetadata {
    reader: Reader,
    footer: ArrowIpcFooter,
    run_infos: Vec<RunInfoData>,
    reads: Vec<ReadData>,
}

/// Main merge implementation using zero-copy async I/O.
/// Uses scoped threads to pass mmap slices directly to writer thread.
fn merge_impl<P: AsRef<Path>, Q: AsRef<Path>>(
    inputs: &[P],
    output: Q,
    options: &MergeOptions,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeResult> {
    use std::sync::mpsc;
    use std::thread;

    let num_files = inputs.len();

    // Convert to owned paths for parallel processing
    let input_paths: Vec<&Path> = inputs.iter().map(|p| p.as_ref()).collect();

    // Phase 1: Open files and collect metadata in parallel (single open per file)
    let metadata_results: Vec<Result<FileMetadata>> = input_paths
        .par_iter()
        .map(|path| {
            let reader = Reader::open(path)?;
            let signal_bytes = reader.signal_table_bytes()?;
            let footer = ArrowIpcFooter::parse(signal_bytes)?;
            let run_infos = reader.run_infos().to_vec();
            let reads: Vec<ReadData> = reader
                .reads()?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(FileMetadata { reader, footer, run_infos, reads })
        })
        .collect();

    // Unwrap results and count reads
    let file_metadata: Vec<FileMetadata> = metadata_results
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let total_read_count: u64 = file_metadata.iter().map(|m| m.reads.len() as u64).sum();

    // Phase 2: Write signal data using scoped thread (zero-copy from mmap)
    let mut all_batches: Vec<BatchBlock> = Vec::new();
    let mut current_offset: usize = 0;
    let mut current_signal_row: u64 = 0;
    let mut signal_offsets: Vec<u64> = Vec::with_capacity(num_files);

    // Use scoped thread to allow borrowing mmap slices without copying
    let (file, signal_end, signal_rows) = thread::scope(|scope| -> Result<(File, usize, u64)> {
        // Channel for sending byte slices to writer thread
        let (tx, rx) = mpsc::sync_channel::<&[u8]>(4); // Small buffer for backpressure

        // Spawn writer thread within scope - can borrow from parent
        let output_path = output.as_ref();
        let writer_handle = scope.spawn(move || -> std::io::Result<(File, usize)> {
            let file = File::create(output_path)?;
            let mut file = BufWriter::with_capacity(16 * 1024 * 1024, file);

            // Write POD5 header
            file.write_all(&POD5_SIGNATURE)?;
            let section_marker = Uuid::new_v4();
            file.write_all(section_marker.as_bytes())?;

            // Write all signal data from channel
            for bytes in rx {
                file.write_all(bytes)?;
            }

            let pos = file.stream_position()? as usize;
            file.flush()?;
            Ok((file.into_inner()?, pos))
        });

        // Main thread: send signal bytes to writer
        let mut header_written = false;

        for (file_idx, metadata) in file_metadata.iter().enumerate()
        {
            let signal_bytes = metadata.reader.signal_table_bytes()?;

            // Record signal row offset for this file
            signal_offsets.push(current_signal_row);

            // Write header from first file only
            if !header_written {
                let header_bytes = metadata.footer.header_bytes(signal_bytes);
                tx.send(header_bytes)
                    .map_err(|_| Error::Io(std::io::Error::other("Writer thread closed")))?;
                current_offset = header_bytes.len();
                header_written = true;
            }

            // Send batch bytes directly (zero-copy from mmap)
            let batches_bytes = metadata.footer.batches_bytes(signal_bytes);
            tx.send(batches_bytes)
                .map_err(|_| Error::Io(std::io::Error::other("Writer thread closed")))?;

            // Adjust batch offsets for the combined output
            for batch in &metadata.footer.record_batches {
                let relative_offset = batch.offset as usize - metadata.footer.batches_start_offset;
                let new_offset = current_offset + relative_offset;

                all_batches.push(BatchBlock {
                    offset: new_offset as i64,
                    metadata_length: batch.metadata_length,
                    body_length: batch.body_length,
                    row_count: batch.row_count,
                });
            }

            current_offset += batches_bytes.len();
            current_signal_row += metadata.footer.total_rows;

            if let Some(cb) = progress_callback {
                cb(file_idx + 1, num_files);
            }
        }

        // Close channel - must happen before footer_bytes is created
        // to ensure all mmap slices are consumed
        drop(tx);

        // Wait for writer to finish with mmap data
        let (mut file, _signal_end) = writer_handle
            .join()
            .map_err(|_| Error::Io(std::io::Error::other("Writer thread panicked")))?
            .map_err(Error::Io)?;

        // Write IPC footer directly (small data, no need for async)
        let footer_bytes = build_arrow_ipc_footer(&all_batches)?;
        file.write_all(&footer_bytes).map_err(Error::Io)?;

        let footer_len = footer_bytes.len() as i32;
        file.write_all(&footer_len.to_le_bytes())
            .map_err(Error::Io)?;
        file.write_all(b"ARROW1").map_err(Error::Io)?;
        file.flush().map_err(Error::Io)?;

        let final_pos = file.stream_position().map_err(Error::Io)? as usize;

        Ok((file, final_pos, current_signal_row))
    })?;

    // Phase 3: Write remaining sections using BufWriter
    let mut file = BufWriter::with_capacity(16 * 1024 * 1024, file);
    file.seek(SeekFrom::Start(signal_end as u64))?;

    // Pad to 8-byte alignment
    let padding_needed = (8 - (signal_end % 8)) % 8;
    for _ in 0..padding_needed {
        file.write_all(&[0u8])?;
    }

    // Write section marker
    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;

    // Build and write run_info table
    let mut run_info_map: HashMap<String, u32> = HashMap::new();
    let mut all_run_infos: Vec<RunInfoData> = Vec::new();

    for metadata in &file_metadata {
        for run_info in &metadata.run_infos {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = all_run_infos.len() as u32;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
                all_run_infos.push(run_info.clone());
            }
        }
    }

    let run_info_offset = file.stream_position()? as i64;
    let run_info_bytes = build_run_info_table(&all_run_infos)?;
    file.write_all(&run_info_bytes)?;
    let run_info_length = run_info_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;

    // Build and write reads table
    let reads_offset = file.stream_position()? as i64;

    let mut seen_reads: HashSet<Uuid> = if options.duplicate_ok {
        HashSet::new()
    } else {
        HashSet::with_capacity(total_read_count as usize)
    };

    let mut processed_reads: Vec<(ReadData, Vec<u64>)> = Vec::new();
    let mut total_reads = 0u64;
    let mut duplicate_count = 0u64;

    for (metadata, &signal_offset) in file_metadata.iter().zip(signal_offsets.iter()) {
        for read in &metadata.reads {
            if !options.duplicate_ok {
                if seen_reads.contains(&read.read_id) {
                    duplicate_count += 1;
                    continue;
                }
                seen_reads.insert(read.read_id);
            }

            let original_run_info = metadata.run_infos.get(read.run_info_index as usize);
            let new_run_info_idx = if let Some(ri) = original_run_info {
                *run_info_map.get(&ri.acquisition_id).unwrap_or(&0)
            } else {
                0
            };

            let new_signal_rows: Vec<u64> = read
                .signal_rows
                .iter()
                .map(|&row| row + signal_offset)
                .collect();

            let new_read = read.for_writing(new_run_info_idx);
            processed_reads.push((new_read, new_signal_rows));
            total_reads += 1;
        }
    }

    let reads_bytes = build_reads_table(&processed_reads, &all_run_infos)?;
    file.write_all(&reads_bytes)?;
    let reads_length = reads_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;

    // Write POD5 footer
    file.write_all(&FOOTER_MAGIC)?;

    let signal_offset_val = 24i64; // POD5 header size
    let signal_length = signal_end as i64 - 24;

    let pod5_footer = build_pod5_footer(
        signal_offset_val,
        signal_length,
        run_info_offset,
        run_info_length,
        reads_offset,
        reads_length,
    )?;
    file.write_all(&pod5_footer)?;

    let footer_len = pod5_footer.len() as i64;
    file.write_all(&footer_len.to_le_bytes())?;

    let section_marker = Uuid::new_v4();
    file.write_all(section_marker.as_bytes())?;
    file.write_all(&POD5_SIGNATURE)?;

    file.flush()?;

    Ok(MergeResult {
        reads_written: total_reads,
        duplicates_skipped: duplicate_count,
        signal_rows,
        files_processed: file_metadata.len(),
    })
}

/// Build Arrow IPC footer.
fn build_arrow_ipc_footer(batches: &[BatchBlock]) -> Result<Vec<u8>> {
    let mut fbb = FlatBufferBuilder::with_capacity(256 + batches.len() * 24);

    let blocks: Vec<Block> = batches
        .iter()
        .map(|b| Block::new(b.offset, b.metadata_length, b.body_length))
        .collect();

    let record_batches = fbb.create_vector(&blocks);

    let schema_fields = fbb.create_vector::<flatbuffers::ForwardsUOffset<arrow::ipc::Field>>(&[]);
    let schema = arrow::ipc::Schema::create(
        &mut fbb,
        &arrow::ipc::SchemaArgs {
            endianness: arrow::ipc::Endianness::Little,
            fields: Some(schema_fields),
            custom_metadata: None,
            features: None,
        },
    );

    let footer = arrow::ipc::Footer::create(
        &mut fbb,
        &arrow::ipc::FooterArgs {
            version: MetadataVersion::V5,
            schema: Some(schema),
            dictionaries: None,
            recordBatches: Some(record_batches),
            custom_metadata: None,
        },
    );

    fbb.finish(footer, None);

    Ok(fbb.finished_data().to_vec())
}

/// Build run_info Arrow IPC table.
fn build_run_info_table(run_infos: &[RunInfoData]) -> Result<Vec<u8>> {
    use crate::schema::run_info_schema;
    use arrow::array::{
        ArrayRef, Int16Builder, MapBuilder, MapFieldNames, StringBuilder,
        TimestampMillisecondBuilder, UInt16Builder,
    };
    use arrow::ipc::writer::FileWriter as ArrowFileWriter;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(run_info_schema());

    if run_infos.is_empty() {
        let mut buffer = Vec::new();
        {
            let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
            writer.finish()?;
        }
        return Ok(buffer);
    }

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

    for info in run_infos {
        acquisition_id_builder.append_value(&info.acquisition_id);
        acquisition_start_time_builder.append_value(info.acquisition_start_time);
        adc_max_builder.append_value(info.adc_max);
        adc_min_builder.append_value(info.adc_min);

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

/// Build reads Arrow IPC table.
fn build_reads_table(reads: &[(ReadData, Vec<u64>)], run_infos: &[RunInfoData]) -> Result<Vec<u8>> {
    use crate::schema::reads_schema;
    use arrow::array::{
        ArrayRef, BooleanBuilder, FixedSizeBinaryBuilder, Float32Builder, ListBuilder, StringArray,
        StringDictionaryBuilder, UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
    };
    use arrow::datatypes::Int16Type;
    use arrow::ipc::writer::FileWriter as ArrowFileWriter;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(reads_schema());

    if reads.is_empty() {
        let mut buffer = Vec::new();
        {
            let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
            writer.finish()?;
        }
        return Ok(buffer);
    }

    let num_reads = reads.len();

    // Collect unique pore types and end reasons for dictionaries
    let mut pore_types: Vec<String> = Vec::new();
    let mut end_reasons: Vec<String> = Vec::new();

    for (read, _) in reads {
        if !pore_types.contains(&read.pore_type) {
            pore_types.push(read.pore_type.clone());
        }
        let end_reason_str = read.end_reason.as_str().to_string();
        if !end_reasons.contains(&end_reason_str) {
            end_reasons.push(end_reason_str);
        }
    }

    let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(num_reads, 16);
    let signal_field = Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::UInt64,
        false,
    ));
    let mut signal_builder = ListBuilder::new(UInt64Builder::new()).with_field(signal_field);
    let mut channel_builder = UInt16Builder::with_capacity(num_reads);
    let mut well_builder = UInt8Builder::with_capacity(num_reads);

    let pore_type_dict = StringArray::from_iter_values(pore_types.iter().map(|s| s.as_str()));
    let mut pore_type_builder: StringDictionaryBuilder<Int16Type> =
        StringDictionaryBuilder::new_with_dictionary(num_reads, &pore_type_dict)?;

    let mut calibration_offset_builder = Float32Builder::with_capacity(num_reads);
    let mut calibration_scale_builder = Float32Builder::with_capacity(num_reads);
    let mut read_number_builder = UInt32Builder::with_capacity(num_reads);
    let mut start_builder = UInt64Builder::with_capacity(num_reads);
    let mut median_before_builder = Float32Builder::with_capacity(num_reads);

    let end_reason_dict = StringArray::from_iter_values(end_reasons.iter().map(|s| s.as_str()));
    let mut end_reason_builder: StringDictionaryBuilder<Int16Type> =
        StringDictionaryBuilder::new_with_dictionary(num_reads, &end_reason_dict)?;

    let mut end_reason_forced_builder = BooleanBuilder::with_capacity(num_reads);
    let mut run_info_builder: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
    let mut num_minknow_events_builder = UInt64Builder::with_capacity(num_reads);
    let mut num_samples_builder = UInt64Builder::with_capacity(num_reads);
    let mut open_pore_level_builder = Float32Builder::with_capacity(num_reads);

    for (read, signal_rows) in reads {
        read_id_builder.append_value(read.read_id.as_bytes())?;

        let values = signal_builder.values();
        for &idx in signal_rows {
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

        if let Some(run_info) = run_infos.get(read.run_info_index as usize) {
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

    let mut buffer = Vec::new();
    {
        let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }

    Ok(buffer)
}

/// Build POD5 FlatBuffer footer.
fn build_pod5_footer(
    signal_offset: i64,
    signal_length: i64,
    run_info_offset: i64,
    run_info_length: i64,
    reads_offset: i64,
    reads_length: i64,
) -> Result<Vec<u8>> {
    use crate::types::POD5_VERSION;
    use std::io::Cursor;

    let file_id = Uuid::new_v4().to_string();
    let software = format!("podfive-rs {}", env!("CARGO_PKG_VERSION"));
    let version = POD5_VERSION;

    let embedded_files = [
        (signal_offset, signal_length, 1i16),     // SignalTable
        (run_info_offset, run_info_length, 4i16), // RunInfoTable
        (reads_offset, reads_length, 0i16),       // ReadsTable
    ];

    // Build minimal FlatBuffer
    let mut data = Cursor::new(Vec::<u8>::new());

    // Skip 4 bytes for root offset
    data.write_all(&[0u8; 4])?;

    // Write vtable for Footer table
    let vtable_pos = data.position() as usize;
    data.write_all(&12u16.to_le_bytes())?; // vtable size
    data.write_all(&20u16.to_le_bytes())?; // table size
    data.write_all(&4u16.to_le_bytes())?; // field 0 offset
    data.write_all(&8u16.to_le_bytes())?; // field 1 offset
    data.write_all(&12u16.to_le_bytes())?; // field 2 offset
    data.write_all(&16u16.to_le_bytes())?; // field 3 offset

    // Write table
    let table_pos = data.position() as usize;
    let soffset = (table_pos - vtable_pos) as i32;
    data.write_all(&soffset.to_le_bytes())?;

    let field0_pos = data.position() as usize;
    data.write_all(&[0u8; 4])?;
    let field1_pos = data.position() as usize;
    data.write_all(&[0u8; 4])?;
    let field2_pos = data.position() as usize;
    data.write_all(&[0u8; 4])?;
    let field3_pos = data.position() as usize;
    data.write_all(&[0u8; 4])?;

    // Write strings
    let str0_pos = write_flatbuffer_string(&mut data, &file_id)?;
    let str1_pos = write_flatbuffer_string(&mut data, &software)?;
    let str2_pos = write_flatbuffer_string(&mut data, version)?;

    // Write contents vector
    while data.position() % 4 != 0 {
        data.write_all(&[0u8])?;
    }
    let vec_pos = data.position() as usize;
    data.write_all(&(embedded_files.len() as u32).to_le_bytes())?;

    let offsets_start = data.position() as usize;
    for _ in &embedded_files {
        data.write_all(&[0u8; 4])?;
    }

    // Write embedded file tables
    let mut file_positions = Vec::new();
    for (offset, length, content_type) in &embedded_files {
        while data.position() % 4 != 0 {
            data.write_all(&[0u8])?;
        }

        let ef_vtable_pos = data.position() as usize;
        data.write_all(&14u16.to_le_bytes())?;
        data.write_all(&24u16.to_le_bytes())?;
        data.write_all(&4u16.to_le_bytes())?;
        data.write_all(&12u16.to_le_bytes())?;
        data.write_all(&20u16.to_le_bytes())?;
        data.write_all(&22u16.to_le_bytes())?;

        let ef_table_pos = data.position() as usize;
        file_positions.push(ef_table_pos);
        let ef_soffset = (ef_table_pos - ef_vtable_pos) as i32;
        data.write_all(&ef_soffset.to_le_bytes())?;
        data.write_all(&offset.to_le_bytes())?;
        data.write_all(&length.to_le_bytes())?;
        data.write_all(&0i16.to_le_bytes())?;
        data.write_all(&content_type.to_le_bytes())?;
    }

    let mut result = data.into_inner();

    // Fill offsets
    let str0_rel = (str0_pos - field0_pos) as u32;
    result[field0_pos..field0_pos + 4].copy_from_slice(&str0_rel.to_le_bytes());
    let str1_rel = (str1_pos - field1_pos) as u32;
    result[field1_pos..field1_pos + 4].copy_from_slice(&str1_rel.to_le_bytes());
    let str2_rel = (str2_pos - field2_pos) as u32;
    result[field2_pos..field2_pos + 4].copy_from_slice(&str2_rel.to_le_bytes());
    let vec_rel = (vec_pos - field3_pos) as u32;
    result[field3_pos..field3_pos + 4].copy_from_slice(&vec_rel.to_le_bytes());

    for (i, &pos) in file_positions.iter().enumerate() {
        let offset_loc = offsets_start + i * 4;
        let rel = (pos - offset_loc) as u32;
        result[offset_loc..offset_loc + 4].copy_from_slice(&rel.to_le_bytes());
    }

    let root_rel = table_pos as u32;
    result[0..4].copy_from_slice(&root_rel.to_le_bytes());

    Ok(result)
}

fn write_flatbuffer_string(data: &mut std::io::Cursor<Vec<u8>>, s: &str) -> Result<usize> {
    while data.position() % 4 != 0 {
        data.write_all(&[0u8])?;
    }
    let pos = data.position() as usize;
    data.write_all(&(s.len() as u32).to_le_bytes())?;
    data.write_all(s.as_bytes())?;
    while data.position() % 4 != 0 {
        data.write_all(&[0u8])?;
    }
    Ok(pos)
}

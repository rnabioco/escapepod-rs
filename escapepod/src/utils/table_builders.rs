//! Table building utilities for POD5 file construction.
//!
//! Shared functions for building Arrow IPC tables and POD5 footers,
//! used by both merge and filter operations.

use crate::arrow_ipc::BatchBlock;
use crate::error::Result;
use crate::schema::{reads_schema, run_info_schema};
use crate::types::{ReadData, RunInfoData, Uuid, POD5_VERSION};
use arrow::array::{
    ArrayRef, BooleanBuilder, FixedSizeBinaryBuilder, Float32Builder, Int16Builder, ListBuilder,
    MapBuilder, MapFieldNames, StringArray, StringBuilder, StringDictionaryBuilder,
    TimestampMillisecondBuilder, UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow::datatypes::Int16Type;
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use arrow::ipc::{Block, MetadataVersion};
use arrow::record_batch::RecordBatch;
use flatbuffers::FlatBufferBuilder;
use std::collections::HashSet;
use std::io::{Cursor, Write};
use std::sync::Arc;

/// Build Arrow IPC footer from batch blocks.
pub(crate) fn build_arrow_ipc_footer(batches: &[BatchBlock]) -> Result<Vec<u8>> {
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
pub(crate) fn build_run_info_table(run_infos: &[RunInfoData]) -> Result<Vec<u8>> {
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
pub(crate) fn build_reads_table(
    reads: &[(ReadData, Vec<u64>)],
    run_infos: &[RunInfoData],
) -> Result<Vec<u8>> {
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

    // Collect unique pore types and end reasons for dictionaries using O(1) HashSet lookups
    let mut pore_type_set: HashSet<&str> = HashSet::new();
    let mut end_reason_set: HashSet<&str> = HashSet::new();

    for (read, _) in reads {
        pore_type_set.insert(&read.pore_type);
        end_reason_set.insert(read.end_reason.as_str());
    }

    // Convert to Vec for Arrow dictionary (order doesn't matter for correctness)
    let pore_types: Vec<&str> = pore_type_set.into_iter().collect();
    let end_reasons: Vec<&str> = end_reason_set.into_iter().collect();

    let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(num_reads, 16);
    let signal_field = Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::UInt64,
        false,
    ));
    let mut signal_builder = ListBuilder::new(UInt64Builder::new()).with_field(signal_field);
    let mut channel_builder = UInt16Builder::with_capacity(num_reads);
    let mut well_builder = UInt8Builder::with_capacity(num_reads);

    let pore_type_dict = StringArray::from_iter_values(pore_types.iter().copied());
    let mut pore_type_builder: StringDictionaryBuilder<Int16Type> =
        StringDictionaryBuilder::new_with_dictionary(num_reads, &pore_type_dict)?;

    let mut calibration_offset_builder = Float32Builder::with_capacity(num_reads);
    let mut calibration_scale_builder = Float32Builder::with_capacity(num_reads);
    let mut read_number_builder = UInt32Builder::with_capacity(num_reads);
    let mut start_builder = UInt64Builder::with_capacity(num_reads);
    let mut median_before_builder = Float32Builder::with_capacity(num_reads);

    let end_reason_dict = StringArray::from_iter_values(end_reasons.iter().copied());
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
pub(crate) fn build_pod5_footer(
    signal_offset: i64,
    signal_length: i64,
    run_info_offset: i64,
    run_info_length: i64,
    reads_offset: i64,
    reads_length: i64,
) -> Result<Vec<u8>> {
    let file_id = Uuid::new_v4().to_string();
    let software = format!("escapepod-rs {}", env!("CARGO_PKG_VERSION"));
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

fn write_flatbuffer_string(data: &mut Cursor<Vec<u8>>, s: &str) -> Result<usize> {
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

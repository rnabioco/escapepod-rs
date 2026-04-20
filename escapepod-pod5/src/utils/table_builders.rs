//! Table building utilities for POD5 file construction.
//!
//! Shared functions for building Arrow IPC tables and POD5 footers,
//! used by both merge and filter operations.

use crate::arrow_ipc::BatchBlock;
use crate::error::Result;
use crate::schema::{reads_schema, run_info_schema};
use crate::types::{POD5_VERSION, ReadData, RunInfoData, Uuid};
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, DictionaryArray, FixedSizeBinaryArray,
    FixedSizeBinaryBuilder, Float32Array, Float32Builder, Int16Array, Int16Builder, ListArray,
    ListBuilder, MapBuilder, MapFieldNames, StringArray, StringBuilder,
    TimestampMillisecondBuilder, UInt8Array, UInt8Builder, UInt16Array, UInt16Builder, UInt32Array,
    UInt32Builder, UInt64Array, UInt64Builder,
};
use arrow::compute::concat;
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use arrow::ipc::{Block, MetadataVersion};
use arrow::record_batch::RecordBatch;
use flatbuffers::FlatBufferBuilder;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use std::sync::Arc;

/// Metadata applied to Arrow IPC schemas for POD5 compatibility.
///
/// The C++ POD5 reader requires `MINKNOW:file_identifier`, `MINKNOW:software`,
/// and `MINKNOW:pod5_version` keys in each embedded Arrow table's schema metadata.
pub(crate) struct SchemaMetadata {
    pub file_identifier: String,
    pub software: String,
    pub pod5_version: String,
}

impl SchemaMetadata {
    /// Create metadata with a new random file identifier.
    pub fn new() -> Self {
        Self {
            file_identifier: Uuid::new_v4().to_string(),
            software: format!("escapepod-rs {}", env!("CARGO_PKG_VERSION")),
            pod5_version: POD5_VERSION.to_string(),
        }
    }

    /// Apply this metadata to an Arrow schema.
    pub fn apply(&self, schema: arrow::datatypes::Schema) -> arrow::datatypes::Schema {
        let mut metadata = schema.metadata().clone();
        metadata.insert(
            "MINKNOW:file_identifier".to_string(),
            self.file_identifier.clone(),
        );
        metadata.insert("MINKNOW:software".to_string(), self.software.clone());
        metadata.insert(
            "MINKNOW:pod5_version".to_string(),
            self.pod5_version.clone(),
        );
        schema.with_metadata(metadata)
    }
}

/// Build Arrow IPC footer from batch blocks.
///
/// The footer's `schema` field is what Arrow's `ArrowFileReader` trusts when
/// it decodes batches — an empty schema here makes every batch come back with
/// zero columns, which silently breaks any consumer that looks up columns by
/// name (e.g. `Reader::get_signal`, which calls `column_by_name("signal")`).
/// Pass the real Arrow schema of the batches so the footer matches the
/// schema message embedded in the file header.
pub(crate) fn build_arrow_ipc_footer(
    batches: &[BatchBlock],
    schema: &arrow::datatypes::Schema,
) -> Result<Vec<u8>> {
    let mut fbb = FlatBufferBuilder::with_capacity(256 + batches.len() * 24);

    let blocks: Vec<Block> = batches
        .iter()
        .map(|b| Block::new(b.offset, b.metadata_length, b.body_length))
        .collect();

    let record_batches = fbb.create_vector(&blocks);

    let schema_offset = arrow::ipc::convert::schema_to_fb_offset(&mut fbb, schema);

    let footer = arrow::ipc::Footer::create(
        &mut fbb,
        &arrow::ipc::FooterArgs {
            version: MetadataVersion::V5,
            schema: Some(schema_offset),
            dictionaries: None,
            recordBatches: Some(record_batches),
            custom_metadata: None,
        },
    );

    fbb.finish(footer, None);

    Ok(fbb.finished_data().to_vec())
}

/// Build run_info Arrow IPC table.
pub(crate) fn build_run_info_table(
    run_infos: &[RunInfoData],
    meta: &SchemaMetadata,
) -> Result<Vec<u8>> {
    let schema = Arc::new(meta.apply(run_info_schema()));

    if run_infos.is_empty() {
        let mut buffer = Vec::new();
        {
            let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
            writer.finish()?;
        }
        return Ok(buffer);
    }

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
    let mut protocol_start_time_builder = TimestampMillisecondBuilder::new().with_timezone("UTC");
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

/// Arrays built for a single partition of reads.
/// Dictionary columns are stored as keys only (Int16Array) to allow
/// concatenation with a shared dictionary.
struct PartitionArrays {
    read_id: FixedSizeBinaryArray,
    signal: ListArray,
    read_number: UInt32Array,
    start: UInt64Array,
    median_before: Float32Array,
    num_minknow_events: UInt64Array,
    tracked_scaling_scale: Float32Array,
    tracked_scaling_shift: Float32Array,
    predicted_scaling_scale: Float32Array,
    predicted_scaling_shift: Float32Array,
    num_reads_since_mux_change: UInt32Array,
    time_since_mux_change: Float32Array,
    num_samples: UInt64Array,
    channel: UInt16Array,
    well: UInt8Array,
    pore_type_keys: Int16Array,
    calibration_offset: Float32Array,
    calibration_scale: Float32Array,
    end_reason_keys: Int16Array,
    end_reason_forced: BooleanArray,
    run_info_keys: Int16Array,
    open_pore_level: Float32Array,
}

/// Build arrays for a single partition of reads.
fn build_partition(
    reads: &[(ReadData, Vec<u64>)],
    pore_type_map: &HashMap<&str, i16>,
    end_reason_map: &HashMap<&str, i16>,
    run_info_map: &HashMap<&str, i16>,
    run_infos: &[RunInfoData],
) -> PartitionArrays {
    let num_reads = reads.len();

    let mut read_id_builder = FixedSizeBinaryBuilder::with_capacity(num_reads, 16);
    let signal_field = Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::UInt64,
        true,
    ));
    let mut signal_builder = ListBuilder::new(UInt64Builder::new()).with_field(signal_field);
    let mut read_number_builder = UInt32Builder::with_capacity(num_reads);
    let mut start_builder = UInt64Builder::with_capacity(num_reads);
    let mut median_before_builder = Float32Builder::with_capacity(num_reads);
    let mut num_minknow_events_builder = UInt64Builder::with_capacity(num_reads);
    let mut tracked_scaling_scale_builder = Float32Builder::with_capacity(num_reads);
    let mut tracked_scaling_shift_builder = Float32Builder::with_capacity(num_reads);
    let mut predicted_scaling_scale_builder = Float32Builder::with_capacity(num_reads);
    let mut predicted_scaling_shift_builder = Float32Builder::with_capacity(num_reads);
    let mut num_reads_since_mux_change_builder = UInt32Builder::with_capacity(num_reads);
    let mut time_since_mux_change_builder = Float32Builder::with_capacity(num_reads);
    let mut num_samples_builder = UInt64Builder::with_capacity(num_reads);
    let mut channel_builder = UInt16Builder::with_capacity(num_reads);
    let mut well_builder = UInt8Builder::with_capacity(num_reads);
    let mut pore_type_keys_builder = Int16Builder::with_capacity(num_reads);
    let mut calibration_offset_builder = Float32Builder::with_capacity(num_reads);
    let mut calibration_scale_builder = Float32Builder::with_capacity(num_reads);
    let mut end_reason_keys_builder = Int16Builder::with_capacity(num_reads);
    let mut end_reason_forced_builder = BooleanBuilder::with_capacity(num_reads);
    let mut run_info_keys_builder = Int16Builder::with_capacity(num_reads);
    let mut open_pore_level_builder = Float32Builder::with_capacity(num_reads);

    for (read, signal_rows) in reads {
        // read_id is infallible for 16-byte slices
        let _ = read_id_builder.append_value(read.read_id.as_bytes());

        let values = signal_builder.values();
        for &idx in signal_rows {
            values.append_value(idx);
        }
        signal_builder.append(true);

        // V0 fields
        read_number_builder.append_value(read.read_number);
        start_builder.append_value(read.start_sample);
        median_before_builder.append_value(read.median_before);

        // V1 fields
        num_minknow_events_builder.append_value(read.num_minknow_events);
        tracked_scaling_scale_builder.append_value(read.tracked_scaling_scale);
        tracked_scaling_shift_builder.append_value(read.tracked_scaling_shift);
        predicted_scaling_scale_builder.append_value(read.predicted_scaling_scale);
        predicted_scaling_shift_builder.append_value(read.predicted_scaling_shift);
        num_reads_since_mux_change_builder.append_value(read.num_reads_since_mux_change);
        time_since_mux_change_builder.append_value(read.time_since_mux_change);

        // V2 fields
        num_samples_builder.append_value(read.num_samples);

        // V3 fields
        channel_builder.append_value(read.channel);
        well_builder.append_value(read.well);
        let pore_key = pore_type_map
            .get(read.pore_type.as_str())
            .copied()
            .unwrap_or(0);
        pore_type_keys_builder.append_value(pore_key);
        calibration_offset_builder.append_value(read.calibration_offset);
        calibration_scale_builder.append_value(read.calibration_scale);
        let end_key = end_reason_map
            .get(read.end_reason.as_str())
            .copied()
            .unwrap_or(0);
        end_reason_keys_builder.append_value(end_key);
        end_reason_forced_builder.append_value(read.end_reason_forced);
        let run_info_key = run_infos
            .get(read.run_info_index as usize)
            .and_then(|ri| run_info_map.get(ri.acquisition_id.as_str()).copied())
            .unwrap_or(0);
        run_info_keys_builder.append_value(run_info_key);

        // V4 fields
        open_pore_level_builder.append_value(read.open_pore_level);
    }

    PartitionArrays {
        read_id: read_id_builder.finish(),
        signal: signal_builder.finish(),
        read_number: read_number_builder.finish(),
        start: start_builder.finish(),
        median_before: median_before_builder.finish(),
        num_minknow_events: num_minknow_events_builder.finish(),
        tracked_scaling_scale: tracked_scaling_scale_builder.finish(),
        tracked_scaling_shift: tracked_scaling_shift_builder.finish(),
        predicted_scaling_scale: predicted_scaling_scale_builder.finish(),
        predicted_scaling_shift: predicted_scaling_shift_builder.finish(),
        num_reads_since_mux_change: num_reads_since_mux_change_builder.finish(),
        time_since_mux_change: time_since_mux_change_builder.finish(),
        num_samples: num_samples_builder.finish(),
        channel: channel_builder.finish(),
        well: well_builder.finish(),
        pore_type_keys: pore_type_keys_builder.finish(),
        calibration_offset: calibration_offset_builder.finish(),
        calibration_scale: calibration_scale_builder.finish(),
        end_reason_keys: end_reason_keys_builder.finish(),
        end_reason_forced: end_reason_forced_builder.finish(),
        run_info_keys: run_info_keys_builder.finish(),
        open_pore_level: open_pore_level_builder.finish(),
    }
}

/// Build reads Arrow IPC table.
///
/// Uses parallel partition-based building for performance:
/// 1. Parallel dictionary collection (unique pore types, end reasons)
/// 2. Create O(1) lookup maps for dictionary keys
/// 3. Parallel partition building (split reads, build arrays per partition)
/// 4. Concatenate partition arrays and create final RecordBatch
pub(crate) fn build_reads_table(
    reads: &[(ReadData, Vec<u64>)],
    run_infos: &[RunInfoData],
    meta: &SchemaMetadata,
) -> Result<Vec<u8>> {
    let schema = Arc::new(meta.apply(reads_schema()));

    if reads.is_empty() {
        let mut buffer = Vec::new();
        {
            let mut writer = ArrowFileWriter::try_new(&mut buffer, &schema)?;
            writer.finish()?;
        }
        return Ok(buffer);
    }

    // Phase 1: Parallel dictionary collection
    let (pore_type_set, end_reason_set): (HashSet<&str>, HashSet<&str>) = reads
        .par_iter()
        .fold(
            || (HashSet::new(), HashSet::new()),
            |(mut pores, mut ends), (read, _)| {
                pores.insert(read.pore_type.as_str());
                ends.insert(read.end_reason.as_str());
                (pores, ends)
            },
        )
        .reduce(
            || (HashSet::new(), HashSet::new()),
            |(mut a_pores, mut a_ends), (b_pores, b_ends)| {
                a_pores.extend(b_pores);
                a_ends.extend(b_ends);
                (a_pores, a_ends)
            },
        );

    let pore_types: Vec<&str> = pore_type_set.into_iter().collect();
    let end_reasons: Vec<&str> = end_reason_set.into_iter().collect();
    let run_info_ids: Vec<&str> = run_infos
        .iter()
        .map(|ri| ri.acquisition_id.as_str())
        .collect();

    // Phase 2: Create O(1) lookup maps for dictionary keys
    let pore_type_map: HashMap<&str, i16> = pore_types
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as i16))
        .collect();
    let end_reason_map: HashMap<&str, i16> = end_reasons
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as i16))
        .collect();
    let run_info_map: HashMap<&str, i16> = run_info_ids
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as i16))
        .collect();

    // Phase 3: Parallel partition building
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = reads.len().div_ceil(num_threads);

    let partition_arrays: Vec<PartitionArrays> = reads
        .par_chunks(chunk_size)
        .map(|chunk| {
            build_partition(
                chunk,
                &pore_type_map,
                &end_reason_map,
                &run_info_map,
                run_infos,
            )
        })
        .collect();

    // Phase 4: Concatenate partition arrays and create DictionaryArrays

    // Helper to concatenate arrays of a specific type
    macro_rules! concat_arrays {
        ($field:ident, $array_type:ty) => {{
            let refs: Vec<&dyn Array> = partition_arrays
                .iter()
                .map(|p| &p.$field as &dyn Array)
                .collect();
            Arc::new(
                concat(&refs)?
                    .as_any()
                    .downcast_ref::<$array_type>()
                    .unwrap()
                    .clone(),
            ) as ArrayRef
        }};
    }

    // Concatenate primitive arrays (in schema order)
    let read_id_array = concat_arrays!(read_id, FixedSizeBinaryArray);
    let signal_array = concat_arrays!(signal, ListArray);
    let read_number_array = concat_arrays!(read_number, UInt32Array);
    let start_array = concat_arrays!(start, UInt64Array);
    let median_before_array = concat_arrays!(median_before, Float32Array);
    let num_minknow_events_array = concat_arrays!(num_minknow_events, UInt64Array);
    let tracked_scaling_scale_array = concat_arrays!(tracked_scaling_scale, Float32Array);
    let tracked_scaling_shift_array = concat_arrays!(tracked_scaling_shift, Float32Array);
    let predicted_scaling_scale_array = concat_arrays!(predicted_scaling_scale, Float32Array);
    let predicted_scaling_shift_array = concat_arrays!(predicted_scaling_shift, Float32Array);
    let num_reads_since_mux_change_array = concat_arrays!(num_reads_since_mux_change, UInt32Array);
    let time_since_mux_change_array = concat_arrays!(time_since_mux_change, Float32Array);
    let num_samples_array = concat_arrays!(num_samples, UInt64Array);
    let channel_array = concat_arrays!(channel, UInt16Array);
    let well_array = concat_arrays!(well, UInt8Array);
    let calibration_offset_array = concat_arrays!(calibration_offset, Float32Array);
    let calibration_scale_array = concat_arrays!(calibration_scale, Float32Array);
    let end_reason_forced_array = concat_arrays!(end_reason_forced, BooleanArray);
    let open_pore_level_array = concat_arrays!(open_pore_level, Float32Array);

    // Concatenate dictionary key arrays and create DictionaryArrays
    let pore_type_keys_refs: Vec<&dyn Array> = partition_arrays
        .iter()
        .map(|p| &p.pore_type_keys as &dyn Array)
        .collect();
    let pore_type_keys = concat(&pore_type_keys_refs)?
        .as_any()
        .downcast_ref::<Int16Array>()
        .unwrap()
        .clone();
    let pore_type_dict = StringArray::from_iter_values(pore_types.iter().copied());
    let pore_type_array: ArrayRef = Arc::new(DictionaryArray::new(
        pore_type_keys,
        Arc::new(pore_type_dict),
    ));

    let end_reason_keys_refs: Vec<&dyn Array> = partition_arrays
        .iter()
        .map(|p| &p.end_reason_keys as &dyn Array)
        .collect();
    let end_reason_keys = concat(&end_reason_keys_refs)?
        .as_any()
        .downcast_ref::<Int16Array>()
        .unwrap()
        .clone();
    let end_reason_dict = StringArray::from_iter_values(end_reasons.iter().copied());
    let end_reason_array: ArrayRef = Arc::new(DictionaryArray::new(
        end_reason_keys,
        Arc::new(end_reason_dict),
    ));

    let run_info_keys_refs: Vec<&dyn Array> = partition_arrays
        .iter()
        .map(|p| &p.run_info_keys as &dyn Array)
        .collect();
    let run_info_keys = concat(&run_info_keys_refs)?
        .as_any()
        .downcast_ref::<Int16Array>()
        .unwrap()
        .clone();
    let run_info_dict = StringArray::from_iter_values(run_info_ids.iter().copied());
    let run_info_array: ArrayRef =
        Arc::new(DictionaryArray::new(run_info_keys, Arc::new(run_info_dict)));

    let arrays: Vec<ArrayRef> = vec![
        // V0
        read_id_array,
        signal_array,
        read_number_array,
        start_array,
        median_before_array,
        // V1
        num_minknow_events_array,
        tracked_scaling_scale_array,
        tracked_scaling_shift_array,
        predicted_scaling_scale_array,
        predicted_scaling_shift_array,
        num_reads_since_mux_change_array,
        time_since_mux_change_array,
        // V2
        num_samples_array,
        // V3
        channel_array,
        well_array,
        pore_type_array,
        calibration_offset_array,
        calibration_scale_array,
        end_reason_array,
        end_reason_forced_array,
        run_info_array,
        // V4
        open_pore_level_array,
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

/// Build POD5 FlatBuffer footer using the generated FlatBuffer types.
pub(crate) fn build_pod5_footer(
    signal_offset: i64,
    signal_length: i64,
    run_info_offset: i64,
    run_info_length: i64,
    reads_offset: i64,
    reads_length: i64,
    meta: &SchemaMetadata,
) -> Result<Vec<u8>> {
    use crate::flatbuffers_gen::{
        ContentType, EmbeddedFile, EmbeddedFileArgs, Footer, FooterArgs, Format,
    };

    let file_id = &meta.file_identifier;
    let software = &meta.software;
    let version = &meta.pod5_version;

    let mut fbb = flatbuffers::FlatBufferBuilder::with_capacity(256);

    // Create embedded file entries
    let signal_entry = EmbeddedFile::create(
        &mut fbb,
        &EmbeddedFileArgs {
            offset: signal_offset,
            length: signal_length,
            format: Format::FeatherV2,
            content_type: ContentType::SignalTable,
        },
    );
    let run_info_entry = EmbeddedFile::create(
        &mut fbb,
        &EmbeddedFileArgs {
            offset: run_info_offset,
            length: run_info_length,
            format: Format::FeatherV2,
            content_type: ContentType::RunInfoTable,
        },
    );
    let reads_entry = EmbeddedFile::create(
        &mut fbb,
        &EmbeddedFileArgs {
            offset: reads_offset,
            length: reads_length,
            format: Format::FeatherV2,
            content_type: ContentType::ReadsTable,
        },
    );

    let contents = fbb.create_vector(&[signal_entry, run_info_entry, reads_entry]);

    let file_id_str = fbb.create_string(file_id);
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

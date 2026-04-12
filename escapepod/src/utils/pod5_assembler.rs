//! Shared POD5 file assembly for post-signal sections.
//!
//! Both `filter` and `merge` operations write signal data differently
//! (filter builds from extracted chunks, merge streams from source files),
//! but they share identical logic for everything that comes after:
//! padding, run info dedup, reads table building, and footer writing.

use crate::error::Result;
use crate::types::{FOOTER_MAGIC, POD5_SIGNATURE, ReadData, RunInfoData, Uuid};
use crate::utils::table_builders::{
    SchemaMetadata, build_pod5_footer, build_reads_table, build_run_info_table,
};
use std::collections::HashMap;
use std::io::{Seek, Write};

/// Metadata collected per source file, used to deduplicate run infos
/// and remap `run_info_index` fields across files.
pub(crate) struct SourceFileMetadata {
    pub run_infos: Vec<RunInfoData>,
}

/// Writes the shared post-signal sections of a POD5 file.
///
/// Call this after all signal data has been written. It handles:
/// - Padding to 8-byte alignment + section marker after signal
/// - Deduplicated run info table
/// - Reads table with remapped signal rows and run_info indices
/// - POD5 footer (flatbuffer + length + closing signature)
pub(crate) fn write_post_signal_sections<W: Write + Seek>(
    file: &mut W,
    section_marker: &Uuid,
    schema_meta: &SchemaMetadata,
    signal_end: usize,
    file_metadata: &[SourceFileMetadata],
    processed_reads: &[(ReadData, Vec<u64>)],
) -> Result<()> {
    // Pad signal section to 8-byte alignment
    let padding_needed = (8 - (signal_end % 8)) % 8;
    for _ in 0..padding_needed {
        file.write_all(&[0u8])?;
    }
    file.write_all(section_marker.as_bytes())?;

    // Build deduplicated run info table
    let (all_run_infos, _run_info_map) = deduplicate_run_infos(file_metadata);

    let run_info_offset = file.stream_position()? as i64;
    let run_info_bytes = build_run_info_table(&all_run_infos, schema_meta)?;
    file.write_all(&run_info_bytes)?;
    let run_info_length = run_info_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    file.write_all(section_marker.as_bytes())?;

    // Write reads table
    let reads_offset = file.stream_position()? as i64;
    let reads_bytes = build_reads_table(processed_reads, &all_run_infos, schema_meta)?;
    file.write_all(&reads_bytes)?;
    let reads_length = reads_bytes.len() as i64;

    // Pad and section marker
    while file.stream_position()? % 8 != 0 {
        file.write_all(&[0u8])?;
    }
    file.write_all(section_marker.as_bytes())?;

    // Write POD5 footer
    file.write_all(&FOOTER_MAGIC)?;

    let signal_offset_val = 24i64; // POD5 header size (signature + section marker)
    let signal_length = signal_end as i64 - 24;

    let pod5_footer = build_pod5_footer(
        signal_offset_val,
        signal_length,
        run_info_offset,
        run_info_length,
        reads_offset,
        reads_length,
        schema_meta,
    )?;
    file.write_all(&pod5_footer)?;

    let footer_len = pod5_footer.len() as i64;
    file.write_all(&footer_len.to_le_bytes())?;

    file.write_all(section_marker.as_bytes())?;
    file.write_all(&POD5_SIGNATURE)?;

    file.flush()?;

    Ok(())
}

/// Deduplicate run infos from multiple source files by `acquisition_id`.
///
/// Returns the deduplicated list and a map from `acquisition_id` to index.
pub(crate) fn deduplicate_run_infos(
    file_metadata: &[SourceFileMetadata],
) -> (Vec<RunInfoData>, HashMap<String, u32>) {
    let mut run_info_map: HashMap<String, u32> = HashMap::new();
    let mut all_run_infos: Vec<RunInfoData> = Vec::new();

    for metadata in file_metadata {
        for run_info in &metadata.run_infos {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = all_run_infos.len() as u32;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
                all_run_infos.push(run_info.clone());
            }
        }
    }

    (all_run_infos, run_info_map)
}

/// Remap a read's `run_info_index` using the dedup map, and compute new sequential
/// signal rows starting from `signal_row_cursor`.
///
/// Returns `(new_read, new_signal_rows, updated_cursor)`.
pub(crate) fn remap_read(
    read: &ReadData,
    source_run_infos: &[RunInfoData],
    run_info_map: &HashMap<String, u32>,
    signal_row_cursor: u64,
) -> (ReadData, Vec<u64>, u64) {
    let original_run_info = source_run_infos.get(read.run_info_index as usize);
    let new_run_info_idx = if let Some(ri) = original_run_info {
        *run_info_map.get(&ri.acquisition_id).unwrap_or(&0)
    } else {
        0
    };

    let num_signal_rows = read.signal_rows.len() as u64;
    let new_signal_rows: Vec<u64> =
        (signal_row_cursor..signal_row_cursor + num_signal_rows).collect();

    let new_read = read.for_writing(new_run_info_idx);
    (
        new_read,
        new_signal_rows,
        signal_row_cursor + num_signal_rows,
    )
}

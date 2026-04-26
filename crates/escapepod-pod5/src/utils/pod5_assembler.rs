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
use std::io::Write;

/// A read paired with the signal-row indices it should reference in the
/// output reads table. Used by the writers in merge and filter.
pub(crate) type ProcessedRead = (ReadData, Vec<u64>);

/// Up to 7 zero bytes for 8-byte alignment padding.
const PADDING_ZEROS: [u8; 8] = [0u8; 8];

/// Writes the shared post-signal sections of a POD5 file.
///
/// Call this after all signal data has been written. It handles:
/// - Padding to 8-byte alignment + section marker after signal
/// - Run info table (caller already deduplicated)
/// - Reads table with remapped signal rows and run_info indices
/// - POD5 footer (flatbuffer + length + closing signature)
///
/// `signal_end` is the absolute byte offset of the end of the signal section
/// (i.e. how many bytes have been written so far). The function tracks the
/// position internally rather than calling `stream_position()`, which would
/// force the underlying `BufWriter` to flush.
pub(crate) fn write_post_signal_sections<W: Write>(
    file: &mut W,
    section_marker: &Uuid,
    schema_meta: &SchemaMetadata,
    signal_end: usize,
    all_run_infos: &[RunInfoData],
    processed_reads: &[ProcessedRead],
) -> Result<()> {
    let mut pos = signal_end;

    // Pad signal section to 8-byte alignment, then section marker.
    pos += write_padding_to_align8(file, pos)?;
    file.write_all(section_marker.as_bytes())?;
    pos += 16;

    // Run info table.
    let run_info_offset = pos as i64;
    let run_info_bytes = build_run_info_table(all_run_infos, schema_meta)?;
    file.write_all(&run_info_bytes)?;
    pos += run_info_bytes.len();
    let run_info_length = run_info_bytes.len() as i64;

    pos += write_padding_to_align8(file, pos)?;
    file.write_all(section_marker.as_bytes())?;
    pos += 16;

    // Reads table.
    let reads_offset = pos as i64;
    let reads_bytes = build_reads_table(processed_reads, all_run_infos, schema_meta)?;
    file.write_all(&reads_bytes)?;
    pos += reads_bytes.len();
    let reads_length = reads_bytes.len() as i64;

    let _ = write_padding_to_align8(file, pos)?;
    file.write_all(section_marker.as_bytes())?;
    // pos no longer needed after this point.

    // POD5 footer.
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

/// Write zero bytes to reach the next 8-byte alignment boundary.
/// Returns the number of padding bytes written (0..=7).
fn write_padding_to_align8<W: Write>(file: &mut W, pos: usize) -> Result<usize> {
    let padding = (8 - (pos % 8)) % 8;
    if padding > 0 {
        file.write_all(&PADDING_ZEROS[..padding])?;
    }
    Ok(padding)
}

/// Deduplicate run infos from multiple source files by `acquisition_id`.
///
/// Returns the deduplicated list and a map from `acquisition_id` to index.
/// Each input slice is the run-info table of a single source file. We borrow
/// rather than own to avoid deep-cloning every `RunInfoData` (which carries
/// two `HashMap<String,String>` plus 13 Strings).
pub(crate) fn deduplicate_run_infos(
    per_file_run_infos: &[&[RunInfoData]],
) -> (Vec<RunInfoData>, HashMap<String, u32>) {
    let mut run_info_map: HashMap<String, u32> = HashMap::new();
    let mut all_run_infos: Vec<RunInfoData> = Vec::new();

    for run_infos in per_file_run_infos {
        for run_info in run_infos.iter() {
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

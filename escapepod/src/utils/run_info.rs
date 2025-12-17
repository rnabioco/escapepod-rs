//! Run info handling utilities.

use crate::{Reader, Result, Writer};
use std::collections::HashMap;

/// Add run infos from a reader to a writer, deduplicating by acquisition_id.
///
/// This function iterates through run infos in the reader and adds them to the writer,
/// but only if an entry with the same acquisition_id hasn't already been added.
///
/// Returns the mapping from acquisition_id to writer index via the provided `run_info_map`.
///
/// # Arguments
///
/// * `reader` - The POD5 reader to get run infos from
/// * `writer` - The POD5 writer to add run infos to
/// * `run_info_map` - A mutable map that tracks acquisition_id -> writer index
///
/// # Example
///
/// ```no_run
/// use escapepod::{Reader, Writer, WriterOptions};
/// use escapepod::utils::add_run_infos_deduplicated;
/// use std::collections::HashMap;
///
/// let reader = Reader::open("input.pod5")?;
/// let mut writer = Writer::create("output.pod5", WriterOptions::default())?;
/// let mut run_info_map = HashMap::new();
///
/// add_run_infos_deduplicated(&reader, &mut writer, &mut run_info_map)?;
/// # Ok::<(), escapepod::Error>(())
/// ```
pub fn add_run_infos_deduplicated(
    reader: &Reader,
    writer: &mut Writer,
    run_info_map: &mut HashMap<String, u32>,
) -> Result<()> {
    for run_info in reader.run_infos() {
        if !run_info_map.contains_key(&run_info.acquisition_id) {
            let idx = writer.add_run_info(run_info.clone())?;
            run_info_map.insert(run_info.acquisition_id.clone(), idx);
        }
    }
    Ok(())
}

/// Get the new run_info index for a read, using the run_info_map.
///
/// This maps from the original run_info index in the reader to the new index
/// in the writer, using the acquisition_id as the key.
///
/// # Arguments
///
/// * `reader` - The POD5 reader containing the original run info
/// * `read_run_info_index` - The run_info index from the read being processed
/// * `run_info_map` - Map from acquisition_id to new writer index
///
/// # Returns
///
/// The new run_info index to use in the writer, or 0 if not found.
pub fn map_run_info_index(
    reader: &Reader,
    read_run_info_index: u32,
    run_info_map: &HashMap<String, u32>,
) -> u32 {
    if let Some(original_run_info) = reader.get_run_info(read_run_info_index as usize) {
        *run_info_map
            .get(&original_run_info.acquisition_id)
            .unwrap_or(&0)
    } else {
        0
    }
}

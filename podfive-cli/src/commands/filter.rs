//! Filter command implementation.
//!
//! Filters reads from a POD5 file based on a list of read IDs.

use podfive_core::{ReadData, Reader, Uuid, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub fn run(input: PathBuf, ids_file: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    println!(
        "Filtering {} using IDs from {}",
        input.display(),
        ids_file.display()
    );
    println!("Output: {}", output.display());

    // Read IDs from file
    let ids = read_ids_from_file(&ids_file)?;
    println!("Loaded {} read IDs to filter", ids.len());

    if ids.is_empty() {
        anyhow::bail!("No read IDs found in {}", ids_file.display());
    }

    // Open input file
    let reader = Reader::open(&input)?;

    // Create writer
    let options = WriterOptions::default();
    let mut writer = Writer::create(&output, options)?;

    // Copy run infos
    let mut run_info_map: HashMap<String, u32> = HashMap::new();
    for run_info in reader.run_infos() {
        let idx = writer.add_run_info(run_info.clone())?;
        run_info_map.insert(run_info.acquisition_id.clone(), idx);
    }

    // Filter reads
    let mut matched = 0u64;
    let mut total = 0u64;

    for read_result in reader.reads()? {
        let read = read_result?;
        total += 1;

        // Check if this read's ID is in the filter list
        if ids.contains(&read.read_id) {
            // Get signal for this read
            let signal = reader.get_signal(&read.signal_rows)?;

            // Map run_info index
            let new_run_info_idx = if let Some(original_run_info) =
                reader.get_run_info(read.run_info_index as usize)
            {
                *run_info_map
                    .get(&original_run_info.acquisition_id)
                    .unwrap_or(&0)
            } else {
                0
            };

            // Create new read data
            let new_read = ReadData {
                read_id: read.read_id,
                read_number: read.read_number,
                start_sample: read.start_sample,
                channel: read.channel,
                well: read.well,
                pore_type: read.pore_type.clone(),
                calibration_offset: read.calibration_offset,
                calibration_scale: read.calibration_scale,
                median_before: read.median_before,
                end_reason: read.end_reason,
                end_reason_forced: read.end_reason_forced,
                run_info_index: new_run_info_idx,
                num_minknow_events: read.num_minknow_events,
                num_samples: read.num_samples,
                open_pore_level: read.open_pore_level,
                signal_rows: Vec::new(),
            };

            writer.add_read(new_read, &signal)?;
            matched += 1;
        }
    }

    // Finalize output
    writer.finish()?;

    println!(
        "Filtered {} reads from {} total ({:.1}%)",
        matched,
        total,
        (matched as f64 / total as f64) * 100.0
    );

    let not_found = ids.len() as u64 - matched;
    if not_found > 0 {
        println!(
            "Warning: {} requested IDs were not found in the input file",
            not_found
        );
    }

    Ok(())
}

/// Read read IDs from a text file (one per line).
///
/// Supports UUIDs in various formats:
/// - Standard: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
/// - No dashes: `a1b2c3d4e5f67890abcdef1234567890`
fn read_ids_from_file(path: &PathBuf) -> anyhow::Result<HashSet<Uuid>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut ids = HashSet::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Try to parse as UUID
        match Uuid::parse_str(line) {
            Ok(uuid) => {
                ids.insert(uuid);
            }
            Err(e) => {
                anyhow::bail!("Invalid UUID on line {}: '{}' ({})", line_num + 1, line, e);
            }
        }
    }

    Ok(ids)
}

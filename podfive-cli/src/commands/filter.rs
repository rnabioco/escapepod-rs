//! Filter command implementation.
//!
//! Filters reads from a POD5 file based on a list of read IDs.
//! Uses lazy signal loading and block-level copying for maximum performance.

use crate::util::{parse_uuid_flexible, resolve_pod5_inputs};
use podfive_core::{Reader, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use uuid::Uuid;

pub fn run(input: PathBuf, ids_file: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    // Resolve input to list of POD5 files (supports directories)
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!(
        "Filtering {} using IDs from {}",
        if is_directory {
            format!("{} ({} files)", input.display(), files.len())
        } else {
            input.display().to_string()
        },
        ids_file.display()
    );
    println!("Output: {}", output.display());

    // Read IDs from file
    let ids = read_ids_from_file(&ids_file)?;
    println!("Loaded {} read IDs to filter", ids.len());

    if ids.is_empty() {
        anyhow::bail!("No read IDs found in {}", ids_file.display());
    }

    // Create writer with optimized batch sizes
    // Signal batches can be smaller for frequent direct file writes
    // Read batch must hold all reads to avoid dictionary replacement in Arrow IPC
    let mut options = WriterOptions::default();
    options.signal_batch_size = 1_000;
    options.read_batch_size = 500_000;
    let mut writer = Writer::create(&output, options)?;

    // Track run infos across all files
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Filter reads from all files
    let mut matched = 0u64;
    let mut total = 0u64;
    let mut read_errors = 0u64;
    let mut signal_errors = 0u64;

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(e) => {
                if is_directory {
                    eprintln!(
                        "Warning: skipping {} ({})",
                        file_path.file_name().unwrap_or_default().to_string_lossy(),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        // Add run infos (deduplicated by acquisition_id)
        for run_info in reader.run_infos() {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = writer.add_run_info(run_info.clone())?;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
            }
        }

        // NOTE: Signal is loaded lazily per-read, not all upfront
        // This is much more efficient when filtering a small subset of reads

        let reads_iter = match reader.reads() {
            Ok(iter) => iter,
            Err(e) => {
                if is_directory {
                    eprintln!(
                        "Warning: cannot read {} ({})",
                        file_path.file_name().unwrap_or_default().to_string_lossy(),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        for read_result in reads_iter {
            let read = match read_result {
                Ok(r) => r,
                Err(e) => {
                    read_errors += 1;
                    if read_errors <= 3 {
                        eprintln!("Warning: error reading read record: {}", e);
                    }
                    continue;
                }
            };
            total += 1;

            // Check if this read's ID is in the filter list
            if ids.contains(&read.read_id) {
                // Lazy load: only fetch signal for matching reads (O(1) batch lookup + LRU cache)
                let compressed_signal = match reader.get_compressed_signal_for_rows(&read.signal_rows)
                {
                    Ok(s) => s,
                    Err(e) => {
                        signal_errors += 1;
                        if signal_errors <= 3 {
                            eprintln!(
                                "Warning: cannot read signal for read {}: {}",
                                read.read_id, e
                            );
                        }
                        continue;
                    }
                };

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

                // Create new read data for writing
                let new_read = read.for_writing(new_run_info_idx);

                writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;
                matched += 1;
            }
        }
    }

    // Finalize output
    writer.finish()?;

    println!(
        "Filtered {} reads from {} total ({:.1}%)",
        matched,
        total,
        if total > 0 {
            (matched as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    );

    let not_found = (ids.len() as u64).saturating_sub(matched);
    if not_found > 0 {
        println!(
            "Warning: {} requested IDs were not found in the input",
            not_found
        );
    }
    if matched > ids.len() as u64 {
        println!(
            "Note: {} duplicate reads matched across multiple files",
            matched - ids.len() as u64
        );
    }

    // Report any errors encountered
    if read_errors > 0 || signal_errors > 0 {
        eprintln!(
            "Warning: encountered {} read error(s) and {} signal error(s)",
            read_errors, signal_errors
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

        // Try to parse as UUID (supports both standard and compact formats)
        match parse_uuid_flexible(line) {
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

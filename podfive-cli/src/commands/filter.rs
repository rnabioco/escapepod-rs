//! Filter command implementation.
//!
//! Filters reads from a POD5 file based on a list of read IDs.
//! Uses lazy signal loading and block-level copying for maximum performance.

use crate::progress::{create_progress_bar, create_spinner};
use crate::style;
use crate::util::{
    batch_sizes, get_reads_iter_with_warning, open_reader_with_warning, resolve_pod5_inputs,
    LimitedWarningReporter, OpenResult,
};
use podfive_core::operations::read_ids_from_file;
use podfive_core::utils::{
    add_run_infos_deduplicated, map_run_info_index, scan_dictionary_values,
};
use podfive_core::{PredefinedDictionaries, Writer, WriterOptions};
use std::collections::HashMap;
use std::path::PathBuf;

pub fn run(input: PathBuf, ids_file: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    // Resolve input to list of POD5 files (supports directories)
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!(
        "{} {} using IDs from {}",
        style::action("Filtering"),
        if is_directory {
            format!(
                "{} ({} files)",
                style::path(input.display()),
                style::value(files.len())
            )
        } else {
            style::path(input.display())
        },
        style::path(ids_file.display())
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(output.display())
    );

    // Read IDs from file (using core library)
    let ids = read_ids_from_file(&ids_file)?;
    println!("Loaded {} read IDs to filter", style::count(ids.len()));

    if ids.is_empty() {
        anyhow::bail!("No read IDs found in {}", ids_file.display());
    }

    // Pre-scan files to collect unique dictionary values and count total reads
    let spinner = create_spinner("Scanning")?;
    spinner.set_message("files for dictionary values...");
    let scanned = scan_dictionary_values(&files, Some(&ids));
    spinner.finish_with_message(format!(
        "{} reads found",
        style::count(scanned.total_read_count)
    ));

    // Create writer with predefined dictionaries for consistent multi-batch writes
    let options = WriterOptions {
        signal_batch_size: batch_sizes::SIGNAL_BATCH_SIZE,
        read_batch_size: batch_sizes::READ_BATCH_SIZE,
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(scanned.pore_types.into_iter().collect()),
            end_reasons: Some(scanned.end_reasons.into_iter().collect()),
        }),
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&output, options)?;

    // Track run infos across all files
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Filter reads from all files
    let filter_bar = create_progress_bar(scanned.total_read_count, "Filtering")?;
    let mut matched = 0u64;
    let mut total = 0u64;
    let mut read_warnings = LimitedWarningReporter::new(3);
    let mut signal_warnings = LimitedWarningReporter::new(3);

    for file_path in &files {
        let reader = match open_reader_with_warning(file_path, is_directory) {
            OpenResult::Ok(r) => r,
            OpenResult::Skip => continue,
            OpenResult::Err(e) => return Err(e),
        };

        // Add run infos (deduplicated by acquisition_id)
        add_run_infos_deduplicated(&reader, &mut writer, &mut run_info_map)?;

        // NOTE: Signal is loaded lazily per-read, not all upfront
        // This is much more efficient when filtering a small subset of reads

        let reads_iter = match get_reads_iter_with_warning(&reader, file_path, is_directory) {
            OpenResult::Ok(iter) => iter,
            OpenResult::Skip => continue,
            OpenResult::Err(e) => return Err(e),
        };

        for read_result in reads_iter {
            let read = match read_result {
                Ok(r) => r,
                Err(e) => {
                    read_warnings.warn(&format!("error reading read record: {}", e));
                    continue;
                }
            };
            total += 1;
            filter_bar.inc(1);
            filter_bar.set_message(format!("{} matched", matched));

            // Check if this read's ID is in the filter list
            if ids.contains(&read.read_id) {
                // Lazy load: only fetch signal for matching reads (O(1) batch lookup + LRU cache)
                let compressed_signal =
                    match reader.get_compressed_signal_for_rows(&read.signal_rows) {
                        Ok(s) => s,
                        Err(e) => {
                            signal_warnings.warn(&format!(
                                "cannot read signal for read {}: {}",
                                read.read_id, e
                            ));
                            continue;
                        }
                    };

                // Map run_info index
                let new_run_info_idx =
                    map_run_info_index(&reader, read.run_info_index, &run_info_map);

                // Create new read data for writing
                let new_read = read.for_writing(new_run_info_idx);

                writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;
                matched += 1;
            }
        }
    }

    filter_bar.finish_with_message(format!("{} matched", matched));

    // Finalize output
    writer.finish()?;

    let percentage = if total > 0 {
        (matched as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{} {} reads from {} total ({})",
        style::action("Filtered"),
        style::count(matched),
        total,
        style::percentage(format!("{:.1}%", percentage))
    );

    let not_found = (ids.len() as u64).saturating_sub(matched);
    if not_found > 0 {
        println!(
            "{} {} requested IDs were not found in the input",
            style::warning_label("Warning:"),
            style::warning(not_found)
        );
    }
    if matched > ids.len() as u64 {
        println!(
            "{} {} duplicate reads matched across multiple files",
            style::note_label("Note:"),
            style::warning(matched - ids.len() as u64)
        );
    }

    // Report any errors encountered
    let read_errors = read_warnings.count();
    let signal_errors = signal_warnings.count();
    if read_errors > 0 || signal_errors > 0 {
        eprintln!(
            "{} encountered {} read error(s) and {} signal error(s)",
            style::error_label("Warning:"),
            style::error(read_errors),
            style::error(signal_errors)
        );
    }

    Ok(())
}

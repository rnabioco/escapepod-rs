//! Merge command implementation.
//!
//! Merges multiple POD5 files into a single output file.
//! Uses batch-level signal copying to avoid memory copies for maximum performance.

use crate::progress::{create_progress_bar, create_spinner};
use crate::style;
use crate::util::{batch_sizes, resolve_pod5_inputs};
use podfive_core::{Reader, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use uuid::Uuid;

pub fn run(
    inputs: Vec<PathBuf>,
    output: PathBuf,
    duplicate_ok: bool,
    _threads: Option<usize>,
) -> anyhow::Result<()> {
    if inputs.is_empty() {
        anyhow::bail!("No input files specified");
    }

    // Expand any directories to individual POD5 files
    let mut all_files = Vec::new();
    for input in &inputs {
        let files = resolve_pod5_inputs(input)?;
        all_files.extend(files);
    }

    if all_files.is_empty() {
        anyhow::bail!("No POD5 files found in specified inputs");
    }

    let num_files = all_files.len();
    eprintln!(
        "{} {} files into {}",
        style::action("Merging"),
        style::count(num_files),
        style::path(output.display())
    );

    let options = WriterOptions {
        signal_batch_size: 100,
        // Use large batch size to avoid Arrow IPC dictionary replacement issues
        // (all reads should fit in a single batch for typical merge operations)
        read_batch_size: batch_sizes::MERGE_READ_BATCH_SIZE,
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&output, options)?;

    // Track run infos by acquisition_id to avoid duplicates
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Track signal offset for each file (where its signal rows start in output)
    let mut file_signal_offsets: Vec<u64> = Vec::with_capacity(num_files);

    // Phase 1: Copy signal batches directly from each file
    // This is zero-copy from mmap - batches reference the memory-mapped input directly
    let signal_spinner = create_spinner("Copying")?;
    signal_spinner.set_message("signal batches...");

    for (file_idx, path) in all_files.iter().enumerate() {
        let reader = match Reader::open(path) {
            Ok(r) => r,
            Err(e) => {
                signal_spinner.suspend(|| {
                    eprintln!(
                        "{} failed to read {}: {}",
                        style::warning_label("Warning:"),
                        style::path(path.display()),
                        e
                    );
                });
                // Use u64::MAX as sentinel for failed files
                file_signal_offsets.push(u64::MAX);
                continue;
            }
        };

        // Record where this file's signal rows start
        let start_row = writer.current_signal_row();
        file_signal_offsets.push(start_row);

        // Copy signal batches directly (zero-copy from mmap)
        for batch in reader.signal_batches()? {
            writer.write_signal_batch(&batch)?;
        }

        signal_spinner.set_message(format!(
            "signal batches... ({}/{})",
            file_idx + 1,
            num_files
        ));
    }

    signal_spinner.finish_with_message(format!(
        "{} signal rows written",
        style::count(writer.current_signal_row())
    ));

    // Phase 2: Copy reads with remapped signal indices
    // We need to re-open files to read the reads (signal batches consumed the reader)
    let mut total_read_count = 0u64;
    for path in &all_files {
        if let Ok(reader) = Reader::open(path) {
            total_read_count += reader.read_count()? as u64;
        }
    }

    let write_bar = create_progress_bar(total_read_count, "Writing")?;
    write_bar.set_message("reads");

    // Track read IDs for duplicate detection
    let mut seen_reads: HashSet<Uuid> = if duplicate_ok {
        HashSet::new()
    } else {
        HashSet::with_capacity(100_000)
    };

    let mut total_reads = 0u64;
    let mut duplicate_count = 0u64;

    for (file_idx, path) in all_files.iter().enumerate() {
        // Skip files that failed during signal copying
        let signal_offset = file_signal_offsets[file_idx];
        if signal_offset == u64::MAX {
            continue;
        }

        let reader = match Reader::open(path) {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Add run infos (deduplicated by acquisition_id)
        let run_infos = reader.run_infos().to_vec();
        for run_info in &run_infos {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = writer.add_run_info(run_info.clone())?;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
            }
        }

        // Copy reads with offset-adjusted signal rows
        for read_result in reader.reads()? {
            let read = read_result?;

            // Check for duplicates
            if !duplicate_ok {
                if seen_reads.contains(&read.read_id) {
                    duplicate_count += 1;
                    write_bar.inc(1);
                    continue;
                }
                seen_reads.insert(read.read_id);
            }

            // Map run_info index
            let original_run_info = run_infos.get(read.run_info_index as usize);
            let new_run_info_idx = if let Some(ri) = original_run_info {
                *run_info_map.get(&ri.acquisition_id).unwrap_or(&0)
            } else {
                0
            };

            // Remap signal rows by adding this file's offset
            let new_signal_rows: Vec<u64> = read
                .signal_rows
                .iter()
                .map(|&row| row + signal_offset)
                .collect();

            let new_read = read.for_writing(new_run_info_idx);
            writer.add_read_with_signal_rows(new_read, new_signal_rows)?;
            total_reads += 1;
            write_bar.inc(1);
        }
    }

    write_bar.finish_with_message("done");

    // Finalize output file
    writer.finish()?;

    println!(
        "{} {} reads into {}",
        style::action("Merged"),
        style::count(total_reads),
        style::path(output.display())
    );

    if duplicate_count > 0 {
        println!(
            "{} {} duplicate reads",
            style::note_label("Skipped"),
            style::warning(duplicate_count)
        );
    }

    Ok(())
}

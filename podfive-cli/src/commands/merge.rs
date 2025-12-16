//! Merge command implementation.
//!
//! Merges multiple POD5 files into a single output file.
//! Uses single-pass batch-level signal copying for maximum performance.

use crate::progress::{create_progress_bar, create_spinner};
use crate::style;
use crate::util::{batch_sizes, resolve_pod5_inputs};
use podfive_core::{ReadData, Reader, RunInfoData, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use uuid::Uuid;

/// Context collected from a single input file during the signal copy phase.
/// This allows us to process each file only once.
struct FileContext {
    signal_offset: u64,
    run_infos: Vec<RunInfoData>,
    reads: Vec<ReadData>,
}

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

    // Single-pass: copy signal batches AND collect reads/run_infos from each file
    // This avoids re-opening files in a second pass
    let spinner = create_spinner("Processing")?;
    spinner.set_message("files...");

    let mut file_contexts: Vec<FileContext> = Vec::with_capacity(num_files);
    let mut total_read_count = 0u64;

    for (file_idx, path) in all_files.iter().enumerate() {
        let reader = match Reader::open(path) {
            Ok(r) => r,
            Err(e) => {
                spinner.suspend(|| {
                    eprintln!(
                        "{} failed to read {}: {}",
                        style::warning_label("Warning:"),
                        style::path(path.display()),
                        e
                    );
                });
                continue;
            }
        };

        // Record where this file's signal rows start
        let signal_offset = writer.current_signal_row();

        // Copy signal batches directly (zero-copy from mmap)
        for batch in reader.signal_batches()? {
            writer.write_signal_batch(&batch)?;
        }

        // Collect run_infos and reads (lightweight metadata, stays in memory)
        let run_infos = reader.run_infos().to_vec();
        let reads: Vec<ReadData> = reader.reads()?.collect::<Result<Vec<_>, _>>()?;
        total_read_count += reads.len() as u64;

        file_contexts.push(FileContext {
            signal_offset,
            run_infos,
            reads,
        });

        spinner.set_message(format!("files... ({}/{})", file_idx + 1, num_files));
    }

    spinner.finish_with_message(format!(
        "{} signal rows, {} reads collected",
        style::count(writer.current_signal_row()),
        style::count(total_read_count)
    ));

    // Phase 2: Write reads from collected data (no file I/O needed)
    let write_bar = create_progress_bar(total_read_count, "Writing")?;
    write_bar.set_message("reads");

    // Track run infos by acquisition_id to avoid duplicates
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Track read IDs for duplicate detection
    let mut seen_reads: HashSet<Uuid> = if duplicate_ok {
        HashSet::new()
    } else {
        HashSet::with_capacity(total_read_count as usize)
    };

    let mut total_reads = 0u64;
    let mut duplicate_count = 0u64;

    for ctx in &file_contexts {
        // Add run infos (deduplicated by acquisition_id)
        for run_info in &ctx.run_infos {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = writer.add_run_info(run_info.clone())?;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
            }
        }

        // Write reads with offset-adjusted signal rows
        for read in &ctx.reads {
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
            let original_run_info = ctx.run_infos.get(read.run_info_index as usize);
            let new_run_info_idx = if let Some(ri) = original_run_info {
                *run_info_map.get(&ri.acquisition_id).unwrap_or(&0)
            } else {
                0
            };

            // Remap signal rows by adding this file's offset
            let new_signal_rows: Vec<u64> = read
                .signal_rows
                .iter()
                .map(|&row| row + ctx.signal_offset)
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

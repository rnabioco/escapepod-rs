//! Merge command implementation.
//!
//! Merges multiple POD5 files into a single output file.
//! Uses block-level copying of compressed signal data for maximum performance.

use crate::util::resolve_pod5_inputs;
use podfive_core::{CompressedSignalChunk, Reader, Writer, WriterOptions};
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
    println!("Merging {} files into {}", num_files, output.display());

    // Create writer with optimized batch sizes
    // Signal batches can be smaller for frequent direct file writes
    // Read batch must hold all reads to avoid dictionary replacement in Arrow IPC
    let mut options = WriterOptions::default();
    options.signal_batch_size = 1_000; // Smaller batches for streaming to file
    options.read_batch_size = 500_000; // Large enough for typical merge operations
    let mut writer = Writer::create(&output, options)?;

    // Track run infos by acquisition_id to avoid duplicates
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Track read IDs for duplicate detection
    let mut seen_reads: HashSet<Uuid> = if duplicate_ok {
        HashSet::new()
    } else {
        HashSet::with_capacity(100_000)
    };

    let mut total_reads = 0u64;
    let mut duplicate_count = 0u64;

    // Reusable buffer for signal chunks (avoids repeated allocation)
    let mut compressed_signal: Vec<CompressedSignalChunk> = Vec::with_capacity(64);

    for (file_idx, file_path) in all_files.iter().enumerate() {
        let reader = Reader::open(file_path)?;

        // Add run infos (deduplicated by acquisition_id)
        for run_info in reader.run_infos() {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = writer.add_run_info(run_info.clone())?;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
            }
        }

        // Load all compressed signal data once
        let all_signal = reader.get_all_signal_compressed()?;

        for read_result in reader.reads()? {
            let read = read_result?;

            // Check for duplicates
            if !duplicate_ok {
                if seen_reads.contains(&read.read_id) {
                    duplicate_count += 1;
                    continue;
                }
                seen_reads.insert(read.read_id);
            }

            // Get compressed signal by looking up row indices
            // Reuse the buffer instead of allocating new Vec each time
            compressed_signal.clear();
            for &idx in &read.signal_rows {
                if let Some(chunk) = all_signal.get(idx as usize) {
                    compressed_signal.push(chunk.clone());
                }
            }

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
            total_reads += 1;
        }

        eprint!("\rProcessed {}/{} files", file_idx + 1, num_files);
    }
    eprintln!();

    // Finalize output file
    writer.finish()?;

    println!(
        "Successfully merged {} reads into {}",
        total_reads,
        output.display()
    );

    if duplicate_count > 0 {
        println!("Skipped {} duplicate reads", duplicate_count);
    }

    Ok(())
}

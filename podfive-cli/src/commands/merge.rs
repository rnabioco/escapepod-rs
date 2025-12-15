//! Merge command implementation.
//!
//! Merges multiple POD5 files into a single output file.
//! Uses parallel file reading and block-level signal copying for maximum performance.

use crate::util::resolve_pod5_inputs;
use podfive_core::{CompressedSignalChunk, ReadData, Reader, RunInfoData, Writer, WriterOptions};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use uuid::Uuid;

/// Data extracted from a single input file for merging.
struct FileData {
    run_infos: Vec<RunInfoData>,
    reads: Vec<(ReadData, Vec<CompressedSignalChunk>)>,
}

/// Read a single file and extract all data needed for merging.
fn read_file_data(path: &PathBuf) -> anyhow::Result<FileData> {
    let reader = Reader::open(path)?;

    // Collect run infos
    let run_infos: Vec<RunInfoData> = reader.run_infos().to_vec();

    // Collect reads with their signal data
    let mut reads = Vec::new();
    for read_result in reader.reads()? {
        let read = read_result?;
        // Use lazy signal loading with O(1) batch lookup
        let signal = reader.get_compressed_signal_for_rows(&read.signal_rows)?;
        reads.push((read, signal));
    }

    Ok(FileData { run_infos, reads })
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
    eprintln!("Merging {} files into {}", num_files, output.display());

    // Phase 1: Read all files in parallel
    eprintln!("Reading {} files in parallel...", num_files);
    let file_results: Vec<Result<FileData, anyhow::Error>> = all_files
        .par_iter()
        .map(|path| read_file_data(path))
        .collect();

    // Check for errors and collect successful results
    let mut file_data_vec = Vec::with_capacity(num_files);
    for (i, result) in file_results.into_iter().enumerate() {
        match result {
            Ok(data) => file_data_vec.push(data),
            Err(e) => {
                eprintln!(
                    "Warning: failed to read {}: {}",
                    all_files[i].display(),
                    e
                );
            }
        }
    }

    // Phase 2: Write all data sequentially (Writer is not thread-safe)
    eprintln!("Writing merged output...");

    let mut options = WriterOptions::default();
    options.signal_batch_size = 10_000; // Larger batches reduce flush overhead
    options.read_batch_size = 500_000;
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

    for file_data in file_data_vec {
        // Add run infos (deduplicated by acquisition_id)
        for run_info in file_data.run_infos {
            if !run_info_map.contains_key(&run_info.acquisition_id) {
                let idx = writer.add_run_info(run_info.clone())?;
                run_info_map.insert(run_info.acquisition_id.clone(), idx);
            }
        }

        // Write reads
        for (read, compressed_signal) in file_data.reads {
            // Check for duplicates
            if !duplicate_ok {
                if seen_reads.contains(&read.read_id) {
                    duplicate_count += 1;
                    continue;
                }
                seen_reads.insert(read.read_id);
            }

            // Map run_info index
            let new_run_info_idx = run_info_map
                .values()
                .next()
                .copied()
                .unwrap_or(0);

            let new_read = read.for_writing(new_run_info_idx);
            writer.add_read_with_compressed_signal(new_read, &compressed_signal)?;
            total_reads += 1;
        }
    }

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

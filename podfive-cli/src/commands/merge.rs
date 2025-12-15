//! Merge command implementation.
//!
//! Merges multiple POD5 files into a single output file using parallel I/O.

use crate::util::resolve_pod5_inputs;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use podfive_core::{ReadData, Reader, RunInfoData, Writer, WriterOptions};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

/// Message sent from reader tasks to the writer task
struct ReadMessage {
    read: ReadData,
    signal: Vec<i16>,
    run_info: RunInfoData,
}

pub fn run(
    inputs: Vec<PathBuf>,
    output: PathBuf,
    duplicate_ok: bool,
    threads: Option<usize>,
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

    let inputs = all_files;

    let num_threads = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    });

    // Build tokio runtime
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .enable_all()
        .build()?;

    runtime.block_on(async_merge(inputs, output, duplicate_ok, num_threads))
}

async fn async_merge(
    inputs: Vec<PathBuf>,
    output: PathBuf,
    duplicate_ok: bool,
    num_threads: usize,
) -> anyhow::Result<()> {
    let num_files = inputs.len();
    println!(
        "Merging {} files into {} (using {} threads)",
        num_files,
        output.display(),
        num_threads
    );

    // Set up progress bars
    let multi_progress = MultiProgress::new();
    let overall_style = ProgressStyle::default_bar()
        .template("{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} files ({msg})")?
        .progress_chars("━━─");
    let overall_bar = multi_progress.add(ProgressBar::new(num_files as u64));
    overall_bar.set_style(overall_style);
    overall_bar.set_prefix("Merging");

    // Channel for sending reads from readers to writer
    // Use bounded channel for backpressure
    let (tx, mut rx) = mpsc::channel::<ReadMessage>(1000);

    // Shared inputs for reader tasks
    let inputs = Arc::new(inputs);

    // Spawn reader tasks with semaphore to limit concurrency
    let semaphore = Arc::new(tokio::sync::Semaphore::new(num_threads));
    let mut reader_handles = Vec::new();

    for input_path in inputs.iter() {
        let tx = tx.clone();
        let semaphore = semaphore.clone();
        let input_path = input_path.clone();
        let bar = overall_bar.clone();

        let handle = tokio::spawn(async move {
            let _permit = semaphore
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("Semaphore acquire failed: {}", e))?;

            // Run blocking file I/O in spawn_blocking
            let result = tokio::task::spawn_blocking(move || read_file_contents(input_path, tx))
                .await
                .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?;

            bar.inc(1);
            result
        });

        reader_handles.push(handle);
    }

    // Drop the original sender so the channel closes when all readers finish
    drop(tx);

    // Writer task - runs in spawn_blocking since Writer is synchronous
    let output_clone = output.clone();
    let writer_handle =
        tokio::task::spawn_blocking(move || write_merged_file(output_clone, &mut rx, duplicate_ok));

    // Wait for all readers to complete
    let mut read_errors = Vec::new();
    for handle in reader_handles {
        if let Err(e) = handle.await {
            read_errors.push(format!("Reader task error: {}", e));
        }
    }

    // Wait for writer to complete
    let write_result = writer_handle.await?;

    overall_bar.finish_with_message("done");

    // Report any reader errors
    if !read_errors.is_empty() {
        for err in &read_errors {
            eprintln!("Error: {}", err);
        }
    }

    write_result
}

fn read_file_contents(input_path: PathBuf, tx: mpsc::Sender<ReadMessage>) -> anyhow::Result<u64> {
    let reader = Reader::open(&input_path)?;
    let mut count = 0u64;

    // Cache run infos for this file
    let run_infos: Vec<RunInfoData> = reader.run_infos().to_vec();

    for read_result in reader.reads()? {
        let read = read_result?;

        // Get signal data
        let signal = reader.get_signal(&read.signal_rows)?;

        // Get the run info for this read
        let run_info = run_infos
            .get(read.run_info_index as usize)
            .or_else(|| run_infos.first())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No run info available for read"))?;

        let msg = ReadMessage {
            read,
            signal,
            run_info,
        };

        // Send to writer (blocking send in sync context)
        if tx.blocking_send(msg).is_err() {
            // Channel closed, writer likely errored
            break;
        }

        count += 1;
    }

    Ok(count)
}

fn write_merged_file(
    output: PathBuf,
    rx: &mut mpsc::Receiver<ReadMessage>,
    duplicate_ok: bool,
) -> anyhow::Result<()> {
    let options = WriterOptions::default();
    let mut writer = Writer::create(&output, options)?;

    // Track run infos by acquisition_id to avoid duplicates
    let mut run_info_map: HashMap<String, u32> = HashMap::new();

    // Track read IDs for duplicate detection
    let mut seen_reads: HashSet<Uuid> = if duplicate_ok {
        HashSet::new() // Won't be used but need to initialize
    } else {
        HashSet::with_capacity(100_000)
    };

    let mut total_reads = 0u64;
    let mut duplicate_count = 0u64;

    // Process messages from the channel
    while let Some(msg) = rx.blocking_recv() {
        // Check for duplicates
        if !duplicate_ok {
            if seen_reads.contains(&msg.read.read_id) {
                duplicate_count += 1;
                continue;
            }
            seen_reads.insert(msg.read.read_id);
        }

        // Add or get run info index
        let run_info_idx = if let Some(&idx) = run_info_map.get(&msg.run_info.acquisition_id) {
            idx
        } else {
            let idx = writer.add_run_info(msg.run_info.clone())?;
            run_info_map.insert(msg.run_info.acquisition_id.clone(), idx);
            idx
        };

        // Create new read with correct run_info index
        let new_read = msg.read.for_writing(run_info_idx);

        writer.add_read(new_read, &msg.signal)?;
        total_reads += 1;
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

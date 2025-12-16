//! Repack command implementation.
//!
//! Repacks POD5 files to optimize storage and apply current compression settings.

use crate::progress::create_progress_bar;
use crate::style;
use podfive_core::{Reader, Writer, WriterOptions};
use std::path::PathBuf;
use tempfile::NamedTempFile;

pub fn run(inputs: Vec<PathBuf>, output_dir: PathBuf, force: bool) -> anyhow::Result<()> {
    if inputs.is_empty() {
        anyhow::bail!("No input files specified");
    }

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    println!(
        "{} {} file(s) to {}",
        style::action("Repacking"),
        style::count(inputs.len()),
        style::path(output_dir.display())
    );

    let overall_bar = create_progress_bar(inputs.len() as u64, "Repacking")?;

    let mut total_reads = 0u64;

    for input_path in &inputs {
        let file_name = input_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid input path"))?;
        let output_path = output_dir.join(file_name);

        // Check if output exists
        if output_path.exists() && !force {
            anyhow::bail!(
                "Output file {} already exists. Use --force to overwrite.",
                output_path.display()
            );
        }

        overall_bar.set_message(format!("{}", file_name.to_string_lossy()));

        let reads = repack_file(input_path, &output_path)?;
        total_reads += reads;

        overall_bar.inc(1);
    }

    overall_bar.finish_with_message("done");

    println!(
        "{} {} reads across {} file(s)",
        style::action("Repacked"),
        style::count(total_reads),
        style::value(inputs.len())
    );

    Ok(())
}

fn repack_file(input: &PathBuf, output: &PathBuf) -> anyhow::Result<u64> {
    // Check if input and output resolve to the same file
    // This prevents bus errors from truncating a memory-mapped file
    let input_canonical = std::fs::canonicalize(input)?;
    let same_file = output.exists() && std::fs::canonicalize(output)? == input_canonical;

    // Use a temp file if writing to the same location as input
    let (actual_output, temp_file): (PathBuf, Option<NamedTempFile>) = if same_file {
        let temp = NamedTempFile::new_in(output.parent().unwrap_or(std::path::Path::new(".")))?;
        (temp.path().to_path_buf(), Some(temp))
    } else {
        (output.clone(), None)
    };

    let reader = Reader::open(input)?;

    let options = WriterOptions::default();
    let mut writer = Writer::create(&actual_output, options)?;

    // Copy run infos
    let run_infos = reader.run_infos().to_vec();
    for run_info in &run_infos {
        writer.add_run_info(run_info.clone())?;
    }

    let mut count = 0u64;

    // Copy reads with their signals
    for read_result in reader.reads()? {
        let read = read_result?;
        let signal = reader.get_signal(&read.signal_rows)?;

        let new_read = read.for_writing_same_run();

        writer.add_read(new_read, &signal)?;
        count += 1;
    }

    writer.finish()?;

    // If we used a temp file, close the reader and rename temp to output
    if let Some(temp) = temp_file {
        drop(reader); // Release the memory map
        temp.persist(output)?;
    }

    Ok(count)
}

//! Archive command implementation.
//!
//! Creates archival copies of POD5 files with downsampled signal data.
//! This reduces storage requirements while preserving the ability to
//! re-basecall with appropriate models (typically fast or HAC).

use crate::progress::{create_progress_bar, create_spinner};
use crate::style;
use crate::util::{
    batch_sizes, get_reads_iter_with_warning, map_run_info_index, open_reader_with_warning,
    resolve_pod5_inputs, scan_dictionary_values, LimitedWarningReporter, OpenResult,
};
use podfive_core::signal::{downsample, downsample_average, downsampled_rate};
use podfive_core::{PredefinedDictionaries, Reader, RunInfoData, Writer, WriterOptions};
use std::collections::HashMap;
use std::path::PathBuf;

/// Downsampling method to use.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum DownsampleMethod {
    /// Simple decimation - keep every Nth sample (fast)
    #[default]
    Decimate,
    /// Average groups of N samples (better quality, slower)
    Average,
}

impl std::fmt::Display for DownsampleMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownsampleMethod::Decimate => write!(f, "decimate"),
            DownsampleMethod::Average => write!(f, "average"),
        }
    }
}

pub fn run(
    input: PathBuf,
    output: PathBuf,
    factor: u32,
    method: DownsampleMethod,
) -> anyhow::Result<()> {
    // Validate factor
    if factor < 2 {
        anyhow::bail!("Downsample factor must be at least 2 (got {})", factor);
    }
    if factor > 16 {
        anyhow::bail!(
            "Downsample factor {} is too aggressive (max 16). \
            Higher factors would likely make the data unusable for basecalling.",
            factor
        );
    }

    // Resolve input to list of POD5 files (supports directories)
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!(
        "{} {} with {}x downsampling ({})",
        style::action("Archiving"),
        if is_directory {
            format!(
                "{} ({} files)",
                style::path(input.display()),
                style::value(files.len())
            )
        } else {
            style::path(input.display())
        },
        style::value(factor),
        style::value(method)
    );
    println!(
        "{} {}",
        style::label("Output:"),
        style::path(output.display())
    );

    // Print warning about basecalling impact
    print_basecalling_warning(factor);

    // Pre-scan files to collect unique dictionary values and count total reads
    let spinner = create_spinner("Scanning")?;
    spinner.set_message("files...");
    let scanned = scan_dictionary_values(&files, None);
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

    // Track statistics
    let mut total_reads = 0u64;
    let mut total_original_samples = 0u64;
    let mut total_downsampled_samples = 0u64;

    // Process reads from all files
    let progress_bar = create_progress_bar(scanned.total_read_count, "Archiving")?;
    let mut read_warnings = LimitedWarningReporter::new(3);
    let mut signal_warnings = LimitedWarningReporter::new(3);

    for file_path in &files {
        let reader = match open_reader_with_warning(file_path, is_directory) {
            OpenResult::Ok(r) => r,
            OpenResult::Skip => continue,
            OpenResult::Err(e) => return Err(e),
        };

        // Add run infos with archive metadata (deduplicated by acquisition_id)
        add_archived_run_infos(&reader, &mut writer, &mut run_info_map, factor, method)?;

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

            progress_bar.inc(1);
            total_reads += 1;

            // Get decompressed signal
            let signal = match reader.get_signal(&read.signal_rows) {
                Ok(s) => s,
                Err(e) => {
                    signal_warnings.warn(&format!(
                        "cannot read signal for read {}: {}",
                        read.read_id, e
                    ));
                    continue;
                }
            };

            total_original_samples += signal.len() as u64;

            // Downsample the signal
            let downsampled = match method {
                DownsampleMethod::Decimate => downsample(&signal, factor),
                DownsampleMethod::Average => downsample_average(&signal, factor),
            };

            total_downsampled_samples += downsampled.len() as u64;

            // Map run_info index
            let new_run_info_idx = map_run_info_index(&reader, read.run_info_index, &run_info_map);

            // Create new read data with updated sample count
            let mut new_read = read.for_writing(new_run_info_idx);
            new_read.num_samples = downsampled.len() as u64;

            // Write the read with downsampled signal
            writer.add_read(new_read, &downsampled)?;
        }
    }

    progress_bar.finish_with_message("done");

    // Finalize output
    writer.finish()?;

    // Print summary
    let reduction = if total_original_samples > 0 {
        100.0 * (1.0 - (total_downsampled_samples as f64 / total_original_samples as f64))
    } else {
        0.0
    };

    println!();
    println!(
        "{} {} reads archived",
        style::action("Complete:"),
        style::count(total_reads)
    );
    println!(
        "{} {} -> {} samples ({})",
        style::label("Signal:"),
        style::value(format_samples(total_original_samples)),
        style::value(format_samples(total_downsampled_samples)),
        style::percentage(format!("{:.1}% reduction", reduction))
    );

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

/// Add run infos with archive metadata.
///
/// This function copies run infos from the source file to the writer,
/// adding metadata about the downsampling operation to context_tags.
fn add_archived_run_infos(
    reader: &Reader,
    writer: &mut Writer,
    run_info_map: &mut HashMap<String, u32>,
    factor: u32,
    method: DownsampleMethod,
) -> anyhow::Result<()> {
    for run_info in reader.run_infos() {
        if !run_info_map.contains_key(&run_info.acquisition_id) {
            // Create a modified run_info with archive metadata
            let archived_run_info = create_archived_run_info(run_info, factor, method);
            let idx = writer.add_run_info(archived_run_info)?;
            run_info_map.insert(run_info.acquisition_id.clone(), idx);
        }
    }
    Ok(())
}

/// Create an archived run_info with downsampling metadata.
fn create_archived_run_info(
    original: &RunInfoData,
    factor: u32,
    method: DownsampleMethod,
) -> RunInfoData {
    let mut archived = original.clone();

    // Add archive metadata to context_tags (preserve original for reference)
    archived.context_tags.insert(
        "podfive.archive.original_sample_rate".to_string(),
        original.sample_rate.to_string(),
    );
    archived.context_tags.insert(
        "podfive.archive.downsample_factor".to_string(),
        factor.to_string(),
    );
    archived.context_tags.insert(
        "podfive.archive.downsample_method".to_string(),
        method.to_string(),
    );

    // Update sample_rate to reflect the effective rate after downsampling.
    // This ensures timing calculations (read duration, start time) are correct.
    // Note: This will cause dorado's chemistry lookup to return UNKNOWN since
    // the (flowcell, kit, new_rate) tuple won't be in its chemistry map,
    // but dorado handles this gracefully with a warning and continues.
    archived.sample_rate = downsampled_rate(original.sample_rate, factor);

    archived
}

/// Format sample count with appropriate suffix.
fn format_samples(samples: u64) -> String {
    if samples >= 1_000_000_000 {
        format!("{:.2}G", samples as f64 / 1_000_000_000.0)
    } else if samples >= 1_000_000 {
        format!("{:.2}M", samples as f64 / 1_000_000.0)
    } else if samples >= 1_000 {
        format!("{:.2}K", samples as f64 / 1_000.0)
    } else {
        format!("{}", samples)
    }
}

/// Print a warning about the impact on basecalling.
fn print_basecalling_warning(factor: u32) {
    println!();
    println!(
        "{} {}x downsampling reduces effective sample rate:",
        style::warning_label("Note:"),
        factor
    );
    println!("  - 4000 Hz -> {} Hz", 4000 / factor);
    println!("  - 5000 Hz -> {} Hz", 5000 / factor);
    println!();
    println!("  Basecalling impact:");
    match factor {
        2 => {
            println!("  - HAC models: minimal impact (~1-2% accuracy loss)");
            println!("  - SUP models: moderate impact (~3-5% accuracy loss)");
            println!("  - Modified bases: noticeable impact (~5-10% accuracy loss)");
        }
        3..=4 => {
            println!("  - HAC models: noticeable impact (~3-5% accuracy loss)");
            println!("  - SUP models: significant impact (~5-10% accuracy loss)");
            println!("  - Modified bases: likely unusable");
        }
        _ => {
            println!("  - All models: significant accuracy degradation expected");
            println!("  - Recommended only for basic QC or storage savings");
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_samples() {
        assert_eq!(format_samples(500), "500");
        assert_eq!(format_samples(1500), "1.50K");
        assert_eq!(format_samples(1_500_000), "1.50M");
        assert_eq!(format_samples(1_500_000_000), "1.50G");
    }
}

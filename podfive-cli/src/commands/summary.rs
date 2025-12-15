//! Summary command implementation.
//!
//! Generates a comprehensive summary of POD5 file(s) with statistics and QC metrics.

use crate::util::{format_bytes, format_duration_hours, format_number, resolve_pod5_inputs};
use chrono::{TimeZone, Utc};
use owo_colors::OwoColorize;
use podfive_core::{Reader, RunInfoData};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Arguments for the summary command.
#[derive(Debug, clap::Args)]
pub struct SummaryArgs {
    /// Input POD5 file or directory containing POD5 files.
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output as JSON instead of formatted table.
    #[arg(long)]
    pub json: bool,
}

/// Aggregated statistics for reads.
#[derive(Debug, Default, Serialize)]
struct ReadStats {
    count: u64,
    total_samples: u64,
    lengths: Vec<u64>,
    channels: HashMap<u16, u64>,
    wells: [u64; 5], // Index 0 unused, wells 1-4
    end_reasons: HashMap<String, u64>,
}

/// File-level summary data.
#[derive(Debug, Serialize)]
struct FileSummary {
    path: String,
    size_bytes: u64,
    read_count: u64,
    batch_count: usize,
    pod5_version: String,
    software: String,
    file_identifier: String,
}

/// Complete summary output.
#[derive(Debug, Serialize)]
struct Summary {
    files: Vec<FileSummary>,
    run_info: Option<RunInfoSummary>,
    statistics: StatisticsSummary,
    end_reasons: HashMap<String, u64>,
    warnings: Vec<String>,
}

/// Run info summary (from first file's first run info).
#[derive(Debug, Serialize)]
struct RunInfoSummary {
    acquisition_id: String,
    acquisition_start_time: Option<String>,
    sample_rate: u16,
    flow_cell_id: String,
    flow_cell_product_code: String,
    sequencing_kit: String,
    sample_id: String,
    experiment_name: String,
    protocol_name: String,
    software: String,
    system_name: String,
    system_type: String,
}

/// Statistics summary.
#[derive(Debug, Serialize)]
struct StatisticsSummary {
    total_samples: u64,
    length_min: u64,
    length_max: u64,
    length_mean: f64,
    length_median: u64,
    length_n50: u64,
    active_channels: usize,
    total_channels: u16,
}

/// Current POD5 version for comparison.
const CURRENT_POD5_VERSION: &str = "1.0";

/// Run the summary command.
pub fn run(args: SummaryArgs) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&args.input)?;
    let is_directory = files.len() > 1;

    let mut stats = ReadStats::default();
    let mut file_summaries = Vec::new();
    let mut run_info: Option<RunInfoData> = None;
    let mut sample_rate: u16 = 0;
    let mut warnings = Vec::new();

    // Track corrupted files
    let mut corrupted_files = Vec::new();

    // Process each file
    for path in &files {
        // Try to open the file, skip if corrupted
        let reader = match Reader::open(path) {
            Ok(r) => r,
            Err(e) => {
                corrupted_files.push(path.display().to_string());
                warnings.push(format!(
                    "Corrupted/unreadable file: {} ({})",
                    path.display(),
                    e
                ));
                continue;
            }
        };

        // Check version
        let version = reader.pod5_version();
        if !version.starts_with(CURRENT_POD5_VERSION) && !version.is_empty() {
            warnings.push(format!(
                "Old POD5 version: {} uses v{} (current: {})",
                path.file_name().unwrap_or_default().to_string_lossy(),
                version,
                CURRENT_POD5_VERSION
            ));
        }

        // Get file size
        let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        // Get batch count (handle errors gracefully)
        let batch_count = reader.read_batch_count().unwrap_or(0);

        // Get run info from first file
        if run_info.is_none() {
            if let Some(ri) = reader.run_infos().first() {
                run_info = Some(ri.clone());
                sample_rate = ri.sample_rate;
            }
        }

        // Count reads and collect stats
        let mut file_read_count = 0u64;
        let mut read_errors = 0u64;

        match reader.reads() {
            Ok(reads_iter) => {
                for read_result in reads_iter {
                    match read_result {
                        Ok(read) => {
                            file_read_count += 1;
                            stats.count += 1;
                            stats.total_samples += read.num_samples;
                            stats.lengths.push(read.num_samples);

                            *stats.channels.entry(read.channel).or_insert(0) += 1;

                            if read.well >= 1 && read.well <= 4 {
                                stats.wells[read.well as usize] += 1;
                            }

                            *stats
                                .end_reasons
                                .entry(read.end_reason.as_str().to_string())
                                .or_insert(0) += 1;
                        }
                        Err(_) => {
                            read_errors += 1;
                        }
                    }
                }
            }
            Err(e) => {
                warnings.push(format!(
                    "Cannot read reads from {}: {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    e
                ));
            }
        }

        if read_errors > 0 {
            warnings.push(format!(
                "{} read errors in {}",
                read_errors,
                path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }

        file_summaries.push(FileSummary {
            path: path.display().to_string(),
            size_bytes,
            read_count: file_read_count,
            batch_count,
            pod5_version: reader.pod5_version().to_string(),
            software: reader.software().to_string(),
            file_identifier: reader.file_identifier().to_string(),
        });
    }

    // Add summary of corrupted files
    if !corrupted_files.is_empty() {
        warnings.insert(
            0,
            format!("{} file(s) could not be read", corrupted_files.len()),
        );
    }

    // Compute statistics
    let statistics = compute_statistics(&mut stats);

    // Build run info summary
    let run_info_summary = run_info.map(|ri| RunInfoSummary {
        acquisition_id: ri.acquisition_id,
        acquisition_start_time: if ri.acquisition_start_time > 0 {
            Some(format_timestamp(ri.acquisition_start_time))
        } else {
            None
        },
        sample_rate: ri.sample_rate,
        flow_cell_id: ri.flow_cell_id,
        flow_cell_product_code: ri.flow_cell_product_code,
        sequencing_kit: ri.sequencing_kit,
        sample_id: ri.sample_id,
        experiment_name: ri.experiment_name,
        protocol_name: ri.protocol_name,
        software: ri.software,
        system_name: ri.system_name,
        system_type: ri.system_type,
    });

    let summary = Summary {
        files: file_summaries,
        run_info: run_info_summary,
        statistics,
        end_reasons: stats.end_reasons,
        warnings,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_summary(&summary, &args.input, is_directory, sample_rate, &stats.lengths);
    }

    Ok(())
}

/// Compute statistics from collected read data.
fn compute_statistics(stats: &mut ReadStats) -> StatisticsSummary {
    if stats.lengths.is_empty() {
        return StatisticsSummary {
            total_samples: 0,
            length_min: 0,
            length_max: 0,
            length_mean: 0.0,
            length_median: 0,
            length_n50: 0,
            active_channels: 0,
            total_channels: 512,
        };
    }

    // Sort for median and N50
    stats.lengths.sort_unstable();

    let length_min = *stats.lengths.first().unwrap_or(&0);
    let length_max = *stats.lengths.last().unwrap_or(&0);
    let length_mean = stats.total_samples as f64 / stats.count as f64;

    let length_median = if stats.lengths.len() % 2 == 0 {
        let mid = stats.lengths.len() / 2;
        (stats.lengths[mid - 1] + stats.lengths[mid]) / 2
    } else {
        stats.lengths[stats.lengths.len() / 2]
    };

    let length_n50 = compute_n50(&stats.lengths);

    StatisticsSummary {
        total_samples: stats.total_samples,
        length_min,
        length_max,
        length_mean,
        length_median,
        length_n50,
        active_channels: stats.channels.len(),
        total_channels: 512,
    }
}

/// Compute N50 from sorted lengths.
fn compute_n50(sorted_lengths: &[u64]) -> u64 {
    if sorted_lengths.is_empty() {
        return 0;
    }

    let total: u64 = sorted_lengths.iter().sum();
    let half = total / 2;
    let mut cumsum = 0u64;

    // N50 requires reverse iteration (longest to shortest)
    for &len in sorted_lengths.iter().rev() {
        cumsum += len;
        if cumsum >= half {
            return len;
        }
    }

    0
}

/// Format a timestamp in milliseconds to ISO 8601.
fn format_timestamp(ms: i64) -> String {
    Utc.timestamp_millis_opt(ms)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| "Invalid timestamp".to_string())
}

/// Unicode sparkline characters.
const SPARK_CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Generate a sparkline from values.
fn sparkline(values: &[u64], width: usize) -> String {
    if values.is_empty() || width == 0 {
        return String::new();
    }

    // Create histogram buckets
    let min = *values.iter().min().unwrap_or(&0);
    let max = *values.iter().max().unwrap_or(&1);
    let range = (max - min).max(1);

    let mut buckets = vec![0u64; width];
    for &v in values {
        let bucket = (((v - min) as f64 / range as f64) * (width - 1) as f64).round() as usize;
        buckets[bucket.min(width - 1)] += 1;
    }

    let max_count = *buckets.iter().max().unwrap_or(&1) as f64;
    buckets
        .iter()
        .map(|&count| {
            let idx = ((count as f64 / max_count) * 7.0).round() as usize;
            SPARK_CHARS[idx.min(7)]
        })
        .collect()
}

/// Generate a progress bar.
fn progress_bar(pct: f64, width: usize) -> String {
    let filled = (pct / 100.0 * width as f64).round() as usize;
    format!(
        "{}{}",
        "█".repeat(filled.min(width)),
        "░".repeat(width.saturating_sub(filled))
    )
}

/// Print the formatted summary.
fn print_summary(summary: &Summary, input: &PathBuf, is_directory: bool, sample_rate: u16, lengths: &[u64]) {
    let width = 77;
    let border = "─".repeat(width);

    // Title
    let title = if is_directory {
        format!(
            "POD5 Summary: {} ({} files)",
            input.display(),
            summary.files.len()
        )
    } else {
        format!("POD5 Summary: {}", input.display())
    };

    println!("┌{}┐", border);
    println!("│ {:<width$} │", title.bold().cyan(), width = width - 1);
    println!("├{}┤", border);

    // File info row
    let total_size: u64 = summary.files.iter().map(|f| f.size_bytes).sum();
    let total_reads: u64 = summary.files.iter().map(|f| f.read_count).sum();
    let duration = format_duration_hours(summary.statistics.total_samples, sample_rate);

    println!(
        "│ {:>6} {} │ {:>7} {} │ {:>6} {} │ {:>8} {} │",
        format_bytes(total_size).bold(),
        "Size".dimmed(),
        format_number(total_reads).bold(),
        "Reads".dimmed(),
        format!("{} kHz", sample_rate / 1000).bold(),
        "Rate".dimmed(),
        duration.bold(),
        "Duration".dimmed(),
    );

    // Run info
    if let Some(ri) = &summary.run_info {
        println!("├{}┤", border);

        let flow_cell = if ri.flow_cell_product_code.is_empty() {
            ri.flow_cell_id.clone()
        } else {
            format!("{} ({})", ri.flow_cell_id, ri.flow_cell_product_code)
        };

        println!(
            "│ {:12} {:<26} │ {:12} {:<20} │",
            "Flow Cell".dimmed(),
            truncate(&flow_cell, 26).bold(),
            "Kit".dimmed(),
            truncate(&ri.sequencing_kit, 20).bold(),
        );
        println!(
            "│ {:12} {:<26} │ {:12} {:<20} │",
            "Sample".dimmed(),
            truncate(&ri.sample_id, 26).bold(),
            "Protocol".dimmed(),
            truncate(&ri.protocol_name, 20).bold(),
        );
        if let Some(start) = &ri.acquisition_start_time {
            println!(
                "│ {:12} {:<26} │ {:12} {:<20} │",
                "Started".dimmed(),
                start.bold(),
                "Software".dimmed(),
                truncate(&ri.software, 20).bold(),
            );
        }
    }

    // Read length statistics
    println!("├{}┤", border);
    println!(
        "│ {} {:width$} │",
        "READ LENGTH (samples)".cyan(),
        "",
        width = width - 23
    );

    let s = &summary.statistics;
    println!(
        "│   {:6} {:>10} │ {:6} {:>10} │ {:6} {:>10} │ {:6} {:>12} │",
        "N50".dimmed(),
        format_number(s.length_n50).bold(),
        "Mean".dimmed(),
        format_number(s.length_mean as u64).bold(),
        "Median".dimmed(),
        format_number(s.length_median).bold(),
        "Range".dimmed(),
        format!("{}-{}", format_compact(s.length_min), format_compact(s.length_max)).bold(),
    );

    // Add sparkline for length distribution
    if !lengths.is_empty() {
        let spark = sparkline(lengths, 40);
        let label = "length distribution";
        let padding = width.saturating_sub(spark.len() + label.len() + 6);
        println!(
            "│   {} {}{:padding$} │",
            spark,
            label.dimmed(),
            "",
            padding = padding
        );
    }

    // Channel usage
    println!("├{}┤", border);
    let channel_pct = s.active_channels as f64 / s.total_channels as f64 * 100.0;
    println!(
        "│ {:10} {}/{} active ({:.1}%) {:width$} │",
        "CHANNELS".cyan(),
        s.active_channels.to_string().bold(),
        s.total_channels,
        channel_pct,
        "",
        width = width - 42
    );

    // End reasons
    println!("├{}┤", border);
    println!(
        "│ {} {:width$} │",
        "END REASONS".cyan(),
        "",
        width = width - 13
    );

    let total_reads_f = total_reads as f64;
    let mut reasons: Vec<_> = summary.end_reasons.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1));

    for (reason, count) in reasons.iter().take(4) {
        let pct = **count as f64 / total_reads_f * 100.0;
        let bar = progress_bar(pct, 20);
        println!(
            "│   {:24} {} {:>5.1}%  ({:>7}) {:width$} │",
            reason.bold(),
            bar,
            pct,
            format_number(**count),
            "",
            width = width - 60
        );
    }

    // Warnings
    if !summary.warnings.is_empty() {
        println!("├{}┤", border);
        println!(
            "│ {} {:width$} │",
            "⚠ WARNINGS".yellow().bold(),
            "",
            width = width - 12
        );
        for warning in &summary.warnings {
            println!(
                "│   {} {:width$} │",
                warning.yellow(),
                "",
                width = width - warning.len() - 4
            );
        }
    }

    println!("└{}┘", border);
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

/// Format a number in compact form (e.g., 1.2K, 500K).
fn format_compact(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

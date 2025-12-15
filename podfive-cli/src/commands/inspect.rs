//! Inspect command implementation.

use crate::util::resolve_pod5_inputs;
use podfive_core::Reader;
use std::path::PathBuf;

pub fn summary(input: PathBuf) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!("POD5 File Summary");
    println!("=================");
    println!();

    if is_directory {
        println!("Directory: {}", input.display());
        println!("Files: {}", files.len());
    }

    let mut total_reads = 0usize;
    let mut total_batches = 0usize;

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(e) => {
                if is_directory {
                    eprintln!(
                        "  Warning: skipping {} ({})",
                        file_path.file_name().unwrap_or_default().to_string_lossy(),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        if !is_directory {
            println!("File: {}", file_path.display());
            println!("File ID: {}", reader.file_identifier());
            println!("POD5 version: {}", reader.pod5_version());
            println!("Software: {}", reader.software());
            println!();
        }

        let read_count = reader.read_count().unwrap_or(0);
        let batch_count = reader.read_batch_count().unwrap_or(0);
        total_reads += read_count;
        total_batches += batch_count;

        if is_directory {
            println!(
                "  {}: {} reads, {} batches",
                file_path.file_name().unwrap_or_default().to_string_lossy(),
                read_count,
                batch_count
            );
        } else {
            println!("Reads: {}", read_count);
            println!("Read batches: {}", batch_count);
            println!();

            println!("Run info entries: {}", reader.run_info_count());
            for (i, run_info) in reader.run_infos().iter().enumerate() {
                println!("  [{}] acquisition_id: {}", i, run_info.acquisition_id);
                println!("      sample_rate: {} Hz", run_info.sample_rate);
                println!("      flow_cell_id: {}", run_info.flow_cell_id);
            }
        }
    }

    if is_directory {
        println!();
        println!("Total reads: {}", total_reads);
        println!("Total batches: {}", total_batches);
    }

    Ok(())
}

pub fn reads(input: PathBuf) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!(
        "{:<36} {:>8} {:>4} {:>10} {:>12}",
        "read_id", "channel", "well", "samples", "end_reason"
    );
    println!("{}", "-".repeat(76));

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
                Err(_) => continue,
            };
            println!(
                "{:<36} {:>8} {:>4} {:>10} {:>12}",
                read.read_id, read.channel, read.well, read.num_samples, read.end_reason
            );
        }
    }

    Ok(())
}

pub fn read(input: PathBuf, read_id: String) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;
    let target_id: uuid::Uuid = read_id.parse()?;

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(_) if is_directory => continue,
            Err(e) => return Err(e.into()),
        };

        let reads_iter = match reader.reads() {
            Ok(iter) => iter,
            Err(_) if is_directory => continue,
            Err(e) => return Err(e.into()),
        };

        for read_result in reads_iter {
            let read = match read_result {
                Ok(r) => r,
                Err(_) => continue,
            };
            if read.read_id == target_id {
                println!("Read Details");
                println!("============");
                println!();
                if is_directory {
                    println!("file: {}", file_path.display());
                }
                println!("read_id: {}", read.read_id);
                println!("read_number: {}", read.read_number);
                println!("channel: {}", read.channel);
                println!("well: {}", read.well);
                println!("start_sample: {}", read.start_sample);
                println!("num_samples: {}", read.num_samples);
                println!("num_minknow_events: {}", read.num_minknow_events);
                println!();
                println!("pore_type: {}", read.pore_type);
                println!("calibration_offset: {}", read.calibration_offset);
                println!("calibration_scale: {}", read.calibration_scale);
                println!("median_before: {}", read.median_before);
                println!("open_pore_level: {}", read.open_pore_level);
                println!();
                println!("end_reason: {}", read.end_reason);
                println!("end_reason_forced: {}", read.end_reason_forced);
                return Ok(());
            }
        }
    }

    anyhow::bail!("Read not found: {}", read_id)
}

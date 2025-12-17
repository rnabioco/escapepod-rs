//! Inspect command implementation.

use crate::style;
use crate::util::resolve_pod5_inputs;
use escapepod::Reader;
use std::path::PathBuf;

pub fn summary(input: PathBuf) -> anyhow::Result<()> {
    let files = resolve_pod5_inputs(&input)?;
    let is_directory = files.len() > 1;

    println!("{}", style::header("POD5 File Summary"));
    println!("=================");
    println!();

    if is_directory {
        println!(
            "{} {}",
            style::key("Directory:"),
            style::path(input.display())
        );
        println!("{} {}", style::key("Files:"), style::count(files.len()));
    }

    let mut total_reads = 0usize;
    let mut total_batches = 0usize;

    for file_path in &files {
        let reader = match Reader::open(file_path) {
            Ok(r) => r,
            Err(e) => {
                if is_directory {
                    eprintln!(
                        "  {} skipping {} ({})",
                        style::warning_label("Warning:"),
                        style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
                        e
                    );
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };

        if !is_directory {
            println!(
                "{} {}",
                style::key("File:"),
                style::path(file_path.display())
            );
            println!(
                "{} {}",
                style::key("File ID:"),
                style::value(reader.file_identifier())
            );
            println!(
                "{} {}",
                style::key("POD5 version:"),
                style::value(reader.pod5_version())
            );
            println!(
                "{} {}",
                style::key("Software:"),
                style::value(reader.software())
            );
            println!();
        }

        let read_count = reader.read_count().unwrap_or(0);
        let batch_count = reader.read_batch_count().unwrap_or(0);
        total_reads += read_count;
        total_batches += batch_count;

        if is_directory {
            println!(
                "  {}: {} reads, {} batches",
                style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
                style::count(read_count),
                batch_count
            );
        } else {
            println!("{} {}", style::key("Reads:"), style::count(read_count));
            println!("{} {}", style::key("Read batches:"), batch_count);
            println!();

            println!(
                "{} {}",
                style::key("Run info entries:"),
                style::value(reader.run_info_count())
            );
            for (i, run_info) in reader.run_infos().iter().enumerate() {
                println!(
                    "  [{}] {}: {}",
                    i,
                    style::key("acquisition_id"),
                    style::value(&run_info.acquisition_id)
                );
                println!(
                    "      {}: {} Hz",
                    style::key("sample_rate"),
                    style::value(run_info.sample_rate)
                );
                println!(
                    "      {}: {}",
                    style::key("flow_cell_id"),
                    style::value(&run_info.flow_cell_id)
                );
            }
        }
    }

    if is_directory {
        println!();
        println!(
            "{} {}",
            style::key("Total reads:"),
            style::count(total_reads)
        );
        println!("{} {}", style::key("Total batches:"), total_batches);
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
                        "{} skipping {} ({})",
                        style::warning_label("Warning:"),
                        style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
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
                        "{} cannot read {} ({})",
                        style::warning_label("Warning:"),
                        style::path(file_path.file_name().unwrap_or_default().to_string_lossy()),
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
                println!("{}", style::header("Read Details"));
                println!("============");
                println!();
                if is_directory {
                    println!(
                        "{}: {}",
                        style::key("file"),
                        style::path(file_path.display())
                    );
                }
                println!("{}: {}", style::key("read_id"), style::value(read.read_id));
                println!(
                    "{}: {}",
                    style::key("read_number"),
                    style::value(read.read_number)
                );
                println!("{}: {}", style::key("channel"), style::value(read.channel));
                println!("{}: {}", style::key("well"), style::value(read.well));
                println!(
                    "{}: {}",
                    style::key("start_sample"),
                    style::value(read.start_sample)
                );
                println!(
                    "{}: {}",
                    style::key("num_samples"),
                    style::count(read.num_samples)
                );
                println!(
                    "{}: {}",
                    style::key("num_minknow_events"),
                    style::value(read.num_minknow_events)
                );
                println!();
                println!(
                    "{}: {}",
                    style::key("pore_type"),
                    style::value(&read.pore_type)
                );
                println!(
                    "{}: {}",
                    style::key("calibration_offset"),
                    style::value(read.calibration_offset)
                );
                println!(
                    "{}: {}",
                    style::key("calibration_scale"),
                    style::value(read.calibration_scale)
                );
                println!(
                    "{}: {}",
                    style::key("median_before"),
                    style::value(read.median_before)
                );
                println!(
                    "{}: {}",
                    style::key("open_pore_level"),
                    style::value(read.open_pore_level)
                );
                println!();
                println!(
                    "{}: {}",
                    style::key("end_reason"),
                    style::value(read.end_reason)
                );
                println!(
                    "{}: {}",
                    style::key("end_reason_forced"),
                    style::value(read.end_reason_forced)
                );
                return Ok(());
            }
        }
    }

    anyhow::bail!("Read not found: {}", read_id)
}

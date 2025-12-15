//! Inspect command implementation.

use podfive_core::Reader;
use std::path::PathBuf;

pub fn summary(input: PathBuf) -> anyhow::Result<()> {
    let reader = Reader::open(&input)?;

    println!("POD5 File Summary");
    println!("=================");
    println!();
    println!("File: {}", input.display());
    println!("File ID: {}", reader.file_identifier());
    println!("POD5 version: {}", reader.pod5_version());
    println!("Software: {}", reader.software());
    println!();

    let read_count = reader.read_count()?;
    let batch_count = reader.read_batch_count()?;
    println!("Reads: {}", read_count);
    println!("Read batches: {}", batch_count);
    println!();

    println!("Run info entries: {}", reader.run_info_count());
    for (i, run_info) in reader.run_infos().iter().enumerate() {
        println!("  [{}] acquisition_id: {}", i, run_info.acquisition_id);
        println!("      sample_rate: {} Hz", run_info.sample_rate);
        println!("      flow_cell_id: {}", run_info.flow_cell_id);
    }

    Ok(())
}

pub fn reads(input: PathBuf) -> anyhow::Result<()> {
    let reader = Reader::open(&input)?;

    println!(
        "{:<36} {:>8} {:>4} {:>10} {:>12}",
        "read_id", "channel", "well", "samples", "end_reason"
    );
    println!("{}", "-".repeat(76));

    for read_result in reader.reads()? {
        let read = read_result?;
        println!(
            "{:<36} {:>8} {:>4} {:>10} {:>12}",
            read.read_id, read.channel, read.well, read.num_samples, read.end_reason
        );
    }

    Ok(())
}

pub fn read(input: PathBuf, read_id: String) -> anyhow::Result<()> {
    let reader = Reader::open(&input)?;

    let target_id: uuid::Uuid = read_id.parse()?;

    for read_result in reader.reads()? {
        let read = read_result?;
        if read.read_id == target_id {
            println!("Read Details");
            println!("============");
            println!();
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

    anyhow::bail!("Read not found: {}", read_id)
}

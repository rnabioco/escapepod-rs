//! Write a synthetic POD5 file with realistic metadata.
//!
//! This example creates a POD5 file that mimics real Oxford Nanopore data
//! closely enough for tools like Dorado and the Python `pod5` library to
//! parse successfully.
//!
//! Usage:
//!     cargo run --release --example write_pod5 -- output.pod5

use std::collections::HashMap;
use std::path::PathBuf;

use escapepod_signal::{EndReason, ReadData, RunInfoData, Writer, WriterOptions};
use uuid::Uuid;

fn main() -> escapepod_signal::Result<()> {
    let output_path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: write_pod5 <output.pod5>");
            std::process::exit(1);
        })
        .into();

    let options = WriterOptions::default();
    let mut writer = Writer::create(&output_path, options)?;

    // Use realistic metadata matching R10.4.1 E8.2 5kHz chemistry so that
    // modern Dorado (>=1.0) can basecall these reads.
    let run_info = RunInfoData {
        acquisition_id: "synthetic_acq_001".to_string(),
        acquisition_start_time: 1700000000000,
        adc_max: 2047,
        adc_min: -2048,
        context_tags: HashMap::from([
            ("experiment_type".to_string(), "genomic_dna".to_string()),
            (
                "basecall_config_filename".to_string(),
                "dna_r10.4.1_e8.2_400bps_fast@v5.0.0".to_string(),
            ),
            ("sample_frequency".to_string(), "5000".to_string()),
        ]),
        experiment_name: "synthetic_dorado_compat".to_string(),
        flow_cell_id: "PAM00001".to_string(),
        flow_cell_product_code: "FLO-MIN114".to_string(),
        protocol_name: "sequencing/sequencing_MIN114_DNA".to_string(),
        protocol_run_id: "synthetic_proto_001".to_string(),
        protocol_start_time: 1699999000000,
        sample_id: "synthetic_sample".to_string(),
        sample_rate: 5000,
        sequencing_kit: "SQK-LSK114".to_string(),
        sequencer_position: "MN00001".to_string(),
        sequencer_position_type: "MinION".to_string(),
        software: "MinKNOW 23.11.1".to_string(),
        system_name: "synthetic_host".to_string(),
        system_type: "linux".to_string(),
        tracking_id: HashMap::from([
            ("device_id".to_string(), "MN00001".to_string()),
            ("run_id".to_string(), "synthetic_run_001".to_string()),
        ]),
    };
    let run_info_idx = writer.add_run_info(run_info)?;

    // Write several reads with varying signal lengths.
    let signal_sizes: &[usize] = &[1000, 5000, 20000, 50000, 100000];
    for (i, &size) in signal_sizes.iter().enumerate() {
        let read = ReadData {
            read_id: Uuid::new_v4(),
            read_number: (i + 1) as u32,
            start_sample: (i as u64) * 100_000,
            channel: (i as u16 % 512) + 1,
            well: 1,
            pore_type: "not_set".into(),
            calibration_offset: -220.0,
            calibration_scale: 0.15,
            median_before: 200.0,
            end_reason: EndReason::SignalPositive,
            end_reason_forced: false,
            run_info_index: run_info_idx,
            num_minknow_events: (size / 10) as u64,
            tracked_scaling_scale: 1.0,
            tracked_scaling_shift: 0.0,
            predicted_scaling_scale: 1.0,
            predicted_scaling_shift: 0.0,
            num_reads_since_mux_change: 0,
            time_since_mux_change: 0.0,
            num_samples: size as u64,
            open_pore_level: 220.0,
            signal_rows: Vec::new(),
        };

        // Generate a sinusoidal signal that looks vaguely like nanopore current.
        let signal: Vec<i16> = (0..size)
            .map(|s| ((s as f64 * 0.05).sin() * 100.0) as i16 + 512)
            .collect();

        writer.add_read(read, &signal)?;
    }

    writer.finish()?;
    eprintln!(
        "Wrote {} reads to {}",
        signal_sizes.len(),
        output_path.display()
    );
    Ok(())
}

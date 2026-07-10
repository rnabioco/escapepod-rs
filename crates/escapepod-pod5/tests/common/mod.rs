//! Shared helpers for escapepod-pod5 integration tests.

use std::collections::HashMap;
use std::path::Path;

use escapepod_pod5::{EndReason, ReadData, RunInfoData, Uuid, Writer, WriterOptions};

pub fn make_run_info(acq_id: &str) -> RunInfoData {
    RunInfoData {
        acquisition_id: acq_id.to_string(),
        acquisition_start_time: 1_609_459_200_000,
        adc_max: 2047,
        adc_min: -2048,
        context_tags: HashMap::from([("experiment_type".to_string(), "genomic_dna".to_string())]),
        experiment_name: "itest".to_string(),
        flow_cell_id: "FAK_ITEST".to_string(),
        flow_cell_product_code: "FLO-MIN106".to_string(),
        protocol_name: "itest_protocol".to_string(),
        protocol_run_id: "protocol_itest".to_string(),
        protocol_start_time: 1_609_459_200_000,
        sample_id: "itest_sample".to_string(),
        sample_rate: 4_000,
        sequencing_kit: "SQK-LSK109".to_string(),
        sequencer_position: "MN00000".to_string(),
        sequencer_position_type: "minion".to_string(),
        software: "escapepod-itest".to_string(),
        system_name: "itest_system".to_string(),
        system_type: "minion".to_string(),
        tracking_id: HashMap::new(),
    }
}

pub fn make_read(run_info_idx: u32, read_number: u32, num_samples: u64) -> ReadData {
    ReadData {
        read_id: Uuid::new_v4(),
        read_number,
        start_sample: (read_number as u64 - 1) * num_samples,
        channel: 1,
        well: 1,
        pore_type: "not_set".into(),
        calibration_offset: 0.5,
        calibration_scale: 0.95,
        median_before: 200.0,
        end_reason: EndReason::SignalPositive,
        end_reason_forced: false,
        run_info_index: run_info_idx,
        num_minknow_events: 100,
        tracked_scaling_scale: 1.0,
        tracked_scaling_shift: 0.0,
        predicted_scaling_scale: 1.0,
        predicted_scaling_shift: 0.0,
        num_reads_since_mux_change: 0,
        time_since_mux_change: 0.0,
        num_samples,
        open_pore_level: 220.0,
        expected_open_pore_level: 0.0,
        selected_read_level: 0.0,
        signal_rows: Vec::new(),
    }
}

pub fn synth_signal(n: usize, seed: u64) -> Vec<i16> {
    let mut state = seed | 1;
    let mut base: i32 = 500;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state as i32) >> 24) % 15;
            if i % 200 == 0 {
                base = 400 + ((i as i32 / 200) % 4) * 50;
            }
            (base + noise).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
        .collect()
}

pub struct WrittenFile {
    pub read_ids: Vec<Uuid>,
}

pub fn write_fixture(
    path: &Path,
    acq_id: &str,
    n_reads: usize,
    samples_per_read: usize,
) -> WrittenFile {
    let mut writer = Writer::create(path, WriterOptions::default()).expect("writer::create");
    let run_idx = writer
        .add_run_info(make_run_info(acq_id))
        .expect("add_run_info");
    let mut ids = Vec::with_capacity(n_reads);
    for i in 0..n_reads {
        let read = make_read(run_idx, i as u32 + 1, samples_per_read as u64);
        ids.push(read.read_id);
        let signal = synth_signal(samples_per_read, 0xA110 + i as u64);
        writer.add_read(read, &signal).expect("add_read");
    }
    writer.finish().expect("writer.finish");
    WrittenFile { read_ids: ids }
}

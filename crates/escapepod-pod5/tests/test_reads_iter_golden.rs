//! Golden for `Reader::reads()` (#467): the row iterator resolves the reads
//! table's columns once per batch, and must yield exactly what the per-row
//! extractor (`Reader::read_from_batch`, which looks every column up by name
//! and parses the end-reason string per row) yields — every field, every row.
//!
//! The reference is the by-name extractor rather than a stored expectation, so
//! it is independent of the code under test and needs no real file. The file is
//! multi-batch with a signal that spans several chunks per read, both pore-type
//! and end-reason dictionaries (every `EndReason` variant, listed in a different
//! order from the enum, referenced out of dictionary order), non-round floats
//! and NaN / -0.0 in the level fields.

use escapepod_pod5::{
    EndReason, PredefinedDictionaries, ReadData, Reader, RunInfoData, Uuid, Writer, WriterOptions,
};
use std::collections::HashMap;
use tempfile::TempDir;

const ALL_END_REASONS: [EndReason; 10] = [
    EndReason::Paused,
    EndReason::SignalPositive,
    EndReason::Unknown,
    EndReason::MuxChange,
    EndReason::ApiRequest,
    EndReason::SignalNegative,
    EndReason::DeviceDataError,
    EndReason::UnblockMuxChange,
    EndReason::AnalysisConfigChange,
    EndReason::DataServiceUnblockMuxChange,
];

fn run_info() -> RunInfoData {
    RunInfoData {
        acquisition_id: "acq-golden".to_string(),
        acquisition_start_time: 1_700_000_123_456,
        adc_max: 4095,
        adc_min: -4096,
        context_tags: HashMap::new(),
        experiment_name: "golden".to_string(),
        flow_cell_id: "FC".to_string(),
        flow_cell_product_code: "FLO-PRO114M".to_string(),
        protocol_name: "p".to_string(),
        protocol_run_id: "r".to_string(),
        protocol_start_time: 1_700_000_122_000,
        sample_id: "s".to_string(),
        sample_rate: 5000,
        sequencing_kit: "k".to_string(),
        sequencer_position: "1A".to_string(),
        sequencer_position_type: "promethion".to_string(),
        software: "golden".to_string(),
        system_name: "sys".to_string(),
        system_type: "promethion".to_string(),
        tracking_id: HashMap::new(),
    }
}

fn read(i: u32) -> ReadData {
    let f = i as f32;
    ReadData {
        read_id: Uuid::from_u128(0x1234_5678_9abc_def0_u128 * (i as u128 + 1)),
        read_number: 1000 + i * 3,
        start_sample: 1_234_567 + u64::from(i) * 977,
        // Spans the u16 range the V3-V5 writer accepts.
        channel: 1 + (i * 811) % 3000,
        well: 1 + (i % 4) as u8,
        pore_type: ["rna_004", "not_set", "dna_r10.4.1"][(i % 3) as usize].into(),
        calibration_offset: -220.25 + f * 0.125,
        calibration_scale: 0.19 + f * 1e-4,
        median_before: if i.is_multiple_of(7) { f32::NAN } else { 200.5 + f },
        end_reason: ALL_END_REASONS[(i as usize * 7) % ALL_END_REASONS.len()],
        end_reason_forced: i.is_multiple_of(3),
        run_info_index: 0,
        num_minknow_events: u64::from(i) * 11,
        tracked_scaling_scale: 1.25 + f,
        tracked_scaling_shift: -3.5 - f,
        predicted_scaling_scale: 1.5 + f,
        predicted_scaling_shift: -2.0 - f,
        num_reads_since_mux_change: i * 5,
        time_since_mux_change: 12.5 + f,
        num_samples: 0, // set by the writer from the signal
        open_pore_level: if i.is_multiple_of(5) { f32::NAN } else { 220.0 + f },
        expected_open_pore_level: if i.is_multiple_of(11) { -0.0 } else { 221.0 + f },
        selected_read_level: if i.is_multiple_of(13) { f32::NAN } else { 222.0 + f },
        signal_rows: Vec::new(),
    }
}

/// `Debug` shows a float's exact shortest round-trip form, NaN and signed zero
/// included, so equal strings are equal bits for every field.
fn dump(r: &ReadData) -> String {
    format!("{r:?}")
}

#[test]
fn reads_iterator_matches_the_by_name_extractor_for_every_field_of_every_row() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("golden.pod5");

    let opts = WriterOptions {
        read_batch_size: 7, // 50 reads => 8 reads-table batches
        signal_batch_size: 5,
        max_signal_chunk_size: 64, // several signal rows per read
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(vec![
                "not_set".into(),
                "dna_r10.4.1".into(),
                "rna_004".into(),
            ]),
            end_reasons: Some(ALL_END_REASONS.iter().map(|e| e.as_str().into()).collect()),
            run_infos: None,
        }),
        ..Default::default()
    };
    let mut writer = Writer::create(&path, opts).unwrap();
    writer.add_run_info(run_info()).unwrap();
    for i in 0..50u32 {
        let n = 40 + (i as usize * 37) % 400;
        let signal: Vec<i16> = (0..n).map(|s| (s as i16).wrapping_mul(3) - 100).collect();
        writer.add_read(read(i), &signal).unwrap();
    }
    writer.finish().unwrap();

    let reader = Reader::open(&path).unwrap();

    let mut want = Vec::new();
    for batch in reader.read_batches().unwrap() {
        let batch = batch.unwrap();
        for row in 0..batch.num_rows() {
            want.push(dump(&Reader::read_from_batch(&batch, row).unwrap()));
        }
    }
    let got: Vec<String> = reader.reads().unwrap().map(|r| dump(&r.unwrap())).collect();

    assert_eq!(want.len(), 50);
    assert!(
        reader.read_batches().unwrap().count() > 1,
        "fixture must be multi-batch"
    );
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "row {i}");
    }
    // The fixture exercises what it claims to.
    assert!(got.iter().any(|g| g.contains("NaN")));
    assert!(got.iter().any(|g| g.contains("-0.0")));
    for er in ALL_END_REASONS {
        assert!(
            got.iter()
                .any(|g| g.contains(&format!("end_reason: {er:?}"))),
            "{er:?} never read back"
        );
    }
    assert!(
        reader
            .reads()
            .unwrap()
            .any(|r| r.unwrap().signal_rows.len() > 1),
        "signal must span several rows"
    );
}

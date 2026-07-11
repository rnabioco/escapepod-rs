//! Field-fidelity round-trips ported from the upstream POD5 test harness
//! (`read_table_tests.cpp`, `run_info_table_tests.cpp`, `test_writer.py`).
//!
//! The existing `read_columns_matches_collect_all_reads` only proves the two
//! *read* paths agree on the same file — a writer column-swap or a unit/timezone
//! bug would pass because both paths read the same wrong bytes. These tests
//! instead assert every field against **known written input**, covering the
//! spots a reimplementation is most likely to get wrong:
//!
//! * RunInfo int64-millisecond timestamps (Arrow `Timestamp(Millisecond)` unit /
//!   timezone), signed `adc_min`, `sample_rate`, and the string/map columns.
//! * ReadData calibration/scaling floats and **NaN preservation** for the
//!   nullable-ish level fields (a coerce-NaN-to-0 bug is otherwise invisible).
//! * Arrow **dictionary index consistency across record batches** for the
//!   `pore_type` / `end_reason` columns, where a declared value is only first
//!   *referenced* in a later batch — the reader must decode each batch's indices
//!   against the single file-level dictionary the IPC format allows.

mod common;

use std::collections::HashMap;

use escapepod_pod5::{
    EndReason, PredefinedDictionaries, ReadData, Reader, RunInfoData, Uuid, Writer, WriterOptions,
};
use tempfile::TempDir;

/// f32 bit-compare so NaN == NaN (and -0.0 != 0.0), used for every float field.
fn bits_eq(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

/// A RunInfoData with a distinctive value in every field — nothing left at a
/// type default, so a dropped/mis-mapped column shows up on read-back.
fn distinctive_run_info() -> RunInfoData {
    RunInfoData {
        acquisition_id: "acq-fidelity-42".to_string(),
        // Deliberately non-round ms-since-epoch: 2023-11-14T22:15:23.456Z.
        acquisition_start_time: 1_700_000_123_456,
        adc_max: 4095,
        adc_min: -4096, // signed, negative
        context_tags: HashMap::from([
            ("experiment_type".to_string(), "rna".to_string()),
            ("basecall_config".to_string(), "sup".to_string()),
        ]),
        experiment_name: "exp-fidelity".to_string(),
        flow_cell_id: "FCID-XYZ-1".to_string(),
        flow_cell_product_code: "FLO-PRO114M".to_string(),
        protocol_name: "seq-protocol-fidelity".to_string(),
        protocol_run_id: "prun-abcdef".to_string(),
        // A different timestamp from acquisition, earlier by ~1s.
        protocol_start_time: 1_700_000_122_000,
        sample_id: "sample-fidelity".to_string(),
        sample_rate: 5000,
        sequencing_kit: "SQK-RNA004".to_string(),
        sequencer_position: "1A".to_string(),
        sequencer_position_type: "promethion".to_string(),
        software: "writer-under-test".to_string(),
        system_name: "sys-fidelity".to_string(),
        system_type: "promethion".to_string(),
        tracking_id: HashMap::from([("device_id".to_string(), "PC24B".to_string())]),
    }
}

/// Assert every RunInfoData field matches, one at a time (RunInfoData has no
/// `PartialEq`, and a per-field assert names the offender on failure).
fn assert_run_info_eq(got: &RunInfoData, want: &RunInfoData) {
    assert_eq!(got.acquisition_id, want.acquisition_id, "acquisition_id");
    assert_eq!(
        got.acquisition_start_time, want.acquisition_start_time,
        "acquisition_start_time (ms timestamp)"
    );
    assert_eq!(
        got.protocol_start_time, want.protocol_start_time,
        "protocol_start_time (ms timestamp)"
    );
    assert_eq!(got.adc_max, want.adc_max, "adc_max");
    assert_eq!(got.adc_min, want.adc_min, "adc_min (signed)");
    assert_eq!(got.sample_rate, want.sample_rate, "sample_rate");
    assert_eq!(got.context_tags, want.context_tags, "context_tags");
    assert_eq!(got.tracking_id, want.tracking_id, "tracking_id");
    assert_eq!(got.experiment_name, want.experiment_name, "experiment_name");
    assert_eq!(got.flow_cell_id, want.flow_cell_id, "flow_cell_id");
    assert_eq!(
        got.flow_cell_product_code, want.flow_cell_product_code,
        "flow_cell_product_code"
    );
    assert_eq!(got.protocol_name, want.protocol_name, "protocol_name");
    assert_eq!(got.protocol_run_id, want.protocol_run_id, "protocol_run_id");
    assert_eq!(got.sample_id, want.sample_id, "sample_id");
    assert_eq!(got.sequencing_kit, want.sequencing_kit, "sequencing_kit");
    assert_eq!(
        got.sequencer_position, want.sequencer_position,
        "sequencer_position"
    );
    assert_eq!(
        got.sequencer_position_type, want.sequencer_position_type,
        "sequencer_position_type"
    );
    assert_eq!(got.software, want.software, "software");
    assert_eq!(got.system_name, want.system_name, "system_name");
    assert_eq!(got.system_type, want.system_type, "system_type");
}

/// Gap #1 — every RunInfo column survives write → read, timestamps included.
#[test]
fn run_info_all_fields_round_trip() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("runinfo.pod5");
    let want = distinctive_run_info();

    let mut writer = Writer::create(&path, WriterOptions::default()).unwrap();
    writer.add_run_info(want.clone()).unwrap();
    // A run info is only persisted if at least one read references it.
    writer
        .add_read(
            make_full_read(0, Uuid::new_v4(), "pore-1", EndReason::SignalPositive),
            &[10, 11, 12],
        )
        .unwrap();
    writer.finish().unwrap();

    let reader = Reader::open(&path).unwrap();
    assert_eq!(reader.run_info_count(), 1);
    assert_run_info_eq(reader.get_run_info(0).unwrap(), &want);
}

/// Build a ReadData with a distinctive value in every field. `read_number`
/// seeds the distinctive integers; floats are set explicitly by the caller-
/// facing tests where NaN matters.
fn make_full_read(read_number: u32, id: Uuid, pore_type: &str, end_reason: EndReason) -> ReadData {
    ReadData {
        read_id: id,
        read_number,
        start_sample: 1_234_567 + read_number as u64,
        channel: 37 + read_number as u16,
        well: 2,
        pore_type: pore_type.into(),
        calibration_offset: 0.5,
        calibration_scale: 0.95,
        median_before: 200.5,
        end_reason,
        end_reason_forced: true,
        run_info_index: 0,
        num_minknow_events: 4242,
        tracked_scaling_scale: 1.25,
        tracked_scaling_shift: -3.5,
        predicted_scaling_scale: 1.5,
        predicted_scaling_shift: -2.0,
        num_reads_since_mux_change: 7,
        time_since_mux_change: 12.5,
        num_samples: 3,
        open_pore_level: 220.0,
        expected_open_pore_level: 221.0,
        selected_read_level: 222.0,
        signal_rows: Vec::new(),
    }
}

/// Gap #2 — every ReadData field survives write → read against known input,
/// including NaN in the level/median fields (upstream `test_read_copy` tolerates
/// NaN there; a NaN→0 coercion must not happen).
#[test]
fn read_all_fields_round_trip_including_nan() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("readfields.pod5");

    let id = Uuid::new_v4();
    let mut want = make_full_read(9, id, "pore-nan", EndReason::SignalNegative);
    // NaN in every field upstream marks as nullable/NaN-tolerant.
    want.median_before = f32::NAN;
    want.open_pore_level = f32::NAN;
    want.expected_open_pore_level = f32::NAN;
    want.selected_read_level = f32::NAN;
    let signal: Vec<i16> = vec![-5, 0, 5];

    let mut writer = Writer::create(&path, WriterOptions::default()).unwrap();
    writer.add_run_info(distinctive_run_info()).unwrap();
    writer.add_read(want.clone(), &signal).unwrap();
    writer.finish().unwrap();

    let reader = Reader::open(&path).unwrap();
    let reads = reader
        .reads()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(reads.len(), 1);
    let got = &reads[0];

    assert_eq!(got.read_id, want.read_id, "read_id");
    assert_eq!(got.read_number, want.read_number, "read_number");
    assert_eq!(got.start_sample, want.start_sample, "start_sample");
    assert_eq!(got.channel, want.channel, "channel");
    assert_eq!(got.well, want.well, "well");
    assert_eq!(got.pore_type.as_str(), want.pore_type.as_str(), "pore_type");
    assert!(
        bits_eq(got.calibration_offset, want.calibration_offset),
        "calibration_offset"
    );
    assert!(
        bits_eq(got.calibration_scale, want.calibration_scale),
        "calibration_scale"
    );
    assert!(
        bits_eq(got.median_before, want.median_before),
        "median_before (NaN) got={}",
        got.median_before
    );
    assert_eq!(
        got.end_reason.as_str(),
        want.end_reason.as_str(),
        "end_reason"
    );
    assert_eq!(
        got.end_reason_forced, want.end_reason_forced,
        "end_reason_forced"
    );
    assert_eq!(got.run_info_index, want.run_info_index, "run_info_index");
    assert_eq!(
        got.num_minknow_events, want.num_minknow_events,
        "num_minknow_events"
    );
    assert!(
        bits_eq(got.tracked_scaling_scale, want.tracked_scaling_scale),
        "tracked_scaling_scale"
    );
    assert!(
        bits_eq(got.tracked_scaling_shift, want.tracked_scaling_shift),
        "tracked_scaling_shift"
    );
    assert!(
        bits_eq(got.predicted_scaling_scale, want.predicted_scaling_scale),
        "predicted_scaling_scale"
    );
    assert!(
        bits_eq(got.predicted_scaling_shift, want.predicted_scaling_shift),
        "predicted_scaling_shift"
    );
    assert_eq!(
        got.num_reads_since_mux_change, want.num_reads_since_mux_change,
        "num_reads_since_mux_change"
    );
    assert!(
        bits_eq(got.time_since_mux_change, want.time_since_mux_change),
        "time_since_mux_change"
    );
    assert_eq!(got.num_samples, want.num_samples, "num_samples");
    assert!(
        bits_eq(got.open_pore_level, want.open_pore_level),
        "open_pore_level (NaN) got={}",
        got.open_pore_level
    );
    assert!(
        bits_eq(got.expected_open_pore_level, want.expected_open_pore_level),
        "expected_open_pore_level (NaN)"
    );
    assert!(
        bits_eq(got.selected_read_level, want.selected_read_level),
        "selected_read_level (NaN)"
    );

    // And the signal itself.
    assert_eq!(reader.get_signal(&got.signal_rows).unwrap(), signal);
}

/// Gap #3 — dictionary-encoded columns (`pore_type`, `end_reason`,
/// `run_info_index`) must decode correctly in **every** record batch, including
/// when a dictionary entry is first *referenced* in a later batch. Forces small
/// batches so the Arrow IPC stream carries several record batches that index
/// into the file-level dictionary.
///
/// Note on escapepod's writer: the Arrow IPC **file** format permits only one
/// dictionary per field across all batches (no replacement). The reads table
/// stores `pore_type`, `end_reason` *and* `run_info` as `Dictionary<Int16,Utf8>`.
/// `pore_type`/`end_reason` can be pinned up front with `PredefinedDictionaries`
/// (so every batch emits the identical dictionary); `run_info` cannot, so a
/// multi-batch write must reference a single run_info (its per-batch dictionary
/// is then identical everywhere). This is the writable multi-batch shape, and
/// the one where the reader's per-batch index-decoding gotcha lives: values
/// like `mux_change` are declared once but only *referenced* in later batches.
#[test]
fn dictionary_columns_consistent_across_batches() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("dicts.pod5");

    let opts = WriterOptions {
        read_batch_size: 2, // 6 reads => 3 record batches
        signal_batch_size: 2,
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(vec!["poreA".into(), "poreB".into()]),
            end_reasons: Some(vec![
                EndReason::SignalPositive.as_str().into(),
                EndReason::SignalNegative.as_str().into(),
                EndReason::MuxChange.as_str().into(),
            ]),
        }),
        ..Default::default()
    };
    let mut writer = Writer::create(&path, opts).unwrap();

    let mut run0 = distinctive_run_info();
    run0.acquisition_id = "acq-run-0".into();
    let ri0 = writer.add_run_info(run0).unwrap();

    // (pore_type, end_reason) per read; `mux_change` is only referenced in
    // batches 1 and 2 even though it's declared up front — the reader must
    // decode those batches' dictionary indices against the file-level dictionary.
    let plan = [
        ("poreA", EndReason::SignalPositive), // batch 0
        ("poreB", EndReason::SignalNegative), // batch 0
        ("poreB", EndReason::MuxChange),      // batch 1 — mux_change first used
        ("poreA", EndReason::SignalPositive), // batch 1
        ("poreA", EndReason::MuxChange),      // batch 2
        ("poreB", EndReason::SignalNegative), // batch 2
    ];

    let mut expected: HashMap<Uuid, (String, String)> = HashMap::new();
    for (i, (pore, end)) in plan.iter().enumerate() {
        let id = Uuid::new_v4();
        let mut read = make_full_read(i as u32, id, pore, *end);
        read.run_info_index = ri0;
        expected.insert(id, (pore.to_string(), end.as_str().to_string()));
        writer.add_read(read, &[1, 2, 3]).unwrap();
    }
    writer.finish().unwrap();

    let reader = Reader::open(&path).unwrap();
    assert!(
        reader.read_batch_count().unwrap() >= 3,
        "expected ≥3 batches"
    );
    let reads = reader
        .reads()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(reads.len(), 6);

    for read in &reads {
        let (pore, end) = &expected[&read.read_id];
        assert_eq!(
            read.pore_type.as_str(),
            pore,
            "pore_type for {}",
            read.read_id
        );
        assert_eq!(
            read.end_reason.as_str(),
            end,
            "end_reason for {}",
            read.read_id
        );
        assert_eq!(
            read.run_info_index, ri0,
            "run_info_index for {}",
            read.read_id
        );
    }
}

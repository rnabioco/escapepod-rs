//! Integration tests for the escapepod library.
//!
//! These tests verify end-to-end workflows including reading, writing,
//! merging, and filtering POD5 files.

use escapepod_signal::{EndReason, ReadData, Reader, RunInfoData, Writer, WriterOptions};
use std::collections::HashMap;
use tempfile::NamedTempFile;
use uuid::Uuid;

/// Create a test run info with the given acquisition ID.
fn create_test_run_info(acquisition_id: &str) -> RunInfoData {
    RunInfoData {
        acquisition_id: acquisition_id.to_string(),
        acquisition_start_time: 1609459200000,
        adc_max: 2047,
        adc_min: -2048,
        context_tags: HashMap::from([("experiment_type".to_string(), "genomic_dna".to_string())]),
        experiment_name: "test_experiment".to_string(),
        flow_cell_id: "FAK12345".to_string(),
        flow_cell_product_code: "FLO-MIN106".to_string(),
        protocol_name: "test_protocol".to_string(),
        protocol_run_id: "protocol_123".to_string(),
        protocol_start_time: 1609459200000,
        sample_id: "sample_001".to_string(),
        sample_rate: 4000,
        sequencing_kit: "SQK-LSK109".to_string(),
        sequencer_position: "MN00001".to_string(),
        sequencer_position_type: "minion".to_string(),
        software: "MinKNOW 21.0.0".to_string(),
        system_name: "test_system".to_string(),
        system_type: "minion".to_string(),
        tracking_id: HashMap::from([("run_id".to_string(), "run_456".to_string())]),
    }
}

/// Create a test read with the given parameters.
fn create_test_read(run_info_idx: u32, read_number: u32, num_samples: u64) -> ReadData {
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
        signal_rows: Vec::new(),
    }
}

/// Generate test signal data.
fn generate_test_signal(num_samples: usize, offset: i16) -> Vec<i16> {
    (0..num_samples)
        .map(|i| ((i as f64 * 0.1).sin() * 100.0) as i16 + 200 + offset)
        .collect()
}

#[test]
fn test_full_round_trip() -> escapepod_signal::Result<()> {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path();

    // Write a POD5 file
    let options = WriterOptions::default();
    let mut writer = Writer::create(path, options)?;

    let run_info = create_test_run_info("integration_test_run");
    let run_info_idx = writer.add_run_info(run_info)?;

    let num_reads = 5;
    let mut original_ids = Vec::new();
    for i in 0..num_reads {
        let read = create_test_read(run_info_idx, i + 1, 500);
        original_ids.push(read.read_id);
        let signal = generate_test_signal(500, i as i16 * 10);
        writer.add_read(read, &signal)?;
    }

    writer.finish()?;

    // Read back and verify
    let reader = Reader::open(path)?;

    assert_eq!(reader.run_info_count(), 1);
    let read_run_info = reader.get_run_info(0).expect("Should have run info");
    assert_eq!(read_run_info.acquisition_id, "integration_test_run");

    let read_count = reader.read_count()?;
    assert_eq!(read_count, num_reads as usize);

    let reads: Vec<_> = reader.reads()?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(reads.len(), num_reads as usize);

    // Verify all original IDs are present
    for original_id in &original_ids {
        assert!(
            reads.iter().any(|r| &r.read_id == original_id),
            "Read ID {:?} not found in read back data",
            original_id
        );
    }

    // Verify signal data can be retrieved
    for read in &reads {
        let signal = reader.get_signal(&read.signal_rows)?;
        assert_eq!(signal.len(), 500);
    }

    Ok(())
}

#[test]
fn test_read_with_context_tags_and_tracking_id() -> escapepod_signal::Result<()> {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path();

    // Write a POD5 file with context_tags and tracking_id
    let options = WriterOptions::default();
    let mut writer = Writer::create(path, options)?;

    let mut run_info = create_test_run_info("context_tracking_test");
    run_info
        .context_tags
        .insert("custom_key".to_string(), "custom_value".to_string());
    run_info
        .tracking_id
        .insert("tracking_key".to_string(), "tracking_value".to_string());

    let run_info_idx = writer.add_run_info(run_info)?;

    let read = create_test_read(run_info_idx, 1, 100);
    let signal = generate_test_signal(100, 0);
    writer.add_read(read, &signal)?;
    writer.finish()?;

    // Read back
    let reader = Reader::open(path)?;
    let read_run_info = reader.get_run_info(0).expect("Should have run info");

    // Verify context_tags are preserved
    assert_eq!(
        read_run_info.context_tags.get("experiment_type"),
        Some(&"genomic_dna".to_string())
    );
    assert_eq!(
        read_run_info.context_tags.get("custom_key"),
        Some(&"custom_value".to_string())
    );

    // Verify tracking_id is preserved
    assert_eq!(
        read_run_info.tracking_id.get("run_id"),
        Some(&"run_456".to_string())
    );
    assert_eq!(
        read_run_info.tracking_id.get("tracking_key"),
        Some(&"tracking_value".to_string())
    );

    Ok(())
}

#[test]
fn test_large_signal_chunking() -> escapepod_signal::Result<()> {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path();

    // Write with smaller chunk size to force chunking
    let options = WriterOptions {
        max_signal_chunk_size: 10000,
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(path, options)?;

    let run_info = create_test_run_info("chunking_test");
    let run_info_idx = writer.add_run_info(run_info)?;

    // 50k samples should create multiple chunks
    let signal = generate_test_signal(50000, 0);
    let read = create_test_read(run_info_idx, 1, 50000);
    writer.add_read(read, &signal)?;
    writer.finish()?;

    // Read back and verify
    let reader = Reader::open(path)?;
    let reads: Vec<_> = reader.reads()?.collect::<Result<Vec<_>, _>>()?;

    assert_eq!(reads.len(), 1);
    let read_back = &reads[0];

    // Should have multiple signal chunks
    assert!(
        read_back.signal_rows.len() > 1,
        "Expected multiple signal chunks"
    );

    // Verify signal data is complete
    let signal_back = reader.get_signal(&read_back.signal_rows)?;
    assert_eq!(signal_back.len(), 50000);
    assert_eq!(signal_back, signal);

    Ok(())
}

#[test]
fn test_all_end_reasons() -> escapepod_signal::Result<()> {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path();

    let options = WriterOptions::default();
    let mut writer = Writer::create(path, options)?;

    let run_info = create_test_run_info("end_reason_test");
    let run_info_idx = writer.add_run_info(run_info)?;

    let end_reasons = vec![
        EndReason::Unknown,
        EndReason::MuxChange,
        EndReason::UnblockMuxChange,
        EndReason::DataServiceUnblockMuxChange,
        EndReason::SignalPositive,
        EndReason::SignalNegative,
        EndReason::ApiRequest,
        EndReason::DeviceDataError,
        EndReason::AnalysisConfigChange,
        EndReason::Paused,
    ];

    for (i, end_reason) in end_reasons.iter().enumerate() {
        let mut read = create_test_read(run_info_idx, i as u32 + 1, 100);
        read.end_reason = *end_reason;
        let signal = generate_test_signal(100, i as i16);
        writer.add_read(read, &signal)?;
    }

    writer.finish()?;

    // Read back
    let reader = Reader::open(path)?;
    let reads: Vec<_> = reader.reads()?.collect::<Result<Vec<_>, _>>()?;

    assert_eq!(reads.len(), end_reasons.len());

    // Verify all end reasons are preserved
    for end_reason in &end_reasons {
        assert!(
            reads.iter().any(|r| r.end_reason == *end_reason),
            "End reason {:?} not found",
            end_reason
        );
    }

    Ok(())
}

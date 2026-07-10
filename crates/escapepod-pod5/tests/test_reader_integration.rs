//! Integration tests for Reader and SignalExtractor.
//!
//! Covers the memory-mapped read path end-to-end: open → iterate reads →
//! decompress signal via both the sequential `get_signal` API and the
//! thread-safe `SignalExtractor` used by rayon-parallel consumers.

mod common;

use std::collections::HashSet;

use escapepod_pod5::{Reader, Uuid};
use rayon::prelude::*;
use tempfile::TempDir;

use common::{synth_signal, write_fixture};

const N_READS: usize = 25;
const SAMPLES_PER_READ: usize = 1_200;

#[test]
fn reader_iterates_all_reads() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("reader.pod5");
    let fx = write_fixture(&path, "reader_itest", N_READS, SAMPLES_PER_READ);

    let reader = Reader::open(&path).expect("open");
    assert_eq!(reader.read_count().unwrap(), N_READS);

    let reads: Vec<_> = reader
        .reads()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(reads.len(), N_READS);

    let ids: HashSet<Uuid> = reads.iter().map(|r| r.read_id).collect();
    let expected: HashSet<Uuid> = fx.read_ids.iter().copied().collect();
    assert_eq!(ids, expected);

    for read in &reads {
        assert_eq!(read.num_samples, SAMPLES_PER_READ as u64);
        assert!(!read.signal_rows.is_empty());
    }
}

#[test]
fn reader_extracts_signal_correctly() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("reader.pod5");
    write_fixture(&path, "reader_sig", 5, SAMPLES_PER_READ);

    let reader = Reader::open(&path).unwrap();
    for (idx, read_result) in reader.reads().unwrap().enumerate() {
        let read = read_result.unwrap();
        let signal = reader.get_signal(&read.signal_rows).unwrap();
        assert_eq!(signal.len(), SAMPLES_PER_READ);
        // Our fixture uses deterministic seeds 0xA110 + i; reproduce and compare.
        let expected = synth_signal(SAMPLES_PER_READ, 0xA110 + idx as u64);
        assert_eq!(signal, expected, "signal mismatch for read {idx}");
    }
}

#[test]
fn signal_extractor_is_thread_safe() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("parallel.pod5");
    write_fixture(&path, "reader_par", 50, 800);

    let reader = Reader::open(&path).unwrap();
    let reads: Vec<_> = reader
        .reads()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let extractor = reader.signal_extractor().unwrap();

    let sample_sums: Vec<i64> = reads
        .par_iter()
        .map(|r| {
            let sig = extractor.get_signal(&r.signal_rows).unwrap();
            assert_eq!(sig.len(), 800);
            sig.iter().map(|&s| s as i64).sum::<i64>()
        })
        .collect();

    // Sequentially compute the same sums and compare — catches any data race.
    let sequential: Vec<i64> = reads
        .iter()
        .map(|r| {
            let sig = reader.get_signal(&r.signal_rows).unwrap();
            sig.iter().map(|&s| s as i64).sum::<i64>()
        })
        .collect();
    assert_eq!(sample_sums, sequential);
}

#[test]
fn reader_reads_by_ids_fast_path() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("by_ids.pod5");
    let fx = write_fixture(&path, "reader_by_ids", 40, 500);

    let reader = Reader::open(&path).unwrap();
    let targets: HashSet<Uuid> = fx.read_ids.iter().step_by(3).copied().collect();
    let matched = reader.reads_by_ids(&targets).unwrap();
    assert_eq!(matched.len(), targets.len());
    let matched_ids: HashSet<Uuid> = matched.iter().map(|r| r.read_id).collect();
    assert_eq!(matched_ids, targets);
}

#[test]
fn reader_reports_metadata() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("meta.pod5");
    write_fixture(&path, "reader_meta", 3, 300);

    let reader = Reader::open(&path).unwrap();
    assert_eq!(reader.run_info_count(), 1);
    assert_eq!(
        reader.get_run_info(0).unwrap().acquisition_id,
        "reader_meta"
    );
    assert_eq!(reader.read_batch_count().unwrap(), 1);
    assert!(!reader.file_identifier().is_empty());
    assert!(reader.software().contains("escapepod"));
}

#[test]
fn read_columns_matches_collect_all_reads() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("columns.pod5");
    write_fixture(&path, "reader_columns", N_READS, SAMPLES_PER_READ);

    let reader = Reader::open(&path).unwrap();
    let reads = reader.collect_all_reads().unwrap();
    let cols = reader.read_columns().unwrap();

    assert_eq!(cols.len(), reads.len());
    // Compare floats by bit pattern so NaN (common in scaling fields) counts as
    // equal to itself — the columnar path must be byte-identical to `read()`.
    let bits = |x: f32| x.to_bits();
    for (i, r) in reads.iter().enumerate() {
        assert_eq!(cols.read_id[i], r.read_id, "read_id[{i}]");
        assert_eq!(cols.read_number[i], r.read_number, "read_number[{i}]");
        assert_eq!(cols.start_sample[i], r.start_sample, "start_sample[{i}]");
        assert_eq!(cols.channel[i], r.channel, "channel[{i}]");
        assert_eq!(cols.well[i], r.well, "well[{i}]");
        assert_eq!(
            cols.pore_type[i].as_str(),
            r.pore_type.as_str(),
            "pore_type[{i}]"
        );
        assert_eq!(bits(cols.calibration_offset[i]), bits(r.calibration_offset));
        assert_eq!(bits(cols.calibration_scale[i]), bits(r.calibration_scale));
        assert_eq!(bits(cols.median_before[i]), bits(r.median_before));
        assert_eq!(
            cols.end_reason[i].as_str(),
            r.end_reason.as_str(),
            "end_reason[{i}]"
        );
        assert_eq!(cols.end_reason_forced[i], r.end_reason_forced);
        assert_eq!(cols.run_info_index[i], r.run_info_index);
        assert_eq!(cols.num_minknow_events[i], r.num_minknow_events);
        assert_eq!(cols.num_samples[i], r.num_samples);
        assert_eq!(
            bits(cols.tracked_scaling_scale[i]),
            bits(r.tracked_scaling_scale)
        );
        assert_eq!(
            bits(cols.tracked_scaling_shift[i]),
            bits(r.tracked_scaling_shift)
        );
        assert_eq!(
            bits(cols.predicted_scaling_scale[i]),
            bits(r.predicted_scaling_scale)
        );
        assert_eq!(
            bits(cols.predicted_scaling_shift[i]),
            bits(r.predicted_scaling_shift)
        );
        assert_eq!(
            cols.num_reads_since_mux_change[i],
            r.num_reads_since_mux_change
        );
        assert_eq!(
            bits(cols.time_since_mux_change[i]),
            bits(r.time_since_mux_change)
        );
        assert_eq!(bits(cols.open_pore_level[i]), bits(r.open_pore_level));
        assert_eq!(
            bits(cols.expected_open_pore_level[i]),
            bits(r.expected_open_pore_level)
        );
        assert_eq!(
            bits(cols.selected_read_level[i]),
            bits(r.selected_read_level)
        );
    }
}

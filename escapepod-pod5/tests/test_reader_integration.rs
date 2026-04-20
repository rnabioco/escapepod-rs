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

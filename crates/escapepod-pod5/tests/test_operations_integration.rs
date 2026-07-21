//! End-to-end tests for the file-level operations (`repack_files`,
//! `filter_files`, `subset_files`), ported from the upstream POD5 tool tests
//! (`test_repack.py`, `test_filter.py`, `test_subset.py`).
//!
//! Before this, only `FilterCriteria::matches` and the CSV/UUID parsers had unit
//! tests — the actual multi-file pipeline functions (which slice compressed
//! signal straight out of the mmap and reassemble it) were never exercised
//! end-to-end. These check the load-bearing invariants: signal is preserved
//! without recompression, the exact requested read set is produced, and
//! `subset_files` partitions reads (across multiple inputs) into the right
//! output with no duplicates.

mod common;

use std::collections::{HashMap, HashSet};

use escapepod_pod5::operations::{FilterOptions, filter_files, subset_files};
use escapepod_pod5::{Reader, RepackOptions, Uuid, Writer, WriterOptions, repack_files};
use tempfile::TempDir;

use common::{make_read, make_run_info};

fn filter_opts() -> FilterOptions {
    FilterOptions {
        signal_batch_size: 100,
        read_batch_size: 1000,
        ..Default::default()
    }
}

/// Deterministic signal so we can compare exact values after an operation.
fn sig(seed: i16, n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| seed.wrapping_add(i as i16).wrapping_mul(7))
        .collect()
}

/// Write a file of `n` reads with distinct signals; return id -> signal.
fn write_file(path: &std::path::Path, acq: &str, n: usize) -> HashMap<Uuid, Vec<i16>> {
    let mut writer = Writer::create(path, WriterOptions::default()).unwrap();
    let run = writer.add_run_info(make_run_info(acq)).unwrap();
    let mut map = HashMap::new();
    for i in 0..n {
        let read = make_read(run, i as u32 + 1, (100 + i) as u64);
        let id = read.read_id;
        let signal = sig(i as i16 * 3, 100 + i);
        writer.add_read(read, &signal).unwrap();
        map.insert(id, signal);
    }
    writer.finish().unwrap();
    map
}

fn read_signals(path: &std::path::Path) -> HashMap<Uuid, Vec<i16>> {
    let reader = Reader::open(path).unwrap();
    let mut map = HashMap::new();
    for r in reader.reads().unwrap() {
        let r = r.unwrap();
        let s = reader.get_signal(&r.signal_rows).unwrap();
        map.insert(r.read_id, s);
    }
    map
}

/// Repack must preserve every read and its signal exactly (block-level copy, no
/// recompression). cf. `test_repack.py::test_works`.
#[test]
fn repack_preserves_all_reads_and_signal() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("in.pod5");
    let output = tmp.path().join("out.pod5");
    let original = write_file(&input, "acq_repack", 12);

    let opts = RepackOptions {
        signal_batch_size: 100,
        read_batch_size: 1000,
        force: false,
        ..Default::default()
    };
    let result = repack_files(&[(&input, &output)], opts, None);
    assert_eq!(result.files_processed, 1);
    assert_eq!(result.files_skipped, 0);
    assert_eq!(result.total_reads, original.len() as u64);

    let repacked = read_signals(&output);
    assert_eq!(repacked, original, "repack changed reads or signal");
}

/// Filtering to a subset of ids must yield exactly those reads, signal intact.
/// cf. `test_filter.py::test_all_in_out`.
#[test]
fn filter_selects_exact_read_subset() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("in.pod5");
    let output = tmp.path().join("out.pod5");
    let original = write_file(&input, "acq_filter", 20);

    // Keep every third read.
    let keep: HashSet<Uuid> = original
        .keys()
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, id)| id)
        .collect();

    let result = filter_files(&[&input], &output, &keep, filter_opts(), None).unwrap();
    assert_eq!(result.matched_reads, keep.len() as u64);

    let got = read_signals(&output);
    let got_ids: HashSet<Uuid> = got.keys().copied().collect();
    assert_eq!(got_ids, keep, "filter produced the wrong read set");
    for id in &keep {
        assert_eq!(got[id], original[id], "signal mismatch for kept read {id}");
    }
}

/// `subset_files` must partition reads into the right output files, assembling a
/// group whose reads span multiple inputs, with no duplicate reads.
/// cf. `test_subset.py::test_subset_base`.
#[test]
fn subset_partitions_reads_across_inputs() {
    let tmp = TempDir::new().unwrap();
    let in_a = tmp.path().join("a.pod5");
    let in_b = tmp.path().join("b.pod5");
    let out_dir = tmp.path().join("out");
    std::fs::create_dir(&out_dir).unwrap();

    let a = write_file(&in_a, "acq_a", 6);
    let b = write_file(&in_b, "acq_b", 6);

    // Assign reads to two groups, interleaved and spanning both inputs so each
    // output file is assembled from A and B.
    let mut read_to_group: HashMap<Uuid, String> = HashMap::new();
    let mut want: HashMap<&str, HashSet<Uuid>> = HashMap::new();
    for (src_idx, src) in [&a, &b].iter().enumerate() {
        for (i, id) in src.keys().enumerate() {
            let group = if (i + src_idx) % 2 == 0 {
                "even.pod5"
            } else {
                "odd.pod5"
            };
            read_to_group.insert(*id, group.to_string());
            want.entry(group).or_default().insert(*id);
        }
    }

    let results = subset_files(&[&in_a, &in_b], &read_to_group, &out_dir, filter_opts()).unwrap();
    assert!(results.failures.is_empty(), "{:?}", results.failures);
    let counts: HashMap<String, u64> = results.groups.into_iter().collect();

    for (group, ids) in &want {
        assert_eq!(
            counts.get(*group).copied().unwrap_or(0),
            ids.len() as u64,
            "wrong reads_written for {group}"
        );
        let got = read_signals(&out_dir.join(group));
        let got_ids: HashSet<Uuid> = got.keys().copied().collect();
        assert_eq!(&got_ids, ids, "wrong read set in {group}");
        assert_eq!(got.len(), ids.len(), "duplicate reads in {group}");
        // Signal preserved regardless of which input the read came from.
        for id in ids {
            let expected = a.get(id).or_else(|| b.get(id)).unwrap();
            assert_eq!(&got[id], expected, "signal mismatch for {id} in {group}");
        }
    }
}

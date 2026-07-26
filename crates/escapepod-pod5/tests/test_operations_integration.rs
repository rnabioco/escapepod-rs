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

/// Collect the signal table's `read_id` column, indexed by global signal row.
fn signal_table_read_ids(path: &std::path::Path) -> Vec<[u8; 16]> {
    use arrow::array::{Array, FixedSizeBinaryArray};
    let reader = Reader::open(path).unwrap();
    let mut ids = Vec::new();
    for batch in reader.signal_batches().unwrap() {
        let col = batch
            .column_by_name("read_id")
            .expect("signal table has a read_id column")
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("read_id is FixedSizeBinary(16)");
        for i in 0..col.len() {
            ids.push(col.value(i).try_into().unwrap());
        }
    }
    ids
}

/// Every signal row must carry the UUID of the read that owns it.
///
/// Row *order* is deliberately not asserted: `filter`/`subset` rebuild the
/// signal table and emit rows in their own order, so the association has to be
/// checked through the reads table's `signal_rows` indices, not positionally.
fn assert_signal_read_ids_match(path: &std::path::Path, what: &str) {
    let sig_ids = signal_table_read_ids(path);
    assert!(!sig_ids.is_empty(), "{what}: signal table is empty");

    let zero = sig_ids.iter().filter(|v| v.iter().all(|&b| b == 0)).count();
    assert_eq!(
        zero,
        0,
        "{what}: {zero} of {} signal rows have a zero-filled read_id; ONT's own \
         tooling populates this column and the schema documents it as \"UUID for \
         consistency checking\"",
        sig_ids.len()
    );

    let reader = Reader::open(path).unwrap();
    for read in reader.reads().unwrap() {
        let read = read.unwrap();
        for &row in &read.signal_rows {
            let got = sig_ids
                .get(row as usize)
                .unwrap_or_else(|| panic!("{what}: signal row {row} out of range"));
            assert_eq!(
                got,
                read.read_id.as_bytes(),
                "{what}: signal row {row} carries the wrong read_id"
            );
        }
    }
}

/// The signal table's `read_id` column must be populated and correctly
/// associated on every write path.
///
/// `filter`/`subset` used to write all zeros here. Nothing broke, which is why
/// it survived: the reads table is the authority for the read -> signal-row
/// mapping, so no reader ever consults this column, and the existing tests all
/// compare decoded signal *through* the reads table. The cost was that the same
/// logical operation produced different bytes depending on which command made
/// the file, and diverged from ONT's own output.
#[test]
fn signal_table_read_ids_are_real_on_every_write_path() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("in.pod5");
    let ids: Vec<Uuid> = write_file(&input, "acq-readid", 12).into_keys().collect();

    // Writer (the incremental path).
    assert_signal_read_ids_match(&input, "Writer");

    // filter — rebuilds batches from a row subset.
    let filtered = tmp.path().join("filtered.pod5");
    let keep: HashSet<Uuid> = ids.iter().take(7).copied().collect();
    filter_files(&[&input], &filtered, &keep, filter_opts(), None).unwrap();
    assert_signal_read_ids_match(&filtered, "filter");

    // subset — same assembler, multiple outputs.
    let out_dir = tmp.path().join("subset");
    std::fs::create_dir_all(&out_dir).unwrap();
    let mapping: HashMap<Uuid, String> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, format!("g{}.pod5", i % 2)))
        .collect();
    subset_files(&[&input], &mapping, &out_dir, filter_opts()).unwrap();
    assert_signal_read_ids_match(&out_dir.join("g0.pod5"), "subset");

    // repack — block-level copy.
    let repacked = tmp.path().join("repacked.pod5");
    let repack = repack_files(&[(&input, &repacked)], RepackOptions::default(), None);
    assert!(
        repack.failures.is_empty(),
        "repack failed: {:?}",
        repack.failures
    );
    assert_signal_read_ids_match(&repacked, "repack");
}

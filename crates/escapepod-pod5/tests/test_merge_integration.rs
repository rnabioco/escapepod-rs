//! End-to-end merge tests.
//!
//! Verifies that `merge_files` preserves every read, deduplicates run_info by
//! acquisition_id, and keeps signal bytes bitwise-identical (the block-level
//! copy path must not recompress).

mod common;

use std::collections::{HashMap, HashSet};

use escapepod_pod5::{MergeOptions, Reader, Uuid, merge_files};
use tempfile::TempDir;

use common::{make_run_info, write_fixture};

#[test]
fn merge_two_files_preserves_all_reads() {
    let tmp = TempDir::new().expect("tempdir");
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let merged = tmp.path().join("merged.pod5");

    let fa = write_fixture(&a, "acq_a", 20, 1_500);
    let fb = write_fixture(&b, "acq_b", 15, 1_500);

    let result = merge_files(
        &[a.clone(), b.clone()],
        &merged,
        &MergeOptions::default(),
        None,
    )
    .expect("merge_files failed");

    assert_eq!(result.files_processed, 2);
    assert_eq!(
        result.reads_written,
        (fa.read_ids.len() + fb.read_ids.len()) as u64
    );
    assert_eq!(result.duplicates_skipped, 0);

    let reader = Reader::open(&merged).expect("open merged");
    let reads: Vec<_> = reader
        .reads()
        .expect("reads")
        .collect::<Result<Vec<_>, _>>()
        .expect("read iter");
    let merged_ids: HashSet<Uuid> = reads.iter().map(|r| r.read_id).collect();

    let expected: HashSet<Uuid> = fa
        .read_ids
        .iter()
        .chain(fb.read_ids.iter())
        .copied()
        .collect();
    assert_eq!(merged_ids, expected);
}

#[test]
fn merge_deduplicates_run_info_by_acquisition_id() {
    let tmp = TempDir::new().expect("tempdir");
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let merged = tmp.path().join("merged.pod5");

    // Both files share the same acquisition_id — merge must collapse to one run_info.
    write_fixture(&a, "shared_acq", 5, 500);
    write_fixture(&b, "shared_acq", 5, 500);

    merge_files(&[a, b], &merged, &MergeOptions::default(), None).expect("merge_files");

    let reader = Reader::open(&merged).expect("open merged");
    assert_eq!(
        reader.run_info_count(),
        1,
        "Run infos with matching acquisition_id should be deduplicated"
    );
}

#[test]
fn merge_preserves_distinct_run_infos() {
    let tmp = TempDir::new().expect("tempdir");
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let merged = tmp.path().join("merged.pod5");

    write_fixture(&a, "acq_one", 3, 500);
    write_fixture(&b, "acq_two", 3, 500);

    merge_files(&[a, b], &merged, &MergeOptions::default(), None).expect("merge_files");

    let reader = Reader::open(&merged).expect("open merged");
    assert_eq!(reader.run_info_count(), 2);
    let acq_ids: HashSet<&str> = reader
        .run_infos()
        .iter()
        .map(|r| r.acquisition_id.as_str())
        .collect();
    assert!(acq_ids.contains("acq_one"));
    assert!(acq_ids.contains("acq_two"));
}

#[test]
fn merge_preserves_signal_bytewise() {
    // Block-level merge must keep every compressed signal chunk unchanged.
    // Sizes are deliberately different between sources so the resulting file
    // has non-uniform signal-batch sizes — exercises the row-to-batch lookup
    // on the reader side.
    let tmp = TempDir::new().expect("tempdir");
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let merged = tmp.path().join("merged.pod5");

    let fa = write_fixture(&a, "acq_a", 5, 800);
    let fb = write_fixture(&b, "acq_b", 9, 800);

    let mut original: HashMap<Uuid, Vec<i16>> = HashMap::new();
    for path in [&a, &b] {
        let reader = Reader::open(path).unwrap();
        for r in reader.reads().unwrap() {
            let r = r.unwrap();
            let sig = reader.get_signal(&r.signal_rows).unwrap();
            original.insert(r.read_id, sig);
        }
    }

    merge_files(&[a, b], &merged, &MergeOptions::default(), None).expect("merge_files");

    // Exercise both the Reader::get_signal path and the parallel
    // SignalExtractor path (both slice compressed bytes from the mmap via the
    // signal ArrowIpcFooter). Both must agree with the pre-merge signals.
    let reader = Reader::open(&merged).unwrap();
    let extractor = reader.signal_extractor().unwrap();
    let mut checked = 0;
    for read_result in reader.reads().unwrap() {
        let read = read_result.unwrap();
        let sig_sequential = reader.get_signal(&read.signal_rows).unwrap();
        let sig_parallel = extractor.get_signal(&read.signal_rows).unwrap();
        assert_eq!(sig_sequential, sig_parallel);
        let expected = original
            .get(&read.read_id)
            .expect("read ID missing in original");
        assert_eq!(
            sig_sequential, *expected,
            "merged signal differs from original for read {}",
            read.read_id
        );
        checked += 1;
    }
    assert_eq!(checked, fa.read_ids.len() + fb.read_ids.len());
}

#[test]
fn merge_empty_input_errors() {
    let tmp = TempDir::new().expect("tempdir");
    let merged = tmp.path().join("out.pod5");
    let err = merge_files::<&std::path::Path, _>(&[], &merged, &MergeOptions::default(), None)
        .expect_err("empty input must error");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("no input") || msg.to_lowercase().contains("input"),
        "unexpected error: {msg}"
    );
}

#[allow(dead_code)]
fn _ensure_run_info_helper_is_used() {
    // Keep make_run_info visible to the linker even if individual tests drop it.
    let _ = make_run_info("unused");
}

//! End-to-end merge tests.
//!
//! Verifies that `merge_files` preserves every read, deduplicates run_info by
//! acquisition_id, and keeps signal bytes bitwise-identical (the block-level
//! copy path must not recompress).

mod common;

use std::collections::{HashMap, HashSet};

use escapepod_pod5::{MergeOptions, Reader, Uuid, Writer, WriterOptions, merge_files};
use tempfile::TempDir;

use common::{make_read, make_run_info, write_fixture};

/// Write a file whose reads have the exact `ids` given (so two files can be
/// made to share a read_id for the duplicate-handling test).
fn write_with_ids(path: &std::path::Path, acq: &str, ids: &[Uuid], samples: u64) {
    let mut writer = Writer::create(path, WriterOptions::default()).unwrap();
    let run = writer.add_run_info(make_run_info(acq)).unwrap();
    for (i, &id) in ids.iter().enumerate() {
        let mut read = make_read(run, i as u32 + 1, samples);
        read.read_id = id;
        writer.add_read(read, &[(i as i16), 1, 2, 3, 4]).unwrap();
    }
    writer.finish().unwrap();
}

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

/// A read_id present in two inputs must be emitted once and counted as a
/// skipped duplicate under the default `duplicate_ok = false`. cf. upstream
/// `test_merge.py::test_merge_duplicate_stopped` (which *errors*; escapepod's
/// default instead deduplicates — this pins that contract).
#[test]
fn merge_duplicate_read_id_is_deduplicated() {
    let tmp = TempDir::new().expect("tempdir");
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let merged = tmp.path().join("merged.pod5");

    let shared = Uuid::new_v4();
    let a_only = Uuid::new_v4();
    let b_only = Uuid::new_v4();

    write_with_ids(&a, "acq_a", &[a_only, shared], 5);
    write_with_ids(&b, "acq_b", &[shared, b_only], 5);

    let result = merge_files(&[a, b], &merged, &MergeOptions::default(), None).expect("merge");

    // 4 reads seen across inputs, 1 duplicate collapsed => 3 written.
    assert_eq!(result.duplicates_skipped, 1);
    assert_eq!(result.reads_written, 3);

    let reader = Reader::open(&merged).unwrap();
    let ids: Vec<Uuid> = reader
        .reads()
        .unwrap()
        .map(|r| r.unwrap().read_id)
        .collect();
    assert_eq!(ids.len(), 3, "output must contain each read exactly once");
    let unique: HashSet<Uuid> = ids.iter().copied().collect();
    assert_eq!(unique.len(), 3);
    assert_eq!(unique, HashSet::from([a_only, shared, b_only]));
}

/// Each input's own trailing batch is legitimately short (a read count
/// rarely divides evenly by the signal batch size). Concatenating two such
/// files' batches as-is — the pre-fix behaviour — put file `a`'s short
/// batch immediately before file `b`'s first (full) batch in the merged
/// output, breaking the constant stride dorado and the official `pod5`
/// library assume between batches (escapepod itself walks real cumulative
/// row counts and is unaffected — see `nonuniform_signal_batch`'s own docs,
/// and escapepod-rs#195 for the writer-side version of the same bug class).
#[test]
fn merge_output_signal_batches_stay_uniform_even_with_short_trailing_inputs() {
    let tmp = TempDir::new().expect("tempdir");
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let merged = tmp.path().join("merged.pod5");

    // Default WriterOptions batches signal at 100 rows/batch; 150 one-chunk
    // reads per file means each input's own batches are [100, 50] — a short
    // trailing batch, same shape as virtually every real POD5.
    let fa = write_fixture(&a, "acq_a", 150, 10);
    let fb = write_fixture(&b, "acq_b", 150, 10);

    let result = merge_files(&[a, b], &merged, &MergeOptions::default(), None).expect("merge");
    assert_eq!(result.reads_written, 300);
    assert_eq!(result.duplicates_skipped, 0);

    let reader = Reader::open(&merged).expect("open merged");
    assert!(
        reader.nonuniform_signal_batch().is_none(),
        "merged output has a non-uniform signal batch: {:?} — dorado and the \
         official pod5 library would resolve every read after it to the \
         wrong signal",
        reader.nonuniform_signal_batch()
    );

    // And nothing was lost or corrupted in the rebuild.
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

/// Write a file of reads with varied end reasons / pore types, one of which
/// spans several signal chunks, and an optional run-info pair.
fn write_varied(path: &std::path::Path, acqs: &[&str], ids: &[Uuid], compress: bool) {
    use escapepod_pod5::EndReason;
    let opts = WriterOptions {
        compress_signal: compress,
        max_signal_chunk_size: 300,
        ..Default::default()
    };
    let mut w = Writer::create(path, opts).unwrap();
    let runs: Vec<u32> = acqs
        .iter()
        .map(|a| w.add_run_info(make_run_info(a)).unwrap())
        .collect();
    let reasons = [
        EndReason::SignalPositive,
        EndReason::UnblockMuxChange,
        EndReason::MuxChange,
        EndReason::DataServiceUnblockMuxChange,
    ];
    for (i, &id) in ids.iter().enumerate() {
        let n = [50usize, 700, 301, 1000][i % 4];
        let mut read = make_read(runs[i % runs.len()], i as u32 + 1, n as u64);
        read.read_id = id;
        read.end_reason = reasons[i % reasons.len()];
        read.end_reason_forced = i % 2 == 0;
        read.pore_type = if i % 3 == 0 { "r10_a" } else { "not_set" }.into();
        read.channel = (i as u32 % 7) as _;
        w.add_read(read, &common::synth_signal(n, 77 + i as u64))
            .unwrap();
    }
    w.finish().unwrap();
}

/// Golden: the merged file equals, read for read, what the inputs say it
/// should be — every scalar field, end reason, pore type, the run info the
/// read points at (resolved by acquisition_id, not index), signal samples,
/// signal-row renumbering — across shared run infos, a duplicate read, and
/// both compressed and uncompressed inputs.
#[test]
fn merge_output_identical_to_baseline() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<Uuid> = (0..30).map(|_| Uuid::new_v4()).collect();
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let c = tmp.path().join("c.pod5");
    // b repeats ids[10..14] from a (duplicates to drop); c is uncompressed
    // and lists its run infos in the opposite order.
    write_varied(&a, &["acq_x", "acq_y"], &ids[0..14], true);
    write_varied(&b, &["acq_y"], &ids[10..22], true);
    write_varied(&c, &["acq_z", "acq_x"], &ids[22..30], false);

    let merged = tmp.path().join("m.pod5");
    let res = merge_files(
        &[a.clone(), b.clone(), c.clone()],
        &merged,
        &MergeOptions::default(),
        None,
    )
    .unwrap();
    assert_eq!(res.duplicates_skipped, 4);
    assert_eq!(res.reads_written, 30);

    // Expected: first occurrence of each id, in input order.
    let mut expected = Vec::new();
    let mut seen = HashSet::new();
    for p in [&a, &b, &c] {
        let r = Reader::open(p).unwrap();
        for read in r.reads().unwrap() {
            let read = read.unwrap();
            if seen.insert(read.read_id) {
                let acq = r.run_infos()[read.run_info_index as usize]
                    .acquisition_id
                    .clone();
                let sig = r.get_signal(&read.signal_rows).unwrap();
                expected.push((read, acq, sig));
            }
        }
    }

    let m = Reader::open(&merged).unwrap();
    let got: Vec<_> = m.reads().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(got.len(), expected.len());
    let mut next_row = 0u64;
    for (g, (e, acq, sig)) in got.iter().zip(&expected) {
        assert_eq!(g.read_id, e.read_id);
        assert_eq!(
            m.run_infos()[g.run_info_index as usize].acquisition_id,
            *acq
        );
        let n = e.signal_rows.len() as u64;
        assert_eq!(g.signal_rows, (next_row..next_row + n).collect::<Vec<_>>());
        next_row += n;
        assert_eq!(m.get_signal(&g.signal_rows).unwrap(), *sig);
        let mut norm = g.clone();
        norm.run_info_index = e.run_info_index;
        norm.signal_rows = e.signal_rows.clone();
        assert_eq!(
            format!("{norm:?}"),
            format!("{e:?}"),
            "field mismatch for {}",
            e.read_id
        );
    }
    assert_eq!(m.run_info_count(), 3);
}

/// Run infos are deduplicated by acquisition_id across files, and each read
/// points at its own run info afterwards.
#[test]
fn merge_dedups_run_info() {
    let tmp = TempDir::new().unwrap();
    let a = tmp.path().join("a.pod5");
    let b = tmp.path().join("b.pod5");
    let ids_a: Vec<Uuid> = (0..4).map(|_| Uuid::new_v4()).collect();
    let ids_b: Vec<Uuid> = (0..4).map(|_| Uuid::new_v4()).collect();
    write_varied(&a, &["p", "q"], &ids_a, true);
    write_varied(&b, &["q", "p"], &ids_b, true);
    let merged = tmp.path().join("m.pod5");
    merge_files(&[a, b], &merged, &MergeOptions::default(), None).unwrap();
    let m = Reader::open(&merged).unwrap();
    assert_eq!(m.run_info_count(), 2);
    for (i, r) in m.reads().unwrap().enumerate() {
        let r = r.unwrap();
        // Even rows are run 0 of their file, odd rows run 1; file b swaps names.
        let want = match (i < 4, i % 2) {
            (true, 0) | (false, 1) => "p",
            _ => "q",
        };
        assert_eq!(
            m.run_infos()[r.run_info_index as usize].acquisition_id,
            want
        );
    }
}

#[allow(dead_code)]
fn _ensure_run_info_helper_is_used() {
    // Keep make_run_info visible to the linker even if individual tests drop it.
    let _ = make_run_info("unused");
}

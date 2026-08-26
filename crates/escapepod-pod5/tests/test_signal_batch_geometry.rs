//! The `.p5s` cache of the POD5 signal table's per-batch row counts.
//!
//! The counts are the one field the Arrow IPC footer does not carry, so
//! recovering them means a scattered read of every batch header. Caching them
//! in the sidecar is only sound if a cache that does not match the file is
//! *detected* rather than believed, so most of what is tested here is the
//! rejection path: a wrong geometry must change nothing except how long the
//! open takes.

mod common;

use std::collections::HashSet;

use escapepod_pod5::sidecar::{
    Pod5Identity, decode_batch_rows, encode_batch_rows, read_sidecar_file, read_sidecar_metadata,
    sidecar_path, write_sidecar_file,
};
use escapepod_pod5::{Reader, Uuid, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info, synth_signal};

const N_READS: usize = 43;
const SIGNAL_BATCH: u32 = 5;
const SAMPLES: usize = 400;

/// A fixture with several signal batches and a deliberately short last one, so
/// the geometry has something to get wrong. 43 reads at 5 rows per batch is
/// 8 full batches plus a final 3.
fn fixture() -> (TempDir, std::path::PathBuf, Vec<Uuid>) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("reads.pod5");
    let options = WriterOptions {
        signal_batch_size: SIGNAL_BATCH,
        ..Default::default()
    };
    let mut writer = Writer::create(&path, options).expect("writer::create");
    let run_idx = writer
        .add_run_info(make_run_info("geometry_acq"))
        .expect("add_run_info");
    let mut ids = Vec::with_capacity(N_READS);
    for i in 0..N_READS {
        let read = make_read(run_idx, i as u32 + 1, SAMPLES as u64);
        ids.push(read.read_id);
        writer
            .add_read(read, &synth_signal(SAMPLES, 0xB0 + i as u64))
            .expect("add_read");
    }
    writer.finish().expect("finish");
    (tmp, path, ids)
}

fn identity(path: &std::path::Path) -> Pod5Identity {
    Reader::open(path).unwrap().sidecar_identity().unwrap()
}

/// Read every read's signal, keyed by id, so two readers can be compared
/// byte-for-byte regardless of how they resolved the batch geometry.
fn all_signal(path: &std::path::Path, ids: &[Uuid]) -> Vec<Vec<i16>> {
    let reader = Reader::open(path).unwrap();
    let targets: HashSet<Uuid> = ids.iter().copied().collect();
    let mut rows = reader.find_signal_rows_by_ids(&targets).unwrap();
    rows.sort_by_key(|(id, _)| *id);
    reader
        .get_signal_bulk(&rows)
        .unwrap()
        .into_iter()
        .map(|(_, s)| s)
        .collect()
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

#[test]
fn run_length_encoding_round_trips() {
    // The shape a conformant POD5 actually has: one long run, one short tail.
    let uniform: Vec<u64> = std::iter::repeat_n(100, 8865).chain([37]).collect();
    let encoded = encode_batch_rows(&uniform).unwrap();
    assert_eq!(encoded, "8865x100,1x37", "uniform files must stay tiny");
    assert_eq!(decode_batch_rows(&encoded).unwrap(), uniform);

    // Non-uniform is recorded as it is, not rejected.
    let ragged = vec![10, 10, 3, 7, 7, 7, 1];
    let encoded = encode_batch_rows(&ragged).unwrap();
    assert_eq!(encoded, "2x10,1x3,3x7,1x1");
    assert_eq!(decode_batch_rows(&encoded).unwrap(), ragged);

    let single = vec![42];
    assert_eq!(
        decode_batch_rows(&encode_batch_rows(&single).unwrap()).unwrap(),
        single
    );

    assert_eq!(encode_batch_rows(&[]), None);
}

#[test]
fn malformed_geometry_decodes_to_none_rather_than_erroring() {
    // Every one of these must be a cache miss, never a panic and never a
    // partial value: the caller's fallback is to read the file.
    for bad in [
        "",
        "garbage",
        "5",
        "x5",
        "5x",
        "-1x4",
        "4x-1",
        "0x100",
        "2x100,",
        // A run count large enough to be an allocation attack.
        "99999999999x1",
    ] {
        assert_eq!(decode_batch_rows(bad), None, "{bad:?} must not decode");
    }
}

// ---------------------------------------------------------------------------
// The happy path
// ---------------------------------------------------------------------------

#[test]
fn index_records_geometry_and_reader_agrees_with_an_unindexed_one() {
    let (_tmp, path, _ids) = fixture();

    // Before indexing there is no sidecar, so the counts come from the walk.
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    assert!(truth.len() > 2, "fixture must have several signal batches");
    assert_eq!(truth.iter().sum::<u64>(), N_READS as u64);
    assert_ne!(
        truth.last(),
        truth.first(),
        "fixture must have a short final batch, or the spot check is untested"
    );

    let reader = Reader::open(&path).unwrap();
    reader.build_and_write_index(sidecar_path(&path)).unwrap();

    let meta = read_sidecar_metadata(sidecar_path(&path), &identity(&path))
        .unwrap()
        .expect("sidecar exists");
    assert_eq!(
        meta.signal_batch_rows.as_deref(),
        Some(truth.as_slice()),
        "`escpod index` must record the geometry it walked"
    );

    // The cached path must produce exactly the same footer.
    assert_eq!(
        Reader::open(&path).unwrap().signal_batch_row_counts(),
        truth
    );
    assert_eq!(Reader::open(&path).unwrap().read_count().unwrap(), N_READS);
}

#[test]
fn geometry_survives_an_annotation_write() {
    use escapepod_pod5::operations::{AnnotateOptions, write_annotation};

    let (_tmp, path, ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    Reader::open(&path)
        .unwrap()
        .build_and_write_index(sidecar_path(&path))
        .unwrap();

    let assignments = ids
        .iter()
        .take(5)
        .map(|id| (*id, "BC01".to_string()))
        .collect();
    write_annotation(&path, &assignments, &AnnotateOptions::default()).unwrap();

    let meta = read_sidecar_metadata(sidecar_path(&path), &identity(&path))
        .unwrap()
        .unwrap();
    assert_eq!(
        meta.signal_batch_rows.as_deref(),
        Some(truth.as_slice()),
        "annotating describes reads, not the signal table — it must not drop \
         the geometry and force the next open to walk again"
    );
}

// ---------------------------------------------------------------------------
// The rejection path — the reason this cache is safe to have
// ---------------------------------------------------------------------------

/// Overwrite the sidecar's geometry with `counts`, keeping everything else.
fn poison_geometry(path: &std::path::Path, counts: Vec<u64>) {
    let id = identity(path);
    let mut sc = read_sidecar_file(sidecar_path(path), &id).unwrap().unwrap();
    sc.set_signal_batch_rows(counts);
    write_sidecar_file(sidecar_path(path), &id, &sc).unwrap();
}

#[test]
fn a_wrong_geometry_is_detected_and_the_file_still_reads_correctly() {
    let (_tmp, path, ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    let expected_signal = all_signal(&path, &ids);
    Reader::open(&path)
        .unwrap()
        .build_and_write_index(sidecar_path(&path))
        .unwrap();

    // Right number of batches, wrong counts — the case a length check alone
    // would wave through, and the one that would silently mis-resolve every
    // global signal row index into the wrong batch.
    poison_geometry(&path, vec![1; truth.len()]);
    assert_eq!(
        Reader::open(&path).unwrap().signal_batch_row_counts(),
        truth,
        "a poisoned geometry must be rejected in favour of the real headers"
    );
    assert_eq!(
        all_signal(&path, &ids),
        expected_signal,
        "and the signal must be byte-identical to the uncached read"
    );

    // Plausible-but-wrong: uniform stride, no short tail. This is exactly what
    // deriving the stride from batch 0 instead of measuring it would produce,
    // so it is the assumption this design refused, written down as a test.
    poison_geometry(&path, vec![truth[0]; truth.len()]);
    assert_eq!(
        Reader::open(&path).unwrap().signal_batch_row_counts(),
        truth
    );
    assert_eq!(all_signal(&path, &ids), expected_signal);

    // Wrong length in both directions.
    poison_geometry(&path, vec![SIGNAL_BATCH as u64; truth.len() + 3]);
    assert_eq!(
        Reader::open(&path).unwrap().signal_batch_row_counts(),
        truth
    );
    poison_geometry(&path, vec![SIGNAL_BATCH as u64; 1]);
    assert_eq!(
        Reader::open(&path).unwrap().signal_batch_row_counts(),
        truth
    );
    assert_eq!(all_signal(&path, &ids), expected_signal);
}

#[test]
fn a_geometry_bound_to_another_pod5_is_never_used() {
    let (_tmp_a, path_a, ids_a) = fixture();
    let (_tmp_b, path_b, _ids_b) = fixture();

    let truth_a = Reader::open(&path_a).unwrap().signal_batch_row_counts();
    // Read A's signal through the index while A's own sidecar is still absent,
    // because once B's is planted the index path (correctly) refuses to load.
    let expected_a = all_signal(&path_a, &ids_a);
    Reader::open(&path_b)
        .unwrap()
        .build_and_write_index(sidecar_path(&path_b))
        .unwrap();

    // B's sidecar next to A. Both fixtures have the same shape, so B's
    // geometry would *fit* A — length and stride and short tail all match.
    // Only identity binding separates them, which is the point: the geometry
    // is trusted because the sidecar is bound, not because it looks plausible.
    std::fs::copy(sidecar_path(&path_b), sidecar_path(&path_a)).unwrap();
    assert_eq!(
        Reader::open(&path_a).unwrap().signal_batch_row_counts(),
        truth_a,
        "a foreign geometry must not be consulted"
    );

    // Two different reactions to the same bad sidecar, both deliberate. The
    // signal path degrades quietly because it can always recover the answer
    // from the POD5; the index path fails loudly because it cannot, and
    // stepping over it would turn a stale index into a slow wrong answer.
    assert!(
        Reader::open(&path_a).unwrap().read_index().is_err(),
        "a foreign sidecar must still be an error for the read index"
    );

    // With the foreign sidecar gone, A reads exactly as it did before.
    std::fs::remove_file(sidecar_path(&path_a)).unwrap();
    assert_eq!(all_signal(&path_a, &ids_a), expected_a);
}

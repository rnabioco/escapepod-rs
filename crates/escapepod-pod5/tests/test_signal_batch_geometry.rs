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
// Every write path records it — the gap that made the cache near-unreachable
// ---------------------------------------------------------------------------

/// Label the first five reads, the cheapest way to exercise `write_columns`
/// (which `write_annotation` delegates to, and which `demux --annotate` calls).
fn annotate_a_few(path: &std::path::Path, ids: &[Uuid], label: &str) {
    use escapepod_pod5::operations::{AnnotateOptions, write_annotation};
    let assignments = ids
        .iter()
        .take(5)
        .map(|id| (*id, label.to_string()))
        .collect();
    write_annotation(path, &assignments, &AnnotateOptions::default()).unwrap();
}

#[test]
fn annotating_a_file_with_no_sidecar_records_the_geometry() {
    // The workflow that actually produces sidecars in the wild:
    // `demux --annotate` / `escpod annotate -a`, on a file nobody indexed.
    // Before this was wired up, the sidecar such a run created had a read
    // index and no geometry, so every later open re-walked the batch headers
    // — and the obvious remedy, `escpod index`, skipped the file as "already
    // indexed".
    let (_tmp, path, ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    assert!(
        !sidecar_path(&path).exists(),
        "fixture must start with none"
    );

    annotate_a_few(&path, &ids, "BC01");

    let meta = read_sidecar_metadata(sidecar_path(&path), &identity(&path))
        .unwrap()
        .expect("annotating must create a sidecar");
    assert_eq!(
        meta.signal_batch_rows.as_deref(),
        Some(truth.as_slice()),
        "a sidecar created by annotation must carry the geometry too"
    );
}

#[test]
fn writing_a_design_records_the_geometry() {
    use escapepod_pod5::operations::{DesignOptions, write_design};

    let (tmp, path, ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    annotate_a_few(&path, &ids, "BC01");

    let csv = tmp.path().join("design.csv");
    std::fs::write(&csv, "barcode,condition\nBC01,treated\n").unwrap();
    write_design(&path, &csv, &DesignOptions::default()).unwrap();

    let meta = read_sidecar_metadata(sidecar_path(&path), &identity(&path))
        .unwrap()
        .unwrap();
    assert_eq!(meta.signal_batch_rows.as_deref(), Some(truth.as_slice()));
}

#[test]
fn a_sidecar_written_without_geometry_is_backfilled_by_the_next_write() {
    use escapepod_pod5::operations::read_annotation;

    let (_tmp, path, ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();

    // What an older escpod left behind: index, no geometry.
    let id = identity(&path);
    let entries = Reader::open(&path)
        .unwrap()
        .read_index()
        .unwrap()
        .entries()
        .to_vec();
    let legacy = escapepod_pod5::sidecar::Sidecar::new(entries);
    write_sidecar_file(sidecar_path(&path), &id, &legacy).unwrap();
    assert!(
        read_sidecar_metadata(sidecar_path(&path), &id)
            .unwrap()
            .unwrap()
            .signal_batch_rows
            .is_none(),
        "precondition: the legacy sidecar must have no geometry"
    );

    annotate_a_few(&path, &ids, "BC02");

    assert_eq!(
        read_sidecar_metadata(sidecar_path(&path), &id)
            .unwrap()
            .unwrap()
            .signal_batch_rows
            .as_deref(),
        Some(truth.as_slice()),
        "an existing sidecar missing the geometry must gain it on the next write"
    );
    assert_eq!(
        read_annotation(&path, Some("barcode")).unwrap().len(),
        5,
        "and the annotation that triggered the backfill must survive it"
    );
}

#[test]
fn index_then_demux_reuses_the_geometry_and_keeps_it() {
    use escapepod_pod5::operations::read_annotation;

    // The sequence the whole design is for: pre-warm with `escpod index`, then
    // run the pipeline. The geometry must be reused (not re-measured into
    // something different) and must survive the annotation write, while the
    // annotation lands.
    let (_tmp, path, ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    Reader::open(&path)
        .unwrap()
        .build_and_write_index(sidecar_path(&path))
        .unwrap();

    // Reuse: a fresh reader resolves the footer through the sidecar and lands
    // on exactly the counts the walk produced.
    assert_eq!(
        Reader::open(&path).unwrap().signal_batch_row_counts(),
        truth
    );

    annotate_a_few(&path, &ids, "BC03");

    let meta = read_sidecar_metadata(sidecar_path(&path), &identity(&path))
        .unwrap()
        .unwrap();
    assert_eq!(
        meta.signal_batch_rows.as_deref(),
        Some(truth.as_slice()),
        "the pipeline must update the sidecar, not trade its caches away"
    );
    assert_eq!(read_annotation(&path, Some("barcode")).unwrap().len(), 5);
    assert_eq!(
        Reader::open(&path).unwrap().read_index().unwrap().len(),
        N_READS,
        "and the read index must still be there"
    );
}

#[test]
fn a_failed_measure_does_not_erase_a_recorded_geometry() {
    // `Reader::measure_signal_batch_rows` returns an empty vec for every
    // failure it has, so an empty value means "could not measure", never
    // "there are no batches". Treating it as a value let one failed re-measure
    // silently drop a geometry an earlier run recorded correctly — silently,
    // because the sidecar stays valid and merely gets slow again.
    let (_tmp, path, _ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    Reader::open(&path)
        .unwrap()
        .build_and_write_index(sidecar_path(&path))
        .unwrap();

    let id = identity(&path);
    let mut sc = read_sidecar_file(sidecar_path(&path), &id)
        .unwrap()
        .unwrap();
    assert!(sc.signal_batch_rows().is_some(), "precondition");
    sc.set_signal_batch_rows(Vec::new());
    assert_eq!(
        sc.signal_batch_rows(),
        Some(truth.as_slice()),
        "an empty measurement must leave the recorded geometry alone"
    );

    // Discarding it is still possible, but only by saying so.
    sc.clear_signal_batch_rows();
    assert!(sc.signal_batch_rows().is_none());
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

// ---------------------------------------------------------------------------
// Completing a sidecar in place — what the ungated `escpod index` does to the
// sidecar every pre-geometry `demux --annotate` left behind.
// ---------------------------------------------------------------------------

/// Build the sidecar an older escpod would have written: a correct read index,
/// annotations, and no geometry.
fn legacy_sidecar_with_annotation(path: &std::path::Path, ids: &[Uuid]) {
    use escapepod_pod5::operations::{AnnotateOptions, write_annotation};

    let assignments = ids
        .iter()
        .take(5)
        .map(|id| (*id, "BC01".to_string()))
        .collect();
    write_annotation(path, &assignments, &AnnotateOptions::default()).unwrap();

    // Strip the geometry the write just recorded, leaving the rest as-is.
    let id = identity(path);
    let mut sc = read_sidecar_file(sidecar_path(path), &id).unwrap().unwrap();
    sc.clear_signal_batch_rows();
    write_sidecar_file(sidecar_path(path), &id, &sc).unwrap();
    assert!(
        read_sidecar_metadata(sidecar_path(path), &id)
            .unwrap()
            .unwrap()
            .signal_batch_rows
            .is_none(),
        "precondition: the legacy sidecar must have no geometry"
    );
}

#[test]
fn completing_a_sidecar_adds_the_geometry_and_keeps_everything_else() {
    use escapepod_pod5::operations::read_annotation;

    // The reason this exists instead of just calling `build_and_write_index`:
    // a rebuild re-scans the reads table and rewrites every column, and the
    // columns it would carry through are barcode assignments that took hours to
    // compute. Adding one metadata key cannot lose them.
    let (_tmp, path, ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    legacy_sidecar_with_annotation(&path, &ids);

    let index_before = Reader::open(&path).unwrap().read_index().unwrap().len();

    assert!(
        Reader::open(&path)
            .unwrap()
            .complete_sidecar_geometry(sidecar_path(&path))
            .unwrap(),
        "a sidecar missing the geometry must be reported as completed"
    );

    let id = identity(&path);
    assert_eq!(
        read_sidecar_metadata(sidecar_path(&path), &id)
            .unwrap()
            .unwrap()
            .signal_batch_rows
            .as_deref(),
        Some(truth.as_slice()),
        "and must now carry the measured geometry"
    );
    assert_eq!(
        read_annotation(&path, Some("barcode")).unwrap().len(),
        5,
        "the annotation must survive completion untouched"
    );
    assert_eq!(
        Reader::open(&path).unwrap().read_index().unwrap().len(),
        index_before,
        "and so must the read index"
    );
}

#[test]
fn completing_is_a_no_op_when_there_is_nothing_to_complete() {
    let (_tmp, path, _ids) = fixture();

    // No sidecar at all: this is not the command that creates one, and saying
    // so beats quietly building one behind the caller's back.
    assert!(!sidecar_path(&path).exists());
    assert!(
        !Reader::open(&path)
            .unwrap()
            .complete_sidecar_geometry(sidecar_path(&path))
            .unwrap()
    );
    assert!(
        !sidecar_path(&path).exists(),
        "completing must not conjure a sidecar"
    );

    // Already complete: nothing to do, and the file must not be rewritten.
    Reader::open(&path)
        .unwrap()
        .build_and_write_index(sidecar_path(&path))
        .unwrap();
    let before = std::fs::read(sidecar_path(&path)).unwrap();
    assert!(
        !Reader::open(&path)
            .unwrap()
            .complete_sidecar_geometry(sidecar_path(&path))
            .unwrap()
    );
    assert_eq!(
        std::fs::read(sidecar_path(&path)).unwrap(),
        before,
        "a sidecar that already has the geometry must be left byte-identical"
    );
}

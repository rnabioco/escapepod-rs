//! The reads table's record-batch geometry in files escpod *writes*.
//!
//! `Writer` honours `read_batch_size`, and `test_writer_integration` pins that.
//! The operations — `filter`, `merge`, `subset` — do not go through `Writer`;
//! they assemble the table with `build_reads_table{,_remapped}` and write it
//! with a raw Arrow writer. Those built one record batch for the whole file
//! however large it was, so a 40,000-read filter output came back as a single
//! batch — where real writers use a bounded size (measured: MinKNOW ~10,000
//! reads per batch, the pod5 Python package exactly 1,000).
//!
//! Nothing caught it. Round-trip tests compare read *content*, which is
//! identical either way, and the only tests that assert a batch count use
//! `Writer`. The geometry is invisible unless something looks at it directly,
//! which is what this file does.
//!
//! It is not cosmetic. `demux`'s reader shards by batch index, so a
//! single-batch file is read by exactly one thread whatever
//! `ESCAPEPOD_DEMUX_FILLERS` says, and blocks reach the GPU only at batch
//! boundaries — one batch means the whole file is decoded before any work is
//! handed downstream (#297).

mod common;

use escapepod_pod5::{Reader, Uuid, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info, synth_signal};

/// Enough reads to need several batches at the 1,000 ONT uses, with a short
/// final batch so an off-by-one in the slicing has somewhere to show.
const N_READS: usize = 2_500;
const SAMPLES: usize = 40;

fn source(dir: &TempDir, name: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut writer = Writer::create(&path, WriterOptions::default()).expect("writer::create");
    let run_idx = writer
        .add_run_info(make_run_info("read_geometry_acq"))
        .expect("add_run_info");
    for i in 0..N_READS {
        let read = make_read(run_idx, i as u32 + 1, SAMPLES as u64);
        writer
            .add_read(read, &synth_signal(SAMPLES, 0x50 + i as u64))
            .expect("add_read");
    }
    writer.finish().expect("finish");
    path
}

fn batches(path: &std::path::Path) -> usize {
    Reader::open(path).unwrap().read_batch_count().unwrap()
}

fn reads(path: &std::path::Path) -> usize {
    Reader::open(path).unwrap().read_count().unwrap()
}

/// `Writer` — the path that already honoured the option. Here as the control:
/// if this one ever regresses, the operations below are not the culprit.
#[test]
fn writer_splits_the_reads_table_into_batches() {
    let tmp = TempDir::new().unwrap();
    let path = source(&tmp, "src.pod5");
    assert_eq!(reads(&path), N_READS);
    assert_eq!(
        batches(&path),
        N_READS.div_ceil(1_000),
        "Writer must honour the default read_batch_size of 1000"
    );
}

/// `filter` must not collapse its output into one record batch.
///
/// Before the fix this returned 1 for any input size.
#[test]
fn filter_splits_the_reads_table_into_batches() {
    let tmp = TempDir::new().unwrap();
    let src = source(&tmp, "src.pod5");
    let out = tmp.path().join("filtered.pod5");

    let keep: Vec<_> = Reader::open(&src)
        .unwrap()
        .reads()
        .unwrap()
        .take(1_500)
        .map(|r| r.unwrap().read_id)
        .collect();
    assert_eq!(keep.len(), 1_500);

    escapepod_pod5::operations::filter_files(
        &[&src],
        &out,
        &keep.iter().copied().collect(),
        Default::default(),
        None,
    )
    .expect("filter");

    assert_eq!(reads(&out), 1_500);
    assert_eq!(
        batches(&out),
        1_500_usize.div_ceil(1_000),
        "filter collapsed the reads table into one record batch"
    );
}

/// `merge` must not collapse its output into one record batch either — it had
/// the same builder and a `read_batch_size` default of 100,000, so every merged
/// file under that size was a single batch.
#[test]
fn merge_splits_the_reads_table_into_batches() {
    let tmp = TempDir::new().unwrap();
    let a = source(&tmp, "a.pod5");
    let b = source(&tmp, "b.pod5");
    let out = tmp.path().join("merged.pod5");

    escapepod_pod5::merge::merge_files(&[a, b], &out, &Default::default(), None).expect("merge");

    let n = reads(&out);
    assert_eq!(n, 2 * N_READS);
    assert_eq!(
        batches(&out),
        n.div_ceil(1_000),
        "merge collapsed the reads table into one record batch"
    );
}

/// Striding must partition the table: every read exactly once across the
/// shards, in a shard-local ascending order.
///
/// The loop this replaced decoded every batch in every shard and discarded the
/// ones it did not own, so a bug here would previously have been masked by the
/// discard rather than surfacing as a missing read.
#[test]
fn strided_batches_partition_the_reads_table() {
    use std::collections::HashSet;

    let tmp = TempDir::new().unwrap();
    let path = source(&tmp, "src.pod5");
    let reader = Reader::open(&path).unwrap();
    let total = reader.read_batch_count().unwrap();
    assert!(total > 1, "fixture must have several batches, got {total}");

    let all: Vec<Uuid> = reader
        .reads()
        .unwrap()
        .map(|r| r.unwrap().read_id)
        .collect();
    assert_eq!(all.len(), N_READS);

    for shards in [1usize, 2, 3, 5, 16, 64] {
        let mut seen: Vec<Uuid> = Vec::new();
        let mut batches_seen = 0usize;
        for shard in 0..shards {
            let mut prev: Option<u32> = None;
            for batch in reader.read_batches_strided(shard, shards).unwrap() {
                let batch = batch.unwrap();
                batches_seen += 1;
                let view = escapepod_pod5::ReadsBatchView::new(&batch, false).unwrap();
                for row in 0..view.num_rows() {
                    let read = view.read(row).unwrap();
                    // Ascending within a shard: `set_index` must not rewind.
                    if let Some(p) = prev {
                        assert!(
                            read.read_number > p,
                            "shard {shard}/{shards} went backwards: {} after {p}",
                            read.read_number
                        );
                    }
                    prev = Some(read.read_number);
                    seen.push(read.read_id);
                }
            }
        }
        assert_eq!(
            batches_seen, total,
            "shards={shards} visited {batches_seen} batches, expected {total}"
        );
        assert_eq!(
            seen.len(),
            N_READS,
            "shards={shards} yielded {} reads, expected {N_READS}",
            seen.len()
        );
        assert_eq!(
            seen.iter().copied().collect::<HashSet<_>>().len(),
            N_READS,
            "shards={shards} yielded a duplicate read"
        );
        assert_eq!(
            seen.iter().copied().collect::<HashSet<_>>(),
            all.iter().copied().collect::<HashSet<_>>(),
            "shards={shards} did not cover exactly the source reads"
        );
    }
}

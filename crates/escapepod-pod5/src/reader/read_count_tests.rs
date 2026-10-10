//! `Reader::read_count` and the `.p5s` sidecar's recorded count (#457).
//!
//! In-crate rather than under `tests/` because the interesting assertion —
//! that the sidecar's count was *used* instead of the batch-header walk —
//! needs the reads-table byte offsets, which are crate-private.

#[path = "../../tests/common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};

use crate::arrow_ipc::ArrowIpcFooter;
use crate::sidecar::{
    IndexEntry, Sidecar, read_sidecar_file, read_sidecar_metadata, sidecar_path, write_sidecar_file,
};
use crate::{Reader, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info, synth_signal};

/// 23 reads at 5 per batch: four full batches and a short last one of 3, so
/// the `(batches - 1) * first + last` prediction has a short tail to get right.
const N_READS: usize = 23;
const READ_BATCH: u32 = 5;

fn fixture() -> (TempDir, PathBuf) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("reads.pod5");
    let mut writer = Writer::create(
        &path,
        WriterOptions {
            read_batch_size: READ_BATCH,
            ..Default::default()
        },
    )
    .expect("create");
    let run = writer
        .add_run_info(make_run_info("read_count_acq"))
        .expect("run info");
    for i in 0..N_READS {
        writer
            .add_read(
                make_read(run, i as u32 + 1, 100),
                &synth_signal(100, i as u64),
            )
            .expect("add_read");
    }
    writer.finish().expect("finish");
    (tmp, path)
}

fn index(path: &Path) {
    Reader::open(path)
        .unwrap()
        .build_and_write_index(sidecar_path(path))
        .unwrap();
}

/// Replace the sidecar with one bound to the same POD5 (so identity passes)
/// whose index is `edit`ed, hence whose recorded count is the edited length.
fn rewrite_entries(path: &Path, edit: impl FnOnce(&mut Vec<IndexEntry>)) {
    let id = Reader::open(path).unwrap().sidecar_identity().unwrap();
    let old = read_sidecar_file(sidecar_path(path), &id).unwrap().unwrap();
    let mut entries = old.entries().to_vec();
    edit(&mut entries);
    write_sidecar_file(sidecar_path(path), &id, &Sidecar::new(entries)).unwrap();
}

/// Overwrite one *middle* reads batch's message header, which only a full
/// header walk ever reads. Afterwards the walk errors or disagrees, and only
/// a count taken from elsewhere is still right.
fn break_middle_batch_header(path: &Path) {
    let (start, offset) = {
        let reader = Reader::open(path).unwrap();
        let start = reader.footer.reads_table().unwrap().offset as usize;
        let bytes = reader.reads_table_bytes().unwrap();
        let footer = ArrowIpcFooter::parse(bytes).unwrap();
        assert!(footer.record_batches.len() >= 4, "need a middle batch");
        (start, footer.record_batches[2].offset as usize)
    };
    let mut file = std::fs::read(path).unwrap();
    file[start + offset..start + offset + 16].fill(0xFF);
    std::fs::write(path, file).unwrap();
}

#[test]
fn read_count_uses_sidecar_when_valid() {
    let (_tmp, path) = fixture();
    assert_eq!(Reader::open(&path).unwrap().read_count().unwrap(), N_READS);
    index(&path);
    break_middle_batch_header(&path);

    // Without the sidecar the walk now trips over the damaged header...
    let bare = path.with_file_name("bare.pod5");
    std::fs::copy(&path, &bare).unwrap();
    let walked = Reader::open(&bare).and_then(|r| r.read_count());
    assert!(
        !matches!(walked, Ok(n) if n == N_READS),
        "the damaged header must defeat the walk, or this test proves nothing"
    );

    // ...and with it, the recorded count is confirmed from the first and last
    // headers alone and the damaged one is never read.
    assert_eq!(Reader::open(&path).unwrap().read_count().unwrap(), N_READS);
}

#[test]
fn read_count_ignores_stale_sidecar() {
    type Edit = Box<dyn FnOnce(&mut Vec<IndexEntry>)>;
    let edits: Vec<(&str, Edit)> = vec![
        (
            "one read short",
            Box::new(|e| {
                e.pop();
            }),
        ),
        (
            "one read over",
            Box::new(|e| {
                let extra = *e.last().unwrap();
                e.push(extra);
            }),
        ),
    ];
    for (label, edit) in edits {
        let (_tmp, path) = fixture();
        index(&path);
        rewrite_entries(&path, edit);

        // The sidecar passes identity (same file id and size) and still lies:
        // believing it blindly would report N_READS -/+ 1.
        let id = Reader::open(&path).unwrap().sidecar_identity().unwrap();
        let claimed = read_sidecar_metadata(sidecar_path(&path), &id)
            .unwrap()
            .unwrap()
            .read_count
            .unwrap();
        assert_ne!(claimed as usize, N_READS, "{label}: fixture must lie");
        assert_eq!(
            Reader::open(&path).unwrap().read_count().unwrap(),
            N_READS,
            "{label}: a count that identity cannot vouch for must not be believed"
        );
    }
}

#[test]
fn read_count_ignores_sidecar_bound_to_another_file() {
    let (_tmp_a, a) = fixture();
    let (_tmp_b, b) = fixture();
    index(&b);
    std::fs::copy(sidecar_path(&b), sidecar_path(&a)).unwrap();
    assert_eq!(Reader::open(&a).unwrap().read_count().unwrap(), N_READS);
}

#[test]
fn confirming_total_accepts_the_truth_and_refuses_anything_else() {
    let (_tmp, path) = fixture();
    let reader = Reader::open(&path).unwrap();
    let bytes = reader.reads_table_bytes().unwrap();
    let n = N_READS as u64;
    let ok = ArrowIpcFooter::parse_confirming_total(bytes, n)
        .unwrap()
        .expect("true total confirmed");
    assert_eq!(ok.total_rows, n);
    for wrong in [0, n - 1, n + 1, n + READ_BATCH as u64, 1] {
        assert!(
            ArrowIpcFooter::parse_confirming_total(bytes, wrong)
                .unwrap()
                .is_none(),
            "claim {wrong} must be refuted"
        );
    }
}

#[test]
fn sidecar_metadata_is_opened_once_per_reader() {
    let (_tmp, path) = fixture();
    index(&path);
    let reader = Reader::open(&path).unwrap();
    assert!(reader.has_sidecar_index());
    // Removing the file after the first look: a second ask is answered from
    // the reader's own copy, which is how the signal geometry, the count and
    // `has_sidecar_index` share one parse.
    std::fs::remove_file(sidecar_path(&path)).unwrap();
    assert!(reader.has_sidecar_index());
    assert_eq!(reader.read_count().unwrap(), N_READS);
}

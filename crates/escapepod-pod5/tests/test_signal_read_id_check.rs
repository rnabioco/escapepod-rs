//! The signal table's own `read_id` column is the spec's consistency check:
//! chunks handed back for a read must carry that read's id.

mod common;

use escapepod_pod5::{Reader, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info, synth_signal};

fn fixture() -> (TempDir, std::path::PathBuf) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("reads.pod5");
    let options = WriterOptions {
        signal_batch_size: 5,
        ..Default::default()
    };
    let mut writer = Writer::create(&path, options).expect("writer::create");
    let run_idx = writer
        .add_run_info(make_run_info("rid_acq"))
        .expect("add_run_info");
    for i in 0..12 {
        let read = make_read(run_idx, i as u32 + 1, 400);
        writer
            .add_read(read, &synth_signal(400, 0xD0 + i as u64))
            .expect("add_read");
    }
    writer.finish().expect("finish");
    (tmp, path)
}

#[test]
fn chunk_read_id_is_checked_against_the_requested_read() {
    let (_tmp, path) = fixture();
    let reader = Reader::open(&path).unwrap();
    let reads: Vec<_> = reader.reads().unwrap().map(|r| r.unwrap()).collect();
    let (a, b) = (&reads[3], &reads[4]);

    // Right id, right rows: fine, and identical to the unchecked path.
    let ok = reader
        .get_signal_checked(a.read_id, &a.signal_rows)
        .unwrap();
    assert_eq!(ok, reader.get_signal(&a.signal_rows).unwrap());
    let extractor = reader.signal_extractor().unwrap();
    assert_eq!(
        extractor
            .get_signal_checked(a.read_id, &a.signal_rows)
            .unwrap(),
        ok
    );

    // Another read's rows under this read's id: an error naming both.
    let err = reader
        .get_signal_prefix_checked(a.read_id, &b.signal_rows, 10)
        .unwrap_err()
        .to_string();
    assert!(err.contains(&a.read_id.to_string()), "{err}");
    assert!(err.contains(&b.read_id.to_string()), "{err}");
    assert!(err.contains("reads.pod5"), "{err}");
    assert!(
        extractor
            .get_signal_checked(a.read_id, &b.signal_rows)
            .is_err()
    );
}

#[test]
fn chunks_of_one_read_must_agree_on_read_id() {
    let (_tmp, path) = fixture();
    let reader = Reader::open(&path).unwrap();
    let reads: Vec<_> = reader.reads().unwrap().map(|r| r.unwrap()).collect();
    // Two different reads' rows presented as one read: no id is supplied, but
    // the chunks disagree with each other.
    let mixed = [reads[1].signal_rows[0], reads[2].signal_rows[0]];
    assert!(reader.get_signal(&mixed).is_err());
}

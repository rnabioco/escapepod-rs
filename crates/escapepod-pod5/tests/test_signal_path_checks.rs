//! Per-read signal extraction must fail loudly, never return wrong or short
//! data: out-of-range rows are errors (as on the bulk path), and a sidecar
//! geometry that is wrong in a batch the open does not validate surfaces as an
//! error rather than another read's signal.

mod common;

use escapepod_pod5::sidecar::{read_sidecar_file, sidecar_path, write_sidecar_file};
use escapepod_pod5::{Reader, Uuid, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info, synth_signal};

const N_READS: usize = 43;
const SIGNAL_BATCH: u32 = 5;
const SAMPLES: usize = 400;

/// 43 reads, one chunk each, 5 rows per signal batch: 8 full batches and a
/// short final one.
fn fixture() -> (TempDir, std::path::PathBuf, Vec<Uuid>) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("reads.pod5");
    let options = WriterOptions {
        signal_batch_size: SIGNAL_BATCH,
        ..Default::default()
    };
    let mut writer = Writer::create(&path, options).expect("writer::create");
    let run_idx = writer
        .add_run_info(make_run_info("checks_acq"))
        .expect("add_run_info");
    let mut ids = Vec::new();
    for i in 0..N_READS {
        let read = make_read(run_idx, i as u32 + 1, SAMPLES as u64);
        ids.push(read.read_id);
        writer
            .add_read(read, &synth_signal(SAMPLES, 0xC0 + i as u64))
            .expect("add_read");
    }
    writer.finish().expect("finish");
    (tmp, path, ids)
}

#[test]
fn single_read_oob_signal_row_errors() {
    let (_tmp, path, _ids) = fixture();
    let reader = Reader::open(&path).unwrap();
    let past_end = N_READS as u64 + 7;
    // One good row, one past the end: used to be a silent half-length Ok.
    assert!(reader.get_signal(&[0, past_end]).is_err());
    assert!(reader.get_signal(&[past_end]).is_err());
    let extractor = reader.signal_extractor().unwrap();
    assert!(extractor.get_signal(&[0, past_end]).is_err());
}

#[test]
fn prefix_oob_signal_row_errors() {
    let (_tmp, path, _ids) = fixture();
    let reader = Reader::open(&path).unwrap();
    let past_end = N_READS as u64 + 7;
    // The prefix budget is spent by the first row, so a decode-only check
    // would never even look at the bad second one.
    assert!(reader.get_signal_prefix(&[0, past_end], 10).is_err());
    let extractor = reader.signal_extractor().unwrap();
    assert!(extractor.get_signal_prefix(&[0, past_end], 10).is_err());
}

/// A sidecar whose geometry is right for the first and last batch (the only
/// two the open validates) but wrong in the middle must never hand back
/// another read's signal. Every read either errors or returns its own signal.
#[test]
fn sidecar_middle_batch_geometry_mismatch_fails() {
    let (_tmp, path, _ids) = fixture();
    let truth = Reader::open(&path).unwrap().signal_batch_row_counts();
    assert!(truth.len() > 4);

    let expected: Vec<(Vec<u64>, Vec<i16>)> = {
        let r = Reader::open(&path).unwrap();
        r.reads()
            .unwrap()
            .map(|x| {
                let x = x.unwrap();
                let sig = r.get_signal(&x.signal_rows).unwrap();
                (x.signal_rows, sig)
            })
            .collect()
    };

    Reader::open(&path)
        .unwrap()
        .build_and_write_index(sidecar_path(&path))
        .unwrap();
    let id = Reader::open(&path).unwrap().sidecar_identity().unwrap();
    let mut sc = read_sidecar_file(sidecar_path(&path), &id)
        .unwrap()
        .unwrap();
    // Shift one row from batch 3 to batch 4: same total, same first/last.
    let mut poisoned = truth.clone();
    poisoned[3] -= 1;
    poisoned[4] += 1;
    sc.set_signal_batch_rows(poisoned.clone());
    write_sidecar_file(sidecar_path(&path), &id, &sc).unwrap();

    let reader = Reader::open(&path).unwrap();
    assert_eq!(
        reader.signal_batch_row_counts(),
        poisoned,
        "precondition: the open accepts a geometry only first/last-validated"
    );
    let mut errors = 0;
    for (i, (rows, sig)) in expected.iter().enumerate() {
        match reader.get_signal(rows) {
            Ok(got) => assert_eq!(&got, sig, "read {i}: wrong signal returned"),
            Err(_) => errors += 1,
        }
    }
    assert!(errors > 0, "the poisoned batches must surface as errors");
}

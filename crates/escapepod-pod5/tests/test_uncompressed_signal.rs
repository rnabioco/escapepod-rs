//! `WriterOptions { compress_signal: false }` must produce a readable file
//! (#446): the signal table uses the spec's `LargeList<Int16>` layout, every
//! read path decodes it, and merge/filter/repack of such a file yield valid
//! (VBZ) output rather than an unreadable one.

mod common;

use std::collections::HashSet;
use std::path::Path;

use escapepod_pod5::operations::{FilterOptions, filter_files};

use escapepod_pod5::{
    MergeOptions, Reader, RepackOptions, Uuid, Writer, WriterOptions, merge_files, repack_files,
};
use tempfile::TempDir;

use common::{make_read, make_run_info};

fn signal(n: usize, seed: u64) -> Vec<i16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 48) as i16
        })
        .collect()
}

/// Three reads: multi-chunk, short, and exactly-two-samples (4 bytes, the
/// width of the `samples` column).
fn write_uncompressed(path: &Path) -> Vec<(Uuid, Vec<i16>)> {
    let opts = WriterOptions {
        compress_signal: false,
        max_signal_chunk_size: 1000,
        signal_batch_size: 2,
        ..Default::default()
    };
    let mut w = Writer::create(path, opts).unwrap();
    let run = w.add_run_info(make_run_info("u")).unwrap();
    let mut out = Vec::new();
    for (i, n) in [2_500usize, 17, 2].into_iter().enumerate() {
        let sig = signal(n, 0xABCD + i as u64);
        let read = make_read(run, i as u32 + 1, n as u64);
        out.push((read.read_id, sig.clone()));
        w.add_read(read, &sig).unwrap();
    }
    w.finish().unwrap();
    out
}

fn assert_exact(path: &Path, expected: &[(Uuid, Vec<i16>)]) {
    let reader = Reader::open(path).unwrap();
    let extractor = reader.signal_extractor().unwrap();
    let mut seen = 0;
    for read in reader.reads().unwrap() {
        let read = read.unwrap();
        let want = &expected
            .iter()
            .find(|(id, _)| *id == read.read_id)
            .expect("read present")
            .1;
        assert_eq!(&reader.get_signal(&read.signal_rows).unwrap(), want);
        assert_eq!(&extractor.get_signal(&read.signal_rows).unwrap(), want);
        let prefix = want.len().min(5);
        assert_eq!(
            extractor
                .get_signal_prefix(&read.signal_rows, prefix)
                .unwrap(),
            want[..prefix]
        );
        let bulk = reader
            .get_compressed_signal_bulk(&[(0u8, read.signal_rows.clone())])
            .unwrap();
        let samples: u32 = bulk[0].1.iter().map(|c| c.samples).sum();
        assert_eq!(samples as usize, want.len());
        seen += 1;
    }
    assert_eq!(seen, expected.len());
}

#[test]
fn round_trip_uncompressed_signal() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("u.pod5");
    let expected = write_uncompressed(&path);
    assert_exact(&path, &expected);
}

#[test]
fn uncompressed_signal_survives_merge_filter_repack() {
    let tmp = TempDir::new().unwrap();
    let a = tmp.path().join("a.pod5");
    let expected = write_uncompressed(&a);

    let merged = tmp.path().join("merged.pod5");
    merge_files(&[&a], &merged, &MergeOptions::default(), None).unwrap();
    assert_exact(&merged, &expected);

    let filtered = tmp.path().join("filtered.pod5");
    let ids: HashSet<Uuid> = expected.iter().map(|(id, _)| *id).collect();
    filter_files(&[&a], &filtered, &ids, FilterOptions::default(), None).unwrap();
    assert_exact(&filtered, &expected);

    let repacked = tmp.path().join("repacked.pod5");
    let res = repack_files(
        &[(&a, &repacked)],
        RepackOptions {
            force: true,
            ..Default::default()
        },
        None,
    );
    assert!(res.failures.is_empty(), "{:?}", res.failures);
    assert_exact(&repacked, &expected);
}

//! Signal round-trip edge cases ported from the upstream POD5 test harness
//! (nanoporetech/pod5-file-format). These exercise format gotchas that the
//! small synthetic fixtures in the other integration tests never reach:
//!
//! * A single read whose signal spans **multiple signal-table rows**. The
//!   writer chunks a read at `max_signal_chunk_size` (default 102_400 samples),
//!   so any read longer than that becomes several rows; the reader must
//!   concatenate them back **in order**. cf. python
//!   `test_signal_tools.test_round_trip_chunked`.
//! * A **zero-sample read** (no signal at all), which must survive a write/read
//!   round-trip without disturbing its neighbours. cf. `test_round_trip_empty`
//!   / `test_round_trip_chunked_empty`.
//! * **Full int16-range** random signal, matching conftest `_random_signal`
//!   (`randint(-32768, 32767)`) and `svb16_scalar_tests.cpp`, across a spread of
//!   sizes with odd tails.

mod common;

use std::collections::HashMap;

use escapepod_pod5::{Reader, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info};

/// Deterministic full-range i16 generator (xorshift64), so these tests are
/// reproducible without pulling in `rand`. Mirrors the upstream fixtures that
/// draw from the entire `[i16::MIN, i16::MAX]` range rather than a narrow band.
fn rand_i16_signal(n: usize, seed: u64) -> Vec<i16> {
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

fn write_one(path: &std::path::Path, opts: WriterOptions, num_samples: u64, signal: &[i16]) {
    let mut writer = Writer::create(path, opts).expect("create");
    let run = writer
        .add_run_info(make_run_info("roundtrip"))
        .expect("run_info");
    let read = make_read(run, 1, num_samples);
    writer.add_read(read, signal).expect("add_read");
    writer.finish().expect("finish");
}

/// A read longer than the default chunk size must be split across several
/// signal rows on write and stitched back together, exactly, on read.
#[test]
fn read_signal_spans_multiple_chunks() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("multichunk.pod5");

    // Default chunk size is 102_400 samples; 250_000 samples => 3 signal rows.
    let n = 250_000usize;
    let signal = rand_i16_signal(n, 0x00C0_FFEE);
    write_one(&path, WriterOptions::default(), n as u64, &signal);

    let reader = Reader::open(&path).unwrap();
    let reads = reader
        .reads()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(reads.len(), 1);
    let read = &reads[0];

    // The writer must have split the signal across multiple signal-table rows...
    assert_eq!(
        read.signal_rows.len(),
        3,
        "expected 3 chunks for {n} samples at the default 102_400 chunk size"
    );
    assert_eq!(read.num_samples, n as u64);

    // ...and the reader must concatenate them back into the exact original.
    let got = reader.get_signal(&read.signal_rows).unwrap();
    assert_eq!(got.len(), n);
    assert_eq!(got, signal);
}

/// Force many tiny chunks and use a strictly monotone ramp so that any
/// mis-ordering of the concatenated rows is immediately visible.
#[test]
fn tiny_chunks_preserve_signal_order() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("tinychunks.pod5");

    let n = 1000usize;
    let signal: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(31)).collect();
    let opts = WriterOptions {
        max_signal_chunk_size: 64,
        ..Default::default()
    };
    write_one(&path, opts, n as u64, &signal);

    let reader = Reader::open(&path).unwrap();
    let read = reader.reads().unwrap().next().unwrap().unwrap();
    assert_eq!(read.signal_rows.len(), n.div_ceil(64));
    let got = reader.get_signal(&read.signal_rows).unwrap();
    assert_eq!(got, signal);
}

/// A read that recorded no signal (0 samples) must round-trip and leave its
/// neighbours untouched. Ports the intent of upstream `test_round_trip_empty`
/// to the file layer, where an empty read yields zero signal rows.
#[test]
fn zero_sample_read_among_normal_reads() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("empty_mixed.pod5");

    let mut writer = Writer::create(&path, WriterOptions::default()).unwrap();
    let run = writer.add_run_info(make_run_info("empty_mixed")).unwrap();

    let sig_a = rand_i16_signal(300, 1);
    let read_a = make_read(run, 1, 300);
    let id_a = read_a.read_id;
    writer.add_read(read_a, &sig_a).unwrap();

    let read_empty = make_read(run, 2, 0);
    let id_empty = read_empty.read_id;
    writer.add_read(read_empty, &[]).unwrap();

    let sig_c = rand_i16_signal(300, 2);
    let read_c = make_read(run, 3, 300);
    let id_c = read_c.read_id;
    writer.add_read(read_c, &sig_c).unwrap();

    writer.finish().unwrap();

    let reader = Reader::open(&path).unwrap();
    let by_id: HashMap<_, _> = reader
        .reads()
        .unwrap()
        .map(|r| r.unwrap())
        .map(|r| (r.read_id, r))
        .collect();
    assert_eq!(by_id.len(), 3);

    let empty = &by_id[&id_empty];
    assert_eq!(empty.num_samples, 0);
    assert!(
        empty.signal_rows.is_empty(),
        "a zero-sample read must have no signal rows"
    );
    assert!(reader.get_signal(&empty.signal_rows).unwrap().is_empty());

    // Neighbours are unaffected.
    assert_eq!(reader.get_signal(&by_id[&id_a].signal_rows).unwrap(), sig_a);
    assert_eq!(reader.get_signal(&by_id[&id_c].signal_rows).unwrap(), sig_c);
}

/// Full int16-range random signal across a spread of sizes, several of which
/// straddle a (deliberately small) chunk boundary and leave odd tails. Every
/// read must decompress back to its exact input.
#[test]
fn full_range_random_signal_roundtrips() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("fullrange.pod5");

    let sizes = [1usize, 7, 63, 64, 65, 1000, 4096, 5001];
    let opts = WriterOptions {
        max_signal_chunk_size: 256,
        ..Default::default()
    };

    let mut writer = Writer::create(&path, opts).unwrap();
    let run = writer.add_run_info(make_run_info("fullrange")).unwrap();
    let mut expected = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        let sig = rand_i16_signal(n, 0xABCD_0000 + i as u64);
        let read = make_read(run, i as u32 + 1, n as u64);
        expected.push((read.read_id, sig.clone()));
        writer.add_read(read, &sig).unwrap();
    }
    writer.finish().unwrap();

    let reader = Reader::open(&path).unwrap();
    let by_id: HashMap<_, _> = reader
        .reads()
        .unwrap()
        .map(|r| r.unwrap())
        .map(|r| (r.read_id, r))
        .collect();
    for (id, sig) in &expected {
        let got = reader.get_signal(&by_id[id].signal_rows).unwrap();
        assert_eq!(&got, sig, "mismatch for read {id}");
    }
}

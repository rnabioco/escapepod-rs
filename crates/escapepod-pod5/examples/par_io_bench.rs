//! Single-threaded vs. parallel compressed-signal read throughput on a POD5
//! file. Isolates the reader's I/O access pattern — no decode, no classify, no
//! writers — to see whether parallel access tanks throughput on a network FS
//! (the suspected cause of `demux` reading at ~4 MB/s on BeeGFS).
//!
//! The parallel phase runs FIRST on a cold file region so it isn't helped by
//! the single phase's readahead; the single phase uses a disjoint region.
//!
//! Run: cargo run --release --example par_io_bench -- <file.pod5> [n_reads]

use std::time::Instant;

use escapepod_pod5::{CompressedSignalChunk, Reader};
use rayon::prelude::*;

fn sum_bytes(chunks: &[CompressedSignalChunk]) -> usize {
    chunks.iter().map(|c| c.data.len()).sum()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: par_io_bench <file.pod5> [n_reads]");
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);

    let reader = Reader::open(&path).expect("open");

    // Collect signal_rows for the first 2n reads so the two phases use disjoint,
    // cold file regions (neither benefits from the other's page cache).
    let mut rows: Vec<Vec<u64>> = Vec::with_capacity(2 * n);
    for r in reader.reads().expect("reads") {
        let r = r.expect("read");
        if !r.signal_rows.is_empty() {
            rows.push(r.signal_rows);
        }
        if rows.len() >= 2 * n {
            break;
        }
    }
    let nn = (rows.len() / 2).min(n);
    let threads = rayon::current_num_threads();
    println!(
        "collected {} reads; {} per phase; rayon threads {}",
        rows.len(),
        nn,
        threads
    );

    // PARALLEL phase on reads [0, nn) — cold region, the real test.
    let t0 = Instant::now();
    let bytes_p: usize = rows[0..nn]
        .par_iter()
        .map(|sr| {
            reader
                .get_compressed_signal_for_rows(sr)
                .map(|c| sum_bytes(&c))
                .unwrap_or(0)
        })
        .sum();
    let dt_p = t0.elapsed().as_secs_f64();

    // SINGLE-THREAD phase on reads [nn, 2nn) — disjoint region.
    let t1 = Instant::now();
    let mut bytes_s = 0usize;
    for sr in &rows[nn..2 * nn] {
        if let Ok(c) = reader.get_compressed_signal_for_rows(sr) {
            bytes_s += sum_bytes(&c);
        }
    }
    let dt_s = t1.elapsed().as_secs_f64();

    let rate = |b: usize, dt: f64| b as f64 / 1e6 / dt;
    println!(
        "parallel({threads}):  {:.2} GB in {:.1}s = {:.0} MB/s",
        bytes_p as f64 / 1e9,
        dt_p,
        rate(bytes_p, dt_p)
    );
    println!(
        "single-thread: {:.2} GB in {:.1}s = {:.0} MB/s",
        bytes_s as f64 / 1e9,
        dt_s,
        rate(bytes_s, dt_s)
    );
    println!(
        "parallel/single throughput ratio: {:.2}x  (<<1 => parallel access is the bottleneck)",
        rate(bytes_p, dt_p) / rate(bytes_s, dt_s).max(0.001)
    );
}

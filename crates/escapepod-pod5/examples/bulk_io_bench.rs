//! Compare two cold-read access patterns for compressed signal on a POD5 file
//! (the I/O half of the `#72` demux slowdown):
//!
//!   A. per-read parallel  — N concurrent `get_compressed_signal_for_rows`
//!      calls (what the current demux scan does). Each thread faults a
//!      different batch region, scattering page faults and defeating kernel
//!      readahead on a network FS.
//!   B. batch-grouped bulk — the reads are split into a few contiguous chunks,
//!      each handed to `get_compressed_signal_bulk` (one ascending sweep per
//!      chunk). Sequential per-thread access lets readahead work.
//!
//! The two phases run on DISJOINT, cold file regions so neither warms the
//! other's page cache. Phase A (the slow one) runs first.
//!
//! Run: cargo run --release --example bulk_io_bench -- <file.pod5> [n_reads]

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
        .expect("usage: bulk_io_bench <file.pod5> [n_reads]");
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4_000);

    let reader = Reader::open(&path).expect("open");

    // Collect signal_rows for the first 2n reads → two disjoint cold regions.
    let mut rows: Vec<(usize, Vec<u64>)> = Vec::with_capacity(2 * n);
    for r in reader.reads().expect("reads") {
        let r = r.expect("read");
        if !r.signal_rows.is_empty() {
            let idx = rows.len();
            rows.push((idx, r.signal_rows));
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

    // ---- Phase A: per-read parallel (current demux pattern), reads [0, nn) ----
    // This path is known-pathological on cold network FS (the subject of #72);
    // set SKIP_A=1 to measure only the bulk path B and save cluster time.
    let skip_a = std::env::var("SKIP_A").is_ok_and(|v| v != "0");
    let (bytes_a, dt_a) = if skip_a {
        (0usize, f64::NAN)
    } else {
        let t0 = Instant::now();
        let b: usize = rows[0..nn]
            .par_iter()
            .map(|(_, sr)| {
                reader
                    .get_compressed_signal_for_rows(sr)
                    .map(|c| sum_bytes(&c))
                    .unwrap_or(0)
            })
            .sum();
        (b, t0.elapsed().as_secs_f64())
    };

    // ---- Phase B: batch-grouped bulk, reads [nn, 2nn) ----
    // Split into `threads` contiguous chunks so each worker does one ascending
    // sweep over its own slice (sequential per thread, no cross-thread scatter).
    let region = &rows[nn..2 * nn];
    // Number of concurrent sequential sweeps. BULK_STREAMS=1 is the real fix
    // pattern: a single ascending sweep (sequential I/O) while CPU work would
    // parallelize separately. Default = rayon threads (the naive parallel-chunk
    // approach, which scatters and defeats readahead).
    let streams: usize = std::env::var("BULK_STREAMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(threads)
        .max(1);
    let chunk_len = region.len().div_ceil(streams);
    let t1 = Instant::now();
    let bytes_b: usize = region
        .par_chunks(chunk_len.max(1))
        .map(|chunk| {
            reader
                .get_compressed_signal_bulk(chunk)
                .map(|v| v.iter().map(|(_, c)| sum_bytes(c)).sum())
                .unwrap_or(0)
        })
        .sum();
    let dt_b = t1.elapsed().as_secs_f64();

    let rate = |b: usize, dt: f64| b as f64 / 1e6 / dt;
    if !skip_a {
        println!(
            "A per-read parallel: {:.2} GB in {:.1}s = {:.0} MB/s",
            bytes_a as f64 / 1e9,
            dt_a,
            rate(bytes_a, dt_a)
        );
    }
    println!(
        "B batch-grouped bulk: {:.2} GB in {:.1}s = {:.0} MB/s",
        bytes_b as f64 / 1e9,
        dt_b,
        rate(bytes_b, dt_b)
    );
    if !skip_a {
        println!(
            "B/A speedup: {:.1}x",
            rate(bytes_b, dt_b) / rate(bytes_a, dt_a).max(0.001)
        );
    }
}

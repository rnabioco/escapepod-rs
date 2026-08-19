//! Microbenchmark for the boundary-CNN input prep — the `prep` stage
//! `escpod demux detect --method cnn --gpu --profile` reports.
//!
//! On a warm 150k-read run that stage measured **151 µs/read**, against an op
//! count (mean-pool ~9.2k samples to ~920, two `median_via_select` passes and
//! two maps over ~920) that says single-digit µs. This bench asks the question
//! two ways, because they have different answers:
//!
//! - `prep/single` — one call, one thread. The op count.
//! - `prep/parallel` — the same call on every rayon worker at once, which is
//!   how the detect producer runs it. If prep allocates per read, this is where
//!   it shows: six heap allocations per read × 16 threads is an allocator
//!   benchmark, not a signal-processing one.
//!
//! Lengths are the two that matter: 10,206 samples (escapepod-models' tRNA
//! mean, so the window clamps to the read) and 16,000 (`max_obs_trace`, the
//! bound a long mRNA read hits).
//!
//! Run with:
//!   cargo bench --bench adapter_prep --features cnn-detect

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;
use std::hint::black_box;

use escapepod_demux::AdapterCnnConfig;

/// Deterministic pseudo-signal in a plausible pA range, so the median/MAD path
/// sees real spread rather than a constant (which would take the `mad == 0`
/// guard and skip the interesting work).
fn pseudo_signal(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            60.0 + ((state >> 40) as f32 / 16_777_216.0) * 80.0
        })
        .collect()
}

fn bench_prep(c: &mut Criterion) {
    let cfg = AdapterCnnConfig::default();
    let threads = rayon::current_num_threads();

    let mut single = c.benchmark_group("prep/single");
    for &len in &[10_206usize, 16_000] {
        let signal = pseudo_signal(len, len as u64);
        single.throughput(Throughput::Elements(1));
        single.bench_with_input(BenchmarkId::from_parameter(len), &signal, |b, s| {
            b.iter(|| black_box(cfg.prep(black_box(s))));
        });
    }
    single.finish();

    // Every worker preps its own read concurrently — the producer's schedule.
    // Throughput is in reads, so the number is directly comparable to the
    // single-threaded arm: if prep scaled perfectly it would be `threads` times
    // the per-call rate.
    let mut par = c.benchmark_group("prep/parallel");
    for &len in &[10_206usize, 16_000] {
        let reads: Vec<Vec<f32>> = (0..threads * 32)
            .map(|i| pseudo_signal(len, (len + i) as u64))
            .collect();
        par.throughput(Throughput::Elements(reads.len() as u64));
        par.bench_with_input(BenchmarkId::from_parameter(len), &reads, |b, rs| {
            b.iter(|| {
                let out: Vec<_> = rs.par_iter().map(|s| cfg.prep(black_box(s))).collect();
                black_box(out)
            });
        });
    }
    par.finish();

    // What `detect --gpu` actually times as "prep": the decoded `i16` window
    // converted to a fresh `f32` vector, then prepped, then both dropped. The
    // conversion is not free — it allocates ~40 KB per read and touches ten
    // fresh pages — and it is inside the stage's timer, so it belongs in any
    // comparison against the 139 µs/read that run reported.
    let mut asdetect = c.benchmark_group("prep/from_i16");
    for &len in &[10_206usize, 16_000] {
        let reads: Vec<Vec<i16>> = (0..threads * 32)
            .map(|i| {
                pseudo_signal(len, (len + i) as u64)
                    .into_iter()
                    .map(|x| x as i16)
                    .collect()
            })
            .collect();
        asdetect.throughput(Throughput::Elements(reads.len() as u64));
        asdetect.bench_with_input(BenchmarkId::from_parameter(len), &reads, |b, rs| {
            b.iter(|| {
                let out: Vec<_> = rs
                    .par_iter()
                    .map(|s| {
                        let f: Vec<f32> = s.iter().map(|&x| x as f32).collect();
                        cfg.prep(black_box(&f))
                    })
                    .collect();
                black_box(out)
            });
        });
    }
    asdetect.finish();
}

criterion_group!(benches, bench_prep);
criterion_main!(benches);

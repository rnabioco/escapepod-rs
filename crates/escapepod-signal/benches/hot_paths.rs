//! Microbenchmarks for hot paths touched by the performance audit.
//!
//! Covers:
//! - DTW distance (classic Viterbi-style 2-row DP used by demux)
//! - Resquiggle DP step (banded Viterbi used by resquiggle)
//! - Fingerprint MAD normalization (demux fingerprint preprocessing)
//! - VBZ roundtrip (SVB16 + ZSTD — signal compression hot path)
//! - DTW distance-matrix (training path)
//!
//! Run with:
//!   cargo bench --bench hot_paths
//!   cargo bench --bench hot_paths -- dtw          # subset
//!
//! Env vars:
//!   ESCAPEPOD_BENCH_THREADS=N     Rayon pool size for the parallel bench
//!                                 (dtw_distance_matrix). Defaults to the
//!                                 number of physical cores seen by rayon.
//!   ESCAPEPOD_BENCH_SAMPLES=N     Override criterion sample_size for the
//!                                 slow groups (dtw_distance_matrix) where
//!                                 the default 100 runs takes forever.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use escapepod_signal::compression::vbz;
use escapepod_signal::dtw::{Fingerprint, NormMethod, dtw_distance, normalize_fingerprint};
use escapepod_signal::resquiggle::adaptive_dp::adaptive_banded_dp;
use escapepod_signal::resquiggle::dp::{ViterbiBuffers, dp_step_buffered};
use escapepod_signal::resquiggle::types::RefineAlgo;

/// Read ESCAPEPOD_BENCH_THREADS and pre-configure the rayon global pool.
/// No-op if the env var isn't set or rayon has already been initialized.
fn configure_rayon() {
    if let Some(n) = std::env::var("ESCAPEPOD_BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }
}

fn bench_sample_size(default: usize) -> usize {
    std::env::var("ESCAPEPOD_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Build a deterministic pseudo-random float sequence (xorshift, seeded).
fn pseudo_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

/// Build a nanopore-like i16 signal: slow drift + per-sample noise.
fn pseudo_signal(n: usize, seed: u64) -> Vec<i16> {
    let mut state = seed | 1;
    let mut base: i32 = 500;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state as i32) >> 24) % 15;
            if i % 200 == 0 {
                base = 400 + ((i as i32 / 200) % 4) * 50;
            }
            (base + noise).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
        .collect()
}

fn bench_dtw(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw_distance");
    for &len in &[100usize, 400, 1000] {
        let a = pseudo_floats(len, 0xA110C8);
        let b = pseudo_floats(len, 0xB33F);
        group.throughput(Throughput::Elements((len * len) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |bench, _| {
            bench.iter(|| dtw_distance(black_box(&a), black_box(&b), black_box(None)));
        });
        group.bench_with_input(BenchmarkId::new("windowed", len), &len, |bench, _| {
            bench.iter(|| dtw_distance(black_box(&a), black_box(&b), black_box(Some(len / 10))));
        });
    }
    group.finish();
}

fn bench_dp_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("resquiggle_dp_step");
    for &band in &[64usize, 256, 1024] {
        let signal = pseudo_floats(band, 0xDEAD);
        let previous = pseudo_floats(band, 0xBEEF);
        let level = 0.25f32;
        let mut buf = ViterbiBuffers::new(band);
        let mut scores = vec![0.0f32; band];
        let mut traceback = vec![0i32; band];
        group.throughput(Throughput::Elements(band as u64));
        group.bench_with_input(BenchmarkId::from_parameter(band), &band, |bench, _| {
            bench.iter(|| {
                dp_step_buffered(
                    black_box(&mut scores),
                    black_box(&mut traceback),
                    black_box(&previous),
                    black_box(level),
                    black_box(&signal),
                    black_box(0),
                    black_box(&mut buf),
                );
            });
        });
    }
    group.finish();
}

/// Full adaptive banded DP over a whole read (many bases), unlike
/// `bench_dp_step` which measures a single band step. This exercises the
/// per-base traceback bookkeeping — the hot path for the traceback-allocation
/// audit item — so the win scales with `n_bases`.
fn bench_adaptive_dp(c: &mut Criterion) {
    let mut group = c.benchmark_group("resquiggle_adaptive_dp");
    let bandwidth = 100usize;
    // (n_bases, samples-per-base) — long reads are where the allocation bites.
    for &(n_bases, dwell) in &[(200usize, 10usize), (1000, 10)] {
        let signal_len = n_bases * dwell;
        let signal = pseudo_floats(signal_len, 0x5165);
        let levels = pseudo_floats(n_bases, 0x1EE1);
        // Monotonic signal-position map of length n_bases + 1, evenly spaced.
        let initial_map: Vec<usize> = (0..=n_bases).map(|i| (i * signal_len) / n_bases).collect();
        group.throughput(Throughput::Elements(n_bases as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_bases}bases")),
            &n_bases,
            |bench, _| {
                bench.iter(|| {
                    adaptive_banded_dp(
                        black_box(&signal),
                        black_box(&levels),
                        black_box(bandwidth),
                        black_box(&initial_map),
                        black_box(&RefineAlgo::Viterbi),
                        black_box(None),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_fingerprint_mad(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint_mad_normalize");
    for &len in &[64usize, 200, 1000] {
        let values = pseudo_floats(len, 0xF1)
            .into_iter()
            .map(|v| v + 1.0)
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |bench, _| {
            bench.iter_batched(
                || Fingerprint::new(values.clone(), escapepod_signal::types::Uuid::nil()),
                |mut fp| {
                    normalize_fingerprint(black_box(&mut fp), NormMethod::Median);
                    fp
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_vbz_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("vbz");
    for &len in &[1000usize, 10_000, 100_000] {
        let samples = pseudo_signal(len, 0xC0DE);
        group.throughput(Throughput::Bytes((len * 2) as u64));

        group.bench_with_input(BenchmarkId::new("encode", len), &len, |bench, _| {
            bench.iter(|| vbz::compress_signal(black_box(&samples)).unwrap());
        });

        let compressed = vbz::compress_signal(&samples).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", len), &len, |bench, _| {
            bench.iter(|| {
                vbz::decompress_signal(black_box(&compressed), black_box(samples.len())).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_dtw_matrix(c: &mut Criterion) {
    use escapepod_signal::dtw::dtw_distance_matrix;
    configure_rayon();
    let mut group = c.benchmark_group("dtw_distance_matrix");
    group.sample_size(bench_sample_size(50));
    // training workload shape: (n_queries x n_refs) with moderate fingerprint length.
    for &(q, r, l) in &[(8usize, 8usize, 150usize), (32, 32, 150)] {
        let queries: Vec<Vec<f32>> = (0..q).map(|i| pseudo_floats(l, 0x100 + i as u64)).collect();
        let refs: Vec<Vec<f32>> = (0..r).map(|i| pseudo_floats(l, 0x200 + i as u64)).collect();
        group.throughput(Throughput::Elements((q * r) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{q}x{r}x{l}")),
            &(q, r, l),
            |bench, _| {
                bench.iter(|| {
                    dtw_distance_matrix(black_box(&queries), black_box(&refs), black_box(None))
                });
            },
        );
    }
    group.finish();
}

/// Direct A/B of median-of-slice via full `sort_unstable` (old) vs
/// `select_nth_unstable` (the Phase 2 change), across the array sizes the real
/// call sites see: 64-1024 (resquiggle `median_dwell`), ~16k (theil_sen
/// pairwise slopes), ~256k (SVM-training kernel median). `select_nth` is O(n)
/// vs O(n log n) but carries a larger constant, so this pins the crossover —
/// both variants pay the same per-iteration clone, so the sort-vs-select gap at
/// each size is apples-to-apples.
fn bench_median_select_vs_sort(c: &mut Criterion) {
    fn median_sort(v: &mut [f32]) -> f32 {
        v.sort_unstable_by(|a, b| a.total_cmp(b));
        let n = v.len();
        if n.is_multiple_of(2) {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        } else {
            v[n / 2]
        }
    }
    fn median_select(v: &mut [f32]) -> f32 {
        let n = v.len();
        let mid = n / 2;
        let (lo, pivot, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let upper = *pivot;
        if n.is_multiple_of(2) {
            let lower = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (lower + upper) / 2.0
        } else {
            upper
        }
    }

    let mut group = c.benchmark_group("median_select_vs_sort");
    for &len in &[64usize, 256, 1024, 16_384, 262_144] {
        let data = pseudo_floats(len, 0x5057 + len as u64);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("sort", len), &len, |b, _| {
            b.iter_batched_ref(
                || data.clone(),
                |v| black_box(median_sort(black_box(v))),
                criterion::BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("select", len), &len, |b, _| {
            b.iter_batched_ref(
                || data.clone(),
                |v| black_box(median_select(black_box(v))),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dtw,
    bench_dp_step,
    bench_adaptive_dp,
    bench_fingerprint_mad,
    bench_vbz_roundtrip,
    bench_dtw_matrix,
    bench_median_select_vs_sort,
);
criterion_main!(benches);

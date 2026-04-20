//! GPU DTW microbenchmarks.
//!
//! Only compiled with `--features gpu`. Times batched DTW distance-matrix
//! computation on the GPU and compares to the CPU rayon implementation at
//! the same problem sizes.
//!
//! Run with:
//!   cargo bench --features gpu --bench hot_paths_gpu
//!   cargo bench --features gpu --bench hot_paths_gpu -- gpu_dtw_matrix/small
//!
//! If no CUDA device is visible the harness prints a skip notice and exits
//! cleanly — no false failures on GPU-less machines.

#![cfg(feature = "gpu")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use escapepod_signal::dtw::{GpuDtwContext, dtw_distance_matrix};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::hint::black_box;

fn make_set(seed: u64, n: usize, len: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..len).map(|_| rng.random::<f32>()).collect())
        .collect()
}

fn bench_gpu_dtw_matrix(c: &mut Criterion) {
    let ctx = match GpuDtwContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[hot_paths_gpu] skipping — GPU init failed: {e}");
            return;
        }
    };

    let mut g = c.benchmark_group("gpu_dtw_matrix");
    // Problem sizes representative of real demux workloads:
    //   small  — model training-matrix shape (all-pairs within the training set)
    //   medium — per-batch classify shape (thousands of reads × tens of refs)
    //   large  — full-run classify shape
    let cases = [
        ("small", 32usize, 32usize, 110usize, 10usize),
        ("medium", 1024, 40, 110, 10),
        ("large", 8192, 40, 110, 10),
    ];

    for (label, n_q, n_r, len, band) in cases {
        let queries = make_set(11, n_q, len);
        let refs = make_set(22, n_r, len);
        g.throughput(Throughput::Elements((n_q * n_r) as u64));

        g.bench_with_input(BenchmarkId::new("cpu", label), &(), |b, _| {
            b.iter(|| {
                let m = dtw_distance_matrix(black_box(&queries), black_box(&refs), Some(band));
                black_box(m);
            })
        });

        g.bench_with_input(BenchmarkId::new("gpu", label), &(), |b, _| {
            b.iter(|| {
                let m = ctx
                    .distance_matrix(black_box(&queries), black_box(&refs), Some(band))
                    .expect("gpu matrix");
                black_box(m);
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench_gpu_dtw_matrix);
criterion_main!(benches);

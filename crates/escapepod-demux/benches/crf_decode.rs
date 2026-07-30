//! Microbenchmark for the CTC-CRF lattice decode.
//!
//! The decode is transcendental-bound — a 200-timestep read costs roughly 770k
//! `exp` and 260k `ln` across the two forward/backward passes and the posterior
//! softmax — so it is not a rounding error next to encoder inference. Measured
//! on the rna partition, the scalar decode was about as expensive as the whole
//! tract encoder (13.5 ms vs 13.9 ms per read), which is what makes the AVX2
//! path a prerequisite for the GPU encoder rather than a nicety.
//!
//! This benchmark exists so that claim stays checkable rather than becoming
//! folklore in a commit message.
//!
//! Run with:
//!   cargo bench --bench crf_decode
//!   cargo bench --bench crf_decode -- avx2

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use escapepod_demux::crf::{Backend, CrfLayout, CrfScratch, decode_with};

/// Encoder-shaped scores from a closed-form expression — same generator shape
/// as the bonito parity fixture, so the benchmark and the correctness test are
/// exercising the same numbers.
fn synthetic_scores(layout: &CrfLayout, t_len: usize) -> Vec<f32> {
    (0..t_len * layout.n_score)
        .map(|i| {
            let t = (i / layout.n_score) as f64;
            let c = (i % layout.n_score) as f64;
            ((0.1 * t + 0.01 * c).sin() * 2.0 + (0.013 * i as f64).cos()) as f32
        })
        .collect()
}

fn bench_decode(c: &mut Criterion) {
    // RNA004 barcode geometry: 4-mer states, 256 states, 1280 scores, and the
    // 2000-sample / stride-10 window the export fixes.
    let layout = CrfLayout::new(4, 4).expect("valid geometry");
    let t_len = 200;
    let scores = synthetic_scores(&layout, t_len);

    let mut group = c.benchmark_group("crf_decode");
    // One element = one read, so the reported throughput is reads/s directly.
    group.throughput(Throughput::Elements(1));

    let mut backends = vec![("scalar", Backend::Scalar)];
    #[cfg(target_arch = "x86_64")]
    if Backend::best_for(&layout) == Backend::Avx2 {
        backends.push(("avx2", Backend::Avx2));
    }

    for (name, backend) in backends {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &backend,
            |b, &backend| {
                let mut scratch = CrfScratch::new();
                b.iter(|| {
                    black_box(
                        decode_with(
                            &layout,
                            b"NACGT",
                            black_box(&scores),
                            t_len,
                            &mut scratch,
                            backend,
                        )
                        .expect("decode succeeds"),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);

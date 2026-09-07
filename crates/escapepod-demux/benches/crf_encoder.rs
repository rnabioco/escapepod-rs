//! Microbenchmark for the CTC-CRF *encoder*: one read's standardised window
//! through the bundle's ONNX graph to `[T, n_score]` scores.
//!
//! The counterpart to `crf_decode.rs`. The encoder is ~91% of the CPU demux
//! path — 13.0 ms/read through tract against 1.2 ms for the AVX-512 decode
//! (#173) — and the number this file pins is the "before" for running the
//! graph's five-layer LSTM stack natively (rnabioco/escapepod-rs#331). It
//! measures the encoder alone (`encode/tract`) and the per-read production
//! cost, encoder plus decode out of tract's own output tensor
//! (`basecall/tract`).
//!
//! Needs a real bundle, which is not redistributable and so does not exist
//! in CI: set `ESCAPEPOD_CRF_BUNDLE` to a directory holding `metadata.json`
//! and the ONNX graph it names, and the group is skipped with a message
//! otherwise. `ESCAPEPOD_BENCH_SAMPLES=N` overrides the sample count.
//!
//! Run with:
//!   ESCAPEPOD_CRF_BUNDLE=/path/to/barcode_crf_ldx32_rna004@v0.2.2 \
//!     cargo bench -p escapepod-demux --features crf-decode --bench crf_encoder

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use escapepod_demux::crf::{CrfEncoder, CrfScratch};

fn bench_sample_size(default: usize) -> usize {
    std::env::var("ESCAPEPOD_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Deterministic pseudo-random stream (xorshift, seeded), so every run of
/// this file encodes the same window.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    /// Roughly standard normal (twelve uniforms, centred), which is what a
    /// standardised window looks like to the graph. The timing does not
    /// depend on the values; the decode's path does, and a plausible
    /// distribution keeps `basecall/tract` off degenerate all-blank paths.
    fn normal(&mut self) -> f32 {
        let sum: f32 = (0..12).map(|_| self.next() as f32 / u32::MAX as f32).sum();
        sum - 6.0
    }
}

fn bench_encoder(c: &mut Criterion) {
    let Ok(dir) = std::env::var("ESCAPEPOD_CRF_BUNDLE") else {
        eprintln!(
            "skipping the `crf_encoder` group: set ESCAPEPOD_CRF_BUNDLE to a \
             CRF bundle directory to measure it"
        );
        return;
    };
    let encoder = match CrfEncoder::load_bundle(&dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping the `crf_encoder` group: {dir} did not load: {e:#}");
            return;
        }
    };
    let meta = encoder.metadata();
    let chunk = meta.signal.chunk;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let window: Vec<f32> = (0..chunk).map(|_| rng.normal()).collect();
    eprintln!(
        "crf_encoder: {dir}: chunk {chunk}, stride {}, T = {}, decode backend {:?}",
        meta.signal.stride,
        meta.t_len(),
        encoder.backend()
    );

    // The bundle names the benchmark, so criterion's saved baseline for one
    // bundle is never "compared" against a run over another with a different
    // `chunk` — that reads as a +15% regression and is nothing of the kind.
    let bundle = match &meta.model {
        Some(m) => match &m.version {
            Some(v) => format!("{}@{v}", m.id),
            None => m.id.clone(),
        },
        None => std::path::Path::new(&dir)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.clone()),
    };

    let mut g = c.benchmark_group("crf_encoder");
    g.sample_size(bench_sample_size(30));
    // One element = one read, so the reported throughput is reads/s directly.
    g.throughput(Throughput::Elements(1));
    g.bench_function(BenchmarkId::new("encode/tract", &bundle), |b| {
        b.iter(|| black_box(encoder.encode(black_box(&window)).expect("encode succeeds")))
    });
    g.bench_function(BenchmarkId::new("basecall/tract", &bundle), |b| {
        let mut scratch = CrfScratch::new();
        b.iter(|| {
            black_box(
                encoder
                    .basecall_prepped(black_box(&window), &mut scratch)
                    .expect("basecall succeeds"),
            )
        })
    });
    g.finish();
}

criterion_group!(benches, bench_encoder);
criterion_main!(benches);

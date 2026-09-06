//! Microbenchmarks for the per-read charging path.
//!
//! This is the step `escpod classify` repeats once per read — millions of
//! times per production sample — and until now nothing measured it. A
//! 2026-09-06 report that the classifier had "got 2x slower" could only be
//! answered by rebuilding an A/B out of released tarballs, because the
//! criterion suites cover pod5 I/O, the signal hot paths and demux, and stop
//! there. The end-to-end counterpart to this file is `benchmarks/charging.sh`.
//!
//! Covers the chain from an anchored read to a feature vector:
//! - `finalize` — offsets to signal spans (the mask and span-mode rules)
//! - `expected_levels_z` — k-mer levels for the read, z-scored
//! - `junction_features` — per-base dwell/mean/std/resid over those spans
//! - `feature_grid` — all three, which is what the pipeline actually calls
//!
//! Everything is synthetic and self-contained on purpose: `ext/` submodule
//! fixtures do not exist in CI (the `.gitmodules` is empty), so a bench that
//! needed a real POD5 would be a bench that never runs there.
//!
//! Run with:
//!   cargo bench -p escapepod-classify --bench charging
//!   cargo bench -p escapepod-classify --bench charging -- feature_grid
//!
//! Env vars:
//!   ESCAPEPOD_BENCH_SAMPLES=N     criterion sample size (default 100).
//!   ESCAPEPOD_CHARGING_BUNDLE=DIR Score a REAL bundle directory as well.
//!                                 Needs `--features fnn-onnx`; without the
//!                                 var the scorer group is skipped, since the
//!                                 weights are not redistributable and CI has
//!                                 none. Adds `scorer/{batch}` groups.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use escapepod_classify::anchor::{AnchorSource, SpanMode};
use escapepod_classify::features::{expected_levels_z, junction_features};
use escapepod_classify::{AnchoredRead, FeatureRecipe, KmerLevels, Orientation, feature_grid};
use std::collections::HashMap;
use uuid::Uuid;

/// The shipped bundles' grid: 33 offsets, -8 into the tRNA body through +24
/// along the common arm.
const OFFSETS: [i32; 33] = [
    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24,
];

/// Bases per read, and samples per base.
///
/// A tRNA read is short — the mature body plus the ligated adapter — and
/// RNA004 at ~130 b/s against a 4 kHz sampler puts roughly 30 samples on a
/// base. 150 x 30 keeps the signal slice at a realistic 4,500 samples, which
/// matters because the per-read median/MAD gauge is O(ns).
const N_BASES: usize = 150;
const SAMPLES_PER_BASE: i64 = 30;

/// The junction sits far enough in that every negative offset resolves.
const Q_JUNCTION: usize = 60;

fn bench_sample_size(default: usize) -> usize {
    std::env::var("ESCAPEPOD_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Deterministic pseudo-random stream (xorshift, seeded) — same shape as the
/// signal-crate benches, so numbers from the two are generated alike.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn float(&mut self) -> f32 {
        (self.next() as u32 as f32) / (u32::MAX as f32)
    }
}

/// A synthetic read in the shape `feature_grid` consumes.
///
/// One base per `SAMPLES_PER_BASE` samples and a fully-aligned `qf`, so the
/// span resolution does its real work rather than bailing on unaligned
/// offsets. `q_div_m1` is set, so the common-arm mask is exercised too —
/// leaving it `None` would skip the branch that decides which samples are
/// even readable.
fn synthetic_read(seed: u64) -> AnchoredRead {
    let mut rng = Rng(seed | 1);
    let seq: Vec<u8> = (0..N_BASES)
        .map(|_| b"ACGT"[(rng.next() % 4) as usize])
        .collect();
    let qf: Vec<i64> = OFFSETS
        .iter()
        .map(|&o| Q_JUNCTION as i64 + o as i64)
        .collect();
    AnchoredRead {
        read_id: Uuid::nil(),
        reference: "trna".into(),
        mapq: 60,
        ns: N_BASES as i64 * SAMPLES_PER_BASE,
        seq,
        seq_to_sig: (0..=N_BASES as i64).map(|i| i * SAMPLES_PER_BASE).collect(),
        nb: N_BASES,
        q_junction: Q_JUNCTION,
        q_cca_a: Q_JUNCTION - 1,
        ts: 0,
        // The arm's first base in time: the mask boundary the recipe applies.
        q_div_m1: Some(Q_JUNCTION + 16),
        q_body_mid: None,
        q_polya_mid: None,
        qf,
        anchor_source: AnchorSource::Exact,
    }
}

fn synthetic_signal(seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    (0..(N_BASES as i64 * SAMPLES_PER_BASE) as usize)
        .map(|_| 80.0 + rng.float() * 40.0)
        .collect()
}

/// A full 9-mer level table, as the shipped bundles carry.
///
/// Built at full size (4^9 = 262,144 entries) rather than only the k-mers the
/// synthetic read contains: `expected_levels_z` looks levels up in a
/// `HashMap<String, f64>`, and a table small enough to sit in L2 measures a
/// cache the production path does not have.
fn kmer_levels() -> KmerLevels {
    const K: usize = 9;
    let mut map = HashMap::with_capacity(4usize.pow(K as u32));
    let mut kmer = [b'A'; K];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for i in 0..4usize.pow(K as u32) {
        let mut v = i;
        for slot in kmer.iter_mut() {
            *slot = b"ACGT"[v % 4];
            v /= 4;
        }
        // Levels sit around the RNA004 range; the absolute scale is divided
        // out by the z-score, so only the spread has to be non-degenerate.
        map.insert(
            String::from_utf8_lossy(&kmer).into_owned(),
            80.0 + rng.float() as f64 * 40.0,
        );
    }
    KmerLevels {
        map,
        k: K,
        center_idx: 4,
    }
}

/// The whole per-read feature step, which is what `classify_reads` calls.
///
/// Both span modes, because a bundle picks one and they resolve offsets
/// differently: `Aligner` takes the aligner's placement only, `Counted`
/// walks the query past the end of the alignment (the shipped
/// `count_arm_bases: 24` rule), so the two do different amounts of work per
/// read and a regression can live in one alone.
fn bench_feature_grid(c: &mut Criterion) {
    let levels = kmer_levels();
    let read = synthetic_read(1);
    let sig = synthetic_signal(2);

    let mut g = c.benchmark_group("feature_grid");
    g.sample_size(bench_sample_size(100));
    g.throughput(Throughput::Elements(1));
    for (name, mode) in [
        ("aligner", SpanMode::Aligner),
        ("counted_arm_24", SpanMode::Counted { arm_bases: 24 }),
    ] {
        let recipe = FeatureRecipe::new(&OFFSETS, mode, Some(&levels));
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(feature_grid(
                    black_box(&recipe),
                    black_box(&read),
                    Orientation::Reversed,
                    black_box(&sig),
                ))
            })
        });
    }
    g.finish();
}

/// The residual half on its own: k-mer lookup plus the per-read z-score.
///
/// Split out because it is the only part that touches the level table, and
/// the table is the thing a bundle pins by sha256 — a change in how it is
/// looked up shows here and nowhere else.
fn bench_expected_levels(c: &mut Criterion) {
    let levels = kmer_levels();
    let read = synthetic_read(1);
    let qf: Vec<i64> = OFFSETS
        .iter()
        .map(|&o| Q_JUNCTION as i64 + o as i64)
        .collect();

    let mut g = c.benchmark_group("expected_levels_z");
    g.sample_size(bench_sample_size(100));
    g.throughput(Throughput::Elements(1));
    g.bench_function("9mer", |b| {
        b.iter(|| {
            black_box(expected_levels_z(
                black_box(&read.seq),
                black_box(&levels.map),
                levels.k,
                levels.center_idx,
                black_box(&qf),
                read.nb,
            ))
        })
    });
    g.finish();
}

/// The span statistics on their own: the per-read median/MAD gauge plus two
/// passes over every offset's samples.
///
/// The gauge is O(signal length) while the statistics are O(window), so this
/// is the group that moves when read length changes rather than when the
/// recipe does.
fn bench_junction_features(c: &mut Criterion) {
    let levels = kmer_levels();
    let read = synthetic_read(1);
    let sig = synthetic_signal(2);
    let recipe = FeatureRecipe::new(&OFFSETS, SpanMode::Counted { arm_bases: 24 }, Some(&levels));
    let coords = escapepod_classify::finalize(
        &read,
        Orientation::Reversed,
        recipe.offsets,
        recipe.span_mode,
    );
    let expected = expected_levels_z(
        &read.seq,
        &levels.map,
        levels.k,
        levels.center_idx,
        &escapepod_classify::query_positions(&read, recipe.offsets, recipe.span_mode),
        read.nb,
    );

    let mut g = c.benchmark_group("junction_features");
    g.sample_size(bench_sample_size(100));
    g.throughput(Throughput::Elements(1));
    g.bench_function("33_offsets", |b| {
        b.iter(|| {
            black_box(junction_features(
                black_box(&sig),
                black_box(&coords),
                Some(black_box(&expected)),
            ))
        })
    });
    g.finish();
}

/// Span resolution alone — offsets to `(start, end)` sample pairs.
fn bench_finalize(c: &mut Criterion) {
    let read = synthetic_read(1);

    let mut g = c.benchmark_group("finalize");
    g.sample_size(bench_sample_size(100));
    g.throughput(Throughput::Elements(1));
    for (name, mode) in [
        ("aligner", SpanMode::Aligner),
        ("counted_arm_24", SpanMode::Counted { arm_bases: 24 }),
    ] {
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                black_box(escapepod_classify::finalize(
                    black_box(&read),
                    Orientation::Reversed,
                    &OFFSETS,
                    mode,
                ))
            })
        });
    }
    g.finish();
}

/// The scorer, against a real bundle named by `ESCAPEPOD_CHARGING_BUNDLE`.
///
/// Skipped without the var, and the group says why rather than failing: the
/// weights are not redistributable, so CI has no bundle, and a bench that
/// required one would simply never run there.
///
/// Two measurements, because they regress for different reasons. `predict` is
/// the ONNX graph through tract — the number that moved 6.1x when the
/// convolution padding was hoisted out of it (#233), and that a re-exported
/// bundle can move again without any code change here. `per_read` adds
/// `select_columns`, the fold from the canonical `offsets x FEAT_STATS` grid
/// to the model's own column order, which is the part this crate owns.
#[cfg(feature = "fnn-onnx")]
fn bench_scorer(c: &mut Criterion) {
    use escapepod_classify::{ChargingBundle, ChargingScorer};

    let Ok(dir) = std::env::var("ESCAPEPOD_CHARGING_BUNDLE") else {
        eprintln!(
            "skipping the `scorer` group: set ESCAPEPOD_CHARGING_BUNDLE to a \
             bundle directory to measure it"
        );
        return;
    };
    let bundle = match ChargingBundle::load(std::path::Path::new(&dir)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping the `scorer` group: {dir} did not load: {e:#}");
            return;
        }
    };
    let ChargingScorer::FeatureNn(net) = &bundle.scorer else {
        eprintln!("skipping the `scorer` group: {dir} is not a `feature_model` bundle");
        return;
    };
    let Ok(recipe) = bundle.recipe() else {
        eprintln!("skipping the `scorer` group: {dir} declares no column feature space");
        return;
    };

    // A grid in the bundle's OWN offset count, not this file's 33: a bundle
    // is free to have been built over a different one, and select_columns
    // indexes into the grid it declares.
    let mut rng = Rng(3);
    let grid: Vec<f32> = (0..recipe.offsets.len() * escapepod_classify::FEAT_STATS.len())
        .map(|_| rng.float() * 2.0 - 1.0)
        .collect();
    let columns = match bundle.select_columns(&grid) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping the `scorer` group: column selection failed: {e:#}");
            return;
        }
    };

    let mut g = c.benchmark_group("scorer");
    g.sample_size(bench_sample_size(50));
    g.throughput(Throughput::Elements(1));
    g.bench_function("predict", |b| {
        b.iter(|| black_box(net.predict(black_box(&columns))))
    });
    g.bench_function("per_read", |b| {
        b.iter(|| {
            let cols = bundle.select_columns(black_box(&grid)).unwrap();
            black_box(net.predict(&cols))
        })
    });
    g.finish();
}

#[cfg(not(feature = "fnn-onnx"))]
fn bench_scorer(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_feature_grid,
    bench_expected_levels,
    bench_junction_features,
    bench_finalize,
    bench_scorer,
);
criterion_main!(benches);

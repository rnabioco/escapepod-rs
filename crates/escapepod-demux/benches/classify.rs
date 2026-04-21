//! Microbenchmarks for the demux classify hot paths.
//!
//! Covers:
//! - `classify_read` (WarpDemuXModel) — per-read DTW + top-2 + threshold.
//! - `compute_distances` (shared) — DTW distances over all training fingerprints.
//! - `SvmPredictor::{kernel_weighted_scores, decision_function}` — OvO scoring
//!   pipeline from kernel values.
//!
//! Synthetic model shape modeled after WDX4 (32 classes × 4 fingerprints per class,
//! 150-sample features). Fingerprint values are deterministic pseudo-random f64.
//!
//! Run with:
//!   cargo bench --bench classify
//!   cargo bench --bench classify -- classify_read

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::collections::HashMap;
use std::hint::black_box;

use escapepod_demux::{
    DtwSvmModel, KernelParams, SvmPredictor, WarpDemuxModel, classify_read, compute_distances,
    distances_to_kernel,
};

fn pseudo_floats_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as u32 as f64) / (u32::MAX as f64)) * 2.0 - 1.0
        })
        .collect()
}

/// Build a synthetic `WarpDemuXModel` with `n_classes × per_class` training
/// fingerprints of length `feature_len`.
fn build_warpdemux_model(n_classes: usize, per_class: usize, feature_len: usize) -> WarpDemuxModel {
    let total = n_classes * per_class;
    let mut fingerprints: Vec<Vec<f64>> = Vec::with_capacity(total);
    let mut labels: Vec<i32> = Vec::with_capacity(total);
    let mut label_map: HashMap<String, i32> = HashMap::new();
    for c in 0..n_classes {
        label_map.insert(format!("BC{:02}", c), c as i32);
        for k in 0..per_class {
            let seed = 0x1000 + (c as u64) * 101 + k as u64;
            fingerprints.push(pseudo_floats_f64(feature_len, seed));
            labels.push(c as i32);
        }
    }
    WarpDemuxModel {
        training_fingerprints: fingerprints,
        training_labels: labels,
        kernel_params: KernelParams {
            gamma: 1.0,
            power: 2.0,
        },
        label_map,
        threshold: 0.5,
        threshold_type: "kernel".to_string(),
    }
}

/// Build a synthetic `DtwSvmModel` with the same shape as the WarpDemuX
/// model, using kernel-weighted voting (so we don't need real SVM coefficients).
fn build_svm_model(n_classes: usize, per_class: usize, feature_len: usize) -> DtwSvmModel {
    let total = n_classes * per_class;
    let mut fingerprints: Vec<Vec<f64>> = Vec::with_capacity(total);
    let mut labels: Vec<i32> = Vec::with_capacity(total);
    let mut label_mapper: HashMap<usize, i32> = HashMap::new();
    for c in 0..n_classes {
        label_mapper.insert(c, c as i32);
        for k in 0..per_class {
            let seed = 0x2000 + (c as u64) * 101 + k as u64;
            fingerprints.push(pseudo_floats_f64(feature_len, seed));
            labels.push(c as i32);
        }
    }
    let classes: Vec<i32> = (0..n_classes as i32).collect();
    let n_pairs = n_classes * (n_classes - 1) / 2;
    DtwSvmModel {
        version: "1.0".to_string(),
        training_fingerprints: fingerprints,
        training_labels: labels,
        support_indices: (0..total).collect(),
        dual_coef: vec![vec![0.0; total]; n_classes.saturating_sub(1)],
        intercept: vec![0.0; n_pairs],
        classes,
        kernel_params: KernelParams {
            gamma: 1.0,
            power: 2.0,
        },
        window: None,
        label_mapper,
        thresholds: None,
        prob_a: None,
        prob_b: None,
        n_classes,
        noise_class: false,
        use_kernel_weighted: true,
    }
}

fn bench_classify_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("classify_read");
    for &(n_classes, per_class, flen) in &[(32usize, 4usize, 150usize), (96, 4, 150)] {
        let model = build_warpdemux_model(n_classes, per_class, flen);
        let query = pseudo_floats_f64(flen, 0xF00DCAFE);
        let total = n_classes * per_class;
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_classes}x{per_class}x{flen}")),
            &(),
            |bench, _| {
                bench.iter(|| classify_read(black_box(&model), black_box(&query)));
            },
        );
    }
    group.finish();
}

fn bench_compute_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("classify_compute_distances");
    for &(n_classes, per_class, flen) in &[(32usize, 4usize, 150usize), (96, 4, 150)] {
        let model = build_warpdemux_model(n_classes, per_class, flen);
        let query = pseudo_floats_f64(flen, 0xF00DCAFE);
        let total = n_classes * per_class;
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_classes}x{per_class}x{flen}")),
            &(),
            |bench, _| {
                bench.iter(|| {
                    compute_distances(
                        black_box(&query),
                        black_box(&model.training_fingerprints),
                        None,
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_svm_pipeline(c: &mut Criterion) {
    // Per-read pipeline after DTW: kernel transform + OvO voting.
    let mut group = c.benchmark_group("classify_svm_pipeline");
    for &(n_classes, per_class, flen) in &[(32usize, 4usize, 150usize), (96, 4, 150)] {
        let model = build_svm_model(n_classes, per_class, flen);
        let query = pseudo_floats_f64(flen, 0xF00DCAFE);
        let distances = compute_distances(&query, &model.training_fingerprints, None);
        let kernel_values = distances_to_kernel(&distances, &model.kernel_params);
        let predictor = SvmPredictor::new(&model);

        group.bench_with_input(
            BenchmarkId::new("kernel_weighted_scores", format!("{n_classes}cls")),
            &(),
            |bench, _| {
                bench.iter(|| predictor.kernel_weighted_scores(black_box(&kernel_values)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("decision_function", format!("{n_classes}cls")),
            &(),
            |bench, _| {
                bench.iter(|| predictor.decision_function(black_box(&kernel_values)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_classify_read,
    bench_compute_distances,
    bench_svm_pipeline
);
criterion_main!(benches);

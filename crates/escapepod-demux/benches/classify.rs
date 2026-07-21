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
    DtwSvmModel, KernelParams, SvmPredictor, SvmWorkspace, WarpDemuxModel, classify_read,
    classify_with_svm, compute_distances, distances_to_kernel,
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

/// Build a synthetic `DtwSvmModel` that exercises the full Platt-scaled SVM
/// coupling path (so we can benchmark `couple_probabilities`). Uses real-ish
/// dual coefficients + Platt parameters so `predict` takes the coupled-
/// probability branch.
fn build_svm_model_platt(n_classes: usize, per_class: usize, feature_len: usize) -> DtwSvmModel {
    let total = n_classes * per_class;
    let mut fingerprints: Vec<Vec<f64>> = Vec::with_capacity(total);
    let mut labels: Vec<i32> = Vec::with_capacity(total);
    let mut label_mapper: HashMap<usize, i32> = HashMap::new();
    for c in 0..n_classes {
        label_mapper.insert(c, c as i32);
        for k in 0..per_class {
            let seed = 0x3000 + (c as u64) * 101 + k as u64;
            fingerprints.push(pseudo_floats_f64(feature_len, seed));
            labels.push(c as i32);
        }
    }
    let classes: Vec<i32> = (0..n_classes as i32).collect();
    let n_pairs = n_classes * (n_classes - 1) / 2;

    // Cheap synthetic dual coefficients: small alternating values so the SVM
    // decision function does real work without blowing up numerically.
    let dual_coef: Vec<Vec<f64>> = (0..n_classes.saturating_sub(1))
        .map(|r| {
            (0..total)
                .map(|i| {
                    let sign = if (r + i) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * 0.01
                })
                .collect()
        })
        .collect();

    DtwSvmModel {
        version: "1.0".to_string(),
        training_fingerprints: fingerprints,
        training_labels: labels,
        support_indices: (0..total).collect(),
        dual_coef,
        intercept: vec![0.0; n_pairs],
        classes,
        kernel_params: KernelParams {
            gamma: 1.0,
            power: 2.0,
        },
        window: None,
        penalty: 0.0,
        label_mapper,
        thresholds: None,
        prob_a: Some(vec![-1.0; n_pairs]),
        prob_b: Some(vec![0.0; n_pairs]),
        n_classes,
        noise_class: false,
        use_kernel_weighted: false,
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
        penalty: 0.0,
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
                        0.0,
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
        let distances = compute_distances(&query, &model.training_fingerprints, None, 0.0);
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

/// Benchmark the full SVM `predict` pipeline on a Platt-scaled model.
/// This exercises `couple_probabilities` — the allocation-heavy coupling
/// loop that runs per read in production — and compares `classify_with_svm`
/// (rebuilds `SvmPredictor` each call) against `predict_with_workspace`
/// (shares a pre-built predictor + reusable scratch buffers).
///
/// End-to-end `predict` is dominated by DTW, so the workspace win is lost in
/// noise here; `bench_svm_pipeline_post_dtw` below isolates the post-DTW
/// SVM path (kernel + decision + coupling) where the per-read allocation
/// savings actually show up.
fn bench_svm_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("classify_svm_predict");
    for &(n_classes, per_class, flen) in &[(32usize, 4usize, 150usize), (96, 4, 150)] {
        let model = build_svm_model_platt(n_classes, per_class, flen);
        let query = pseudo_floats_f64(flen, 0xF00DCAFE);

        group.bench_with_input(
            BenchmarkId::new("classify_with_svm", format!("{n_classes}cls")),
            &(),
            |bench, _| {
                bench.iter(|| classify_with_svm(black_box(&model), black_box(&query)));
            },
        );

        let predictor = SvmPredictor::new(&model);
        let mut ws = SvmWorkspace::for_model(&model);
        group.bench_with_input(
            BenchmarkId::new("predict_with_workspace", format!("{n_classes}cls")),
            &(),
            |bench, _| {
                bench.iter(|| predictor.predict_with_workspace(black_box(&query), &mut ws));
            },
        );
    }
    group.finish();
}

// NOTE: the biggest workspace win — avoiding per-read `k×k` coupling-matrix
// allocations in `couple_probabilities` — doesn't show up in a single-threaded
// microbench. `predict` is ~99% DTW time at 32 classes / 150 features, so
// shaving SVM-side allocations is lost in noise. The win is real in
// production par_iter workloads where per-thread heap contention matters;
// measure that end-to-end on a real pod5 via hyperfine.

criterion_group!(
    benches,
    bench_classify_read,
    bench_compute_distances,
    bench_svm_pipeline,
    bench_svm_predict,
);
criterion_main!(benches);

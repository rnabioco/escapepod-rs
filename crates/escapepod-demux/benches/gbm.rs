//! Microbenchmark for the production GBM scorer: `GbmPredictor::predict_many`
//! (the 8-lane lockstep walk) against per-read `predict`.
//!
//! Self-contained: the model is a synthetic ensemble of complete binary trees
//! shaped like the shipped barcode models (5 classes, 25 features), and the
//! fingerprints are deterministic pseudo-random vectors. Nothing from `ext/`.
//!
//! Run with:
//!   cargo bench -p escapepod-demux --bench gbm
//!   cargo bench -p escapepod-demux --bench gbm -- --quick

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::collections::HashMap;
use std::hint::black_box;

use escapepod_demux::{GbmHead, GbmModel, GbmNode, GbmPredictor, GbmTree};

fn xorshift(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state as u32) as f64) / (u32::MAX as f64)
}

/// A complete binary tree of `depth` levels in breadth-first layout (children
/// always follow their parent, as `GbmModel::validate` requires).
fn complete_tree(depth: usize, n_features: usize, state: &mut u64) -> GbmTree {
    let n_nodes = (1usize << depth) - 1;
    let first_leaf = (1usize << (depth - 1)) - 1;
    let nodes = (0..n_nodes)
        .map(|i| {
            if i >= first_leaf {
                GbmNode {
                    feature: -1,
                    threshold: 0.0,
                    left: 0,
                    right: 0,
                    missing_left: false,
                    value: xorshift(state) - 0.5,
                    leaf: true,
                }
            } else {
                GbmNode {
                    feature: ((xorshift(state) * n_features as f64) as usize % n_features) as i32,
                    threshold: xorshift(state),
                    left: (2 * i + 1) as u32,
                    right: (2 * i + 2) as u32,
                    missing_left: i % 2 == 0,
                    value: 0.0,
                    leaf: false,
                }
            }
        })
        .collect();
    GbmTree { nodes }
}

fn synthetic_model(n_classes: usize, n_features: usize, iterations: usize) -> GbmModel {
    let mut state = 0x6B_4D_u64;
    let trees = (0..iterations)
        .map(|_| {
            (0..n_classes)
                .map(|_| complete_tree(5, n_features, &mut state))
                .collect()
        })
        .collect();
    let model = GbmModel {
        model_type: "gbm".to_string(),
        head: GbmHead::Softmax,
        n_classes,
        n_features,
        baseline: vec![0.0; n_classes],
        trees,
        label_mapper: (0..n_classes)
            .map(|c| (c, c as i32))
            .collect::<HashMap<_, _>>(),
        thresholds: None,
        classes: None,
    };
    model.validate().expect("synthetic model is valid");
    model
}

fn fingerprints(n: usize, n_features: usize) -> Vec<Vec<f64>> {
    let mut state = 0xF1_9E_u64;
    (0..n)
        .map(|_| (0..n_features).map(|_| xorshift(&mut state)).collect())
        .collect()
}

fn bench_gbm_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbm_predict");
    for &(n_classes, n_features, iterations) in &[(5usize, 25usize, 100usize), (96, 25, 20)] {
        let model = synthetic_model(n_classes, n_features, iterations);
        let predictor = GbmPredictor::new(&model);
        let n_reads = 256;
        let fps = fingerprints(n_reads, n_features);
        let refs: Vec<&[f64]> = fps.iter().map(|v| v.as_slice()).collect();

        // Guard the thing being measured: the lockstep walk and the scalar
        // walk must agree, or the speed comparison is meaningless.
        let many = predictor.predict_many(&refs).unwrap();
        for (fp, (probs, _)) in refs.iter().zip(&many) {
            assert_eq!(&predictor.predict(fp).unwrap().0, probs);
        }

        let label = format!("{n_classes}cls_{iterations}it");
        group.throughput(Throughput::Elements(n_reads as u64));
        group.bench_with_input(
            BenchmarkId::new("predict_many_8lane", &label),
            &(),
            |b, _| {
                b.iter(|| predictor.predict_many(black_box(&refs)).unwrap());
            },
        );
        group.bench_with_input(BenchmarkId::new("predict_scalar", &label), &(), |b, _| {
            b.iter(|| {
                for fp in &refs {
                    black_box(predictor.predict(black_box(fp)).unwrap());
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gbm_predict);
criterion_main!(benches);

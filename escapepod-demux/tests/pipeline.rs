//! Integration tests for the demux pipeline orchestration.
//!
//! Exercises `classify_read` (distance-based) and the core SVM-adjacent
//! helpers (`compute_distances`, `distances_to_kernel`, `softmax`)
//! end-to-end on a synthetic `WarpDemuxModel`.

use std::collections::HashMap;
use tempfile::TempDir;

use escapepod_demux::{
    KernelParams, WarpDemuxModel, classify_read, compute_distances, distances_to_kernel,
    load_model, softmax,
};

/// Build a tiny WarpDemuxModel with three barcodes, each represented by
/// a single training fingerprint. The fingerprints are deliberately spread
/// apart so DTW can tell them apart.
fn synth_model() -> WarpDemuxModel {
    let training_fingerprints: Vec<Vec<f64>> = vec![
        (0..64).map(|i| ((i as f64).sin() * 0.5) + 0.5).collect(), // BC01
        (0..64).map(|i| ((i as f64) * 0.05).cos()).collect(),      // BC02
        (0..64).map(|i| if i < 32 { 1.0 } else { -1.0 }).collect(), // BC03 — step
    ];
    let training_labels = vec![1, 2, 3];
    let mut label_map = HashMap::new();
    label_map.insert("BC01".to_string(), 1);
    label_map.insert("BC02".to_string(), 2);
    label_map.insert("BC03".to_string(), 3);

    WarpDemuxModel {
        training_fingerprints,
        training_labels,
        kernel_params: KernelParams {
            gamma: 1.0,
            power: 1.0,
        },
        label_map,
        threshold: 0.9,
        threshold_type: "ratio".to_string(),
    }
}

#[test]
fn classify_read_identifies_best_match() {
    let model = synth_model();
    // Classify each training fingerprint against itself — must pick its own label.
    for (idx, fp) in model.training_fingerprints.iter().enumerate() {
        let result = classify_read(&model, fp);
        assert_eq!(
            result.best_match_index, idx,
            "self-classification picked wrong index for barcode {idx}"
        );
        // Self-DTW distance is 0 ⇒ fully confident.
        assert!(
            result.best_distance < 1e-6,
            "self distance > 0: {}",
            result.best_distance
        );
        assert!(result.is_confident, "self-match should be confident");
    }
}

#[test]
fn classify_read_rejects_ambiguous_query() {
    // Build a query that sits near the midpoint between BC01 and BC02 so the
    // best/second-best ratio stays above the 0.9 threshold.
    let mut model = synth_model();
    model.threshold = 0.1; // Even tighter — only extremely close matches are confident.
    let query: Vec<f64> = (0..64).map(|i| 0.3 + 0.01 * i as f64).collect();
    let result = classify_read(&model, &query);
    assert!(
        !result.is_confident,
        "expected unconfident classification with tight threshold, got {result:?}"
    );
    assert_eq!(result.barcode, "unclassified");
}

#[test]
fn compute_distances_shape_and_ordering() {
    let model = synth_model();
    let query = model.training_fingerprints[0].clone();
    let distances = compute_distances(&query, &model.training_fingerprints, None);
    assert_eq!(distances.len(), model.training_fingerprints.len());
    // Query == training[0], so distances[0] must be the minimum.
    let (min_idx, _) = distances
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(min_idx, 0);
    for &d in &distances {
        assert!(d >= 0.0 && d.is_finite(), "bad distance: {d}");
    }
}

#[test]
fn distances_to_kernel_is_bounded_0_1() {
    let distances = vec![0.0, 0.5, 1.0, 5.0, 20.0];
    let params = KernelParams {
        gamma: 1.0,
        power: 1.0,
    };
    let kernel = distances_to_kernel(&distances, &params);
    assert_eq!(kernel.len(), distances.len());
    // exp(-gamma * 0) = 1 at distance 0, then monotonically decreasing to 0.
    assert!((kernel[0] - 1.0).abs() < 1e-9);
    for w in kernel.windows(2) {
        assert!(w[0] >= w[1], "kernel must be monotonically non-increasing");
    }
    for &k in &kernel {
        assert!((0.0..=1.0).contains(&k), "kernel out of [0,1]: {k}");
    }
}

#[test]
fn softmax_produces_valid_distribution() {
    let logits = vec![1.0, 2.0, 3.0, 0.5];
    let probs = softmax(&logits);
    assert_eq!(probs.len(), logits.len());
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "probs do not sum to 1: {sum}");
    for &p in &probs {
        assert!((0.0..=1.0).contains(&p), "prob out of [0,1]: {p}");
    }
    // argmax(softmax) == argmax(logits)
    let (softmax_argmax, _) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let (logits_argmax, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(softmax_argmax, logits_argmax);
}

#[test]
fn warpdemux_model_save_and_load_round_trip() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("model.json");

    let model = synth_model();
    // Write via serde directly (load_model expects JSON; no explicit save API for legacy model).
    let json = serde_json::to_string_pretty(&model).expect("serialize");
    std::fs::write(&path, json).unwrap();

    let loaded = load_model(&path).expect("load_model");
    loaded.validate().expect("loaded model must validate");
    assert_eq!(loaded.num_samples(), model.training_fingerprints.len());
    assert_eq!(loaded.feature_dim(), 64);
    assert_eq!(loaded.get_barcode_name(2), "BC02");

    // Classification results on the loaded model match the in-memory model.
    for fp in &model.training_fingerprints {
        let r1 = classify_read(&model, fp);
        let r2 = classify_read(&loaded, fp);
        assert_eq!(r1.barcode, r2.barcode);
        assert_eq!(r1.best_match_index, r2.best_match_index);
    }
}

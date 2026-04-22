//! GPU/CPU parity for the batched SVM classifier. Gated on `--features gpu`;
//! skips transparently on hosts without a CUDA device.

#![cfg(feature = "gpu")]

use escapepod_demux::{DtwSvmModel, KernelParams, classify_with_svm};
use escapepod_signal::dtw::{GpuDtwContext, GpuDtwError};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn try_context() -> Option<GpuDtwContext> {
    let result = std::panic::catch_unwind(GpuDtwContext::new);
    match result {
        Ok(Ok(ctx)) => Some(ctx),
        Ok(Err(GpuDtwError::Driver(e))) => {
            eprintln!("[gpu_svm] skipping — CUDA driver not usable: {e}");
            None
        }
        Ok(Err(e)) => {
            eprintln!("[gpu_svm] skipping — GPU context init failed: {e}");
            None
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&'static str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".into());
            eprintln!(
                "[gpu_svm] skipping — GPU init panicked (likely missing libnvrtc / libcuda): {msg}"
            );
            None
        }
    }
}

fn make_svm_model(training: Vec<Vec<f64>>, labels: Vec<i32>) -> DtwSvmModel {
    let classes: Vec<i32> = {
        let mut c = labels.clone();
        c.sort_unstable();
        c.dedup();
        c
    };
    let n_classes = classes.len();
    let n_samples = training.len();
    let label_mapper: std::collections::HashMap<usize, i32> = classes
        .iter()
        .enumerate()
        .map(|(idx, &lab)| (idx, lab))
        .collect();
    let support_indices: Vec<usize> = (0..n_samples).collect();
    let n_pairs = n_classes * (n_classes - 1) / 2;

    DtwSvmModel {
        version: "1.0".to_string(),
        training_fingerprints: training,
        training_labels: labels,
        support_indices,
        // Kernel-weighted voting path exercises the full kernel+decision+probability
        // pipeline without needing libsvm-shaped dual coefficients.
        dual_coef: vec![vec![1.0 / n_samples as f64; n_samples]; n_classes.saturating_sub(1)],
        intercept: vec![0.0; n_pairs.max(1)],
        classes,
        kernel_params: KernelParams {
            gamma: 1.0,
            power: 1.0,
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

#[test]
fn parity_svm_classify_batch() {
    let Some(ctx) = try_context() else { return };

    let training: Vec<Vec<f64>> = vec![
        (0..110).map(|k| 0.1 * ((k % 7) as f64)).collect(),
        (0..110).map(|k| 0.5 + 0.05 * (k as f64).sin()).collect(),
        (0..110).map(|k| 0.9 - 0.02 * ((k % 5) as f64)).collect(),
        (0..110).map(|k| 0.05 * ((k % 11) as f64)).collect(),
    ];
    let labels = vec![0i32, 1, 2, 0];
    let model = make_svm_model(training, labels);

    let mut rng = StdRng::seed_from_u64(42);
    let queries: Vec<Vec<f64>> = (0..32)
        .map(|_| (0..110).map(|_| rng.random::<f64>()).collect())
        .collect();

    let cpu_results: Vec<_> = queries
        .iter()
        .map(|q| classify_with_svm(&model, q))
        .collect();

    let gpu_results = escapepod_demux::classify_with_svm_batch_gpu_with_ctx(
        &ctx,
        &model,
        &queries,
        escapepod_demux::DEFAULT_GPU_CHUNK_CELLS,
    )
    .expect("gpu svm batch ok");

    assert_eq!(gpu_results.len(), cpu_results.len());
    for (i, ((cpu_probs, cpu_res), (gpu_probs, gpu_res))) in
        cpu_results.iter().zip(gpu_results.iter()).enumerate()
    {
        assert_eq!(
            cpu_res.predicted_barcode, gpu_res.predicted_barcode,
            "prediction mismatch on query {i}: cpu={} gpu={}",
            cpu_res.predicted_barcode, gpu_res.predicted_barcode
        );
        assert_eq!(cpu_probs.len(), gpu_probs.len());
        for (k, (&cp, &gp)) in cpu_probs.iter().zip(gpu_probs.iter()).enumerate() {
            let diff = (cp - gp).abs();
            let tol = (1e-3_f64).max(1e-3 * cp.abs());
            assert!(
                diff <= tol,
                "probability mismatch on query {i} class {k}: cpu={cp} gpu={gp} diff={diff} tol={tol}"
            );
        }
    }
}

/// Same parity check, but with the real-OvO dual-coefficient path
/// (`use_kernel_weighted = false`). Exercises the
/// `coef_dev = dual_coef` branch in `GpuSvmContext::new` and the
/// `intercept != 0` path. Synthetic 3-class model with hand-crafted
/// dual coefficients per pair.
#[test]
fn parity_svm_classify_batch_real_ovo() {
    let Some(ctx) = try_context() else { return };

    let training: Vec<Vec<f64>> = vec![
        (0..110).map(|k| 0.1 * ((k % 7) as f64)).collect(),
        (0..110).map(|k| 0.5 + 0.05 * (k as f64).sin()).collect(),
        (0..110).map(|k| 0.9 - 0.02 * ((k % 5) as f64)).collect(),
        (0..110).map(|k| 0.05 * ((k % 11) as f64)).collect(),
        (0..110).map(|k| 0.6 + 0.03 * ((k % 9) as f64)).collect(),
        (0..110).map(|k| 0.2 + 0.07 * ((k % 13) as f64)).collect(),
    ];
    let labels = vec![0i32, 1, 2, 0, 1, 2];
    let mut model = make_svm_model(training, labels);
    // Switch to real OvO coefficients with non-trivial values; intercepts
    // non-zero so we exercise that path through the kernel.
    model.use_kernel_weighted = false;
    model.dual_coef = vec![
        vec![0.5, -0.3, 0.0, 0.4, -0.2, 0.0],
        vec![0.0, 0.6, -0.4, 0.0, 0.5, -0.3],
    ];
    model.intercept = vec![0.1, -0.2, 0.05];

    let mut rng = StdRng::seed_from_u64(7);
    let queries: Vec<Vec<f64>> = (0..16)
        .map(|_| (0..110).map(|_| rng.random::<f64>()).collect())
        .collect();

    let cpu_results: Vec<_> = queries
        .iter()
        .map(|q| classify_with_svm(&model, q))
        .collect();

    let gpu_results = escapepod_demux::classify_with_svm_batch_gpu_with_ctx(
        &ctx,
        &model,
        &queries,
        escapepod_demux::DEFAULT_GPU_CHUNK_CELLS,
    )
    .expect("gpu svm batch (real OvO) ok");

    assert_eq!(gpu_results.len(), cpu_results.len());
    for (i, ((cpu_probs, cpu_res), (gpu_probs, gpu_res))) in
        cpu_results.iter().zip(gpu_results.iter()).enumerate()
    {
        assert_eq!(
            cpu_res.predicted_barcode, gpu_res.predicted_barcode,
            "real-OvO prediction mismatch on query {i}: cpu={} gpu={}",
            cpu_res.predicted_barcode, gpu_res.predicted_barcode
        );
        for (k, (&cp, &gp)) in cpu_probs.iter().zip(gpu_probs.iter()).enumerate() {
            let diff = (cp - gp).abs();
            let tol = (1e-3_f64).max(1e-3 * cp.abs());
            assert!(
                diff <= tol,
                "real-OvO probability mismatch on query {i} class {k}: cpu={cp} gpu={gp} diff={diff} tol={tol}"
            );
        }
    }
}

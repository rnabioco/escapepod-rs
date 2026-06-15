//! Focused profiling harness for the SVM/DTW classify hot path (no POD5 I/O).
//!
//! Loads a real model, then runs `predict_with_workspace` in a tight loop over a
//! batch of synthetic queries so a profiler sees only the classify cost
//! (DTW distances → kernel → decision → coupling), not file I/O.
//!
//! Usage: profile_classify <model.json> [n_reads] [query_len]

use escapepod_demux::{AnyModel, SvmPredictor, SvmWorkspace, load_any_model};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: profile_classify <model.json> [n_reads] [query_len]");
    let n_reads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(2000);
    let qlen: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(0);

    let model = match load_any_model(model_path.as_ref())? {
        AnyModel::Svm(m) => m,
        AnyModel::WarpDemux(_) | AnyModel::Gbm(_) => anyhow::bail!("need an SVM model"),
    };
    let flen = if qlen > 0 {
        qlen
    } else {
        model.training_fingerprints.first().map_or(10, Vec::len)
    };
    eprintln!(
        "model: n_classes={} n_train={} fp_len={} use_kernel_weighted={} qlen={}",
        model.n_classes,
        model.training_fingerprints.len(),
        model.training_fingerprints.first().map_or(0, Vec::len),
        model.use_kernel_weighted,
        flen,
    );

    let predictor = SvmPredictor::new(&model);
    let mut ws = SvmWorkspace::for_model(&model);

    // A small bank of distinct queries so the optimizer can't fold the call.
    let queries: Vec<Vec<f64>> = (0..64)
        .map(|q| {
            (0..flen)
                .map(|i| ((i as f64) * 0.37 + (q as f64) * 0.11).sin())
                .collect()
        })
        .collect();

    let t0 = std::time::Instant::now();
    let mut checksum = 0.0f64;
    for r in 0..n_reads {
        let query = &queries[r % queries.len()];
        let (_probs, result) = predictor.predict_with_workspace(query, &mut ws);
        checksum += result.confidence;
    }
    let dt = t0.elapsed();
    eprintln!(
        "{} reads in {:.3?}  ->  {:.1} reads/s  ({:.1} us/read)   checksum={checksum:.3}",
        n_reads,
        dt,
        n_reads as f64 / dt.as_secs_f64(),
        dt.as_secs_f64() * 1e6 / n_reads as f64,
    );
    Ok(())
}

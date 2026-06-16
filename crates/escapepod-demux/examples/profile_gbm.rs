//! Focused profiling harness for the native GBM (tree-ensemble) classify hot
//! path (no POD5 I/O).
//!
//! Loads a real GBM model + a fingerprints CSV, then runs `GbmPredictor::predict`
//! in a tight single-threaded loop so a profiler / wall-clock sees only the
//! tree-walk + softmax cost, not file I/O or rayon scheduling.
//!
//! Usage: profile_gbm <model.json> <fingerprints.csv> [passes]
//!   passes: how many times to sweep the whole fingerprint set (default 50)

use escapepod_demux::{GbmPredictor, load_gbm_model};

/// Parse a headered CSV whose first column is a read_id and the rest are f64
/// features. Returns just the feature rows.
fn read_fps(path: &str) -> Vec<Vec<f64>> {
    let text = std::fs::read_to_string(path).expect("read fp csv");
    text.lines()
        .skip(1) // header
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .skip(1) // read_id
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: profile_gbm <model.json> <fingerprints.csv> [passes]");
    let fp_path = args
        .next()
        .expect("usage: profile_gbm <model.json> <fingerprints.csv> [passes]");
    let passes: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(50);

    let model = load_gbm_model(model_path.as_ref())?;
    let fps = read_fps(&fp_path);
    eprintln!(
        "model: n_classes={} n_features={} n_iterations={} | fps: {} rows, {} passes ({} predicts)",
        model.n_classes,
        model.n_features,
        model.n_iterations(),
        fps.len(),
        passes,
        fps.len() * passes,
    );

    let predictor = GbmPredictor::new(&model);

    // Warm pass (page in the model + caches) — not timed.
    let mut warm = 0.0f64;
    for fp in &fps {
        let (probs, _r) = predictor.predict(fp).unwrap();
        warm += probs[0];
    }
    std::hint::black_box(warm);

    let n = (fps.len() * passes) as f64;

    // --- scalar per-read path ---
    let t0 = std::time::Instant::now();
    let mut checksum = 0.0f64;
    let mut confident = 0usize;
    for _ in 0..passes {
        for fp in &fps {
            let (probs, result) = predictor.predict(fp).unwrap();
            checksum += probs[result.predicted_index];
            if result.is_confident {
                confident += 1;
            }
        }
    }
    let dt = t0.elapsed();
    eprintln!(
        "scalar : {:.0} predicts in {:.3?}  ->  {:.0} reads/s  ({:.2} us/read)   confident={confident} checksum={checksum:.3}",
        n,
        dt,
        n / dt.as_secs_f64(),
        dt.as_secs_f64() * 1e6 / n,
    );

    // --- batched (K-read lockstep) path, swept over lane counts ---
    let slices: Vec<&[f64]> = fps.iter().map(|v| v.as_slice()).collect();
    let run_k =
        |label: &str, f: &dyn Fn() -> Vec<(Vec<f64>, escapepod_demux::ProbabilityResult)>| {
            let t1 = std::time::Instant::now();
            let mut bchecksum = 0.0f64;
            let mut bconfident = 0usize;
            for _ in 0..passes {
                for (probs, result) in f() {
                    bchecksum += probs[result.predicted_index];
                    if result.is_confident {
                        bconfident += 1;
                    }
                }
            }
            let dt1 = t1.elapsed();
            eprintln!(
                "{label}: {:.0} reads/s  ({:.2} us/read)  speedup={:.2}x  parity: dchk={:.2e} dconf={}",
                n / dt1.as_secs_f64(),
                dt1.as_secs_f64() * 1e6 / n,
                dt.as_secs_f64() / dt1.as_secs_f64(),
                (checksum - bchecksum).abs(),
                confident as i64 - bconfident as i64,
            );
        };
    run_k("batch-4 ", &|| {
        predictor.predict_many_k::<4>(&slices).unwrap()
    });
    run_k("batch-8 ", &|| {
        predictor.predict_many_k::<8>(&slices).unwrap()
    });
    run_k("batch-12", &|| {
        predictor.predict_many_k::<12>(&slices).unwrap()
    });
    run_k("batch-16", &|| {
        predictor.predict_many_k::<16>(&slices).unwrap()
    });
    Ok(())
}

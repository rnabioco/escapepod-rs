//! Measure the headroom for lower-bound pruning of the kernel-weighted SVM
//! classify sum, on REAL reads. For each read we run the exact prep
//! (decode → LLR detect@downscale 10 → t-test fingerprint), then:
//!
//!   * compute every DTW distance d_k and kernel K_k = exp(-gamma*d_k^power),
//!   * report how concentrated the kernel mass is (how few neighbors hold
//!     99.9% / 99.9999% of the per-class sum),
//!   * compute two cheap DTW lower bounds (endpoint-corner and LB_Kim), and for
//!     a few error tolerances `eps` report what fraction of full DTWs could be
//!     skipped (`K_k <= exp(-gamma*LB^power) < eps` ⇒ treat as 0) and whether
//!     the argmax class prediction flips vs the exact computation.
//!
//! Usage: analyze_pruning <model.json> <reads.pod5> [max_reads]

use escapepod_demux::{AnyModel, extract_fingerprint_from_signal, load_any_model};
use escapepod_signal::Reader;
use escapepod_signal::dtw::{NormMethod, dtw_distance};
use escapepod_signal::segmentation::{detect_adapter, downscale, normalize_signal};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: analyze_pruning <model.json> <reads.pod5> [n]");
    let pod5_path = args
        .next()
        .expect("usage: analyze_pruning <model.json> <reads.pod5> [n]");
    let max_reads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(200);

    let model = match load_any_model(model_path.as_ref())? {
        AnyModel::Svm(m) => m,
        AnyModel::WarpDemux(_) => anyhow::bail!("need an SVM model"),
    };
    let gamma = model.kernel_params.gamma;
    let power = model.kernel_params.power;
    let n_classes = model.n_classes;
    // Per-training-fp class index (skip labels not in classes).
    let class_of: Vec<usize> = model
        .training_labels
        .iter()
        .map(|l| {
            model
                .classes
                .iter()
                .position(|c| c == l)
                .unwrap_or(usize::MAX)
        })
        .collect();
    let counts: Vec<f64> = {
        let mut v = vec![0.0; n_classes];
        for &c in &class_of {
            if c != usize::MAX {
                v[c] += 1.0;
            }
        }
        v
    };
    // f32 training set + per-fp endpoints / min / max for the lower bounds.
    let train: Vec<Vec<f32>> = model
        .training_fingerprints
        .iter()
        .map(|fp| fp.iter().map(|&x| x as f32).collect())
        .collect();
    let stats: Vec<(f32, f32, f32, f32)> = train
        .iter()
        .map(|t| {
            let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in t {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            (t[0], t[t.len() - 1], mn, mx)
        })
        .collect();

    let kernel = |d: f64| (-gamma * d.powf(power)).exp();

    // eps tolerances → LB distance cutoff τ such that exp(-gamma*τ^power)=eps.
    let epsilons = [1e-1_f64, 1e-2, 1e-4, 1e-6];
    let tau: Vec<f64> = epsilons
        .iter()
        .map(|&e| (-e.ln() / gamma).powf(1.0 / power))
        .collect();

    let reader = Reader::open(&pod5_path)?;
    let mut n_used = 0usize;
    // accumulators
    let mut mass_90 = 0.0f64;
    let mut mass_99 = 0.0f64;
    let mut mass_999 = 0.0f64; // mean #fps for 99.9% mass (fraction of N)
    let mut mass_9999 = 0.0f64;
    let mut frac_corner = vec![0.0f64; epsilons.len()];
    let mut frac_kim = vec![0.0f64; epsilons.len()];
    let mut flips_corner = vec![0usize; epsilons.len()];
    let mut flips_kim = vec![0usize; epsilons.len()];
    let n = train.len();

    'outer: for batch in reader.read_batches()? {
        let batch = batch?;
        let view = escapepod_signal::ReadsBatchView::new(&batch, false)?;
        for row in 0..view.num_rows() {
            let Ok(read) = view.read(row) else { continue };
            if read.signal_rows.is_empty() {
                continue;
            }
            let Ok(signal) = reader.get_signal(&read.signal_rows) else {
                continue;
            };
            // detect @ downscale 10 (pipeline default)
            let normalized = normalize_signal(&signal);
            let ds = 10usize;
            let trunc = (normalized.len() / ds) * ds;
            if trunc == 0 {
                continue;
            }
            let processed = downscale(&normalized[..trunc], ds);
            let (s, e) = detect_adapter(&processed, (200 / ds).max(1), (50 / ds).max(1));
            let (s, e) = (s * ds, e * ds);
            if e <= s {
                continue;
            }
            let Some(fp) = extract_fingerprint_from_signal(
                &signal,
                s,
                e,
                111,
                12,
                NormMethod::ZScore,
                read.read_id,
                Some(6),
                Some(25),
                false,
            ) else {
                continue;
            };
            let q: Vec<f32> = fp.values.iter().map(|&x| x as f32).collect();
            if q.is_empty() {
                continue;
            }
            let (q0, ql) = (q[0], q[q.len() - 1]);
            let (mut qmn, mut qmx) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in &q {
                qmn = qmn.min(v);
                qmx = qmx.max(v);
            }

            // exact distances + kernels + class scores
            let mut k_vals = vec![0.0f64; n];
            let mut exact_scores = vec![0.0f64; n_classes];
            for (idx, t) in train.iter().enumerate() {
                let d = dtw_distance(&q, t, None) as f64;
                let k = kernel(d);
                k_vals[idx] = k;
                let c = class_of[idx];
                if c != usize::MAX {
                    exact_scores[c] += k;
                }
            }
            let exact_pred = argmax_norm(&exact_scores, &counts);

            // kernel-mass concentration (global, across all classes)
            let total: f64 = k_vals.iter().sum();
            if total > 0.0 {
                let mut sorted = k_vals.clone();
                sorted.sort_unstable_by(|a, b| b.total_cmp(a));
                mass_90 += frac_for_mass(&sorted, total, 0.90) / n as f64;
                mass_99 += frac_for_mass(&sorted, total, 0.99) / n as f64;
                mass_999 += frac_for_mass(&sorted, total, 0.999) / n as f64;
                mass_9999 += frac_for_mass(&sorted, total, 0.999999) / n as f64;
            }

            // lower-bound pruning at each eps, for both bounds
            for (ei, &t_cut) in tau.iter().enumerate() {
                for (bound_kim, frac_acc, flips_acc) in [
                    (false, &mut frac_corner, &mut flips_corner),
                    (true, &mut frac_kim, &mut flips_kim),
                ] {
                    let mut scores = vec![0.0f64; n_classes];
                    let mut computed = 0usize;
                    for (idx, &(c0, cl, cmn, cmx)) in stats.iter().enumerate() {
                        let corner = ((q0 - c0).powi(2) + (ql - cl).powi(2)).sqrt();
                        let lb = if bound_kim {
                            let mx = (qmx - cmx).max(0.0);
                            let mn = (cmn - qmn).max(0.0);
                            corner.max(mx).max(mn) as f64
                        } else {
                            corner as f64
                        };
                        if lb > t_cut {
                            continue; // prune: K ≈ 0
                        }
                        computed += 1;
                        let c = class_of[idx];
                        if c != usize::MAX {
                            scores[c] += k_vals[idx]; // reuse exact K (we'd compute it for real)
                        }
                    }
                    frac_acc[ei] += computed as f64 / n as f64;
                    if argmax_norm(&scores, &counts) != exact_pred {
                        flips_acc[ei] += 1;
                    }
                }
            }

            n_used += 1;
            if n_used >= max_reads {
                break 'outer;
            }
        }
    }

    if n_used == 0 {
        anyhow::bail!("no usable reads");
    }
    let nu = n_used as f64;
    println!("model: n_train={n} n_classes={n_classes} gamma={gamma} power={power}");
    println!("reads analyzed: {n_used}");
    println!("kernel-mass concentration (mean fraction of {n} fps holding the mass):");
    for (label, acc) in [
        ("90%", mass_90),
        ("99%", mass_99),
        ("99.9%", mass_999),
        ("99.9999%", mass_9999),
    ] {
        println!(
            "  {label:<9}  {:>7.3}%   ({:.0} fps)",
            100.0 * acc / nu,
            n as f64 * acc / nu,
        );
    }
    println!("\nlower-bound pruning (mean fraction of DTWs still COMPUTED; lower = more pruned):");
    println!("  eps        corner-LB   (flips)     LB_Kim     (flips)");
    for (ei, &eps) in epsilons.iter().enumerate() {
        println!(
            "  {:<9.0e}  {:>7.2}%    {:>3}/{:<3}    {:>7.2}%    {:>3}/{:<3}",
            eps,
            100.0 * frac_corner[ei] / nu,
            flips_corner[ei],
            n_used,
            100.0 * frac_kim[ei] / nu,
            flips_kim[ei],
            n_used,
        );
    }
    Ok(())
}

fn argmax_norm(scores: &[f64], counts: &[f64]) -> usize {
    scores
        .iter()
        .zip(counts)
        .map(|(s, c)| if *c > 0.0 { s / c } else { 0.0 })
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Number of (descending-sorted) kernels needed to reach `frac` of `total`.
fn frac_for_mass(sorted_desc: &[f64], total: f64, frac: f64) -> f64 {
    let target = total * frac;
    let mut acc = 0.0;
    for (i, &k) in sorted_desc.iter().enumerate() {
        acc += k;
        if acc >= target {
            return (i + 1) as f64;
        }
    }
    sorted_desc.len() as f64
}

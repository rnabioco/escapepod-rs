//! Parity between the Rust CTC-CRF decode and bonito's own.
//!
//! `fixtures/crf_golden.json` was produced by running the real
//! `bonito.crf.model.CTC_CRF` — through `koi`'s CUDA kernels, on an A30 — over a
//! score tensor defined by a closed-form expression rather than an RNG. Because
//! the input is a formula, this test regenerates it in Rust and needs no
//! multi-megabyte array checked into the repository, and because the decode
//! itself has no dependencies, **this runs in CI**: it does not need a GPU, an
//! ONNX file, or anything under `ext/`.
//!
//! The generator and the full `.npy` intermediates (alpha/beta under both
//! semirings, posteriors, the pass-2 input) live at
//! `~/scratch/ldx/crf_golden{,.py}` on Beevol for debugging a mismatch; the
//! committed JSON keeps the sequences, the per-timestep traceback, and
//! checksums.

use std::path::Path;

use escapepod_demux::crf::{Backend, CrfLayout, CrfScratch, decode_with};

fn golden() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/crf_golden.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&raw).expect("fixture is valid JSON")
}

/// Regenerate read `n` of the fixture's score tensor.
///
/// `scores[t, n, c] = sin(a_t*t + a_c*c + a_n*n) * amp + cos(b * (t*C + c))`,
/// evaluated in `f64` and cast to `f32` — the cosine term is deliberately
/// independent of `n` so the two reads share structure without being related by
/// a constant offset.
fn synthesize(f: &serde_json::Value, n: usize) -> Vec<f32> {
    let g = |k: &str| f[k].as_f64().unwrap();
    let (t_len, n_score) = (f["T"].as_u64().unwrap(), f["C"].as_u64().unwrap());
    let (a_t, a_c, a_n, amp, b) = (g("a_t"), g("a_c"), g("a_n"), g("amp"), g("b"));
    let mut out = Vec::with_capacity((t_len * n_score) as usize);
    for t in 0..t_len {
        for c in 0..n_score {
            let s = (a_t * t as f64 + a_c * c as f64 + a_n * n as f64).sin() * amp;
            let k = (b * (t * n_score + c) as f64).cos();
            out.push((s + k) as f32);
        }
    }
    out
}

fn layout_from(fixture: &serde_json::Value) -> CrfLayout {
    let sd = &fixture["seqdist"];
    let layout = CrfLayout::new(
        sd["n_base"].as_u64().unwrap() as usize,
        sd["state_len"].as_u64().unwrap() as usize,
    )
    .unwrap();
    assert_eq!(layout.n_states, sd["n_states"].as_u64().unwrap() as usize);
    assert_eq!(layout.n_score, sd["n_score"].as_u64().unwrap() as usize);
    layout
}

/// The headline check: same sequence, same per-timestep edge, same path, for
/// both reads and every backend this CPU offers.
#[test]
fn decode_matches_bonito() {
    let fixture = golden();
    let layout = layout_from(&fixture);
    let alphabet: Vec<u8> = fixture["seqdist"]["alphabet"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().as_bytes()[0])
        .collect();

    let small = &fixture["small"];
    let f = &fixture["formula"];
    let t_len = f["T"].as_u64().unwrap() as usize;
    let n_reads = f["N"].as_u64().unwrap() as usize;

    // The fixture flattens (T, N) row-major, so read `n` is every N-th entry.
    let flat = |key: &str| -> Vec<i64> {
        small[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect()
    };
    let traceback = flat("traceback_argmax_flat");
    let path = flat("path_flat");
    let seqs = small["seqs"].as_array().unwrap();

    let mut backends = vec![Backend::Scalar];
    #[cfg(target_arch = "x86_64")]
    if Backend::best_for(&layout) == Backend::Avx2 {
        backends.push(Backend::Avx2);
    }

    let mut scratch = CrfScratch::new();
    for backend in backends {
        for n in 0..n_reads {
            let scores = synthesize(f, n);
            let got =
                decode_with(&layout, &alphabet, &scores, t_len, &mut scratch, backend).unwrap();

            assert_eq!(
                got,
                seqs[n].as_str().unwrap(),
                "{backend:?} read {n}: decoded sequence differs from bonito"
            );
            for t in 0..t_len {
                assert_eq!(
                    scratch.traceback()[t] as i64,
                    traceback[t * n_reads + n],
                    "{backend:?} read {n} t={t}: argmax edge differs from bonito"
                );
                assert_eq!(
                    scratch.path()[t] as i64,
                    path[t * n_reads + n],
                    "{backend:?} read {n} t={t}: path differs from bonito"
                );
            }
        }
    }
}

/// Guards the regeneration itself. If `sin`/`cos` ever drifted enough to change
/// the input tensor, `decode_matches_bonito` would fail with a confusing
/// sequence diff; this fails first and says what actually went wrong.
#[test]
fn synthesized_scores_match_the_fixture_checksums() {
    let fixture = golden();
    let f = &fixture["formula"];
    let want = &fixture["small_scores_checksum"];
    let n_reads = f["N"].as_u64().unwrap() as usize;

    let (mut sum, mut min, mut max) = (0f64, f32::INFINITY, f32::NEG_INFINITY);
    let mut count = 0usize;
    for n in 0..n_reads {
        for v in synthesize(f, n) {
            sum += v as f64;
            min = min.min(v);
            max = max.max(v);
            count += 1;
        }
    }
    assert_eq!(
        count as u64,
        want["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap())
            .product::<u64>()
    );
    // Summation order differs from numpy's, so compare the mean with a
    // tolerance rather than the raw sum with equality.
    let mean = sum / count as f64;
    assert!(
        (mean - want["mean"].as_f64().unwrap()).abs() < 1e-9,
        "mean {mean} vs {}",
        want["mean"]
    );
    assert!(
        (min as f64 - want["min"].as_f64().unwrap()).abs() < 1e-6,
        "min {min}"
    );
    assert!(
        (max as f64 - want["max"].as_f64().unwrap()).abs() < 1e-6,
        "max {max}"
    );
}

/// The fixture records that bonito's `SeqdistModel.decode_batch` — the real
/// entry point, not a reimplementation — agrees with the manual
/// forward/backward chain the port follows. If that ever stopped holding, the
/// whole fixture would be measuring the wrong thing.
#[test]
fn fixture_was_cross_checked_against_decode_batch() {
    let fixture = golden();
    for case in ["small", "real"] {
        assert_eq!(
            fixture[case]["decode_batch_matches_manual_chain"].as_bool(),
            Some(true),
            "{case}: fixture generator did not confirm decode_batch parity"
        );
        assert_eq!(
            fixture[case]["seqs"], fixture[case]["decode_batch_seqs"],
            "{case}: manual chain and decode_batch disagree"
        );
    }
}

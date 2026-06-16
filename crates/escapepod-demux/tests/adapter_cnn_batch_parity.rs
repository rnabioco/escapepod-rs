//! Parity: batched adapter-end detection must equal the per-read path.
//!
//! Requires a real boundary-CNN ONNX model — set `ESCAPEPOD_TEST_ADAPTER_ONNX`
//! to its path (e.g. escapepod-models' `adapter_rna004@v1.0.1/adapter_rna004.onnx`).
//! Skips (passes) when unset so the suite stays hermetic in CI.
//!
//! Run with:
//!   ESCAPEPOD_TEST_ADAPTER_ONNX=/path/to/adapter_rna004.onnx \
//!     cargo nextest run -p escapepod-demux --features cnn-detect batch_parity

#![cfg(feature = "cnn-detect")]

use escapepod_demux::AdapterCnn;

/// Deterministic pseudo-random signal generator (no rand dep needed): a simple
/// LCG mapped into a plausible raw-current range, with per-read length.
fn synth_signal(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // top bits -> [0, 1) -> centered current-ish values
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
            400.0 + 200.0 * (u - 0.5)
        })
        .collect()
}

#[test]
fn batched_matches_per_read() {
    let Ok(model_path) = std::env::var("ESCAPEPOD_TEST_ADAPTER_ONNX") else {
        eprintln!("ESCAPEPOD_TEST_ADAPTER_ONNX unset — skipping CNN batch-parity test");
        return;
    };

    let cnn = AdapterCnn::load(&model_path).expect("load CNN model");

    // Varied lengths spanning short/median/long reads (post-min_obs tails),
    // including a couple below the min length so SignalTooShort is exercised.
    let lengths = [
        500, 1100, 9259, 10206, 13251, 26162, 2000, 1100, 30, 1500, 800, 4096,
    ];
    let signals: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(i, &l)| synth_signal(i as u64 + 1, l))
        .collect();
    let refs: Vec<&[f32]> = signals.iter().map(Vec::as_slice).collect();

    // Per-read reference.
    let per_read: Vec<Result<usize, _>> = refs.iter().map(|s| cnn.detect_adapter_end(s)).collect();

    // Batched (all at once, unsorted — exercises mixed-length padding).
    let batched = cnn.detect_adapter_end_batch(&refs);

    assert_eq!(batched.len(), per_read.len());
    for (i, (b, p)) in batched.iter().zip(&per_read).enumerate() {
        match (b, p) {
            (Ok(bv), Ok(pv)) => assert_eq!(
                bv, pv,
                "read {i} (len {}): batched adapter_end {bv} != per-read {pv}",
                lengths[i]
            ),
            (Err(_), Err(_)) => {} // both rejected (too short) — fine
            _ => panic!(
                "read {i} (len {}): ok/err mismatch {b:?} vs {p:?}",
                lengths[i]
            ),
        }
    }
}

#[test]
fn batch_of_one_matches_single() {
    let Ok(model_path) = std::env::var("ESCAPEPOD_TEST_ADAPTER_ONNX") else {
        eprintln!("ESCAPEPOD_TEST_ADAPTER_ONNX unset — skipping CNN batch-of-one test");
        return;
    };
    let cnn = AdapterCnn::load(&model_path).expect("load CNN model");
    let sig = synth_signal(42, 9259);
    let single = cnn.detect_adapter_end(&sig).unwrap();
    let batched = cnn.detect_adapter_end_batch(&[sig.as_slice()]);
    assert_eq!(batched.len(), 1);
    assert_eq!(batched[0].as_ref().unwrap(), &single);
}

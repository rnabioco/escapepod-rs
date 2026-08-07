//! Parity: length-bucketed (padded) detection vs exact-length grouping.
//!
//! Exact grouping fragments badly on real data — [`prep_adapter_signal`] clamps
//! its window to the read end, so every read shorter than `max_obs_trace` gets
//! its own length. A production run showed 680 distinct lengths across 527k
//! reads, and detection cost 401 s of device time in a 425 s wall against
//! ~0.01 ms/read for the same kernel at a steady shape. Bucketing rounds lengths
//! up to a common multiple and zero-pads the tail so a handful of shapes cover
//! everything.
//!
//! **Padding is not free of consequence, and this test is the check.** The head
//! is fixed (the window always starts at `min_obs_adapter`, and the decode
//! searches forward from index 0), so a tail pad never shifts a boundary index,
//! and `valid_len` keeps the argmax out of the padding. None of that helps: the
//! graph is **globally length-dependent**. `examples/cnn_pad_probe` shows raw
//! scores diverging from position 0 — nowhere near the padding — by up to 4.5,
//! which is the signature of a normalisation over the length axis rather than a
//! local receptive-field effect. Appending zeros moves the statistics, and every
//! output moves with them.
//!
//! Requires a real boundary-CNN ONNX model and a GPU:
//! ```text
//! ESCAPEPOD_TEST_ADAPTER_ONNX=/path/to/adapter_rna004.onnx \
//!   cargo nextest run -p escapepod-demux --features cnn-gpu bucket_parity
//! ```
//! Skips (passes) when unset so the suite stays hermetic in CI.

#![cfg(feature = "cnn-gpu")]

use escapepod_demux::AdapterCnnGpu;

/// Deterministic pseudo-random signal, matching `adapter_cnn_batch_parity`'s
/// generator so the two tests exercise comparable input.
fn synth_signal(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
            400.0 + 200.0 * (u - 0.5)
        })
        .collect()
}

/// Read lengths spanning the regimes that matter:
/// * at or above ~9060 samples — prepped length saturates at the 806 cap (68.6%
///   of a real run), already one shape, so padding is a no-op for them;
/// * 1010..9060 — the window clamps, so each length is its own group and these
///   are what bucketing actually collapses;
/// * a few below the minimum, to keep the too-short path in the comparison.
fn spread_of_lengths() -> Vec<usize> {
    let mut v: Vec<usize> = Vec::new();
    // Dense in the variable-length regime, where bucketing has an effect.
    for n in (1100..9060).step_by(137) {
        v.push(n);
    }
    // Saturated regime.
    for n in [9060, 9500, 12000, 16000, 26162, 40000] {
        v.push(n);
    }
    // Below the prep minimum.
    for n in [30, 500, 1000] {
        v.push(n);
    }
    v
}

fn detect_all(detector: &AdapterCnnGpu, signals: &[Vec<f32>]) -> Vec<Option<usize>> {
    let refs: Vec<&[f32]> = signals.iter().map(Vec::as_slice).collect();
    detector
        .detect_adapter_end_batch(&refs)
        .into_iter()
        .map(Result::ok)
        .collect()
}

/// The prepped-length cap, at which bucketing appends **nothing**.
///
/// Reads here are unaffected by construction, not because padding is safe for
/// them: `prep_adapter_signal` truncates to this, so a bucket at or above it adds
/// no zeros at all. That is the only regime where bucketed and exact agree, and
/// it is exactly the regime where bucketing does no work.
const SAFE_PREPPED_LEN: usize = 806;

/// A read's prepped length, mirroring `prep_adapter_signal`:
/// `min((min(len, max_obs_trace) - min_obs_adapter) / downscale, cap)`.
fn prepped_len(raw_len: usize) -> usize {
    let end = raw_len.min(16_000);
    if end <= 1_000 {
        return 0;
    }
    ((end - 1_000) / 10).min(SAFE_PREPPED_LEN)
}

/// The load-bearing check: **reads that receive no padding must be bit-identical
/// under any bucket size.**
///
/// A genuine invariant — those reads are already at the cap, so bucketing changes
/// their tensor not at all, and any difference would be a wiring bug in
/// `pack_batch` or the `valid_len` clamp rather than a model effect.
///
/// Reads below the cap do receive padding and are counted, not asserted: the
/// graph is globally length-dependent (see the module docs), so they are expected
/// to move, and pinning today's values would pin an artefact. The count is the
/// point — it is what says bucketing cannot be turned on.
#[test]
fn bucketing_is_exact_where_padding_cannot_reach() {
    let Ok(model_path) = std::env::var("ESCAPEPOD_TEST_ADAPTER_ONNX") else {
        eprintln!("ESCAPEPOD_TEST_ADAPTER_ONNX unset — skipping CNN bucket-parity test");
        return;
    };

    let lengths = spread_of_lengths();
    let signals: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(i, &n)| synth_signal(i as u64 + 1, n))
        .collect();

    // Exact grouping is the reference: bucket 1 is the pre-existing behaviour.
    unsafe { std::env::set_var("ESCAPEPOD_CNN_GPU_LEN_BUCKET", "1") };
    let reference = {
        let d = AdapterCnnGpu::load_with_threads(&model_path, 4).expect("load CNN on GPU");
        detect_all(&d, &signals)
    };

    for bucket in [16usize, 64, 128, 256, 1024] {
        unsafe { std::env::set_var("ESCAPEPOD_CNN_GPU_LEN_BUCKET", bucket.to_string()) };
        let d = AdapterCnnGpu::load_with_threads(&model_path, 4).expect("load CNN on GPU");
        let got = detect_all(&d, &signals);
        assert_eq!(got.len(), reference.len());

        let (mut unsafe_diffs, mut safe_diffs, mut safe_total) = (0usize, 0usize, 0usize);
        let mut examples: Vec<String> = Vec::new();
        for (k, (want, have)) in reference.iter().zip(&got).enumerate() {
            let safe = prepped_len(lengths[k]) >= SAFE_PREPPED_LEN;
            if safe {
                safe_total += 1;
            }
            if want == have {
                continue;
            }
            if safe {
                safe_diffs += 1;
                examples.push(format!(
                    "read_len={} prepped={} exact={want:?} bucketed={have:?}",
                    lengths[k],
                    prepped_len(lengths[k])
                ));
            } else {
                unsafe_diffs += 1;
            }
        }
        eprintln!(
            "bucket={bucket:<5} below-threshold reads changed: {unsafe_diffs} (expected, padding \
             is inside their receptive field);  at/above threshold: {safe_diffs}/{safe_total}"
        );
        assert_eq!(
            safe_diffs,
            0,
            "bucket={bucket} moved {safe_diffs} call(s) on reads whose prepped length is \
             >= {SAFE_PREPPED_LEN}, where zero-padding the tail cannot reach any searched \
             position. That is a real defect in the padding or the valid_len clamp, not the \
             known short-read effect:\n  {}",
            examples.join("\n  ")
        );
    }
    unsafe { std::env::remove_var("ESCAPEPOD_CNN_GPU_LEN_BUCKET") };
}

/// Bucketing must not change how many reads are detectable — a padded row is
/// still one read in, one answer out, and the too-short ones must stay `None`.
#[test]
fn bucketing_preserves_which_reads_resolve() {
    let Ok(model_path) = std::env::var("ESCAPEPOD_TEST_ADAPTER_ONNX") else {
        eprintln!("ESCAPEPOD_TEST_ADAPTER_ONNX unset — skipping CNN bucket-shape test");
        return;
    };
    let lengths = spread_of_lengths();
    let signals: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(i, &n)| synth_signal(i as u64 + 7, n))
        .collect();

    unsafe { std::env::set_var("ESCAPEPOD_CNN_GPU_LEN_BUCKET", "1") };
    let exact = {
        let d = AdapterCnnGpu::load_with_threads(&model_path, 4).expect("load CNN on GPU");
        detect_all(&d, &signals)
    };
    unsafe { std::env::set_var("ESCAPEPOD_CNN_GPU_LEN_BUCKET", "256") };
    let bucketed = {
        let d = AdapterCnnGpu::load_with_threads(&model_path, 4).expect("load CNN on GPU");
        detect_all(&d, &signals)
    };
    unsafe { std::env::remove_var("ESCAPEPOD_CNN_GPU_LEN_BUCKET") };

    for (k, (a, b)) in exact.iter().zip(&bucketed).enumerate() {
        assert_eq!(
            a.is_some(),
            b.is_some(),
            "read {k} (len {}) changed resolvability: exact={a:?} bucketed={b:?}",
            lengths[k]
        );
    }
}

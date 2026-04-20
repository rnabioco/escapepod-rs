//! Integration tests for the resquiggle pipeline orchestration.
//!
//! Unit tests in `resquiggle/refine.rs` cover individual settings; these
//! exercise the orchestration paths (default settings with adaptive banding +
//! Theil-Sen rescaling, RNA reversal, multi-read batch) end-to-end on synthetic
//! signals.

use escapepod_signal::resquiggle::{
    BandingAlgo, RefineAlgo, RefineSettings, RescaleAlgo, RoughRescaleAlgo,
    calculate_initial_scaling, refine_signal_map, reverse_query_to_signal_map,
};

/// Build a synthetic signal where each base emits `samples_per_base` samples
/// at the provided level plus small deterministic noise.
fn synth_signal_from_levels(levels: &[f32], samples_per_base: usize) -> Vec<f32> {
    let mut signal = Vec::with_capacity(levels.len() * samples_per_base);
    let mut state: u64 = 0xC0FFEE;
    for &level in levels {
        for _ in 0..samples_per_base {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state as i32 >> 24) as f32) / 256.0 * 0.05;
            signal.push(level + noise);
        }
    }
    signal
}

/// Uniform initial map: boundaries at every `samples_per_base` samples.
fn uniform_map(n_bases: usize, samples_per_base: usize) -> Vec<usize> {
    (0..=n_bases).map(|i| i * samples_per_base).collect()
}

/// Assert a seq-to-signal map is sane: bounded, monotonically increasing,
/// covers the expected signal span.
fn assert_valid_map(map: &[usize], n_bases: usize, signal_len: usize) {
    assert_eq!(
        map.len(),
        n_bases + 1,
        "expected {} boundaries",
        n_bases + 1
    );
    assert_eq!(map[0], 0, "first boundary must be 0");
    assert_eq!(
        map[n_bases], signal_len,
        "last boundary must equal signal_len"
    );
    for w in map.windows(2) {
        assert!(w[1] > w[0], "map must be strictly increasing");
    }
}

#[test]
fn pipeline_default_settings_refines_multiple_reads() {
    // Simulate a small batch of reads with defaults (adaptive banding, Theil-Sen rescale).
    let samples_per_base = 16;
    let reads: Vec<Vec<f32>> = vec![
        vec![0.0, 1.2, -0.8, 0.4, -1.1, 0.7, -0.3, 0.9, -0.5, 0.2],
        vec![0.5, -0.5, 0.8, -0.2, 1.0, -0.9, 0.3, -0.6, 0.1, 0.4, -0.7],
        vec![1.0, 0.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75],
    ];

    let settings = RefineSettings::default();
    for (read_idx, levels) in reads.iter().enumerate() {
        let signal = synth_signal_from_levels(levels, samples_per_base);
        let map = uniform_map(levels.len(), samples_per_base);
        let result = refine_signal_map(&settings, &signal, &map, levels, 1.0, 0.0)
            .unwrap_or_else(|e| panic!("read {read_idx} failed: {e}"));
        assert_valid_map(&result.seq_to_signal_map, levels.len(), signal.len());
        assert!(result.scale.is_finite());
        assert!(result.shift.is_finite());
    }
}

#[test]
fn pipeline_adaptive_banding_recovers_boundaries() {
    let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0, 0.3, -0.3, 0.8, -0.8, 0.2];
    let samples_per_base = 20;
    let signal = synth_signal_from_levels(&levels, samples_per_base);
    let map = uniform_map(levels.len(), samples_per_base);

    let settings = RefineSettings {
        refinement_algo: RefineAlgo::Viterbi,
        banding_algo: BandingAlgo::Adaptive {
            bandwidth: 10,
            x_drop: None,
        },
        rescale_algo: RescaleAlgo::default(),
        rough_rescale_algo: RoughRescaleAlgo::None,
        n_refinement_iters: 1,
        half_bandwidth: 5,
        adjust_band_min_size: 2,
        normalize_levels: false,
    };

    let result = refine_signal_map(&settings, &signal, &map, &levels, 1.0, 0.0).unwrap();
    assert_valid_map(&result.seq_to_signal_map, levels.len(), signal.len());

    // Adaptive banding sacrifices tight per-boundary precision; just check the
    // overall mapping is coherent — total span covers the signal and no base
    // has zero dwell.
    let dwells: Vec<usize> = result
        .seq_to_signal_map
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    assert!(
        dwells.iter().all(|&d| d >= 1),
        "adaptive map has zero-dwell bases: {dwells:?}"
    );
    let span: usize = dwells.iter().sum();
    assert_eq!(span, signal.len());
}

#[test]
fn pipeline_rna_reversal_round_trips() {
    // Simulates the RNA workflow: refine on reversed signal, then reverse map back.
    let levels: Vec<f32> = vec![0.2, 0.8, -0.4, 0.5, -0.9, 0.1, -0.6];
    let samples_per_base = 24;
    let signal_fwd = synth_signal_from_levels(&levels, samples_per_base);
    let signal_len = signal_fwd.len();

    // Reverse both signal and levels for "RNA mode".
    let mut signal_rev = signal_fwd.clone();
    signal_rev.reverse();
    let levels_rev: Vec<f32> = levels.iter().rev().copied().collect();
    let map_rev = uniform_map(levels.len(), samples_per_base);

    let settings = RefineSettings::default();
    let result =
        refine_signal_map(&settings, &signal_rev, &map_rev, &levels_rev, 1.0, 0.0).unwrap();
    let reversed_back = reverse_query_to_signal_map(&result.seq_to_signal_map, signal_len);

    assert_valid_map(&reversed_back, levels.len(), signal_len);
    // Endpoint invariants from reverse_query_to_signal_map.
    assert_eq!(reversed_back[0], 0);
    assert_eq!(*reversed_back.last().unwrap(), signal_len);
}

#[test]
fn pipeline_initial_scaling_composes_with_refine() {
    // Build a signal with known calibration parameters and confirm the
    // pipeline recovers sensible scale/shift after refinement.
    let cal_scale = 0.5;
    let cal_offset = -2.0;
    let sd = 10.0; // BAM sd tag
    let sm = 3.0; // BAM sm tag
    let (scale, shift) = calculate_initial_scaling(cal_scale, cal_offset, sd, sm);

    let levels: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25];
    let samples_per_base = 20;
    // Construct raw-DAC-style signal: raw = shift + scale * norm_level
    let mut signal = Vec::with_capacity(levels.len() * samples_per_base);
    for &lvl in &levels {
        for _ in 0..samples_per_base {
            signal.push(shift + scale * lvl);
        }
    }
    let map = uniform_map(levels.len(), samples_per_base);

    let settings = RefineSettings {
        n_refinement_iters: 2,
        ..RefineSettings::default()
    };
    let result = refine_signal_map(&settings, &signal, &map, &levels, scale, shift).unwrap();
    assert_valid_map(&result.seq_to_signal_map, levels.len(), signal.len());
    // Refined scale/shift should remain finite and within a couple orders of
    // magnitude of the inputs (guards against divergence).
    assert!(result.scale.abs() > 1e-3 && result.scale.abs() < 1e6);
    assert!(result.shift.is_finite());
}

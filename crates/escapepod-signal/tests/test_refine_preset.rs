//! Tests for `RefineSettings::move_table_refinement`, the named preset that
//! callers refining a basecaller move table construct instead of transcribing
//! a settings literal.
//!
//! Two copies of that literal — one in escapepod's Python binding, one in a
//! downstream Rust consumer — each carried a comment saying they matched, and
//! drifted in `dwell_target` anyway (a fixed `4.0` against the per-read
//! resolution), so the same reads refined to different boundaries for four
//! releases. The field-by-field test below is the thing that stops the next
//! drift; the dwell-target test pins the specific field that drifted.

use escapepod_signal::resquiggle::{
    BandingAlgo, RefineAlgo, RefineSettings, RescaleAlgo, RescaleFilterParams, RoughRescaleAlgo,
    refine_signal_map,
};

/// Synthetic signal: base `i` emits `dwells[i]` samples at its level plus
/// small deterministic noise, so the DP has real boundaries to find.
fn synth_signal(levels: &[f32], dwells: &[usize]) -> Vec<f32> {
    let mut signal = Vec::with_capacity(dwells.iter().sum());
    let mut state: u64 = 0xC0FFEE;
    for (&level, &dwell) in levels.iter().zip(dwells) {
        for _ in 0..dwell {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state as i32 >> 24) as f32) / 256.0 * 0.35;
            signal.push(level + noise);
        }
    }
    signal
}

/// Samples per base of the uniform move-table map handed to the refiner.
///
/// RNA004 at 130 bases/s and 4 kHz sits near 31 samples/base, which is the
/// whole point: it is nowhere near the `4.0` that drifted in.
const MEDIAN_DWELL: usize = 31;

/// A read whose true dwells are RNA004-like and *variable*, as a real read's
/// are, so the DP must move boundaries away from the uniform map it is handed.
///
/// The dwells sum to `n_bases * MEDIAN_DWELL`, so the input map is uniform at
/// `MEDIAN_DWELL` and its median dwell is exactly `MEDIAN_DWELL`. Several
/// bases are much shorter than that, which is precisely where the two dwell
/// targets disagree: at 31 the quadratic arm of the penalty resists shrinking
/// a base, at 4.0 every one of these dwells sits on the flat logarithmic arm.
fn rna004_like_read() -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let levels: Vec<f32> = vec![
        0.0, 1.2, -0.8, 0.4, -1.1, 0.7, -0.3, 0.9, -0.5, 0.2, 1.0, -0.6, 0.35, -0.95, 0.55,
    ];
    let dwells: Vec<usize> = vec![15, 45, 31, 20, 42, 31, 25, 37, 31, 18, 44, 31, 28, 34, 33];
    assert_eq!(dwells.len(), levels.len());
    assert_eq!(
        dwells.iter().sum::<usize>(),
        levels.len() * MEDIAN_DWELL,
        "fixture must keep the uniform input map's median dwell at {MEDIAN_DWELL}",
    );
    let signal = synth_signal(&levels, &dwells);
    let map: Vec<usize> = (0..=levels.len()).map(|i| i * MEDIAN_DWELL).collect();
    (signal, levels, map)
}

#[test]
fn move_table_refinement_pins_every_field() {
    let s = RefineSettings::move_table_refinement(7, 3, Some(42));

    // The dwell target is per-read, NOT a constant. This is the field that
    // drifted; a `4.0` here silently corrupted every level-derived feature in
    // a production corpus.
    assert_eq!(
        s.refinement_algo,
        RefineAlgo::DwellPenalty {
            target: RefineAlgo::PER_READ_DWELL_TARGET,
            weight: 0.5,
        },
    );
    assert_eq!(RefineAlgo::PER_READ_DWELL_TARGET, 0.0);
    assert_eq!(RefineSettings::MOVE_TABLE_DWELL_WEIGHT, 0.5);

    // Arguments are threaded through, not ignored.
    assert_eq!(s.half_bandwidth, 7);
    assert_eq!(s.n_refinement_iters, 3);

    assert_eq!(s.adjust_band_min_size, 2);

    // Theil-Sen inter-iteration rescale, default filter, 200 points, seeded.
    assert_eq!(
        s.rescale_algo,
        RescaleAlgo::TheilSen {
            filter: RescaleFilterParams::default(),
            max_points: 200,
            seed: Some(42),
        },
    );
    // Spell the filter out too: `RescaleFilterParams::default()` above would
    // follow a change to those defaults instead of catching it.
    let f = s.rescale_algo.filter_params();
    assert_eq!(f.dwell_filter_lower_percentile, 0.1);
    assert_eq!(f.dwell_filter_upper_percentile, 0.9);
    assert_eq!(f.min_abs_level, 0.2);
    assert_eq!(f.n_bases_truncate, 10);
    assert_eq!(f.min_num_filtered_levels, 10);

    // Least-squares rough rescale over the 0.05–0.95 quantiles, clipped 10
    // bases, base-centred. Note this is NOT `RoughRescaleAlgo::default()`,
    // which is Theil-Sen.
    assert_eq!(
        s.rough_rescale_algo,
        RoughRescaleAlgo::LeastSquares {
            quantiles: RoughRescaleAlgo::default_quantiles(),
            clip_bases: 10,
            use_base_center: true,
        },
    );
    let expected_quantiles: Vec<f32> = (1..=19).map(|i| i as f32 * 0.05).collect();
    match &s.rough_rescale_algo {
        RoughRescaleAlgo::LeastSquares { quantiles, .. } => {
            assert_eq!(quantiles.len(), 19);
            for (got, want) in quantiles.iter().zip(&expected_quantiles) {
                assert!(
                    (got - want).abs() < 1e-6,
                    "quantiles drifted: {quantiles:?}"
                );
            }
        }
        other => panic!("rough rescale is not least-squares: {other:?}"),
    }

    assert!(!s.normalize_levels);
    assert_eq!(s.banding_algo, BandingAlgo::Fixed);

    // `seed: None` must survive as None (unseeded Theil-Sen subsample), not be
    // substituted with a default.
    assert_eq!(
        RefineSettings::move_table_refinement(5, 2, None)
            .rescale_algo
            .seed(),
        None,
    );
}

#[test]
fn preset_dwell_target_resolves_to_the_read_median_not_a_constant() {
    let (signal, levels, map) = rna004_like_read();

    let preset = RefineSettings::move_table_refinement(5, 2, Some(0));
    let from_preset = refine_signal_map(&preset, &signal, &map, &levels, 1.0, 0.0).unwrap();

    // The input map is uniform at MEDIAN_DWELL samples/base, so that is its
    // median dwell.
    // Naming that target explicitly must reproduce the preset exactly — that
    // is what "per-read target" means, stated as an observable.
    let explicit = RefineSettings {
        refinement_algo: RefineAlgo::DwellPenalty {
            target: MEDIAN_DWELL as f32,
            weight: 0.5,
        },
        ..RefineSettings::move_table_refinement(5, 2, Some(0))
    };
    let from_explicit = refine_signal_map(&explicit, &signal, &map, &levels, 1.0, 0.0).unwrap();

    assert_eq!(
        from_preset.seq_to_signal_map, from_explicit.seq_to_signal_map,
        "preset did not resolve the dwell target from the read's own median dwell",
    );
}

#[test]
fn preset_disagrees_with_the_old_fixed_target_of_four() {
    // The regression this PR exists for. RNA004 sits near 31 samples/base; a
    // fixed target of 4.0 treats every base as ~8x too long, and because the
    // penalty is asymmetric (quadratic below target, logarithmic above) it
    // drags boundaries toward dwells the pore never produced.
    //
    // If someone restores `dwell_target = 4.0` as the preset's target, this
    // test fails.
    let (signal, levels, map) = rna004_like_read();

    let preset = RefineSettings::move_table_refinement(5, 2, Some(0));
    let from_preset = refine_signal_map(&preset, &signal, &map, &levels, 1.0, 0.0).unwrap();

    let old_default = RefineSettings {
        refinement_algo: RefineAlgo::DwellPenalty {
            target: 4.0,
            weight: 0.5,
        },
        ..RefineSettings::move_table_refinement(5, 2, Some(0))
    };
    let from_old = refine_signal_map(&old_default, &signal, &map, &levels, 1.0, 0.0).unwrap();

    assert_ne!(
        from_preset.seq_to_signal_map, from_old.seq_to_signal_map,
        "a fixed dwell target of 4.0 produced the same map as the per-read \
         target on a 31-samples/base read — either the preset regressed to a \
         constant or the dwell penalty stopped reaching the DP",
    );

    // Both maps must still be well-formed, so the difference above is a real
    // disagreement about boundaries rather than one path failing.
    for m in [&from_preset.seq_to_signal_map, &from_old.seq_to_signal_map] {
        assert_eq!(m.len(), levels.len() + 1);
        assert_eq!(m[0], 0);
        assert_eq!(*m.last().unwrap(), signal.len());
        assert!(m.windows(2).all(|w| w[1] > w[0]));
    }
}

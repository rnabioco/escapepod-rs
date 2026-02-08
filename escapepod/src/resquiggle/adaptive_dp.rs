// SPDX-License-Identifier: GPL-3.0-or-later

//! Adaptive banded DP (Suzuki & Kasahara, 2017).
//!
//! The band center shifts during the forward pass based on edge score
//! comparisons, adapting to local alignment quality while keeping a fixed
//! bandwidth.

use super::bands::Band;
use super::dp::{banded_traceback, dp_step};
use super::types::RefineAlgo;

/// Run adaptive banded DP.
///
/// The band center is seeded from `initial_map` and then steered during the
/// forward pass: if the lower edge of the band has a better score than the
/// upper edge, the band shifts down for the next base, and vice versa.
///
/// Returns a signal-position path of length `levels.len() + 1`.
pub fn adaptive_banded_dp(
    signal: &[f32],
    levels: &[f32],
    bandwidth: usize,
    initial_map: &[usize],
    _method: &RefineAlgo,
) -> Vec<usize> {
    let n_bases = levels.len();
    let signal_len = signal.len();

    if n_bases == 0 || signal_len == 0 {
        return vec![0; n_bases + 1];
    }

    let half_bw = bandwidth / 2;

    // We need to record band boundaries and traceback for the final path
    let mut band_starts: Vec<usize> = Vec::with_capacity(n_bases);
    let mut band_ends: Vec<usize> = Vec::with_capacity(n_bases);
    let mut all_traceback: Vec<Vec<i32>> = Vec::with_capacity(n_bases);

    // Rolling score buffers (sized for maximum possible bandwidth)
    let max_bw = bandwidth + 2;
    let mut prev_scores = vec![f32::INFINITY; max_bw];
    let mut curr_scores = vec![f32::INFINITY; max_bw];
    let mut curr_traceback = vec![0i32; max_bw];

    // --- First base ---
    // Band for base 0: centered around initial_map midpoint, starts at 0
    let center0 = (initial_map[0] + initial_map[1]) / 2;
    let (_bs0, be0) = band_for_center(center0, half_bw, signal_len);
    // First base must start at signal position 0
    let bs0 = 0;
    let be0 = be0.max(bs0 + 1);
    let bw0 = be0 - bs0;

    // Resize working buffers if first-base band exceeds initial allocation
    if bw0 > prev_scores.len() {
        prev_scores.resize(bw0, f32::INFINITY);
        curr_scores.resize(bw0, f32::INFINITY);
        curr_traceback.resize(bw0, -1);
    }

    band_starts.push(bs0);
    band_ends.push(be0);

    // Initialize "previous" scores: single point at position 0 with score 0
    prev_scores[..bw0].fill(f32::INFINITY);
    prev_scores[0] = 0.0;

    curr_scores[..bw0].fill(f32::INFINITY);
    curr_traceback[..bw0].fill(-1);

    dp_step(
        &mut curr_scores[..bw0],
        &mut curr_traceback[..bw0],
        &prev_scores[..bw0],
        levels[0],
        &signal[bs0..be0],
        1, // prev_band_offset = 1 (initial point at position 0, band also starts at 0)
        );

    all_traceback.push(curr_traceback[..bw0].to_vec());
    std::mem::swap(&mut prev_scores, &mut curr_scores);

    let mut prev_bs = bs0;
    let mut prev_bw = bw0;

    // --- Remaining bases ---
    for base_idx in 1..n_bases {
        // Steer: compare lower and upper edge scores of the previous band
        let lower_score = prev_scores[0];
        let upper_score = prev_scores[prev_bw.saturating_sub(1)];

        // Seed center from initial map
        let mid = (initial_map[base_idx] + initial_map[base_idx + 1]) / 2;
        let mut center = mid.min(signal_len.saturating_sub(1));

        // Apply steering
        if lower_score < upper_score {
            center = center.saturating_sub(1);
        } else if upper_score < lower_score {
            center = (center + 1).min(signal_len.saturating_sub(1));
        }

        let (mut bs, mut be) = band_for_center(center, half_bw, signal_len);

        // Enforce that current band start >= previous band start (monotonicity)
        if bs < prev_bs {
            bs = prev_bs;
        }

        // Ensure overlap with previous band: band start must not exceed
        // prev_bs + prev_bw (the end of the previous band)
        let max_bs = prev_bs + prev_bw;
        if bs > max_bs {
            bs = max_bs;
        }

        // For the last base, ensure band end reaches signal_len
        if base_idx == n_bases - 1 {
            be = signal_len;
        }

        // Ensure band end > band start
        be = be.max(bs + 1);

        let bw = be - bs;
        band_starts.push(bs);
        band_ends.push(be);

        // Resize working buffers if needed (check each independently so they
        // only grow — curr_scores swaps with prev_scores but curr_traceback
        // does not, so their lengths can diverge)
        if bw > curr_scores.len() {
            curr_scores.resize(bw, f32::INFINITY);
        }
        if bw > curr_traceback.len() {
            curr_traceback.resize(bw, -1);
        }

        curr_scores[..bw].fill(f32::INFINITY);
        curr_traceback[..bw].fill(-1);

        let prev_band_offset = bs - prev_bs;

        dp_step(
            &mut curr_scores[..bw],
            &mut curr_traceback[..bw],
            &prev_scores[..prev_bw],
            levels[base_idx],
            &signal[bs..be],
            prev_band_offset,
        );

        all_traceback.push(curr_traceback[..bw].to_vec());
        std::mem::swap(&mut prev_scores, &mut curr_scores);

        prev_bs = bs;
        prev_bw = bw;
    }

    // Build Band and traceback
    let band = Band::new(band_starts.clone(), band_ends.clone(), true);

    let mut base_offsets = Vec::with_capacity(n_bases + 1);
    base_offsets.push(0);
    let mut offset = 0;
    for i in 0..n_bases {
        offset += band_ends[i] - band_starts[i];
        base_offsets.push(offset);
    }

    let flat_traceback: Vec<i32> = all_traceback.into_iter().flatten().collect();

    let mut path = vec![0usize; n_bases + 1];
    banded_traceback(&mut path, &band, &base_offsets, &flat_traceback);

    path
}

/// Compute band [start, end) for a given center and half-bandwidth, clamped to signal bounds.
fn band_for_center(center: usize, half_bw: usize, signal_len: usize) -> (usize, usize) {
    let start = center.saturating_sub(half_bw);
    let end = (center + half_bw + 1).min(signal_len);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dp::banded_dp;

    #[test]
    fn test_adaptive_dp_clean_signal() {
        // 5 bases, each with 10 samples at the expected level
        let n_bases = 5;
        let spb = 10;
        let signal_len = n_bases * spb;

        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0];
        let mut signal = vec![0.0f32; signal_len];
        for (i, &level) in levels.iter().enumerate() {
            for j in 0..spb {
                signal[i * spb + j] = level;
            }
        }

        let initial_map: Vec<usize> = (0..=n_bases).map(|i| i * spb).collect();
        let bandwidth = 10; // half_bw = 5

        let path =
            adaptive_banded_dp(&signal, &levels, bandwidth, &initial_map, &RefineAlgo::Viterbi);

        assert_eq!(path.len(), n_bases + 1);
        assert_eq!(path[0], 0);
        assert_eq!(path[n_bases], signal_len);

        // Strictly increasing
        for w in path.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing: {:?}", path);
        }

        // Boundaries should be near the true boundaries
        for i in 0..=n_bases {
            let expected = i * spb;
            let diff = (path[i] as i64 - expected as i64).unsigned_abs() as usize;
            assert!(
                diff <= bandwidth / 2 + 1,
                "path[{}]={} far from expected {}",
                i,
                path[i],
                expected
            );
        }
    }

    #[test]
    fn test_adaptive_dp_poor_initial() {
        // Deliberately bad initial map: all boundaries shifted left by 5
        let n_bases = 5;
        let spb = 10;
        let signal_len = n_bases * spb;

        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0];
        let mut signal = vec![0.0f32; signal_len];
        for (i, &level) in levels.iter().enumerate() {
            for j in 0..spb {
                signal[i * spb + j] = level;
            }
        }

        // Shift map left by 5 (but keep first at 0 and last at signal_len)
        let mut initial_map: Vec<usize> = (0..=n_bases)
            .map(|i| (i * spb).saturating_sub(5))
            .collect();
        initial_map[0] = 0;
        initial_map[n_bases] = signal_len;

        let bandwidth = 14; // wider band to recover from poor initialization

        let path = adaptive_banded_dp(
            &signal,
            &levels,
            bandwidth,
            &initial_map,
            &RefineAlgo::Viterbi,
        );

        assert_eq!(path.len(), n_bases + 1);
        assert_eq!(path[0], 0);
        assert_eq!(path[n_bases], signal_len);

        for w in path.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing: {:?}", path);
        }
    }

    #[test]
    fn test_adaptive_vs_fixed_agree_on_good_input() {
        // With a good initial map, adaptive and fixed should produce similar results
        let n_bases = 5;
        let spb = 10;
        let signal_len = n_bases * spb;

        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0];
        let mut signal = vec![0.0f32; signal_len];
        for (i, &level) in levels.iter().enumerate() {
            for j in 0..spb {
                signal[i * spb + j] = level;
            }
        }

        let initial_map: Vec<usize> = (0..=n_bases).map(|i| i * spb).collect();
        let half_bw = 5;
        let bandwidth = half_bw * 2;

        // Adaptive path
        let adaptive_path = adaptive_banded_dp(
            &signal,
            &levels,
            bandwidth,
            &initial_map,
            &RefineAlgo::Viterbi,
        );

        // Fixed path using standard banded DP
        let start: Vec<usize> = (0..n_bases)
            .map(|i| (i * spb).saturating_sub(half_bw))
            .collect();
        let end: Vec<usize> = (0..n_bases)
            .map(|i| ((i + 1) * spb + half_bw).min(signal_len))
            .collect();
        let band = Band::new(start, end, true);
        let fixed_path = banded_dp(&signal, &levels, &band, &RefineAlgo::Viterbi);

        // Both should be valid
        assert_eq!(adaptive_path.len(), fixed_path.len());
        assert_eq!(adaptive_path[0], 0);
        assert_eq!(fixed_path[0], 0);
        assert_eq!(adaptive_path[n_bases], signal_len);
        assert_eq!(fixed_path[n_bases], signal_len);

        // Paths should be within a few samples of each other
        for i in 0..=n_bases {
            let diff =
                (adaptive_path[i] as i64 - fixed_path[i] as i64).unsigned_abs() as usize;
            assert!(
                diff <= half_bw + 2,
                "paths diverge at boundary {}: adaptive={}, fixed={}",
                i,
                adaptive_path[i],
                fixed_path[i]
            );
        }
    }

    #[test]
    fn test_adaptive_dp_large_first_dwell() {
        // Regression: when initial_map puts the first boundary far from 0,
        // bw0 can exceed the pre-allocated buffer size and panic.
        // initial_map = [0, 60, 70, 80, 90, 100] with bandwidth=10 gives
        // center0=30, be0=36, bw0=36 vs max_bw=12 → out-of-range without fix.
        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0];
        let signal = vec![0.0f32; 100];
        let initial_map = vec![0, 60, 70, 80, 90, 100];
        let bandwidth = 10;

        let path = adaptive_banded_dp(
            &signal,
            &levels,
            bandwidth,
            &initial_map,
            &RefineAlgo::Viterbi,
        );

        assert_eq!(path.len(), levels.len() + 1);
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), signal.len());

        for w in path.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing: {:?}", path);
        }
    }

    #[test]
    fn test_adaptive_dp_traceback_buffer_shrink() {
        // Regression: curr_traceback could shrink via resize() when bw was
        // between curr_traceback.len() and curr_scores.len(), because
        // curr_scores swaps with prev_scores but curr_traceback does not.
        // This led to a later panic when curr_scores (the other, larger swap
        // buffer) skipped the resize but curr_traceback was too small.
        //
        // Trigger: variable band widths across bases — large first dwell
        // forces a big initial resize, then the last-base extension to
        // signal_len creates another large band via a different swap buffer.
        let n_bases = 10;
        let signal_len = 500;
        let levels = vec![0.5f32; n_bases];
        let signal = vec![0.5f32; signal_len];

        // Highly uneven initial map: first base gets most of the signal,
        // remaining bases are tightly packed at the end.
        let mut initial_map = vec![0usize; n_bases + 1];
        initial_map[0] = 0;
        initial_map[1] = 400; // huge first dwell
        for i in 2..=n_bases {
            initial_map[i] = 400 + (i - 1) * (100 / n_bases);
        }
        initial_map[n_bases] = signal_len;

        let bandwidth = 10;

        let path = adaptive_banded_dp(
            &signal,
            &levels,
            bandwidth,
            &initial_map,
            &RefineAlgo::Viterbi,
        );

        assert_eq!(path.len(), n_bases + 1);
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), signal_len);

        for w in path.windows(2) {
            assert!(w[1] > w[0], "path not strictly increasing: {:?}", path);
        }
    }
}

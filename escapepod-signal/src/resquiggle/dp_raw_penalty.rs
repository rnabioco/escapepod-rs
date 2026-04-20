// SPDX-License-Identifier: GPL-3.0-or-later

//! Banded Viterbi DP with a caller-supplied short-dwell penalty table.
//!
//! Variant of [`super::dp::banded_dp`] that accepts a precomputed penalty
//! table (rather than building one from `target`/`weight`) and uses the
//! table length as the check horizon. Tie-break is strict (`<`): equal
//! stay/move scores keep the stay. These semantics match the Remora-style
//! refinement pipeline and are used by `leech_core` for parity.

const LARGE_SCORE: f32 = 100.0;

#[inline(always)]
fn score(level: f32, signal: f32) -> f32 {
    let d = signal - level;
    d * d
}

/// Standard Viterbi forward step for one base (minimizes squared error).
fn viterbi_step(
    curr_scores: &mut [f32],
    curr_tb: &mut [i32],
    prev_scores: &[f32],
    curr_level: f32,
    curr_signal: &[f32],
    band_start_diff: i32,
) {
    let n_curr = curr_scores.len();
    let mut n_prev = prev_scores.len();
    let prev_offset: usize;

    if band_start_diff == 0 {
        curr_scores[0] = LARGE_SCORE + prev_scores[n_prev - 1];
        curr_tb[0] = -1;
        prev_offset = 0;
    } else {
        let bsd = band_start_diff as usize;
        let base_score = score(curr_level, curr_signal[0]);
        curr_scores[0] = prev_scores[bsd - 1] + base_score;
        curr_tb[0] = 0;
        prev_offset = bsd;
        n_prev = n_prev.saturating_sub(bsd);
    }

    let effective_n_prev = if n_prev == n_curr { n_prev - 1 } else { n_prev };

    for bp in 1..=effective_n_prev {
        let base_score = score(curr_level, curr_signal[bp]);
        let move_score = prev_scores[prev_offset + bp - 1] + base_score;
        let stay_score = curr_scores[bp - 1] + base_score;
        if move_score < stay_score {
            curr_scores[bp] = move_score;
            curr_tb[bp] = 0;
        } else {
            curr_scores[bp] = stay_score;
            curr_tb[bp] = curr_tb[bp - 1] + 1;
        }
    }

    for bp in (effective_n_prev + 1)..n_curr {
        let base_score = score(curr_level, curr_signal[bp]);
        curr_scores[bp] = curr_scores[bp - 1] + base_score;
        curr_tb[bp] = curr_tb[bp - 1] + 1;
    }
}

/// Viterbi forward step with a raw short-dwell penalty table.
#[allow(clippy::too_many_arguments)]
fn dwell_penalty_step(
    curr_scores: &mut [f32],
    curr_tb: &mut [i32],
    prev_scores: &[f32],
    curr_level: f32,
    curr_signal: &[f32],
    band_start_diff: i32,
    dwell_penalty: &[f32],
    scratch_scores: &mut Vec<f32>,
    scratch_tb: &mut Vec<i32>,
) {
    let n_curr = curr_scores.len();
    let n_prev = prev_scores.len();
    let n_pen = dwell_penalty.len();

    scratch_scores.clear();
    scratch_scores.resize(n_curr, 0.0f32);
    scratch_tb.clear();
    scratch_tb.resize(n_curr, 0i32);
    viterbi_step(
        scratch_scores,
        scratch_tb,
        prev_scores,
        curr_level,
        curr_signal,
        band_start_diff,
    );
    let unpen_scores = &*scratch_scores;
    let unpen_tb = &*scratch_tb;

    for bp in 0..n_curr {
        if (bp as i32) + band_start_diff - (n_prev as i32) >= (n_pen as i32) {
            curr_scores[bp] = curr_scores[bp - 1] + score(curr_level, curr_signal[bp]);
            curr_tb[bp] = curr_tb[bp - 1] + 1;
            continue;
        }

        curr_scores[bp] = LARGE_SCORE + prev_scores[n_prev - 1];
        curr_tb[bp] = -1;

        if bp == 0 && band_start_diff == 0 {
            continue;
        }

        let mut running_pos_score: f32 = 0.0;
        for dwell_idx in 0..n_pen {
            if dwell_idx > bp || (band_start_diff == 0 && bp == dwell_idx) {
                break;
            }

            running_pos_score += score(curr_level, curr_signal[bp - dwell_idx]);

            let prev_idx = (bp as i32) - (dwell_idx as i32) - 1 + band_start_diff;
            if prev_idx < 0 || prev_idx >= (n_prev as i32) {
                continue;
            }

            let pos_score =
                prev_scores[prev_idx as usize] + running_pos_score + dwell_penalty[dwell_idx];
            if pos_score < curr_scores[bp] {
                curr_scores[bp] = pos_score;
                curr_tb[bp] = dwell_idx as i32;
            }
        }

        if bp >= n_pen {
            let pos_score = unpen_scores[bp - n_pen] + running_pos_score;
            if pos_score < curr_scores[bp] {
                curr_scores[bp] = pos_score;
                curr_tb[bp] = unpen_tb[bp - n_pen] + (n_pen as i32);
            }
        }
    }
}

/// Banded Viterbi DP with a caller-supplied short-dwell penalty table.
///
/// `band_start[i]` / `band_end[i]` are signed-integer signal indices defining
/// the band for base `i` (inclusive start, exclusive end). Returns a path of
/// length `levels.len() + 1`: `path[i]` is the signal index where base `i`
/// begins; the last entry equals `band_end[seq_len - 1]`.
///
/// When `use_dwell_penalty` is `false`, runs plain Viterbi (ignores
/// `penalty_table`). When `true`, dwells in `0..penalty_table.len()` receive
/// explicit penalties and longer dwells fall back to baseline Viterbi.
pub fn banded_dp_with_penalty_table(
    signal: &[f32],
    levels: &[f32],
    band_start: &[i32],
    band_end: &[i32],
    penalty_table: &[f32],
    use_dwell_penalty: bool,
) -> Vec<i32> {
    let seq_len = levels.len();

    let mut base_offsets = vec![0u32; seq_len + 1];
    for i in 0..seq_len {
        let bw = (band_end[i] - band_start[i]) as u32;
        base_offsets[i + 1] = base_offsets[i] + bw;
    }
    let band_len = base_offsets[seq_len] as usize;

    let mut all_scores = vec![0.0f32; band_len];
    let mut traceback = vec![0i32; band_len];

    let first_bw = (band_end[0] - band_start[0]) as usize;
    let mut prev_scores = vec![f32::MAX; first_bw];
    prev_scores[0] = 0.0;

    let max_bw = (0..seq_len)
        .map(|i| (band_end[i] - band_start[i]) as usize)
        .max()
        .unwrap_or(0);
    let mut scratch_scores = Vec::with_capacity(max_bw);
    let mut scratch_tb = Vec::with_capacity(max_bw);

    if use_dwell_penalty {
        dwell_penalty_step(
            &mut all_scores[..first_bw],
            &mut traceback[..first_bw],
            &prev_scores,
            levels[0],
            &signal[..first_bw],
            1,
            penalty_table,
            &mut scratch_scores,
            &mut scratch_tb,
        );
    } else {
        viterbi_step(
            &mut all_scores[..first_bw],
            &mut traceback[..first_bw],
            &prev_scores,
            levels[0],
            &signal[..first_bw],
            1,
        );
    }

    let mut prev_band_st = 0i32;
    let mut prev_bw = first_bw;
    let mut prev_offset = 0usize;

    for base_idx in 1..seq_len {
        let curr_band_st = band_start[base_idx];
        let curr_band_en = band_end[base_idx];
        let curr_bw = (curr_band_en - curr_band_st) as usize;
        let curr_offset = base_offsets[base_idx] as usize;
        let band_start_diff = curr_band_st - prev_band_st;

        let sig_slice = &signal[curr_band_st as usize..curr_band_en as usize];

        let (left, right) = all_scores.split_at_mut(curr_offset);
        let prev_sc = &left[prev_offset..prev_offset + prev_bw];
        let cs = &mut right[..curr_bw];
        let ct = &mut traceback[curr_offset..curr_offset + curr_bw];

        if use_dwell_penalty {
            dwell_penalty_step(
                cs,
                ct,
                prev_sc,
                levels[base_idx],
                sig_slice,
                band_start_diff,
                penalty_table,
                &mut scratch_scores,
                &mut scratch_tb,
            );
        } else {
            viterbi_step(
                cs,
                ct,
                prev_sc,
                levels[base_idx],
                sig_slice,
                band_start_diff,
            );
        }

        prev_band_st = curr_band_st;
        prev_bw = curr_bw;
        prev_offset = curr_offset;
    }

    let sig_len = band_end[seq_len - 1];
    let mut path = vec![0i32; seq_len + 1];
    path[0] = 0;
    path[seq_len] = sig_len;

    for base_idx in (1..seq_len).rev() {
        let sig_lookup_pos = path[base_idx + 1] - 1;
        let band_idx = sig_lookup_pos - band_start[base_idx];
        let offset = base_offsets[base_idx] as usize + band_idx as usize;
        let next_sig_offset = traceback[offset];
        path[base_idx] = sig_lookup_pos - next_sig_offset;
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viterbi_clean_signal_recovers_boundaries() {
        let n_bases = 5;
        let spb = 10;
        let sig_len = n_bases * spb;
        let levels: Vec<f32> = vec![0.0, 1.0, -0.5, 0.5, -1.0];
        let mut signal = vec![0.0f32; sig_len];
        for (i, &lv) in levels.iter().enumerate() {
            for j in 0..spb {
                signal[i * spb + j] = lv;
            }
        }

        let half_bw = 3i32;
        let band_start: Vec<i32> = (0..n_bases)
            .map(|i| ((i * spb) as i32 - half_bw).max(0))
            .collect();
        let band_end: Vec<i32> = (0..n_bases)
            .map(|i| (((i + 1) * spb) as i32 + half_bw).min(sig_len as i32))
            .collect();

        let pen = [0.0f32; 0];
        let path =
            banded_dp_with_penalty_table(&signal, &levels, &band_start, &band_end, &pen, false);

        assert_eq!(path.len(), n_bases + 1);
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), sig_len as i32);
        for w in path.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn dwell_penalty_step_matches_viterbi_with_empty_table() {
        // With an empty penalty table and use_dwell_penalty=true, the dwell
        // check loop never fires, and the `bp >= n_pen` fallback uses the
        // baseline Viterbi score at `bp - 0 = bp`. Because running_pos_score
        // is zero at that point, the function copies the baseline Viterbi
        // result. Path should therefore equal the pure-Viterbi path.
        let n_bases = 4;
        let spb = 8;
        let sig_len = n_bases * spb;
        let levels: Vec<f32> = vec![0.2, -0.3, 0.4, -0.1];
        let mut signal = vec![0.0f32; sig_len];
        for (i, &lv) in levels.iter().enumerate() {
            for j in 0..spb {
                signal[i * spb + j] = lv + 0.01 * (j as f32);
            }
        }

        let half_bw = 2i32;
        let band_start: Vec<i32> = (0..n_bases)
            .map(|i| ((i * spb) as i32 - half_bw).max(0))
            .collect();
        let band_end: Vec<i32> = (0..n_bases)
            .map(|i| (((i + 1) * spb) as i32 + half_bw).min(sig_len as i32))
            .collect();

        let pen: [f32; 0] = [];
        let vit =
            banded_dp_with_penalty_table(&signal, &levels, &band_start, &band_end, &pen, false);
        let dwp =
            banded_dp_with_penalty_table(&signal, &levels, &band_start, &band_end, &pen, true);
        assert_eq!(vit, dwp);
    }
}

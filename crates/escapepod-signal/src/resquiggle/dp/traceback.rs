// SPDX-License-Identifier: GPL-3.0-or-later
// Inspired by fishnet, licensed under the GNU General Public License v3.0.

//! Traceback to reconstruct the optimal alignment path from the DP matrix.

use crate::resquiggle::bands::Band;

/// Traceback to reconstruct the optimal path.
pub fn banded_traceback(
    path: &mut [usize],
    band: &Band,
    base_offsets: &[usize],
    traceback: &[i32],
) {
    let seq_band_start = &band.start;
    let seq_band_end = &band.end;

    path[0] = 0;
    let last = path.len() - 1;
    path[last] = seq_band_end[seq_band_end.len() - 1];

    for base_idx in (1..last).rev() {
        let sig_lookup_pos = path[base_idx + 1] - 1;
        let base_offset = base_offsets[base_idx];
        let band_start = seq_band_start[base_idx];
        let traceback_idx = base_offset + (sig_lookup_pos - band_start);
        let next_sig_offset = traceback[traceback_idx];
        path[base_idx] = if next_sig_offset >= 0 {
            sig_lookup_pos - (next_sig_offset as usize)
        } else {
            // Traceback hit an unreachable cell (band too narrow or bad inputs).
            // Fall back to band start to avoid underflow.
            band_start
        };
    }
}

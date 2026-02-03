// SPDX-License-Identifier: GPL-3.0-or-later
// Inspired by fishnet, licensed under the GNU General Public License v3.0.

//! Signal band and sequence band computation for banded dynamic programming.

use anyhow::{bail, Result};

/// A band constraining the DP search space.
///
/// For a **sequence band**, entry `i` corresponds to base `i`.
/// `start[i]` is the first signal measurement, `end[i]` the last (exclusive)
/// that the base may align to.
#[derive(Debug)]
pub struct Band {
    pub start: Vec<usize>,
    pub end: Vec<usize>,
    pub is_sequence_band: bool,
}

impl Band {
    /// Create a band directly (for testing).
    pub fn new(start: Vec<usize>, end: Vec<usize>, is_sequence_band: bool) -> Self {
        Band {
            start,
            end,
            is_sequence_band,
        }
    }

    /// Compute a signal band from a sequence-to-signal map.
    ///
    /// For each signal measurement, defines which bases it could align to.
    pub fn compute_signal_band(
        map: &[usize],
        seq_len: usize,
        half_bandwidth: usize,
    ) -> Result<Self> {
        let map_len = map.len();
        if seq_len != map_len - 1 {
            bail!("map length {} != seq_len + 1 ({})", map_len, seq_len + 1);
        }
        if half_bandwidth == 0 {
            bail!("half_bandwidth must be > 0");
        }

        let signal_len = map[map_len - 1] - map[0];
        let mut start = vec![0usize; signal_len];
        let mut end = vec![seq_len; signal_len];

        for seq_idx in 0..seq_len {
            let seq_start = map[seq_idx];
            let seq_end = map[seq_idx + 1];
            for sig_idx in seq_start..seq_end {
                let local_idx = sig_idx - map[0];
                if local_idx >= signal_len {
                    break;
                }
                if seq_idx >= half_bandwidth {
                    start[local_idx] = seq_idx - half_bandwidth;
                }
                end[local_idx] = (seq_idx + half_bandwidth + 1).min(seq_len);
            }
        }

        // Ensure monotonicity
        for i in 1..signal_len {
            start[i] = start[i].max(start[i - 1]);
        }
        for i in (0..signal_len - 1).rev() {
            end[i] = end[i].min(end[i + 1]);
        }

        // Validate
        if start[0] != 0 {
            bail!("signal band start[0] != 0");
        }
        if end[signal_len - 1] != seq_len {
            bail!("signal band end[last] != seq_len");
        }
        for i in 0..signal_len {
            if end[i] <= start[i] {
                bail!("signal band has zero-length region at index {}", i);
            }
        }

        Ok(Band {
            start,
            end,
            is_sequence_band: false,
        })
    }

    /// Convert a signal band to a sequence band in-place.
    pub fn convert_to_sequence_band(&mut self, min_step: usize) -> Result<()> {
        if self.is_sequence_band {
            bail!("already a sequence band");
        }

        let signal_len = self.start.len();
        let seq_len = self.end[self.end.len() - 1];

        let mut seq_start = vec![0usize; seq_len];
        let mut seq_end = vec![signal_len; seq_len];

        // Find change points in end array
        for (sig_idx, window) in self.end.windows(2).enumerate() {
            if window[0] != window[1] {
                let lower_sig_pos = sig_idx + 1;
                let lower_base_pos = self.end[sig_idx];
                if lower_base_pos < seq_len {
                    seq_start[lower_base_pos] = lower_sig_pos;
                }
            }
        }

        // Find change points in start array
        for (sig_idx, window) in self.start.windows(2).enumerate() {
            if window[0] != window[1] {
                let upper_sig_pos = sig_idx + 1;
                let upper_base_pos = self.start[upper_sig_pos];
                if upper_base_pos > 0 {
                    seq_end[upper_base_pos - 1] = upper_sig_pos;
                }
            }
        }

        // Enforce monotonicity on start (forward max)
        let mut max_so_far = 0;
        for val in seq_start.iter_mut() {
            max_so_far = max_so_far.max(*val);
            *val = max_so_far;
        }

        // Enforce monotonicity on end (reverse min)
        let mut min_so_far = signal_len;
        for val in seq_end.iter_mut().rev() {
            min_so_far = min_so_far.min(*val);
            *val = min_so_far;
        }

        self.start = seq_start;
        self.end = seq_end;
        self.is_sequence_band = true;

        self.enforce_min_step(min_step)?;

        // Validate
        if self.start[0] != 0 {
            bail!("sequence band start[0] != 0");
        }
        if self.end[seq_len - 1] != signal_len {
            bail!("sequence band end[last] != signal_len");
        }
        for i in 0..seq_len {
            if self.end[i] <= self.start[i] {
                bail!("sequence band has zero-length region at index {}", i);
            }
        }

        Ok(())
    }

    /// Enforce minimum signal step between consecutive bases in a sequence band.
    fn enforce_min_step(&mut self, min_step: usize) -> Result<()> {
        let band_min = self.start[0];
        let band_max = self.end[self.end.len() - 1];
        let seq_len = self.start.len();

        // Fix starts: each start at least min_step less than the next
        for seq_pos in (0..seq_len - 1).rev() {
            if self.start[seq_pos] > self.start[seq_pos + 1].saturating_sub(min_step) {
                self.start[seq_pos] = self.start[seq_pos + 1].saturating_sub(min_step);
            }
        }

        // Restore first start
        self.start[0] = band_min;

        // Ensure monotonically increasing from the beginning
        let mut seq_pos = 1;
        while seq_pos < seq_len && self.start[seq_pos] <= self.start[seq_pos - 1] {
            self.start[seq_pos] = self.start[seq_pos - 1] + 1;
            seq_pos += 1;
        }

        // Fix ends: each end at least min_step more than the previous
        for seq_pos in 1..seq_len {
            if self.end[seq_pos] < self.end[seq_pos - 1] + min_step {
                self.end[seq_pos] = self.end[seq_pos - 1] + min_step;
            }
        }

        // Restore last end
        self.end[seq_len - 1] = band_max;

        // Ensure monotonically increasing from the end
        if seq_len > 1 {
            let mut seq_pos = seq_len - 2;
            while self.end[seq_pos] >= self.end[seq_pos + 1] {
                self.end[seq_pos] = self.end[seq_pos + 1] - 1;
                if seq_pos == 0 {
                    break;
                }
                seq_pos -= 1;
            }
        }

        Ok(())
    }

    /// Number of entries in the band.
    pub fn len(&self) -> usize {
        self.start.len()
    }

    /// Returns true if the band is empty.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }
}

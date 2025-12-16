//! Signal processing utilities for POD5 data.
//!
//! This module provides functions for manipulating nanopore signal data,
//! including downsampling for archival purposes.

/// Downsample signal data by a given factor.
///
/// This function reduces the number of samples by keeping every Nth sample,
/// where N is the downsample factor. For archival purposes, this provides
/// a simple decimation that maintains signal shape while reducing storage.
///
/// # Arguments
/// * `signal` - The input signal samples
/// * `factor` - The downsampling factor (e.g., 2 means keep every other sample)
///
/// # Returns
/// A new vector with downsampled signal data
///
/// # Example
/// ```
/// use podfive_core::signal::downsample;
///
/// let signal = vec![100i16, 110, 105, 115, 108, 118, 112, 122];
/// let downsampled = downsample(&signal, 2);
/// assert_eq!(downsampled, vec![100, 105, 108, 112]);
/// ```
pub fn downsample(signal: &[i16], factor: u32) -> Vec<i16> {
    if factor <= 1 || signal.is_empty() {
        return signal.to_vec();
    }

    let factor = factor as usize;
    signal.iter().step_by(factor).copied().collect()
}

/// Downsample signal data with averaging.
///
/// This function reduces the number of samples by averaging groups of N samples,
/// where N is the downsample factor. This provides better noise reduction
/// compared to simple decimation but is computationally more expensive.
///
/// # Arguments
/// * `signal` - The input signal samples
/// * `factor` - The downsampling factor (e.g., 2 means average pairs of samples)
///
/// # Returns
/// A new vector with downsampled signal data
///
/// # Example
/// ```
/// use podfive_core::signal::downsample_average;
///
/// let signal = vec![100i16, 110, 105, 115, 108, 118, 112, 122];
/// let downsampled = downsample_average(&signal, 2);
/// assert_eq!(downsampled, vec![105, 110, 113, 117]); // averages of pairs
/// ```
pub fn downsample_average(signal: &[i16], factor: u32) -> Vec<i16> {
    if factor <= 1 || signal.is_empty() {
        return signal.to_vec();
    }

    let factor = factor as usize;
    signal
        .chunks(factor)
        .map(|chunk| {
            let sum: i32 = chunk.iter().map(|&s| s as i32).sum();
            (sum / chunk.len() as i32) as i16
        })
        .collect()
}

/// Calculate the number of samples after downsampling.
///
/// # Arguments
/// * `original_samples` - The original number of samples
/// * `factor` - The downsampling factor
///
/// # Returns
/// The number of samples after downsampling
pub fn downsampled_count(original_samples: u64, factor: u32) -> u64 {
    if factor <= 1 {
        return original_samples;
    }
    (original_samples + factor as u64 - 1) / factor as u64
}

/// Calculate the effective sample rate after downsampling.
///
/// # Arguments
/// * `original_rate` - The original sample rate in Hz
/// * `factor` - The downsampling factor
///
/// # Returns
/// The effective sample rate after downsampling
pub fn downsampled_rate(original_rate: u16, factor: u32) -> u16 {
    if factor <= 1 {
        return original_rate;
    }
    (original_rate as u32 / factor) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_factor_1() {
        let signal = vec![100i16, 110, 105, 115];
        let result = downsample(&signal, 1);
        assert_eq!(result, signal);
    }

    #[test]
    fn test_downsample_factor_2() {
        let signal = vec![100i16, 110, 105, 115, 108, 118, 112, 122];
        let result = downsample(&signal, 2);
        assert_eq!(result, vec![100, 105, 108, 112]);
    }

    #[test]
    fn test_downsample_factor_4() {
        let signal = vec![100i16, 110, 105, 115, 108, 118, 112, 122];
        let result = downsample(&signal, 4);
        assert_eq!(result, vec![100, 108]);
    }

    #[test]
    fn test_downsample_empty() {
        let signal: Vec<i16> = vec![];
        let result = downsample(&signal, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_downsample_uneven() {
        // 7 samples with factor 3 should give 3 samples (indices 0, 3, 6)
        let signal = vec![100i16, 110, 105, 115, 108, 118, 112];
        let result = downsample(&signal, 3);
        assert_eq!(result, vec![100, 115, 112]);
    }

    #[test]
    fn test_downsample_average_factor_2() {
        let signal = vec![100i16, 110, 104, 116, 108, 118, 112, 122];
        let result = downsample_average(&signal, 2);
        assert_eq!(result, vec![105, 110, 113, 117]);
    }

    #[test]
    fn test_downsample_average_uneven() {
        // 7 samples with factor 3: [100,110,105], [115,108,118], [112]
        let signal = vec![100i16, 110, 105, 115, 108, 118, 112];
        let result = downsample_average(&signal, 3);
        assert_eq!(result, vec![105, 113, 112]); // (100+110+105)/3=105, (115+108+118)/3=113, 112/1=112
    }

    #[test]
    fn test_downsampled_count() {
        assert_eq!(downsampled_count(1000, 2), 500);
        assert_eq!(downsampled_count(1000, 4), 250);
        assert_eq!(downsampled_count(1001, 2), 501); // ceiling division
        assert_eq!(downsampled_count(1000, 1), 1000);
    }

    #[test]
    fn test_downsampled_rate() {
        assert_eq!(downsampled_rate(4000, 2), 2000);
        assert_eq!(downsampled_rate(5000, 2), 2500);
        assert_eq!(downsampled_rate(4000, 4), 1000);
        assert_eq!(downsampled_rate(4000, 1), 4000);
    }
}

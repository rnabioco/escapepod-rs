//! Configuration types for signal-to-sequence refinement.

/// Algorithm used for mapping refinement.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineAlgo {
    /// Viterbi algorithm (short dwell times are not penalized).
    Viterbi,
    /// Dwell penalty algorithm, which discourages short dwell times.
    DwellPenalty {
        /// Preferred dwell time.
        target: f32,
        /// Maximum dwell time that is penalized.
        limit: f32,
        /// Strength of the penalty.
        weight: f32,
    },
}

impl Default for RefineAlgo {
    fn default() -> Self {
        Self::DwellPenalty {
            target: 4.0,
            limit: 3.0,
            weight: 0.5,
        }
    }
}

/// Algorithm for precise signal rescaling.
#[derive(Debug, Clone, PartialEq)]
pub enum RescaleAlgo {
    /// Least-squares regression-based rescaling.
    LeastSquares {
        dwell_filter_lower_percentile: f32,
        dwell_filter_upper_percentile: f32,
        min_abs_level: f32,
        n_bases_truncate: usize,
        min_num_filtered_levels: usize,
    },
    /// Theil-Sen estimator-based rescaling.
    TheilSen {
        dwell_filter_lower_percentile: f32,
        dwell_filter_upper_percentile: f32,
        min_abs_level: f32,
        n_bases_truncate: usize,
        min_num_filtered_levels: usize,
        max_points: usize,
    },
}

impl Default for RescaleAlgo {
    fn default() -> Self {
        Self::TheilSen {
            dwell_filter_lower_percentile: 0.1,
            dwell_filter_upper_percentile: 0.9,
            min_abs_level: 0.2,
            n_bases_truncate: 10,
            min_num_filtered_levels: 10,
            max_points: 1000,
        }
    }
}

/// Algorithm for initial rough rescaling of signals.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RoughRescaleAlgo {
    /// No rough rescaling applied.
    #[default]
    None,
    /// Least-squares regression-based rough rescaling.
    LeastSquares {
        quantiles: Vec<f32>,
        clip_bases: usize,
        use_base_center: bool,
    },
    /// Theil-Sen estimator-based rough rescaling.
    TheilSen {
        quantiles: Vec<f32>,
        clip_bases: usize,
        use_base_center: bool,
    },
}

impl RoughRescaleAlgo {
    /// Default quantiles used for rough rescaling (0.05 to 0.95 in steps of 0.05).
    pub fn default_quantiles() -> Vec<f32> {
        vec![
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
            0.85, 0.9, 0.95,
        ]
    }
}

/// Settings for the refinement pipeline.
#[derive(Debug, Clone)]
pub struct RefineSettings {
    /// Algorithm used for mapping refinement.
    pub refinement_algo: RefineAlgo,
    /// Number of refinement iterations.
    pub n_refinement_iters: usize,
    /// Half of the bandwidth for banded DP.
    pub half_bandwidth: usize,
    /// Minimum step between bases in band adjustment.
    pub adjust_band_min_size: usize,
    /// Algorithm for precise rescaling.
    pub rescale_algo: RescaleAlgo,
    /// Algorithm for initial rough rescaling.
    pub rough_rescale_algo: RoughRescaleAlgo,
    /// Whether to normalize kmer levels with MAD.
    pub normalize_levels: bool,
}

impl Default for RefineSettings {
    fn default() -> Self {
        Self {
            refinement_algo: RefineAlgo::default(),
            n_refinement_iters: 1,
            half_bandwidth: 5,
            adjust_band_min_size: 2,
            rescale_algo: RescaleAlgo::default(),
            rough_rescale_algo: RoughRescaleAlgo::default(),
            normalize_levels: false,
        }
    }
}

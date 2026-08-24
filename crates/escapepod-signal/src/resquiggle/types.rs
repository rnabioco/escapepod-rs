// SPDX-License-Identifier: MIT
// Algorithm inspired by fishnet (Brickner et al.); independent implementation.

//! Configuration types for signal-to-sequence refinement.

/// Algorithm used for mapping refinement.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineAlgo {
    /// Viterbi algorithm (short dwell times are not penalized).
    Viterbi,
    /// Dwell penalty algorithm with asymmetric penalty: quadratic below target,
    /// logarithmic above. Discourages short dwells strongly while allowing
    /// genuinely long dwells (e.g., aminoacylation) to survive.
    DwellPenalty {
        /// Preferred dwell time. Non-positive values (see
        /// [`RefineAlgo::PER_READ_DWELL_TARGET`]) resolve per read from the
        /// median dwell of the input move table.
        target: f32,
        /// Strength of the penalty.
        weight: f32,
    },
}

impl RefineAlgo {
    /// Sentinel `DwellPenalty::target` meaning "resolve the target from this
    /// read's own move table" rather than pinning a constant.
    ///
    /// [`refine_signal_map`](super::refine::refine_signal_map) replaces any
    /// non-positive `target` with the median dwell of the *input*
    /// `seq_to_signal_map` before the first DP pass, so the penalty is centred
    /// on the read's actual samples-per-base instead of on a number chosen for
    /// some other chemistry. There is no distinct "auto" variant to match on —
    /// the sentinel *is* the representation, so use this constant rather than
    /// writing a bare `0.0` whose meaning is invisible at the call site.
    ///
    /// This matters more than it looks. RNA004 at 130 bases/s and 4 kHz sits
    /// near 31 samples/base; a target of `4.0` therefore treats every base as
    /// roughly 8x too long, and the asymmetric penalty (quadratic below target,
    /// logarithmic above) pushes boundaries toward implausibly short dwells.
    pub const PER_READ_DWELL_TARGET: f32 = 0.0;
}

impl Default for RefineAlgo {
    fn default() -> Self {
        Self::DwellPenalty {
            target: Self::PER_READ_DWELL_TARGET,
            weight: 0.5,
        }
    }
}

/// Shared filter parameters for rescaling algorithms.
#[derive(Debug, Clone, PartialEq)]
pub struct RescaleFilterParams {
    pub dwell_filter_lower_percentile: f32,
    pub dwell_filter_upper_percentile: f32,
    pub min_abs_level: f32,
    pub n_bases_truncate: usize,
    pub min_num_filtered_levels: usize,
}

impl Default for RescaleFilterParams {
    fn default() -> Self {
        Self {
            dwell_filter_lower_percentile: 0.1,
            dwell_filter_upper_percentile: 0.9,
            min_abs_level: 0.2,
            n_bases_truncate: 10,
            min_num_filtered_levels: 10,
        }
    }
}

/// Algorithm for precise signal rescaling.
#[derive(Debug, Clone, PartialEq)]
pub enum RescaleAlgo {
    /// Least-squares regression-based rescaling.
    LeastSquares { filter: RescaleFilterParams },
    /// Theil-Sen estimator-based rescaling.
    TheilSen {
        filter: RescaleFilterParams,
        max_points: usize,
        /// RNG seed for the point subsample when there are more than
        /// `max_points` filtered levels. `Some(seed)` makes the subsample (and
        /// therefore the rescale/refined map) reproducible; `None` samples from
        /// an unseeded RNG, so results vary across runs. Downstream ML pipelines
        /// that need deterministic features should set a seed.
        seed: Option<u64>,
    },
}

impl RescaleAlgo {
    /// Access the shared filter parameters.
    pub fn filter_params(&self) -> &RescaleFilterParams {
        match self {
            Self::LeastSquares { filter } => filter,
            Self::TheilSen { filter, .. } => filter,
        }
    }

    /// Maximum random subset size (only meaningful for Theil-Sen; returns 0 for LeastSquares).
    pub fn max_points(&self) -> usize {
        match self {
            Self::TheilSen { max_points, .. } => *max_points,
            Self::LeastSquares { .. } => 0,
        }
    }

    /// Subsample RNG seed (Theil-Sen only; `None` = unseeded/random).
    pub fn seed(&self) -> Option<u64> {
        match self {
            Self::TheilSen { seed, .. } => *seed,
            Self::LeastSquares { .. } => None,
        }
    }
}

impl Default for RescaleAlgo {
    fn default() -> Self {
        Self::TheilSen {
            filter: RescaleFilterParams::default(),
            max_points: 1000,
            seed: None,
        }
    }
}

/// Algorithm for initial rough rescaling of signals.
#[derive(Debug, Clone, PartialEq)]
pub enum RoughRescaleAlgo {
    /// No rough rescaling applied.
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

impl Default for RoughRescaleAlgo {
    fn default() -> Self {
        Self::TheilSen {
            quantiles: Self::default_quantiles(),
            clip_bases: 10,
            use_base_center: true,
        }
    }
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

/// Algorithm for computing the DP band.
#[derive(Debug, Clone, PartialEq)]
pub enum BandingAlgo {
    /// Fixed band computed from the initial signal-to-sequence map.
    Fixed,
    /// Adaptive banding (Suzuki & Kasahara, 2017): band center shifts during
    /// the forward pass based on edge score comparisons.
    Adaptive {
        /// Full bandwidth (number of signal positions per base in the band).
        bandwidth: usize,
        /// Optional X-drop threshold for early termination.  When the best
        /// per-base score exceeds the global best by more than this value the
        /// DP bails out and returns the initial map.
        x_drop: Option<f32>,
    },
}

impl Default for BandingAlgo {
    fn default() -> Self {
        Self::Adaptive {
            bandwidth: 10,
            x_drop: None,
        }
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
    /// Algorithm for computing the DP band.
    pub banding_algo: BandingAlgo,
}

impl Default for RefineSettings {
    fn default() -> Self {
        Self {
            refinement_algo: RefineAlgo::default(),
            n_refinement_iters: 2,
            half_bandwidth: 5,
            adjust_band_min_size: 2,
            rescale_algo: RescaleAlgo::default(),
            rough_rescale_algo: RoughRescaleAlgo::default(),
            normalize_levels: false,
            banding_algo: BandingAlgo::default(),
        }
    }
}

impl RefineSettings {
    /// The refinement configuration for a basecaller move table: fixed banding,
    /// a least-squares rough rescale over the 0.05–0.95 quantiles clipped 10
    /// bases with `use_base_center`, a Theil-Sen inter-iteration rescale over
    /// at most 200 points, level normalization off, and the asymmetric dwell
    /// penalty at weight 0.5 with a **per-read** target.
    ///
    /// This exists because it was written twice — once in escapepod's own
    /// Python binding and once in a downstream Rust consumer — with a comment
    /// on each saying it matched the other. One field drifted anyway
    /// (`dwell_target`: a fixed `4.0` against the per-read resolution), and the
    /// two paths refined the same reads to different boundaries for four
    /// releases. A settings block that two callers must agree on bit-for-bit is
    /// a value, not a convention; this is that value.
    ///
    /// The dwell target is [`RefineAlgo::PER_READ_DWELL_TARGET`], the sentinel
    /// `0.0` that
    /// [`refine_signal_map`](super::refine::refine_signal_map) replaces with
    /// the median dwell of the read's own input map. A constant target only
    /// ever suits one chemistry at one translocation rate, and the penalty is
    /// asymmetric, so guessing low is not a soft error: it drags boundaries
    /// toward dwells the pore never produced.
    ///
    /// `seed` is the Theil-Sen subsample seed — pass `Some(_)` for
    /// reproducible features, `None` to sample unseeded.
    ///
    /// # Example
    ///
    /// ```
    /// use escapepod_signal::resquiggle::{RefineAlgo, RefineSettings};
    ///
    /// let settings = RefineSettings::move_table_refinement(5, 2, Some(0));
    /// assert_eq!(
    ///     settings.refinement_algo,
    ///     RefineAlgo::DwellPenalty {
    ///         target: RefineAlgo::PER_READ_DWELL_TARGET,
    ///         weight: 0.5,
    ///     }
    /// );
    /// ```
    pub fn move_table_refinement(half_bandwidth: usize, n_iters: usize, seed: Option<u64>) -> Self {
        Self {
            refinement_algo: RefineAlgo::DwellPenalty {
                target: RefineAlgo::PER_READ_DWELL_TARGET,
                weight: Self::MOVE_TABLE_DWELL_WEIGHT,
            },
            n_refinement_iters: n_iters,
            half_bandwidth,
            adjust_band_min_size: 2,
            rescale_algo: RescaleAlgo::TheilSen {
                filter: RescaleFilterParams::default(),
                max_points: 200,
                seed,
            },
            rough_rescale_algo: RoughRescaleAlgo::LeastSquares {
                quantiles: RoughRescaleAlgo::default_quantiles(),
                clip_bases: 10,
                use_base_center: true,
            },
            normalize_levels: false,
            banding_algo: BandingAlgo::Fixed,
        }
    }

    /// Dwell-penalty weight carried by [`RefineSettings::move_table_refinement`].
    pub const MOVE_TABLE_DWELL_WEIGHT: f32 = 0.5;
}

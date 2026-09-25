- **`level_skew` and `level_kurtosis` per-base rows** (#396).
  `escapepod_signal::features::span_stats` gains `skew`/`kurtosis` outputs
  (`SpanStatsOut::with_skew`/`with_kurtosis`) and `chunk::FeatureChannel`
  gains `LevelSkew`/`LevelKurtosis` (`"level_skew"`/`"level_kurtosis"`): the
  third and fourth standardised central moments of a base's signal span,
  population convention and Fisher excess kurtosis, i.e. `scipy.stats.skew`/
  `kurtosis` at their defaults. `level_mean`/`level_std` describe a span by
  its first two moments, which a stall does not change; skew and kurtosis
  read the tail asymmetry and tail weight it does move. Both are computed
  two-pass in `f64` from the per-span gather `median`/`range` already share,
  never from prefix sums of cubes and quartics. A span with zero variance
  (constant, or a single sample) reads `0.0` for both rather than scipy's
  `NaN`; a `NaN` inside the span propagates to both. `escapepod-python`'s
  `span_statistics`/`span_statistics_batch` gain matching `skew=False,
  kurtosis=False` keyword flags, appended after `median`/`range` in that
  fixed order. A `waveform_model` bundle naming the two rows resolves them
  with no change to the classify runtime, and the load error's list of known
  rows is now derived from `FeatureChannel::ALL` rather than typed by hand.
  leech's uptake — its channel list has to become recorded data first — is
  rnabioco/leech#358.

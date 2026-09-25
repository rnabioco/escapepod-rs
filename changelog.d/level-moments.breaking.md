- **`escapepod_signal::features::SpanStatsOut` is `#[non_exhaustive]`** (#396).
  An out-of-crate struct literal (leech's `features_stats.rs` builds one) no
  longer compiles; use `SpanStatsOut::new(dwell, mean, sd)` plus
  `.with_median(..)`/`.with_range(..)`/`.with_skew(..)`/`.with_kurtosis(..)`,
  which the struct's doc has asked for since the optional outputs appeared.
  Every future output is then an addition rather than a break.

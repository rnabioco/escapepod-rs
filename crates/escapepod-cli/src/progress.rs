//! Shared progress bar utilities.

use indicatif::{HumanCount, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};

/// Whether interactive progress indicators should be drawn.
///
/// Progress bars are status output, so they follow the same verbosity gate as
/// `tracing` status events: suppressed once the level drops below INFO (i.e.
/// under `-q`). When hidden, the returned [`ProgressBar`] is a no-op, so call
/// sites need no extra branching.
fn progress_enabled() -> bool {
    tracing::enabled!(tracing::Level::INFO)
}

/// Create a progress bar with the standard style: live throughput
/// (items/s over the whole run), elapsed, and ETA.
pub fn create_progress_bar(total: u64, prefix: &str) -> anyhow::Result<ProgressBar> {
    if !progress_enabled() {
        return Ok(ProgressBar::hidden());
    }
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} ({rate}) {msg} [{elapsed_precise}] ETA: {eta}")?
            // indicatif's stock {per_sec} prints four decimals; render the
            // rate as a rounded human count instead ("12,847/s").
            .with_key("rate", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                let _ = write!(w, "{}/s", HumanCount(state.per_sec().round() as u64));
            })
            .progress_chars("━━─"),
    );
    pb.set_prefix(prefix.to_string());
    Ok(pb)
}

/// Create a spinner for indeterminate progress.
pub fn create_spinner(prefix: &str) -> anyhow::Result<ProgressBar> {
    if !progress_enabled() {
        return Ok(ProgressBar::with_draw_target(
            None,
            ProgressDrawTarget::hidden(),
        ));
    }
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::default_spinner().template("{prefix:.bold} {spinner} {msg}")?);
    spinner.set_prefix(prefix.to_string());
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    Ok(spinner)
}

//! Shared phase-timing helper used by long-running CLI commands.
//!
//! `PhaseTimer` tracks named phases plus total elapsed time. Commands call
//! [`phase`] to start/rotate a phase and [`finish`] at the end. When the
//! caller's `--profile` flag is set, [`report`] prints a breakdown to stderr;
//! otherwise the timer is a cheap no-op beyond the `Instant::now()` calls.
//!
//! Per-phase durations also emit `tracing::debug!` events, so `-vv` will
//! surface them without needing `--profile`.
//!
//! [`phase`]: PhaseTimer::phase
//! [`finish`]: PhaseTimer::finish
//! [`report`]: PhaseTimer::report

use std::time::{Duration, Instant};

use crate::style;

pub struct PhaseTimer {
    phases: Vec<(String, Duration)>,
    current: Option<(String, Instant)>,
    total_start: Instant,
}

impl PhaseTimer {
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            current: None,
            total_start: Instant::now(),
        }
    }

    /// Start a new phase. Any currently-open phase is closed and its duration
    /// recorded.
    pub fn phase(&mut self, name: impl Into<String>) {
        self.close_current();
        self.current = Some((name.into(), Instant::now()));
    }

    fn close_current(&mut self) {
        if let Some((name, start)) = self.current.take() {
            let dur = start.elapsed();
            tracing::debug!(phase = %name, elapsed_ms = dur.as_millis() as u64, "phase complete");
            self.phases.push((name, dur));
        }
    }

    pub fn total(&self) -> Duration {
        self.total_start.elapsed()
    }

    /// Print a per-phase breakdown to stderr if `enabled`. Closes any open
    /// phase first.
    pub fn report(&mut self, enabled: bool) {
        self.close_current();
        if !enabled {
            return;
        }
        let total = self.total();
        eprintln!();
        eprintln!("{}", style::action("Profile"));
        for (name, dur) in &self.phases {
            let pct = if total.as_secs_f64() > 0.0 {
                (dur.as_secs_f64() / total.as_secs_f64()) * 100.0
            } else {
                0.0
            };
            eprintln!("  {:<30} {:>8.2}s ({:>5.1}%)", name, dur.as_secs_f64(), pct);
        }
        eprintln!("  {:<30} {:>8.2}s", "Total", total.as_secs_f64());
    }
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

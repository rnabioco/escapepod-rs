//! Global rayon thread pool configuration.
//!
//! There is exactly **one** call site: [`init`], from `main`, before command
//! dispatch. That is not a style preference — it is the invariant that keeps
//! `-t/--threads` meaningful.
//!
//! rayon's global pool can be built only once per process, and only before
//! anything else touches it: a single `par_iter()` earlier in the run lazily
//! creates the pool at `available_parallelism()` width, after which
//! `build_global()` returns `Err` forever. Commands that sized the pool
//! themselves were one reordered statement away from silently ignoring the
//! flag — which is exactly what happened to `demux detect --method cnn`, where
//! a read-counting `par_iter()` ran first and pinned the pool at full width no
//! matter what `-j` said (#155). Sizing the pool before any command code runs
//! makes that unreachable; the `Err` arm in [`init`] is the tripwire in case it
//! ever becomes reachable again.

use std::sync::OnceLock;
use tracing::{debug, warn};

/// Default pool width when `-t/--threads` is not given.
///
/// Capped by `available_parallelism()`, which honours cgroup quota and CPU
/// affinity — so `srun -c 8` yields 8, not 16. The cap matters more than the
/// constant: these commands are largely I/O-bound and fan out across files, so
/// defaulting to every core monopolizes a shared node for little gain. An
/// explicit `-t/--threads` is **not** capped; a user who names a number owns
/// the machine.
pub const DEFAULT_THREADS: usize = 16;

/// Width the pool was actually built at, once [`init`] has run.
static POOL_WIDTH: OnceLock<usize> = OnceLock::new();

/// Resolve the requested thread count to a concrete pool width.
///
/// Precedence: explicit flag, then `RAYON_NUM_THREADS`, then
/// [`DEFAULT_THREADS`] capped at `available`. The env var stays in the chain
/// because `demux classify --help` documents it, and calling `build_global()`
/// with an explicit width would otherwise override it silently — the same
/// class of invisible no-op this module exists to prevent.
///
/// Pure by construction: `env` and `available` are parameters rather than
/// ambient state so the policy is testable without touching globals.
fn resolve(requested: Option<usize>, env: Option<&str>, available: usize) -> anyhow::Result<usize> {
    // rayon reads `num_threads(0)` as "pick for me", so accepting 0 would turn
    // `-t 0` into another silently-ignored flag. Reject it instead.
    if requested == Some(0) {
        anyhow::bail!("--threads must be at least 1");
    }
    Ok(match requested {
        Some(n) => n,
        None => env
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| DEFAULT_THREADS.min(available.max(1))),
    })
}

/// Build the global rayon pool. Call once, from `main`, before dispatch.
pub fn init(requested: Option<usize>) -> anyhow::Result<()> {
    let available = std::thread::available_parallelism().map_or(1, Into::into);
    let env = std::env::var("RAYON_NUM_THREADS").ok();
    let width = resolve(requested, env.as_deref(), available)?;

    if let Some(&prev) = POOL_WIDTH.get() {
        if prev != width {
            warn!("thread pool already built with {prev} thread(s); ignoring request for {width}");
        }
        return Ok(());
    }

    match rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build_global()
    {
        Ok(()) => {
            // Report the observed width, not the requested one: this line is
            // what the regression test asserts against.
            debug!("rayon pool: {} thread(s)", rayon::current_num_threads());
            let _ = POOL_WIDTH.set(width);
        }
        Err(e) => {
            // Reaching here means something created the global pool before
            // `main` got to it. Fail the test suite rather than whisper.
            debug_assert!(
                false,
                "rayon global pool was built before threads::init: {e}"
            );
            let actual = rayon::current_num_threads();
            warn!(
                "could not size the thread pool to {width}; it is already running \
                 {actual} thread(s), so --threads has no effect ({e})"
            );
            let _ = POOL_WIDTH.set(actual);
        }
    }
    Ok(())
}

/// Width the global pool was built at.
///
/// For callers that must size a *second* pool to match — notably the
/// onnxruntime intra-op pool on the `cnn-gpu` path, which would otherwise
/// spawn its own full-width pool alongside rayon's and put total process
/// threads back out of `--threads`' reach.
#[cfg_attr(not(feature = "cnn-gpu"), allow(dead_code))]
pub fn width() -> usize {
    POOL_WIDTH
        .get()
        .copied()
        .unwrap_or_else(rayon::current_num_threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_flag_wins_over_env_and_default() {
        assert_eq!(resolve(Some(3), Some("8"), 64).unwrap(), 3);
    }

    #[test]
    fn explicit_flag_is_not_capped_by_available() {
        // A user who names a number owns the machine; oversubscribing is their
        // call to make.
        assert_eq!(resolve(Some(64), None, 4).unwrap(), 64);
    }

    #[test]
    fn default_is_capped_at_available() {
        assert_eq!(resolve(None, None, 64).unwrap(), DEFAULT_THREADS);
        assert_eq!(resolve(None, None, 4).unwrap(), 4);
        assert_eq!(resolve(None, None, 0).unwrap(), 1);
    }

    #[test]
    fn env_applies_only_without_a_flag() {
        assert_eq!(resolve(None, Some("5"), 64).unwrap(), 5);
        assert_eq!(resolve(None, Some(" 5 "), 64).unwrap(), 5);
    }

    #[test]
    fn unusable_env_falls_back_to_default() {
        assert_eq!(resolve(None, Some("abc"), 64).unwrap(), DEFAULT_THREADS);
        assert_eq!(resolve(None, Some("0"), 64).unwrap(), DEFAULT_THREADS);
        assert_eq!(resolve(None, Some(""), 64).unwrap(), DEFAULT_THREADS);
    }

    #[test]
    fn zero_threads_is_rejected() {
        // Not a shrug: `num_threads(0)` means "rayon picks", which would make
        // `-t 0` yet another silently-ignored flag.
        assert!(resolve(Some(0), None, 64).is_err());
    }
}

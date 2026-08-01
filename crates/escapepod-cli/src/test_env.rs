//! Serialised environment mutation for tests.
//!
//! Two `cache_dir_precedence` tests (demux models, resquiggle models) set the
//! same kind of process-global variables to assert lookup precedence. Cargo runs
//! tests in parallel threads, and the environment is per-process, not per-thread
//! — so one test's `set_var` lands inside another's assertion. The observed
//! failure was `cache_dir()` returning a third test's `TempDir` path.
//!
//! The old per-module helpers carried `// SAFETY: single-threaded test bodies`,
//! which is not true of a parallel test runner. Both now go through this lock,
//! which makes it true.

use std::sync::{Mutex, MutexGuard, OnceLock};

fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    // A poisoned lock only means some other env test panicked mid-body; the
    // variables are restored below regardless, so the guard is still usable.
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Run `f` with `vars` applied, restoring the previous values afterwards.
///
/// Holds a process-wide lock for the duration, so concurrent tests cannot
/// observe or clobber each other's variables.
pub(crate) fn temp_env(vars: &[(&str, Option<&str>)], f: impl FnOnce()) {
    let _guard = env_lock();
    let saved: Vec<_> = vars
        .iter()
        .map(|(k, _)| (*k, std::env::var_os(k)))
        .collect();
    let apply = |k: &str, v: Option<&str>| match v {
        // SAFETY: the process-wide lock above is what makes this sound — no
        // other test can be reading or writing the environment concurrently.
        Some(v) => unsafe { std::env::set_var(k, v) },
        None => unsafe { std::env::remove_var(k) },
    };
    for (k, v) in vars {
        apply(k, *v);
    }
    f();
    for (k, v) in saved {
        apply(k, v.as_ref().and_then(|s| s.to_str()));
    }
}

//! One rule for `ESCAPEPOD_*` boolean switches and positive-integer knobs
//! (rnabioco/escapepod-rs#410).
//!
//! Before this, a boolean switch tested `var_os(..).is_some()` in five
//! places, so `VAR=0` — meant to turn it off, and what every other flag in
//! this codebase and its own `CLAUDE.md` documentation say — turned it *on*;
//! other switches already accepted `1|true`, or only `1`, three different
//! rules for the same shape of knob. A positive-integer knob repeated
//! `.ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0)` in eight more
//! places and silently fell back to its default on a garbled value, with no
//! way to notice a typo'd override was never applied.
//!
//! [`flag`] and [`positive_usize`] are the one parse for each shape, in the
//! lowest crate every other one already depends on. [`usize_allow_zero`] is
//! the same parse for the rarer knob where `0` is a meaningful setting
//! rather than a typo — `ESCAPEPOD_AUTOINDEX_MAX=0` disables speculative
//! index warm-up, and routing it through [`positive_usize`] silently turned
//! that off (rnabioco/escapepod-rs#421). A caller whose knob's
//! *default* is "on" (`ESCAPEPOD_CRF_GPU_DECODE`, `ESCAPEPOD_CRF_GPU_ZEROCOPY`
//! — `=0` opts out of an on-by-default behaviour) does not fit [`flag`]'s
//! fixed off-when-unset contract and is deliberately left alone; backend-name
//! caps (`*_BACKEND`) and path variables (`*_CACHE`) are a different shape
//! entirely and stay on their own parse too.

use std::collections::HashSet;
use std::sync::Mutex;

/// Every `ESCAPEPOD_*` name this process has already warned about, so a
/// knob read once per read (or per file) does not repeat the same warning
/// per call.
static WARNED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

fn warn_once(name: &str, message: std::fmt::Arguments<'_>) {
    let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    if warned
        .get_or_insert_with(HashSet::new)
        .insert(name.to_string())
    {
        tracing::warn!("{name}: {message}");
    }
}

/// A boolean `ESCAPEPOD_*` switch.
///
/// On for `1`, `true`, `yes` or `on` (case-insensitive); off when the
/// variable is unset, empty, or spells `0`, `false`, `no` or `off`
/// (case-insensitive). Anything else is not a value this ever meant to
/// carry — warns once per name and counts as off, the same as an unset
/// variable, so a typo cannot silently turn a switch on.
// This module is the one home `clippy.toml`'s `disallowed-methods` sends
// every other `ESCAPEPOD_*` switch/knob to; the raw `std::env::var` calls
// below are that parse, not a copy of it.
#[allow(clippy::disallowed_methods)]
pub fn flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "no" | "off" => false,
            "1" | "true" | "yes" | "on" => true,
            _ => {
                warn_once(
                    name,
                    format_args!(
                        "{v:?} is not 1/true/yes/on or 0/false/no/off; treating it as off"
                    ),
                );
                false
            }
        },
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            warn_once(
                name,
                format_args!("value is not valid Unicode; treating it as off"),
            );
            false
        }
    }
}

/// A positive-integer `ESCAPEPOD_*` knob.
///
/// `None` when the variable is unset or empty — the caller supplies its own
/// default via `.unwrap_or(default)`. A value that is set but does not parse
/// as a positive `usize` (not an integer, negative, or exactly `0` — every
/// knob this guards is a count or a size, and `0` is never a valid one)
/// warns once per name and also returns `None`, so a mistyped override
/// falls back to the default instead of silently doing so with no trace.
// Same home as `flag` above.
#[allow(clippy::disallowed_methods)]
pub fn positive_usize(name: &str) -> Option<usize> {
    let v = std::env::var(name).ok()?;
    let trimmed = v.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(n) if n > 0 => Some(n),
        Ok(_) => {
            warn_once(
                name,
                format_args!("{trimmed:?} must be a positive integer, not 0; ignoring it"),
            );
            None
        }
        Err(_) => {
            warn_once(
                name,
                format_args!("{trimmed:?} is not a positive integer; ignoring it"),
            );
            None
        }
    }
}

/// A non-negative-integer `ESCAPEPOD_*` knob: [`positive_usize`] with `0`
/// accepted as a value.
///
/// For the knob where `0` *means* something — a threshold where `0` is
/// "never", not a count or a size where `0` would be nonsense. Same contract
/// otherwise: `None` when unset or empty (the caller supplies its default),
/// and a value that does not parse as a `usize` (not an integer, negative)
/// warns once per name and returns `None`.
///
/// Reach for [`positive_usize`] unless `0` has a documented meaning; this one
/// exists because `ESCAPEPOD_AUTOINDEX_MAX=0` does (rnabioco/escapepod-rs#421).
// Same home as `flag` above.
#[allow(clippy::disallowed_methods)]
pub fn usize_allow_zero(name: &str) -> Option<usize> {
    let v = std::env::var(name).ok()?;
    let trimmed = v.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(n) => Some(n),
        Err(_) => {
            warn_once(
                name,
                format_args!("{trimmed:?} is not a non-negative integer; ignoring it"),
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// `std::env` is process-global, so tests that set a variable serialize
    /// on this lock rather than racing each other under nextest's threads.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// Runs `f` with `name` set to `value` (or removed, if `None`),
    /// restoring whatever the variable held before on the way out.
    // Test-only raw env access: saving/restoring the ambient value to test
    // `flag`/`positive_usize` themselves, not a second parse of a switch.
    #[allow(clippy::disallowed_methods)]
    fn with_var<T>(name: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var(name).ok();
        match value {
            Some(v) => unsafe { std::env::set_var(name, v) },
            None => unsafe { std::env::remove_var(name) },
        }
        let result = f();
        match previous {
            Some(v) => unsafe { std::env::set_var(name, v) },
            None => unsafe { std::env::remove_var(name) },
        }
        result
    }

    #[test]
    fn flag_every_truthy_spelling() {
        for v in ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON"] {
            assert!(
                with_var("ESCAPEPOD_TEST_FLAG_TRUE", Some(v), || flag(
                    "ESCAPEPOD_TEST_FLAG_TRUE"
                )),
                "{v:?} should be on"
            );
        }
    }

    #[test]
    fn flag_every_falsy_spelling() {
        for v in ["0", "false", "FALSE", "False", "no", "NO", "off", "OFF"] {
            assert!(
                !with_var("ESCAPEPOD_TEST_FLAG_FALSE", Some(v), || flag(
                    "ESCAPEPOD_TEST_FLAG_FALSE"
                )),
                "{v:?} should be off"
            );
        }
    }

    #[test]
    fn flag_unset_and_empty_are_off() {
        assert!(!with_var("ESCAPEPOD_TEST_FLAG_UNSET", None, || flag(
            "ESCAPEPOD_TEST_FLAG_UNSET"
        )));
        assert!(!with_var("ESCAPEPOD_TEST_FLAG_EMPTY", Some(""), || flag(
            "ESCAPEPOD_TEST_FLAG_EMPTY"
        )));
    }

    #[test]
    fn flag_garbage_warns_and_is_off() {
        // Not asserting on the warning itself (that's `tracing`'s business);
        // the contract this pins is that garbage is off, not a silent panic
        // or an accidental on.
        assert!(!with_var(
            "ESCAPEPOD_TEST_FLAG_GARBAGE",
            Some("maybe"),
            || flag("ESCAPEPOD_TEST_FLAG_GARBAGE")
        ));
    }

    #[test]
    fn flag_zero_is_off_not_on() {
        // The bug this whole module exists to fix: `var_os(..).is_some()`
        // treats `=0` as "set", hence on. `flag` must not repeat it.
        assert!(!with_var("ESCAPEPOD_TEST_FLAG_ZERO", Some("0"), || flag(
            "ESCAPEPOD_TEST_FLAG_ZERO"
        )));
    }

    #[test]
    fn positive_usize_valid_values() {
        assert_eq!(
            with_var("ESCAPEPOD_TEST_POS", Some("64"), || positive_usize(
                "ESCAPEPOD_TEST_POS"
            )),
            Some(64)
        );
        assert_eq!(
            with_var("ESCAPEPOD_TEST_POS", Some("  128  "), || positive_usize(
                "ESCAPEPOD_TEST_POS"
            )),
            Some(128)
        );
    }

    #[test]
    fn positive_usize_unset_and_empty_are_none() {
        assert_eq!(
            with_var("ESCAPEPOD_TEST_POS_UNSET", None, || positive_usize(
                "ESCAPEPOD_TEST_POS_UNSET"
            )),
            None
        );
        assert_eq!(
            with_var("ESCAPEPOD_TEST_POS_EMPTY", Some(""), || positive_usize(
                "ESCAPEPOD_TEST_POS_EMPTY"
            )),
            None
        );
    }

    #[test]
    fn positive_usize_zero_warns_and_is_none() {
        assert_eq!(
            with_var("ESCAPEPOD_TEST_POS_ZERO", Some("0"), || positive_usize(
                "ESCAPEPOD_TEST_POS_ZERO"
            )),
            None
        );
    }

    #[test]
    fn positive_usize_garbage_warns_and_is_none() {
        for v in ["nope", "-1", "1.5", "4G"] {
            assert_eq!(
                with_var("ESCAPEPOD_TEST_POS_GARBAGE", Some(v), || positive_usize(
                    "ESCAPEPOD_TEST_POS_GARBAGE"
                )),
                None,
                "{v:?} should not parse"
            );
        }
    }

    #[test]
    fn usize_allow_zero_unset_empty_zero_value_garbage() {
        const NAME: &str = "ESCAPEPOD_TEST_ALLOW_ZERO";
        let read = |value: Option<&str>| with_var(NAME, value, || usize_allow_zero(NAME));
        assert_eq!(read(None), None, "unset falls back to the caller's default");
        assert_eq!(
            read(Some("")),
            None,
            "empty falls back to the caller's default"
        );
        // The whole point: `0` is a value here, not a rejected typo.
        assert_eq!(read(Some("0")), Some(0));
        assert_eq!(read(Some("7")), Some(7));
        assert_eq!(read(Some(" 7 ")), Some(7));
        assert_eq!(read(Some("abc")), None, "garbage warns and falls back");
        assert_eq!(read(Some("-1")), None, "negative warns and falls back");
    }
}

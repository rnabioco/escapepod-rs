//! Centralized styling for CLI output.
//!
//! This module provides consistent color and style functions for all CLI
//! output. ANSI escapes are suppressed automatically when:
//!
//! - `NO_COLOR` is set (https://no-color.org/)
//! - `CLICOLOR=0` is set
//! - stderr is not a TTY (e.g. piped or redirected)
//!
//! `CLICOLOR_FORCE=1` overrides the TTY check and always emits ANSI.

use owo_colors::OwoColorize;
use std::fmt::Display;
use std::io::IsTerminal;
use std::sync::OnceLock;

static USE_COLOR: OnceLock<bool> = OnceLock::new();

fn use_color() -> bool {
    *USE_COLOR.get_or_init(|| {
        // Force-on wins over everything.
        if matches!(
            std::env::var("CLICOLOR_FORCE").as_deref(),
            Ok(v) if v != "0" && !v.is_empty()
        ) {
            return true;
        }
        // NO_COLOR: any value, including empty, disables color.
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        // CLICOLOR=0 disables.
        if matches!(std::env::var("CLICOLOR").as_deref(), Ok("0")) {
            return false;
        }
        // Status prints go to stderr; gate on stderr's terminal status.
        std::io::stderr().is_terminal()
    })
}

/// Style for action verbs like "Filtering", "Merging", "Scanning"
pub fn action<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.green().bold())
    } else {
        s.to_string()
    }
}

/// Style for file paths
pub fn path<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.cyan())
    } else {
        s.to_string()
    }
}

/// Style for important counts/numbers (matched reads, etc.)
pub fn count<T: Display>(n: T) -> String {
    if use_color() {
        format!("{}", n.green())
    } else {
        n.to_string()
    }
}

/// Style for percentages
pub fn percentage<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.cyan())
    } else {
        s.to_string()
    }
}

/// Style for labels like "Output:", "Filter:"
pub fn label<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.bold())
    } else {
        s.to_string()
    }
}

/// Style for section headers like "POD5 File Summary"
pub fn header<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.bold())
    } else {
        s.to_string()
    }
}

/// Style for key names in key-value pairs
pub fn key<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.blue())
    } else {
        s.to_string()
    }
}

/// Style for values in key-value pairs
pub fn value<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.cyan())
    } else {
        s.to_string()
    }
}

/// Style for a warning label (only the demux `split` summary still labels
/// inline; other warnings now flow through `tracing::warn!`).
#[cfg_attr(not(feature = "demux"), allow(dead_code))]
pub fn warning_label<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.yellow().bold())
    } else {
        s.to_string()
    }
}

/// Style for warning messages/values
pub fn warning<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.yellow())
    } else {
        s.to_string()
    }
}

/// Style for error messages/values
pub fn error<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.red())
    } else {
        s.to_string()
    }
}

/// Style for note prefix "Note:"
pub fn note_label<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.yellow())
    } else {
        s.to_string()
    }
}

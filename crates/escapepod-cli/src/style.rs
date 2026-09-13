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
//!
//! The functions at the top level gate on **stderr**'s terminal status — right
//! for status/progress lines emitted via `tracing`. Commands whose colored
//! output IS the data product (`summary`, `inspect`) print to **stdout**
//! instead, and must use the [`stdout`] submodule so redirecting stdout
//! (`> out.txt`, `| less`) suppresses ANSI independently of whatever stderr
//! happens to be attached to.

use owo_colors::OwoColorize;
use std::fmt::Display;
use std::io::IsTerminal;
use std::sync::OnceLock;

/// `CLICOLOR_FORCE`/`NO_COLOR`/`CLICOLOR=0` precedence shared by both the
/// stderr gate below and the stdout gate in [`stdout`]. Returns `Some` when an
/// env var settles the question outright; `None` leaves it to the caller's
/// own terminal check.
fn color_env_override() -> Option<bool> {
    // Force-on wins over everything.
    if matches!(
        std::env::var("CLICOLOR_FORCE").as_deref(),
        Ok(v) if v != "0" && !v.is_empty()
    ) {
        return Some(true);
    }
    // NO_COLOR: any value, including empty, disables color.
    if std::env::var_os("NO_COLOR").is_some() {
        return Some(false);
    }
    // CLICOLOR=0 disables.
    if matches!(std::env::var("CLICOLOR").as_deref(), Ok("0")) {
        return Some(false);
    }
    None
}

static USE_COLOR: OnceLock<bool> = OnceLock::new();

fn use_color() -> bool {
    *USE_COLOR.get_or_init(|| {
        // Status prints go to stderr; gate on stderr's terminal status.
        color_env_override().unwrap_or_else(|| std::io::stderr().is_terminal())
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

/// Strip ANSI SGR escape sequences (`\x1b[...m`) from `s`.
///
/// For output built by unconditionally chaining `owo_colors` methods (as
/// `summary`'s table is, since widths are padded before coloring) rather than
/// going through this module's gated functions: render once with color, then
/// strip after the fact when [`stdout::use_color`] says not to, instead of
/// threading a color flag through every cell.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            let mut peek = chars.clone();
            if peek.next() == Some('[') {
                chars.next();
                for c2 in chars.by_ref() {
                    if ('@'..='~').contains(&c2) {
                        break;
                    }
                }
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// Same styling as the top level, gated on **stdout**'s terminal status
/// instead of stderr's. See the module-level doc comment for when to use this.
pub mod stdout {
    use owo_colors::OwoColorize;
    use std::fmt::Display;
    use std::io::IsTerminal;
    use std::sync::OnceLock;

    static USE_COLOR: OnceLock<bool> = OnceLock::new();

    /// Whether stdout-directed output should be colored.
    pub fn use_color() -> bool {
        *USE_COLOR.get_or_init(|| {
            super::color_env_override().unwrap_or_else(|| std::io::stdout().is_terminal())
        })
    }

    pub fn path<T: Display>(s: T) -> String {
        if use_color() {
            format!("{}", s.cyan())
        } else {
            s.to_string()
        }
    }

    pub fn count<T: Display>(n: T) -> String {
        if use_color() {
            format!("{}", n.green())
        } else {
            n.to_string()
        }
    }

    pub fn header<T: Display>(s: T) -> String {
        if use_color() {
            format!("{}", s.bold())
        } else {
            s.to_string()
        }
    }

    pub fn key<T: Display>(s: T) -> String {
        if use_color() {
            format!("{}", s.blue())
        } else {
            s.to_string()
        }
    }

    pub fn value<T: Display>(s: T) -> String {
        if use_color() {
            format!("{}", s.cyan())
        } else {
            s.to_string()
        }
    }

    pub fn warning<T: Display>(s: T) -> String {
        if use_color() {
            format!("{}", s.yellow())
        } else {
            s.to_string()
        }
    }
}

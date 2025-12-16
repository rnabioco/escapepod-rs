//! Centralized styling for CLI output.
//!
//! This module provides consistent color and style functions
//! for all CLI command output.

use owo_colors::OwoColorize;
use std::fmt::Display;

/// Style for action verbs like "Filtering", "Merging", "Scanning"
pub fn action<T: Display>(s: T) -> String {
    format!("{}", s.green().bold())
}

/// Style for file paths
pub fn path<T: Display>(s: T) -> String {
    format!("{}", s.cyan())
}

/// Style for important counts/numbers (matched reads, etc.)
pub fn count<T: Display>(n: T) -> String {
    format!("{}", n.green())
}

/// Style for percentages
pub fn percentage<T: Display>(s: T) -> String {
    format!("{}", s.cyan())
}

/// Style for labels like "Output:", "Filter:"
pub fn label<T: Display>(s: T) -> String {
    format!("{}", s.bold())
}

/// Style for section headers like "POD5 File Summary"
pub fn header<T: Display>(s: T) -> String {
    format!("{}", s.bold())
}

/// Style for key names in key-value pairs
pub fn key<T: Display>(s: T) -> String {
    format!("{}", s.blue())
}

/// Style for values in key-value pairs
pub fn value<T: Display>(s: T) -> String {
    format!("{}", s.cyan())
}

/// Style for warning prefix "Warning:"
pub fn warning_label<T: Display>(s: T) -> String {
    format!("{}", s.yellow().bold())
}

/// Style for warning messages/values
pub fn warning<T: Display>(s: T) -> String {
    format!("{}", s.yellow())
}

/// Style for error prefix "Error:"
pub fn error_label<T: Display>(s: T) -> String {
    format!("{}", s.red().bold())
}

/// Style for error messages/values
pub fn error<T: Display>(s: T) -> String {
    format!("{}", s.red())
}

/// Style for note prefix "Note:"
pub fn note_label<T: Display>(s: T) -> String {
    format!("{}", s.yellow())
}

/// Style for informational messages
pub fn info<T: Display>(s: T) -> String {
    format!("{}", s.blue())
}

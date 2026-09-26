// SPDX-License-Identifier: MIT

use std::fmt;

/// Everything this crate can refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignError {
    /// A `--scoring` string that is not four comma-separated integers.
    ScoringSyntax(String),
    /// Scores that are well-formed but not a scheme this aligner can run.
    ScoringInvalid(String),
    /// A reference panel with nothing to align against.
    EmptyPanel,
    /// A reference with no bases.
    EmptyReference(String),
    /// A requested kernel this machine (or this build) cannot run.
    BackendUnavailable(&'static str),
    /// [`crate::fasta::read_fasta`] could not read another line.
    FastaIo(String),
    /// A FASTA header (`>` line) with nothing but whitespace after it.
    EmptyReferenceName(usize),
    /// A FASTA header name that is not valid UTF-8.
    NonUtf8ReferenceName(usize),
    /// The same reference name given twice.
    DuplicateReferenceName(String),
    /// Sequence data before any `>` header.
    FastaContentBeforeHeader(usize),
}

impl fmt::Display for AlignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScoringSyntax(s) => write!(
                f,
                "scoring {s:?} is not MATCH,MISMATCH,GAP_OPEN,GAP_EXTEND (four integers, e.g. 2,-1,-10,-1)"
            ),
            Self::ScoringInvalid(why) => write!(f, "invalid scoring: {why}"),
            Self::EmptyPanel => f.write_str("the reference panel is empty"),
            Self::EmptyReference(name) => write!(f, "reference {name:?} has no bases"),
            Self::BackendUnavailable(name) => {
                write!(f, "the {name} kernel cannot run on this machine")
            }
            Self::FastaIo(e) => write!(f, "reading FASTA: {e}"),
            Self::EmptyReferenceName(line) => {
                write!(f, "FASTA line {line}: empty reference name")
            }
            Self::NonUtf8ReferenceName(line) => {
                write!(f, "FASTA line {line}: non-UTF-8 reference name")
            }
            Self::DuplicateReferenceName(name) => {
                write!(f, "duplicate reference name {name:?}")
            }
            Self::FastaContentBeforeHeader(line) => {
                write!(
                    f,
                    "FASTA line {line}: sequence data before the first '>' header"
                )
            }
        }
    }
}

impl std::error::Error for AlignError {}

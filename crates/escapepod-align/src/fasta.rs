// SPDX-License-Identifier: MIT

//! The one reference FASTA reader (rnabioco/escapepod-rs#410).
//!
//! Before this, the reference FASTA was parsed in six places — this crate's
//! own `escpod align` caller, `escapepod-classify`'s `geometry::read_fasta`,
//! and four tests/examples — each a slightly different ad hoc loop, and two
//! of them (`escpod align` and `escpod classify`) already disagreed on the
//! same file: `align` read a gzipped reference and refused a duplicate name,
//! `classify` failed on gzip and let a duplicate silently win. [`read_fasta`]
//! is the rule every caller now goes through instead of writing its own.
//!
//! Deliberately **not** the whole story: this function takes an
//! already-open [`BufRead`], so gzip detection and file opening stay out of
//! this std-only, `cargo publish`-able crate. A caller that wants to accept
//! a gzipped reference wraps `flate2::read::MultiGzDecoder` itself first
//! (`escapepod-cli`'s `align.rs::open_text` is the canonical shape).

use std::collections::HashSet;
use std::io::BufRead;

use crate::AlignError;

/// Parse a FASTA already opened as [`BufRead`], returning `(name, sequence)`
/// pairs in file order.
///
/// - A record's name is the first whitespace-delimited word after the `>` on
///   its header line; anything after that word (a description) is dropped.
/// - A sequence may span multiple lines; they are concatenated with every
///   ASCII-whitespace byte on a sequence line stripped, so a CRLF file, a
///   blank line between records, or an indented sequence line all read the
///   same as a tidy one.
/// - A header with an empty name, or a name repeating an earlier record's,
///   is a hard error naming the line or the name — silently keeping the
///   last of two same-named records (as one of this parser's six
///   predecessors did) is exactly the failure this exists to remove.
/// - Content before the first `>` header is an error unless the file has
///   nothing but blank lines before it (leading blank lines are tolerated).
///
/// Case is preserved exactly as written; nothing here upper-cases a sequence
/// or a name — a caller that wants that (as `escapepod-classify`'s geometry
/// lookup does) applies it to this function's output.
pub fn read_fasta<R: BufRead>(mut reader: R) -> Result<Vec<(String, Vec<u8>)>, AlignError> {
    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut current: Option<(String, Vec<u8>)> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut line_no: usize = 0;

    loop {
        buf.clear();
        let n = reader
            .read_until(b'\n', &mut buf)
            .map_err(|e| AlignError::FastaIo(e.to_string()))?;
        if n == 0 {
            break;
        }
        line_no += 1;
        while matches!(buf.last(), Some(b'\n' | b'\r')) {
            buf.pop();
        }

        if let Some(header) = buf.strip_prefix(b">") {
            if let Some(record) = current.take() {
                records.push(record);
            }
            let name_bytes = first_word(header);
            if name_bytes.is_empty() {
                return Err(AlignError::EmptyReferenceName(line_no));
            }
            let name = String::from_utf8(name_bytes.to_vec())
                .map_err(|_| AlignError::NonUtf8ReferenceName(line_no))?;
            if !seen.insert(name.clone()) {
                return Err(AlignError::DuplicateReferenceName(name));
            }
            current = Some((name, Vec::new()));
        } else {
            match current.as_mut() {
                Some((_, seq)) => seq.extend(buf.iter().filter(|b| !b.is_ascii_whitespace())),
                None if buf.iter().all(|b| b.is_ascii_whitespace()) => {}
                None => return Err(AlignError::FastaContentBeforeHeader(line_no)),
            }
        }
    }
    if let Some(record) = current.take() {
        records.push(record);
    }
    Ok(records)
}

/// The first run of non-whitespace bytes, or an empty slice if there is none
/// — `split_whitespace`'s rule, applied to a byte slice.
fn first_word(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|b| !b.is_ascii_whitespace());
    let Some(start) = start else { return &[] };
    let rest = &s[start..];
    let end = rest
        .iter()
        .position(|b| b.is_ascii_whitespace())
        .unwrap_or(rest.len());
    &rest[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<Vec<(String, Vec<u8>)>, AlignError> {
        read_fasta(s.as_bytes())
    }

    #[test]
    fn single_record() {
        let got = parse(">a\nACGT\n").unwrap();
        assert_eq!(got, vec![("a".to_string(), b"ACGT".to_vec())]);
    }

    #[test]
    fn multi_line_sequence_is_joined() {
        let got = parse(">a\nACGT\nACGT\nAC\n").unwrap();
        assert_eq!(got, vec![("a".to_string(), b"ACGTACGTAC".to_vec())]);
    }

    #[test]
    fn header_description_is_dropped() {
        let got = parse(">a description here\nACGT\n").unwrap();
        assert_eq!(got[0].0, "a");
    }

    #[test]
    fn leading_whitespace_before_the_name_is_skipped() {
        let got = parse(">   a desc\nACGT\n").unwrap();
        assert_eq!(got[0].0, "a");
    }

    #[test]
    fn multiple_records_in_file_order() {
        let got = parse(">a\nAC\n>b\nGT\n>c\nTT\n").unwrap();
        assert_eq!(
            got,
            vec![
                ("a".to_string(), b"AC".to_vec()),
                ("b".to_string(), b"GT".to_vec()),
                ("c".to_string(), b"TT".to_vec()),
            ]
        );
    }

    #[test]
    fn crlf_line_endings() {
        let got = parse(">a\r\nAC\r\nGT\r\n").unwrap();
        assert_eq!(got, vec![("a".to_string(), b"ACGT".to_vec())]);
    }

    #[test]
    fn blank_lines_between_and_within_records_are_ignored() {
        let got = parse("\n\n>a\nAC\n\nGT\n\n>b\nTT\n").unwrap();
        assert_eq!(
            got,
            vec![
                ("a".to_string(), b"ACGT".to_vec()),
                ("b".to_string(), b"TT".to_vec()),
            ]
        );
    }

    #[test]
    fn indented_sequence_line_has_whitespace_stripped() {
        let got = parse(">a\n  AC GT \t\n").unwrap();
        assert_eq!(got, vec![("a".to_string(), b"ACGT".to_vec())]);
    }

    #[test]
    fn a_record_with_no_sequence_lines_is_empty_not_missing() {
        let got = parse(">a\n>b\nGT\n").unwrap();
        assert_eq!(
            got,
            vec![
                ("a".to_string(), Vec::new()),
                ("b".to_string(), b"GT".to_vec()),
            ]
        );
    }

    #[test]
    fn duplicate_name_is_refused() {
        let err = parse(">a\nAC\n>a\nGT\n").unwrap_err();
        assert_eq!(err, AlignError::DuplicateReferenceName("a".to_string()));
    }

    #[test]
    fn empty_name_is_refused() {
        let err = parse(">\nACGT\n").unwrap_err();
        assert_eq!(err, AlignError::EmptyReferenceName(1));
        let err = parse(">   \nACGT\n").unwrap_err();
        assert_eq!(err, AlignError::EmptyReferenceName(1));
    }

    #[test]
    fn content_before_the_first_header_is_refused() {
        let err = parse("ACGT\n>a\nACGT\n").unwrap_err();
        assert_eq!(err, AlignError::FastaContentBeforeHeader(1));
    }

    #[test]
    fn leading_blank_lines_before_the_first_header_are_tolerated() {
        let got = parse("\n\n>a\nACGT\n").unwrap();
        assert_eq!(got, vec![("a".to_string(), b"ACGT".to_vec())]);
    }

    #[test]
    fn empty_input_is_an_empty_list() {
        assert_eq!(parse("").unwrap(), Vec::new());
    }

    #[test]
    fn no_trailing_newline_on_the_last_line_still_reads() {
        let got = parse(">a\nACGT").unwrap();
        assert_eq!(got, vec![("a".to_string(), b"ACGT".to_vec())]);
    }
}

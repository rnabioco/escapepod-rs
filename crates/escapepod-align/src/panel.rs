// SPDX-License-Identifier: MIT

//! The reference panel: names, letters, and codes, in file order.

use crate::AlignError;
use crate::alphabet;

/// One reference sequence.
#[derive(Debug, Clone)]
pub struct Reference {
    /// Name as it will appear in `@SQ SN:` (the FASTA header's first word).
    pub name: String,
    /// The letters as given — kept for `MD`, which writes reference letters
    /// verbatim (upper-cased), ambiguity codes included.
    pub seq: Vec<u8>,
    /// [`alphabet`] codes of `seq`, what the DP scores.
    pub codes: Vec<u8>,
}

/// The references every read is scored against, in the order given.
///
/// Order is meaningful: it is the `@SQ` order of the output, and ties are
/// broken toward the lowest index.
#[derive(Debug, Clone)]
pub struct Panel {
    refs: Vec<Reference>,
    max_len: usize,
}

impl Panel {
    /// Build from `(name, letters)` pairs. Refuses an empty panel or an empty
    /// reference; names are not checked for uniqueness here (the SAM header
    /// writer does that).
    pub fn new<I, S>(refs: I) -> Result<Self, AlignError>
    where
        I: IntoIterator<Item = (S, Vec<u8>)>,
        S: Into<String>,
    {
        let refs: Vec<Reference> = refs
            .into_iter()
            .map(|(name, seq)| {
                let name = name.into();
                if seq.is_empty() {
                    return Err(AlignError::EmptyReference(name));
                }
                let codes = alphabet::encode(&seq);
                Ok(Reference { name, seq, codes })
            })
            .collect::<Result<_, _>>()?;
        if refs.is_empty() {
            return Err(AlignError::EmptyPanel);
        }
        let max_len = refs.iter().map(|r| r.seq.len()).max().unwrap_or(0);
        Ok(Self { refs, max_len })
    }

    pub fn len(&self) -> usize {
        self.refs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    pub fn get(&self, i: usize) -> &Reference {
        &self.refs[i]
    }

    pub fn references(&self) -> &[Reference] {
        &self.refs
    }

    /// Length of the longest reference.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Total reference bases, for cell-count reporting.
    pub fn total_len(&self) -> usize {
        self.refs.iter().map(|r| r.seq.len()).sum()
    }
}

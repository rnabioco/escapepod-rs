// SPDX-License-Identifier: MIT

//! The scoring scheme and the two alignment modes.

use std::fmt;
use std::str::FromStr;

use crate::AlignError;
use crate::alphabet::WILDCARD;

/// Largest magnitude any single score may have. Keeps every cell of a panel
/// of ≤ ~1 kb references inside the SIMD kernels' i16 lanes by a wide margin
/// (see [`crate::simd`] for the bound that is actually checked per read).
pub const MAX_SCORE_MAGNITUDE: i32 = 1000;

/// Affine-gap scores, all signed.
///
/// **Convention**, stated once: a gap of length *k* contributes
/// `gap_open + (k − 1) · gap_extend`. `gap_open` is therefore the whole cost
/// of a one-base gap, not an extra charge on top of the first extension. This
/// is parasail's (and EMBOSS's) convention with the signs made explicit, and
/// it is pinned against parasail by `tests/parasail_golden.rs`.
///
/// bwa states its penalties the other way round — a gap of length *k* costs
/// `O + k·E` — so bwa's `-O 1 -E 1` is `gap_open = -2, gap_extend = -1` here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scoring {
    /// Score of a match, and of any pairing with a wildcard. Positive.
    pub match_score: i32,
    /// Score of a mismatch. Below `match_score`.
    pub mismatch: i32,
    /// Score of a one-base gap. Zero or negative.
    pub gap_open: i32,
    /// Score of each further base of the same gap. Zero or negative.
    pub gap_extend: i32,
}

impl Default for Scoring {
    /// `2,-1,-10,-1`: gpu-tRNA-mapper's default, so a user moving from it sees
    /// the same behaviour. Not a recommendation for any particular pipeline.
    fn default() -> Self {
        Self {
            match_score: 2,
            mismatch: -1,
            gap_open: -10,
            gap_extend: -1,
        }
    }
}

impl Scoring {
    /// Build and validate.
    pub fn new(
        match_score: i32,
        mismatch: i32,
        gap_open: i32,
        gap_extend: i32,
    ) -> Result<Self, AlignError> {
        let s = Self {
            match_score,
            mismatch,
            gap_open,
            gap_extend,
        };
        s.validate()?;
        Ok(s)
    }

    /// Refuse a scheme the DP cannot give a meaningful answer for.
    pub fn validate(&self) -> Result<(), AlignError> {
        let bad = |why: String| Err(AlignError::ScoringInvalid(why));
        if self.match_score <= 0 {
            return bad(format!("match must be positive, got {}", self.match_score));
        }
        if self.mismatch >= self.match_score {
            return bad(format!(
                "mismatch ({}) must score below match ({})",
                self.mismatch, self.match_score
            ));
        }
        if self.gap_open > 0 || self.gap_extend > 0 {
            return bad(format!(
                "gap scores are penalties and must be zero or negative, got open {} extend {}",
                self.gap_open, self.gap_extend
            ));
        }
        for v in [
            self.match_score,
            self.mismatch,
            self.gap_open,
            self.gap_extend,
        ] {
            if v.abs() > MAX_SCORE_MAGNITUDE {
                return bad(format!(
                    "|{v}| exceeds the supported magnitude {MAX_SCORE_MAGNITUDE}"
                ));
            }
        }
        Ok(())
    }

    /// Score of pairing two codes: a match if they are equal or either is a
    /// wildcard, a mismatch otherwise.
    #[inline]
    pub fn substitution(&self, a: u8, b: u8) -> i32 {
        if a == b || a == WILDCARD || b == WILDCARD {
            self.match_score
        } else {
            self.mismatch
        }
    }

    /// Score of a gap of `len` bases (`len >= 1`).
    #[inline]
    pub fn gap(&self, len: u32) -> i32 {
        self.gap_open + (len as i32 - 1) * self.gap_extend
    }
}

impl FromStr for Scoring {
    type Err = AlignError;

    /// `MATCH,MISMATCH,GAP_OPEN,GAP_EXTEND`, e.g. `2,-1,-10,-1`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(AlignError::ScoringSyntax(s.to_string()));
        }
        let mut v = [0i32; 4];
        for (slot, p) in v.iter_mut().zip(&parts) {
            *slot = p
                .parse()
                .map_err(|_| AlignError::ScoringSyntax(s.to_string()))?;
        }
        Self::new(v[0], v[1], v[2], v[3])
    }
}

impl fmt::Display for Scoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{},{},{},{}",
            self.match_score, self.mismatch, self.gap_open, self.gap_extend
        )
    }
}

/// Which cells an alignment may start and end in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Smith–Waterman: the alignment may start and end anywhere, and every
    /// cell is floored at zero. Both overhangs of the read are soft-clipped.
    #[default]
    Local,
    /// Overlap alignment: leading and trailing gaps of *either* sequence are
    /// free. The path starts on the top or left border of the matrix and ends
    /// on the bottom or right border; interior gaps are scored normally, and
    /// no cell is floored. The read's unaligned overhangs are soft-clipped and
    /// the reference's are simply not covered.
    ///
    /// This is our definition. It is what parasail calls `sg` (and pinned
    /// against it), not a reproduction of any other tool's semi-global mode.
    SemiGlobal,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::SemiGlobal => "semiglobal",
        }
    }
}

impl FromStr for Mode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "local" => Ok(Self::Local),
            "semiglobal" | "semi-global" | "overlap" => Ok(Self::SemiGlobal),
            other => Err(format!("unknown mode {other:?}, expected local|semiglobal")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_string() {
        let s: Scoring = "2,-1,-10,-1".parse().unwrap();
        assert_eq!(s, Scoring::default());
        assert_eq!(s.to_string(), "2,-1,-10,-1");
    }

    #[test]
    fn gap_convention() {
        let s = Scoring::default();
        assert_eq!(s.gap(1), -10);
        assert_eq!(s.gap(3), -12);
    }

    #[test]
    fn refuses_nonsense() {
        assert!("2,-1,-10".parse::<Scoring>().is_err());
        assert!("2,-1,10,-1".parse::<Scoring>().is_err());
        assert!("0,-1,-10,-1".parse::<Scoring>().is_err());
        assert!("2,3,-10,-1".parse::<Scoring>().is_err());
        assert!("2,-1,-10,-x".parse::<Scoring>().is_err());
    }
}

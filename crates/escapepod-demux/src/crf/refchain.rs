//! Constrained partition functions: `log P(reference | signal)` under the CRF.
//!
//! [`super::lattice`] answers "what did the model emit". This answers "how much
//! of the model's probability mass emits *this* string" — for every barcode
//! reference at once, off the same encoder scores, in the same decode.
//!
//! # Why the decode's own outputs are not enough
//!
//! Demultiplexing calls a barcode by decoding a sequence and matching it to the
//! nearest reference by edit distance, with the margin to the runner-up as the
//! confidence. On a designed panel that margin measures **reference
//! separation, not decode confidence**: the references are ≥12 edits apart by
//! construction, so a *wrong* decode still lands 12–14 from the runner-up. On a
//! production 16-plex flowcell 99% of reads take one of three margin values and
//! 90% of the reads two independently trained models disagree about are exact
//! matches to a reference — the decode does not stumble toward a wrong answer,
//! it cleanly emits a different valid codeword. Neither the margin nor the
//! distance to the best reference can buy precision at any price (#241).
//!
//! What is missing is the lattice's own opinion, and the lattice has one.
//!
//! # The quantity
//!
//! Every path through the CRF emits exactly one string, so the paths partition
//! by emission and `Σ_strings P(string | signal) = 1`. Restricting the forward
//! recursion to the paths that emit a given reference therefore gives a genuine
//! probability:
//!
//! ```text
//! log P(ref | signal) = logZ_target(ref) - logZ_full
//! ```
//!
//! This is the CRF's training objective evaluated per read per reference —
//! `escapepod_models.crf.loss.CtcCrfLoss` computes the same two terms, and its
//! `logZ_target(normalised) == logZ_target(raw) - logZ_full` identity is why
//! neither side has to materialise a normalised score tensor.
//!
//! Unlike the edit-distance margin this is continuous, so it supports the
//! precision/recall trade the margin cannot: it separates *emitted BC07
//! exactly, p=0.93* from *emitted BC07 exactly, p=0.31 with BC05 at p=0.28*,
//! which is the population the disagreements live in.
//!
//! # The chain, and the prefix nobody knows
//!
//! Restricted to one target the lattice collapses to a chain: state `j` is the
//! `state_len`-mer `target[j..j + state_len]`, and each timestep either stays or
//! advances one position. The scores are gathered exactly as
//! `CtcCrfLoss.gather_target_scores` gathers them — stay from edge 0 into that
//! state, move from edge `1 + target[j - 1]` (the base dropped off the front)
//! into the next.
//!
//! The wrinkle is that a bundle ships `barcodes[].sequence` as the sequence the
//! model **emits**, which is `target[state_len..]`: the first `state_len` bases
//! only fix the initial state and are never emitted, so they are not in the
//! bundle and are not recoverable from it. Pinning the chain at some invented
//! prefix would score a different event than the one asked about.
//!
//! So the head is *marginalised* rather than pinned, which is also the honest
//! inference-time question — the full lattice starts free (`alpha_0 = 0` for
//! every state), and what we want is the probability of an emission, not of an
//! emission-and-a-particular-history. Chain position `j < state_len` therefore
//! carries `n_base ** (state_len - j)` substates, one per still-unresolved
//! prefix: 256, 64, 16, 4, then 1 from `state_len` on. A substate at position
//! `j` knows the base the next move drops (its own oldest base), so the head
//! needs no extra bookkeeping — only `n_base` incoming moves per cell instead
//! of one.
//!
//! Those head layers depend only on the reference's first `j` emitted bases, so
//! references sharing a prefix share them: the cells are built through a trie
//! and position 0 (256 substates, and reference-independent) is built once for
//! the whole panel. For a designed panel with a constant leader this collapses
//! the entire head to one copy.
//!
//! # Cost, and where the next factor is
//!
//! For the shipped 16-plex (44-nt references, `t_len = 300`) the panel is 961
//! cells: 256 that can only stay (pure adds, no transcendental), 85 head cells
//! with fan-in `n_base`, and 640 tail cells with fan-in 1. That is ~218k
//! `exp`/`ln` pairs per read against the decode's ~1.15M `exp`.
//!
//! **Measured** on `demux basecall` over 20k RNA004 reads (rna, 48 cores,
//! interleaved arms): 25.34 s without, 31.69 s with — **+25%** on the whole
//! command, not on the decode alone.
//!
//! That is more than the operation count suggests, and the reason is not loop
//! overhead: specialising the two common fan-ins into their own loops moved it
//! only from +25.7% to +25.1%. The scan is **transcendental-bound like the
//! decode is**, but it calls scalar `exp`/`ln_1p` while the decode's own
//! `logsumexp` runs through the AVX2/AVX-512 Cephes kernels in
//! [`super::avx2`]/[`super::avx512`] at a fraction of the per-element cost. So
//! the remaining factor is a vector kernel for this scan, not a tighter loop —
//! the fan-in-1 cells are 65% of the lattice and independent within a timestep,
//! wanting four gathers, a vector `exp` and a vector `ln1p`. Until that exists
//! the scoring is opt-in (`demux basecall --ref-scores`).
//!
//! The scan runs on the *raw* transposed scores, which
//! [`super::lattice::decode_with_refs`] hands over before pass 1 overwrites
//! them with log-posteriors. That ordering is load-bearing and is why this is
//! folded into the decode rather than bolted on after it.
//!
//! # What it buys
//!
//! On those same 20k reads, against the edit-distance confidence it exists to
//! replace: `confidence` takes **15** distinct values with 98.7% of reads in
//! just three of them, while `crf_margin` takes **14,818** over 16,747 reads.
//! 98.4% of reads match a reference exactly — so edit distance has nothing left
//! to say about them — and within that group `P(barcode | signal)` still ranges
//! from below 0.1 (26 reads) through 0.1–0.5 (828, 5.0%) to 0.9–0.99 (91.8%).
//! Those are reads a clean decode cannot distinguish and the lattice can.

use std::collections::HashMap;
use std::fmt;

use super::lattice::CrfLayout;

/// One read's decode plus what the lattice thought of each reference.
///
/// The three fields answer three different questions and none substitutes for
/// another: what the model emitted, how much probability mass each reference
/// carries, and how peaked the lattice was around its own answer.
#[derive(Debug, Clone)]
pub struct ScoredDecode {
    /// The decoded sequence, as [`super::lattice::decode`] returns it.
    pub sequence: String,
    /// `log P(reference | signal)` per reference, in panel order.
    pub ref_logp: Vec<f32>,
    /// The decoded path's mean per-timestep log-posterior — `<= 0`, and closer
    /// to 0 the more the lattice concentrated on the path it reported.
    pub mean_logpost: f32,
}

impl ScoredDecode {
    /// The best-scoring reference: `(index, log P, margin in nats to the
    /// runner-up)`.
    ///
    /// `None` for an empty panel; the margin is `None` with a single reference,
    /// which is the convention [`super::barcode::BarcodeMatch`] already uses
    /// for the edit-distance margin — with no runner-up there is nothing to
    /// compare against, which is not the same as comparing and tying.
    pub fn best(&self) -> Option<(usize, f32, Option<f32>)> {
        let mut best = (usize::MAX, f32::NEG_INFINITY);
        let mut second = f32::NEG_INFINITY;
        for (i, &v) in self.ref_logp.iter().enumerate() {
            if v > best.1 {
                second = best.1;
                best = (i, v);
            } else if v > second {
                second = v;
            }
        }
        (best.0 != usize::MAX).then(|| {
            (
                best.0,
                best.1,
                second.is_finite().then_some(best.1 - second),
            )
        })
    }
}

/// Errors from building a reference chain set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefChainError {
    /// A reference contains a symbol outside `alphabet[1..]`.
    BadSymbol { index: usize, symbol: u8 },
    /// A reference is shorter than `state_len`, so its chain would end inside
    /// the unresolved-prefix head and no single cell would hold its answer.
    TooShort {
        index: usize,
        len: usize,
        state_len: usize,
    },
    /// The alphabet doesn't have `n_base + 1` symbols.
    AlphabetLen { got: usize, expected: usize },
    /// No references were supplied.
    Empty,
}

impl fmt::Display for RefChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSymbol { index, symbol } => write!(
                f,
                "reference {index} contains {:?}, which is not in the model's alphabet",
                *symbol as char
            ),
            Self::TooShort {
                index,
                len,
                state_len,
            } => write!(
                f,
                "reference {index} is {len} nt, shorter than the model's state length {state_len}"
            ),
            Self::AlphabetLen { got, expected } => {
                write!(f, "alphabet has {got} symbols, expected {expected}")
            }
            Self::Empty => write!(f, "no references"),
        }
    }
}

impl std::error::Error for RefChainError {}

/// The constrained lattices for a whole reference panel, sharing every cell the
/// references' common prefixes let them share.
///
/// Built once per run — the structure depends only on the panel and the model
/// geometry, never on a read — and then walked per read by
/// [`super::lattice::decode_with_refs`].
#[derive(Debug, Clone)]
pub struct RefChains {
    /// Cells `0..n_start` are chain position 0: every state is a legal start,
    /// so they begin at 0 while everything else begins at `-inf`.
    n_start: usize,
    n_cells: usize,
    /// Per cell, the index of its stay edge within one timestep's transposed
    /// score row (`0 * n_states + state`).
    stay: Vec<u32>,
    /// CSR offsets into [`Self::move_src`] / [`Self::move_score`].
    move_off: Vec<u32>,
    /// Source cell of each incoming move.
    move_src: Vec<u32>,
    /// Score index of each incoming move (`(1 + dropped) * n_states + dest`).
    move_score: Vec<u32>,
    /// Cell holding the end of each reference's chain, in input order.
    finals: Vec<u32>,
    /// Widest fan-in over all cells, so the scan can size its accumulator once.
    max_fan: usize,
    /// Cells with exactly one incoming move — the tail positions, and ~65% of
    /// the lattice at the RNA004 geometry.
    ///
    /// The two common fan-ins get their own loops so the general CSR path —
    /// with its accumulator round-trip through memory and its dynamic-length
    /// call — runs only over the head. Worth 0.6 points of the 25% the scoring
    /// costs, and no more than that: the scan is transcendental-bound, so this
    /// is the shape a vector kernel wants rather than the win itself.
    ///
    /// Index lists rather than a reordering of the cells, so a cell's index
    /// still means what it meant at build time.
    tail: Vec<u32>,
    /// Cells with a full `n_base` incoming moves — the unresolved-prefix head.
    head: Vec<u32>,
}

impl RefChains {
    /// Build the chains for `seqs` — the sequences the model **emits**, i.e.
    /// what a bundle's `barcodes[].sequence` holds.
    pub fn build(
        layout: &CrfLayout,
        alphabet: &[u8],
        seqs: &[&[u8]],
    ) -> Result<Self, RefChainError> {
        if alphabet.len() != layout.n_edges {
            return Err(RefChainError::AlphabetLen {
                got: alphabet.len(),
                expected: layout.n_edges,
            });
        }
        if seqs.is_empty() {
            return Err(RefChainError::Empty);
        }

        // alphabet[0] is the blank; base b is alphabet[1 + b]. Case-folded so a
        // lowercase reference table is not a silent no-match.
        let mut base_of = [u8::MAX; 256];
        for (b, &sym) in alphabet[1..].iter().enumerate() {
            base_of[usize::from(sym)] = b as u8;
            base_of[usize::from(sym.to_ascii_lowercase())] = b as u8;
        }

        let (n_base, n_states, state_len) = (layout.n_base, layout.n_states, layout.state_len);
        let mut chains = Self {
            n_start: n_states,
            n_cells: 0,
            stay: Vec::new(),
            move_off: vec![0],
            move_src: Vec::new(),
            move_score: Vec::new(),
            finals: Vec::with_capacity(seqs.len()),
            max_fan: 0,
            tail: Vec::new(),
            head: Vec::new(),
        };

        // Chain position 0: every state, no incoming moves, shared by the whole
        // panel because it does not depend on a reference at all.
        for state in 0..n_states {
            chains.push_cell(state as u32, &[]);
        }

        // `(position, emitted prefix) -> first cell of that layer`. Two
        // references agreeing on `seq[..j]` agree on layer `j` entirely, cells
        // and incoming edges alike, so the trie makes the head one copy.
        let mut nodes: HashMap<(usize, &[u8]), u32> = HashMap::new();
        let mut bases: Vec<u8> = Vec::new();
        let mut incoming: Vec<(u32, u32)> = Vec::new();

        // `copied`, not `iter`: the trie keys borrow from the caller's
        // sequences, so they need that lifetime rather than the iterator's.
        for (index, seq) in seqs.iter().copied().enumerate() {
            if seq.len() < state_len {
                return Err(RefChainError::TooShort {
                    index,
                    len: seq.len(),
                    state_len,
                });
            }
            bases.clear();
            for &symbol in seq.iter() {
                let b = base_of[usize::from(symbol)];
                if b == u8::MAX {
                    return Err(RefChainError::BadSymbol { index, symbol });
                }
                bases.push(b);
            }

            // `code` is the emitted prefix packed as a k-mer, oldest base most
            // significant — the same encoding the lattice uses for a state.
            let mut prev_base = 0u32; // first cell of position j - 1
            let mut code = 0u32;
            for j in 0..=seq.len() {
                // Substates still carrying an unresolved prefix base, and the
                // width of the resolved part below them.
                let free = state_len.saturating_sub(j);
                let n_sub = n_base.pow(free as u32);
                // Width of the resolved (emitted) part of the k-mer below the
                // unresolved prefix; `low * n_sub == n_states` at every `j`,
                // which is why the shift below wraps at `n_states`.
                let low = n_base.pow(j.min(state_len) as u32) as u32;
                if j == 0 {
                    prev_base = 0;
                    continue;
                }
                code = (code * n_base as u32 + u32::from(bases[j - 1])) % n_states as u32;

                if let Some(&base) = nodes.get(&(j, &seq[..j])) {
                    prev_base = base;
                    continue;
                }
                let base = chains.n_cells as u32;
                for u in 0..n_sub {
                    let state = u as u32 * low + code;
                    incoming.clear();
                    if j <= state_len {
                        // The move drops the source's oldest prefix base, so
                        // every one of `n_base` prefixes ending in this
                        // substate is a source and names its own edge.
                        for d in 0..n_base {
                            let src = prev_base + (d * n_sub + u) as u32;
                            incoming.push((src, (1 + d) as u32 * n_states as u32 + state));
                        }
                    } else {
                        // Past the head the dropped base is emitted and known.
                        let dropped = u32::from(bases[j - 1 - state_len]);
                        incoming.push((prev_base, (1 + dropped) * n_states as u32 + state));
                    }
                    chains.push_cell(state, &incoming);
                }
                // Interned against the caller's sequence rather than a copy:
                // the borrow lives as long as `seqs`, which outlives the loop.
                nodes.insert((j, &seq[..j]), base);
                prev_base = base;
            }
            debug_assert_eq!(n_base.pow(state_len.saturating_sub(seq.len()) as u32), 1);
            chains.finals.push(prev_base);
        }

        Ok(chains)
    }

    fn push_cell(&mut self, state: u32, incoming: &[(u32, u32)]) {
        match incoming.len() {
            0 => {}
            1 => self.tail.push(self.n_cells as u32),
            _ => self.head.push(self.n_cells as u32),
        }
        self.stay.push(state);
        for &(src, score) in incoming {
            self.move_src.push(src);
            self.move_score.push(score);
        }
        self.move_off.push(self.move_src.len() as u32);
        self.max_fan = self.max_fan.max(incoming.len());
        self.n_cells += 1;
    }

    /// Number of references.
    pub fn len(&self) -> usize {
        self.finals.len()
    }

    /// Whether the panel is empty. Never true — [`Self::build`] rejects it —
    /// but clippy asks for it next to `len`.
    pub fn is_empty(&self) -> bool {
        self.finals.is_empty()
    }

    /// Cells in the shared lattice, for tests and benchmarks.
    pub fn cells(&self) -> usize {
        self.n_cells
    }

    /// Forward scan over the constrained lattices, writing each reference's
    /// `logZ_target` into `out`.
    ///
    /// `scores` is one read in the decoder's transposed `[t][edge][dest]`
    /// order — the *raw* encoder scores, not log-posteriors. `cur`/`next` are
    /// caller-owned scratch so a per-read call allocates nothing.
    pub(super) fn forward(
        &self,
        scores: &[f32],
        t_len: usize,
        n_score: usize,
        cur: &mut Vec<f32>,
        next: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        cur.clear();
        cur.resize(self.n_cells, f32::NEG_INFINITY);
        cur[..self.n_start].fill(0.0);
        next.clear();
        next.resize(self.n_cells, 0.0);
        let mut acc = vec![0.0f32; self.max_fan + 1];

        for t in 0..t_len {
            let row = &scores[t * n_score..(t + 1) * n_score];

            // Chain position 0: it can only ever stay, so a quarter of the
            // lattice is one add and no transcendental at all.
            for cell in 0..self.n_start {
                next[cell] = cur[cell] + row[self.stay[cell] as usize];
            }

            // The tail: stay, or advance from the single position before it.
            for &cell in &self.tail {
                let cell = cell as usize;
                let i = self.move_off[cell] as usize;
                let stay = cur[cell] + row[self.stay[cell] as usize];
                let moved = cur[self.move_src[i] as usize] + row[self.move_score[i] as usize];
                next[cell] = logaddexp(stay, moved);
            }

            // The head, where the unresolved prefix means `n_base` sources.
            // Left on the general path: it is under a tenth of the cells, and
            // its width is `n_base`, which is not a constant here.
            for &cell in &self.head {
                let cell = cell as usize;
                let (lo, hi) = (
                    self.move_off[cell] as usize,
                    self.move_off[cell + 1] as usize,
                );
                acc[0] = cur[cell] + row[self.stay[cell] as usize];
                for (k, i) in (lo..hi).enumerate() {
                    acc[1 + k] = cur[self.move_src[i] as usize] + row[self.move_score[i] as usize];
                }
                next[cell] = logsumexp(&acc[..1 + hi - lo]);
            }

            std::mem::swap(cur, next);
        }

        out.clear();
        out.extend(self.finals.iter().map(|&c| cur[c as usize]));
    }
}

/// `log(exp(a) + exp(b))`, max-shifted, for the fan-in-1 cells that are most of
/// the lattice.
///
/// One `exp` and one `ln_1p`, against the general [`logsumexp`]'s slice walk —
/// the same arithmetic with nothing to spill to memory, which is what the scan
/// is actually bound by.
#[inline]
fn logaddexp(a: f32, b: f32) -> f32 {
    let (m, d) = if a >= b { (a, b - a) } else { (b, a - b) };
    if !m.is_finite() {
        return m;
    }
    m + d.exp().ln_1p()
}

/// `log(Σ exp(x))`, max-shifted. `-inf` in, `-inf` out — an unreachable cell
/// stays unreachable rather than becoming `NaN`.
///
/// The max term's own `exp` is skipped rather than computed: it is `exp(0)`,
/// and this runs over inputs of two or five elements, so that one call is 20%
/// to 50% of the transcendental work. `Semiring::Log::reduce` in
/// [`super::lattice`] does not do this because it feeds SIMD kernels where the
/// branch would cost more than the `exp` saves; here every call is scalar.
fn logsumexp(vals: &[f32]) -> f32 {
    let (mut m, mut at) = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in vals.iter().enumerate() {
        if v > m {
            m = v;
            at = i;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut sum = 1.0f32;
    for (i, &v) in vals.iter().enumerate() {
        if i != at {
            sum += (v - m).exp();
        }
    }
    m + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> CrfLayout {
        CrfLayout::new(4, 4).unwrap()
    }

    /// The panel's head is built once, not once per reference: 16 references
    /// sharing a 4-base leader must not each pay for 256 + 64 + 16 + 4 cells.
    #[test]
    fn shared_prefix_shares_the_head() {
        let l = layout();
        let shared: Vec<Vec<u8>> = (0..4)
            .map(|i| {
                let mut s = b"ACGTACGT".to_vec();
                s[7] = b"ACGT"[i];
                s
            })
            .collect();
        let refs: Vec<&[u8]> = shared.iter().map(Vec::as_slice).collect();
        let chains = RefChains::build(&l, b"NACGT", &refs).unwrap();

        // 256 starts + head layers 1..=4 (64 + 16 + 4 + 1, shared) + tail
        // positions 5..=8. Positions 5..=7 are shared too (the sequences differ
        // only at the last base), position 8 is per reference.
        assert_eq!(chains.cells(), 256 + 64 + 16 + 4 + 1 + 3 + 4);
        assert_eq!(chains.len(), 4);
    }

    /// Distinct leaders cannot share, and the count says so exactly.
    #[test]
    fn distinct_prefixes_do_not_share() {
        let l = layout();
        let refs: Vec<&[u8]> = vec![b"ACGTAC", b"CGTACG"];
        let chains = RefChains::build(&l, b"NACGT", &refs).unwrap();
        assert_eq!(chains.cells(), 256 + 2 * (64 + 16 + 4 + 1 + 2));
    }

    /// The chain's states and edges have to be the ones bonito's
    /// `gather_target_scores` picks: stay on edge 0 into `target[j..j+4]`, move
    /// on edge `1 + target[j-1]` into `target[j..j+4]`. Rebuild both from the
    /// layout's own `source_state` and check every tail cell agrees.
    #[test]
    fn tail_edges_match_the_layout() {
        let l = layout();
        let seq: &[u8] = b"ACGTTGCAAGCT";
        let chains = RefChains::build(&l, b"NACGT", &[seq]).unwrap();
        let bases: Vec<usize> = seq
            .iter()
            .map(|&c| b"ACGT".iter().position(|&x| x == c).unwrap())
            .collect();

        // Cell for chain position j > state_len, in build order: starts after
        // the 256 + 64 + 16 + 4 + 1 head.
        let head = 256 + 64 + 16 + 4 + 1;
        for j in (l.state_len + 1)..=seq.len() {
            let cell = head + (j - l.state_len - 1);
            // The k-mer this position spells, oldest base most significant.
            let state = bases[j - l.state_len..j]
                .iter()
                .fold(0usize, |acc, &b| acc * l.n_base + b);
            assert_eq!(chains.stay[cell] as usize, state, "stay at j={j}");

            let (lo, hi) = (
                chains.move_off[cell] as usize,
                chains.move_off[cell + 1] as usize,
            );
            assert_eq!(hi - lo, 1, "tail fan-in at j={j}");
            let dropped = bases[j - l.state_len - 1];
            let edge = 1 + dropped;
            assert_eq!(
                chains.move_score[lo] as usize,
                edge * l.n_states + state,
                "move score index at j={j}"
            );
            // The layout must agree that this edge runs from the previous
            // position's state into this one.
            let prev = bases[j - l.state_len - 1..j - 1]
                .iter()
                .fold(0usize, |acc, &b| acc * l.n_base + b);
            assert_eq!(l.source_state(state, edge), prev, "source at j={j}");
            assert_eq!(l.emitted_base(state, edge), Some(bases[j - 1]));
        }
    }

    /// Head cells carry `n_base` incoming moves, one per prefix that could have
    /// preceded them, and each names the base it drops.
    #[test]
    fn head_fan_in_is_one_per_unresolved_prefix() {
        let l = layout();
        let chains = RefChains::build(&l, b"NACGT", &[b"ACGTACGT".as_slice()]).unwrap();
        for cell in 256..256 + 64 + 16 + 4 + 1 {
            let (lo, hi) = (
                chains.move_off[cell] as usize,
                chains.move_off[cell + 1] as usize,
            );
            assert_eq!(hi - lo, l.n_base, "head cell {cell} fan-in");
            let mut edges: Vec<usize> = (lo..hi)
                .map(|i| chains.move_score[i] as usize / l.n_states)
                .collect();
            edges.sort_unstable();
            assert_eq!(edges, vec![1, 2, 3, 4]);
        }
    }

    /// A reference shorter than one state has no unambiguous final cell, so it
    /// is refused rather than silently scored against a substate.
    #[test]
    fn short_reference_is_refused() {
        let l = layout();
        let err = RefChains::build(&l, b"NACGT", &[b"ACG".as_slice()]).unwrap_err();
        assert!(matches!(err, RefChainError::TooShort { len: 3, .. }));
    }

    #[test]
    fn bad_symbol_is_refused() {
        let l = layout();
        let err = RefChains::build(&l, b"NACGT", &[b"ACGTX".as_slice()]).unwrap_err();
        assert!(matches!(err, RefChainError::BadSymbol { symbol: b'X', .. }));
    }

    /// Deterministic pseudo-random scores; a fixed LCG so a failure is
    /// reproducible without pulling `rand` into a test of pure arithmetic.
    fn scores(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 6.0 - 3.0
            })
            .collect()
    }

    /// Every path and its emission, by exhaustive enumeration: `(emitted
    /// bases, total score)`. `scores` is in the transposed `[t][edge][dest]`
    /// order the scan consumes.
    ///
    /// Small enough to enumerate (`n_states * (1 + n_base) ** t_len` paths),
    /// which is the point — it is the definition of the quantity, with none of
    /// the recursion under test shared.
    fn all_paths(l: &CrfLayout, scores: &[f32], t_len: usize) -> Vec<(Vec<u8>, f64)> {
        let branch = 1 + l.n_base;
        let mut out = Vec::new();
        for start in 0..l.n_states {
            for code in 0..branch.pow(t_len as u32) {
                let (mut state, mut total, mut emit) = (start, 0.0f64, Vec::new());
                let mut c = code;
                for t in 0..t_len {
                    let row = &scores[t * l.n_score..(t + 1) * l.n_score];
                    let choice = c % branch;
                    c /= branch;
                    if choice == 0 {
                        total += f64::from(row[state]); // edge 0 (stay) into `state`
                    } else {
                        // Outgoing moves of `state`: they all carry edge index
                        // `1 + state / group` and land on the contiguous block
                        // `(state % group) * n_base ..`, which is the inverse
                        // of `CrfLayout::source_state`.
                        let dest = (state % l.group()) * l.n_base + (choice - 1);
                        let edge = 1 + state / l.group();
                        total += f64::from(row[edge * l.n_states + dest]);
                        emit.push((dest % l.n_base) as u8);
                        state = dest;
                    }
                }
                out.push((emit, total));
            }
        }
        out
    }

    fn logsumexp_f64(vals: impl Iterator<Item = f64>) -> f64 {
        let v: Vec<f64> = vals.collect();
        let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return m;
        }
        m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
    }

    /// The scan must equal the sum over every path that emits the reference —
    /// including the marginalisation over the unresolved prefix, which is the
    /// part no gather-and-compare test can check.
    #[test]
    fn chain_equals_brute_force_over_paths() {
        let l = CrfLayout::new(2, 2).unwrap();
        let t_len = 6;
        let sc = scores(t_len * l.n_score, 7);
        let paths = all_paths(&l, &sc, t_len);

        // Reference every emission long enough to have an unambiguous final
        // cell, so the head marginalisation is exercised on all of them.
        let mut wanted: Vec<Vec<u8>> = paths
            .iter()
            .map(|(e, _)| e.clone())
            .filter(|e| e.len() >= l.state_len)
            .collect();
        wanted.sort();
        wanted.dedup();
        assert!(
            wanted.len() > 8,
            "not enough distinct emissions to be a test"
        );

        let seqs: Vec<Vec<u8>> = wanted
            .iter()
            .map(|e| e.iter().map(|&b| b"AC"[usize::from(b)]).collect())
            .collect();
        let refs: Vec<&[u8]> = seqs.iter().map(Vec::as_slice).collect();
        let chains = RefChains::build(&l, b"NAC", &refs).unwrap();

        let (mut cur, mut next, mut out) = (Vec::new(), Vec::new(), Vec::new());
        chains.forward(&sc, t_len, l.n_score, &mut cur, &mut next, &mut out);

        for (k, want) in wanted.iter().enumerate() {
            let expect = logsumexp_f64(
                paths
                    .iter()
                    .filter(|(e, _)| e == want)
                    .map(|&(_, total)| total),
            );
            assert!(
                (f64::from(out[k]) - expect).abs() < 1e-3,
                "reference {k} {want:?}: chain {} vs brute force {expect}",
                out[k]
            );
        }
    }

    /// `logZ` over the whole lattice — the normaliser the scores are divided
    /// by — is the sum over *every* path, whatever it emits.
    #[test]
    fn decode_logz_equals_brute_force() {
        use crate::crf::lattice::{CrfScratch, decode_with_refs};

        let l = CrfLayout::new(2, 2).unwrap();
        let t_len = 6;
        let sc = scores(t_len * l.n_score, 11);
        let expect = logsumexp_f64(all_paths(&l, &sc, t_len).into_iter().map(|(_, s)| s));

        // `decode_*` takes scores in the encoder's `[t][dest][edge]` order and
        // transposes them itself, so hand it the untransposed form.
        let mut native = vec![0.0f32; sc.len()];
        for t in 0..t_len {
            for edge in 0..l.n_edges {
                for dest in 0..l.n_states {
                    native[t * l.n_score + l.score_index(dest, edge)] =
                        sc[t * l.n_score + edge * l.n_states + dest];
                }
            }
        }

        let chains = RefChains::build(&l, b"NAC", &[b"AC".as_slice()]).unwrap();
        let mut scratch = CrfScratch::new();
        let mut out = Vec::new();
        decode_with_refs(
            &l,
            b"NAC",
            &native,
            t_len,
            &mut scratch,
            crate::crf::lattice::Backend::Scalar,
            &chains,
            &mut out,
        )
        .unwrap();

        assert!(
            (f64::from(scratch.logz()) - expect).abs() < 1e-3,
            "logz {} vs brute force {expect}",
            scratch.logz()
        );
        // Normalised: a probability, so never above 1.
        assert!(out[0] <= 0.0, "log P = {} is above zero", out[0]);
    }

    /// The emissions partition the paths, so the probabilities of *all* of them
    /// sum to exactly 1. Any double-count or missed path in the head shows up
    /// here and nowhere else.
    #[test]
    fn probabilities_over_all_emissions_sum_to_one() {
        let l = CrfLayout::new(2, 2).unwrap();
        let t_len = 5;
        let sc = scores(t_len * l.n_score, 23);
        let paths = all_paths(&l, &sc, t_len);
        let logz = logsumexp_f64(paths.iter().map(|&(_, s)| s));

        let mut wanted: Vec<Vec<u8>> = paths.iter().map(|(e, _)| e.clone()).collect();
        wanted.sort();
        wanted.dedup();
        // Emissions shorter than one state have no unambiguous final cell, so
        // the scan cannot score them; account for their mass separately.
        let (long, short): (Vec<_>, Vec<_>) =
            wanted.into_iter().partition(|e| e.len() >= l.state_len);

        let seqs: Vec<Vec<u8>> = long
            .iter()
            .map(|e| e.iter().map(|&b| b"AC"[usize::from(b)]).collect())
            .collect();
        let refs: Vec<&[u8]> = seqs.iter().map(Vec::as_slice).collect();
        let chains = RefChains::build(&l, b"NAC", &refs).unwrap();
        let (mut cur, mut next, mut out) = (Vec::new(), Vec::new(), Vec::new());
        chains.forward(&sc, t_len, l.n_score, &mut cur, &mut next, &mut out);

        let mut total: f64 = out.iter().map(|&v| (f64::from(v) - logz).exp()).sum();
        for e in &short {
            let m = logsumexp_f64(paths.iter().filter(|(x, _)| x == e).map(|&(_, s)| s));
            total += (m - logz).exp();
        }
        assert!(
            (total - 1.0).abs() < 1e-4,
            "emission probabilities sum to {total}, not 1"
        );
    }

    /// Lowercase references are the same references.
    #[test]
    fn lowercase_is_folded() {
        let l = layout();
        let upper = RefChains::build(&l, b"NACGT", &[b"ACGTACGT".as_slice()]).unwrap();
        let lower = RefChains::build(&l, b"NACGT", &[b"acgtacgt".as_slice()]).unwrap();
        assert_eq!(upper.stay, lower.stay);
        assert_eq!(upper.move_score, lower.move_score);
    }
}

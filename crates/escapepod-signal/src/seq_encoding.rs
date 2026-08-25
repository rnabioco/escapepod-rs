//! Sequence encodings for signal-level models.
//!
//! A model that reads raw signal wants the basecall alongside it, and by the
//! time it reaches the network "the basecall" is not a string: it is a one-hot
//! tensor laid out along the **signal** axis, so every sample carries the
//! k-mer context of whichever base the pore was reading at that sample.
//! [`encode_signal_kmer`] is that tensor, and [`sequence_to_int`] is the base
//! alphabet it is written in.
//!
//! This is the natural pair to [`crate::mapping`]: that module *produces* a
//! base→signal map from a move table and a CIGAR, and this one *consumes* one.
//! Until now only the producing half lived here — the consuming half lived
//! downstream in leech, inside a `cdylib` Python extension module that Rust
//! cannot link (#271). So a Rust runtime for a leech classifier trained with
//! `seq_encoding="signal_kmer"` had to transcribe the rule, and the tensor is
//! *not* in the exported ONNX graph — it is computed in the dataset, so the
//! runtime has to build it before it can call the model at all.
//!
//! Transcribing is what this stack keeps getting bitten by, and this rule has
//! the shape that hides it: every wrong version still produces a correctly
//! shaped array of zeros and ones, just with the context shifted, transposed,
//! or a base quietly missing. See [`crate::features`] for the same argument
//! about per-base statistics, which grew two disagreeing copies before it was
//! written down once.
//!
//! The specific traps, all pinned by tests below:
//!
//! * The number of bases is `seq_to_signal.len() - 1`, **not**
//!   `seq_ints.len()` — `seq_ints` is longer by the k-mer context on both ends.
//! * The context width is a *pair*, and its halves are not interchangeable
//!   where the context is cut ([`sequence_ints_with_context`]) even though the
//!   encoder itself only sees their sum; [`KmerContext`] names them so a caller
//!   cannot transpose them silently.
//! * A base whose span is empty (`start == end`) contributes nothing. It is the
//!   branch a golden test set is most likely to miss entirely.
//! * A span that hangs off either end of the signal is **intersected** with it,
//!   not dropped — see [`encode_signal_kmer`].

/// The value [`sequence_to_int`] gives a base that is not `A`/`C`/`G`/`T`/`U`.
///
/// Negative rather than a fifth channel: an ambiguity code is *absence of
/// information*, and [`encode_signal_kmer`] leaves an all-zero column for that
/// k-mer position rather than inventing a base the model never saw in training.
pub const UNKNOWN_BASE: i8 = -1;

/// Encode one nucleotide as `A = 0`, `C = 1`, `G = 2`, `T`/`U` = 3, anything
/// else [`UNKNOWN_BASE`]. Case-insensitive.
///
/// `U` and `T` share an index deliberately: an RNA basecall and a DNA basecall
/// of the same molecule must land on the same channel, or a model fitted on
/// one is being fed the other's one-hot with every uracil blanked out.
#[must_use]
#[inline]
pub fn base_to_int(base: u8) -> i8 {
    match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' | b'U' | b'u' => 3,
        _ => UNKNOWN_BASE,
    }
}

/// Encode a sequence with [`base_to_int`].
///
/// The input is the sequence **with its k-mer context**, i.e. `seq_len +
/// before + after` bases, because that is what [`encode_signal_kmer`] indexes.
///
/// ```
/// use escapepod_signal::seq_encoding::sequence_to_int;
///
/// assert_eq!(sequence_to_int(b"ACGT"), vec![0, 1, 2, 3]);
/// // RNA reads the same as DNA, and an ambiguity code is negative.
/// assert_eq!(sequence_to_int(b"acgu"), vec![0, 1, 2, 3]);
/// assert_eq!(sequence_to_int(b"N"), vec![-1]);
/// ```
#[must_use]
pub fn sequence_to_int(sequence: &[u8]) -> Vec<i8> {
    sequence.iter().copied().map(base_to_int).collect()
}

/// How many bases of context flank the base being encoded.
///
/// A named pair rather than two bare `usize` arguments, for the reason
/// [`crate::mapping::CigarOp`] is a named struct: `(4, 4)` survives a
/// transposition unnoticed and `(4, 0)` does not, so the asymmetric case — the
/// one where it matters — is exactly the one a positional call site gets wrong.
/// The window shifts by `before - after` bases, and the model still returns a
/// number. [`sequence_ints_with_context`] is where that bite lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KmerContext {
    /// Bases before the encoded base.
    pub before: usize,
    /// Bases after it.
    pub after: usize,
}

impl KmerContext {
    /// Construct a context.
    #[must_use]
    pub fn new(before: usize, after: usize) -> Self {
        Self { before, after }
    }

    /// Bases in the k-mer: `before + 1 + after`. Always at least 1.
    #[must_use]
    #[inline]
    pub fn kmer_len(self) -> usize {
        self.before + 1 + self.after
    }

    /// Rows [`encode_signal_kmer`] emits: `4 * kmer_len`, one block of four per
    /// k-mer position. This is the model's input channel count — 36 for the
    /// usual `(4, 4)` — so it is computed here rather than by each caller.
    #[must_use]
    #[inline]
    pub fn channels(self) -> usize {
        4 * self.kmer_len()
    }
}

/// Default `(4, 4)`, the 36-channel context leech's `signal_kmer` models are
/// trained with.
impl Default for KmerContext {
    fn default() -> Self {
        Self::new(4, 4)
    }
}

/// Cut the sequence a signal window covers, **plus** its k-mer context, into
/// the int encoding [`encode_signal_kmer`] takes.
///
/// `core_start` is the first base the signal window covers and `n_bases` is how
/// many it covers — i.e. `seq_to_signal.len() - 1` for the map that goes with
/// it. The result is `n_bases + ctx.before + ctx.after` long, starting
/// `ctx.before` bases *earlier* than `core_start`, so `encode_signal_kmer`'s
/// `seq_ints[seq_pos + kmer_pos]` indexing lands where it should.
///
/// Positions off either end of `sequence` — the first bases of a read have no
/// preceding context, and the last none following — are [`UNKNOWN_BASE`], which
/// the encoder leaves as an all-zero column. Padding rather than truncating is
/// what keeps the output width fixed at `ctx.channels()`, which is what the
/// model's first layer expects.
///
/// This is the step where `before` and `after` are not interchangeable: swap
/// them and every k-mer is read from a window displaced by `before - after`
/// bases, silently. Given an already-cut `seq_ints`, [`encode_signal_kmer`]
/// itself only depends on the *total* width, so a transposition upstream of
/// here is invisible downstream of it.
///
/// ```
/// use escapepod_signal::seq_encoding::{KmerContext, sequence_ints_with_context};
///
/// // Two core bases (`GT`) with one base of context each way.
/// let ints = sequence_ints_with_context(b"ACGTA", 2, 2, KmerContext::new(1, 1));
/// assert_eq!(ints, vec![1, 2, 3, 0]); // C G T A
///
/// // Context that runs off the start is padded, not shifted.
/// let ints = sequence_ints_with_context(b"ACGTA", 0, 2, KmerContext::new(2, 0));
/// assert_eq!(ints, vec![-1, -1, 0, 1]); // . . A C
/// ```
#[must_use]
pub fn sequence_ints_with_context(
    sequence: &[u8],
    core_start: usize,
    n_bases: usize,
    ctx: KmerContext,
) -> Vec<i8> {
    let lo = core_start as i64 - ctx.before as i64;
    let hi = lo + (n_bases + ctx.before + ctx.after) as i64;
    (lo..hi)
        .map(|i| {
            usize::try_from(i)
                .ok()
                .and_then(|u| sequence.get(u))
                .map_or(UNKNOWN_BASE, |&b| base_to_int(b))
        })
        .collect()
}

/// Scatter the one-hot k-mer context along the signal axis.
///
/// Returns a flat row-major array of shape `(ctx.channels(), signal_len)`: row
/// `4 * kmer_pos + base` is hot over exactly the samples owned by the base
/// sitting `kmer_pos` into the window. This is Remora's `encoded_kmers.pyx`,
/// and the `sequence` input of a leech model built with
/// `seq_encoding="signal_kmer"`.
///
/// * `seq_ints` — the sequence **including** context, as
///   [`sequence_ints_with_context`] cuts it: `0..=3`, negative for unknown, and
///   `ctx.before` bases ahead of the window's first base. A base indexed past
///   the end of this slice,
///   or negative, contributes nothing; it is not padded and not an error, so a
///   caller that supplies too little context gets an all-zero block for the
///   affected k-mer positions rather than a panic.
/// * `seq_to_signal` — the base→signal map from [`crate::mapping`], one entry
///   per base **plus a closing boundary**. Base `i` owns samples
///   `seq_to_signal[i]..seq_to_signal[i + 1]`.
/// * `signal_len` — the width of the output, i.e. the length of the signal
///   window the map is expressed in.
///
/// # Which bases, and which samples
///
/// The base count is `seq_to_signal.len() - 1`, **not** `seq_ints.len()` —
/// `seq_ints` is longer by `before + after`. Base `seq_pos` is encoded at k-mer
/// position `kmer_pos` from `seq_ints[seq_pos + kmer_pos]`, so the leftmost
/// k-mer position reads the *earliest* context base and the window slides
/// forward from there; there is no centring offset to apply here, because
/// `before` is already spent in where `seq_ints` starts. This function
/// therefore only depends on `ctx.kmer_len()`; it is
/// [`sequence_ints_with_context`] that has to get the split right.
///
/// Each span is **intersected** with `[0, signal_len)` and whatever survives is
/// filled. A span that starts before the window still contributes its tail, and
/// one that runs past the end still contributes its head — which matters for a
/// reference-anchored map, whose entries legitimately go negative once the
/// aligned region is cropped. Only an empty intersection contributes nothing,
/// and so does an empty span (`start == end`), which a map with unresolved
/// bases is full of.
///
/// The intersection is worth stating because the obvious spelling gets it wrong
/// in one direction only: clamping *after* an `as usize` cast turns a negative
/// start into a huge one, which clamps to `signal_len`, which makes the span
/// empty — the base vanishes instead of being truncated, and nothing about the
/// output says so.
///
/// # Examples
///
/// ```
/// use escapepod_signal::seq_encoding::{KmerContext, encode_signal_kmer, sequence_to_int};
///
/// // One base, `C`, holding the pore for five samples, with no context.
/// let ctx = KmerContext::new(0, 0);
/// let enc = encode_signal_kmer(&sequence_to_int(b"C"), &[0, 5], 5, ctx);
/// assert_eq!(enc.len(), ctx.channels() * 5);
/// // Row 1 is `C`; the other three are cold.
/// assert_eq!(&enc[5..10], &[1.0; 5]);
/// assert!(enc[..5].iter().chain(&enc[10..]).all(|&v| v == 0.0));
/// ```
#[must_use]
pub fn encode_signal_kmer(
    seq_ints: &[i8],
    seq_to_signal: &[i64],
    signal_len: usize,
    ctx: KmerContext,
) -> Vec<f32> {
    let mut out = vec![0.0f32; ctx.channels() * signal_len];
    encode_signal_kmer_into(seq_ints, seq_to_signal, signal_len, ctx, &mut out);
    out
}

/// [`encode_signal_kmer`] into a caller-owned buffer, so a loop over chunks
/// allocates once rather than once per chunk.
///
/// `out` must be exactly `ctx.channels() * signal_len` long, and is zeroed
/// first — the encoding only ever writes ones, so a buffer still holding the
/// previous chunk would union the two.
///
/// # Panics
///
/// If `out` is not `ctx.channels() * signal_len` long. A mis-sized buffer means
/// a transposed or stale shape, and the row stride comes from `signal_len`
/// either way, so the alternative is a silently misaligned tensor.
pub fn encode_signal_kmer_into(
    seq_ints: &[i8],
    seq_to_signal: &[i64],
    signal_len: usize,
    ctx: KmerContext,
    out: &mut [f32],
) {
    assert_eq!(
        out.len(),
        ctx.channels() * signal_len,
        "output buffer must be `channels * signal_len` for the encoding to be laid out row-major"
    );
    out.fill(0.0);
    if signal_len == 0 {
        return;
    }
    let n_bases = seq_to_signal.len().saturating_sub(1);
    let sig_len = signal_len as i64;

    for kmer_pos in 0..ctx.kmer_len() {
        let block = 4 * kmer_pos;
        for seq_pos in 0..n_bases {
            // Too little context supplied: nothing to encode at this position.
            let Some(&base) = seq_ints.get(seq_pos + kmer_pos) else {
                continue;
            };
            if base < 0 {
                continue;
            }
            // Intersect with the window rather than clamping a cast: see the
            // note on `encode_signal_kmer`.
            let start = seq_to_signal[seq_pos].clamp(0, sig_len) as usize;
            let end = seq_to_signal[seq_pos + 1].clamp(0, sig_len) as usize;
            if start < end {
                let row = (block + base as usize) * signal_len;
                out[row + start..row + end].fill(1.0);
            }
        }
    }
}

#[cfg(test)]
mod seq_encoding_tests {
    use super::*;

    /// The four rows of `kmer_pos`, as `[base][sample]`.
    fn block(enc: &[f32], signal_len: usize, kmer_pos: usize) -> Vec<&[f32]> {
        (0..4)
            .map(|b| {
                let row = (4 * kmer_pos + b) * signal_len;
                &enc[row..row + signal_len]
            })
            .collect()
    }

    fn all_cold(rows: &[&[f32]]) -> bool {
        rows.iter().all(|row| row.iter().all(|&v| v == 0.0))
    }

    #[test]
    fn base_alphabet() {
        assert_eq!(sequence_to_int(b"ACGT"), vec![0, 1, 2, 3]);
        assert_eq!(sequence_to_int(b"acgt"), vec![0, 1, 2, 3]);
        // U shares T's channel, in both cases.
        assert_eq!(sequence_to_int(b"Uu"), vec![3, 3]);
        // Everything else is negative, not a fifth channel.
        assert_eq!(sequence_to_int(b"NX-"), vec![-1, -1, -1]);
        assert_eq!(sequence_to_int(b""), Vec::<i8>::new());
    }

    #[test]
    fn context_arithmetic() {
        assert_eq!(KmerContext::default(), KmerContext::new(4, 4));
        assert_eq!(KmerContext::default().kmer_len(), 9);
        // The 36 channels a `signal_kmer` model takes.
        assert_eq!(KmerContext::default().channels(), 36);
        // A bare base is still a 1-mer, never zero rows.
        assert_eq!(KmerContext::new(0, 0).kmer_len(), 1);
        assert_eq!(KmerContext::new(0, 0).channels(), 4);
    }

    #[test]
    fn single_base_is_one_hot_over_its_span() {
        let ctx = KmerContext::new(0, 0);
        let enc = encode_signal_kmer(&sequence_to_int(b"C"), &[0, 5], 5, ctx);
        assert_eq!(enc.len(), 4 * 5);
        let rows = block(&enc, 5, 0);
        assert_eq!(rows[1], [1.0; 5]);
        for b in [0, 2, 3] {
            assert_eq!(rows[b], [0.0; 5]);
        }
    }

    #[test]
    fn centre_position_covers_every_sample_exactly_once() {
        // 3 bases (A, C, G) of 10 samples each, one base of context either side.
        let ctx = KmerContext::new(1, 1);
        let seq = sequence_to_int(b"AACGG");
        let enc = encode_signal_kmer(&seq, &[0, 10, 20, 30], 30, ctx);
        assert_eq!(enc.len(), ctx.channels() * 30);

        // The centre block IS the core bases, so it tiles the whole window.
        let centre = block(&enc, 30, 1);
        assert_eq!(centre[0][..10], [1.0; 10]); // A
        assert_eq!(centre[1][10..20], [1.0; 10]); // C
        assert_eq!(centre[2][20..], [1.0; 10]); // G

        // Every block is at most one-hot per sample, and only ever 0 or 1; the
        // centre one is exactly one-hot.
        for kp in 0..ctx.kmer_len() {
            let rows = block(&enc, 30, kp);
            for s in 0..30 {
                let hot: f32 = rows.iter().map(|row| row[s]).sum();
                assert!(hot <= 1.0, "block {kp} sample {s} has two bases hot");
                if kp == 1 {
                    assert_eq!(hot, 1.0, "sample {s} is uncovered");
                }
            }
        }
        assert!(enc.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn the_window_slides_forward_over_the_context() {
        // Distinct bases either side of a single core base: the leftmost k-mer
        // position holds the EARLIEST base, not the core one.
        let ctx = KmerContext::new(1, 1);
        let enc = encode_signal_kmer(&sequence_to_int(b"ACG"), &[0, 4], 4, ctx);
        assert_eq!(block(&enc, 4, 0)[0], [1.0; 4]); // A, before
        assert_eq!(block(&enc, 4, 1)[1], [1.0; 4]); // C, the base itself
        assert_eq!(block(&enc, 4, 2)[2], [1.0; 4]); // G, after
    }

    #[test]
    fn context_is_cut_around_the_window_and_padded_off_the_ends() {
        let ctx = KmerContext::new(1, 1);
        // Two core bases, one of context each way.
        assert_eq!(
            sequence_ints_with_context(b"ACGTA", 2, 2, ctx),
            vec![1, 2, 3, 0]
        );
        // Off the start and off the end: padded, so the width never moves.
        assert_eq!(
            sequence_ints_with_context(b"ACGTA", 0, 2, ctx),
            vec![-1, 0, 1, 2]
        );
        assert_eq!(
            sequence_ints_with_context(b"ACGTA", 3, 2, ctx),
            vec![2, 3, 0, -1]
        );
        // A window entirely off the end is all padding, not a short vector.
        assert_eq!(sequence_ints_with_context(b"AC", 9, 2, ctx), vec![-1; 4]);
        // Width is always n_bases + before + after.
        for c in [
            KmerContext::new(0, 0),
            KmerContext::new(4, 4),
            KmerContext::new(3, 1),
        ] {
            let ints = sequence_ints_with_context(b"ACGTA", 1, 3, c);
            assert_eq!(ints.len(), 3 + c.before + c.after);
        }
    }

    #[test]
    fn before_and_after_are_not_interchangeable() {
        // The trap `KmerContext` exists to make visible. It bites when the
        // context is CUT: transposing it displaces the window by
        // `before - after` bases and still yields the same shape.
        let (a_ints, b_ints) = (
            sequence_ints_with_context(b"ACGTA", 2, 1, KmerContext::new(2, 0)),
            sequence_ints_with_context(b"ACGTA", 2, 1, KmerContext::new(0, 2)),
        );
        assert_eq!(a_ints.len(), b_ints.len());
        assert_ne!(a_ints, b_ints);

        let a = encode_signal_kmer(&a_ints, &[0, 4], 4, KmerContext::new(2, 0));
        let b = encode_signal_kmer(&b_ints, &[0, 4], 4, KmerContext::new(0, 2));
        assert_eq!(a.len(), b.len());
        assert_ne!(a, b);
        // (2, 0) ends on the core base `G`; (0, 2) starts on it.
        assert_eq!(block(&a, 4, 2)[2], [1.0; 4]);
        assert_eq!(block(&b, 4, 0)[2], [1.0; 4]);
    }

    #[test]
    fn unknown_bases_contribute_nothing() {
        let ctx = KmerContext::new(0, 0);
        let enc = encode_signal_kmer(&sequence_to_int(b"N"), &[0, 10], 10, ctx);
        assert!(enc.iter().all(|&v| v == 0.0));

        // An unknown in the CONTEXT blanks only that k-mer position.
        let ctx = KmerContext::new(1, 0);
        let enc = encode_signal_kmer(&sequence_to_int(b"NA"), &[0, 10], 10, ctx);
        assert!(all_cold(&block(&enc, 10, 0)));
        assert_eq!(block(&enc, 10, 1)[0], [1.0; 10]);
    }

    #[test]
    fn missing_context_is_skipped_not_padded() {
        // `seq_ints` should be seq_len + before + after long; supplying only
        // the core base must not panic, and must not shift the window.
        let ctx = KmerContext::new(0, 2);
        let enc = encode_signal_kmer(&sequence_to_int(b"A"), &[0, 6], 6, ctx);
        assert_eq!(block(&enc, 6, 0)[0], [1.0; 6]);
        for kp in [1, 2] {
            assert!(all_cold(&block(&enc, 6, kp)));
        }
    }

    #[test]
    fn an_empty_span_contributes_nothing() {
        // The branch a golden set misses: base 0 resolves to no samples at all,
        // so it is absent from the encoding while its neighbour is not.
        let ctx = KmerContext::new(0, 0);
        let enc = encode_signal_kmer(&sequence_to_int(b"AC"), &[0, 0, 10], 10, ctx);
        let rows = block(&enc, 10, 0);
        assert!(rows[0].iter().all(|&v| v == 0.0), "A owned no samples");
        assert_eq!(rows[1], [1.0; 10]);
    }

    #[test]
    fn spans_are_intersected_with_the_window_at_both_ends() {
        let ctx = KmerContext::new(0, 0);

        // Past the end: the head still lands.
        let enc = encode_signal_kmer(&sequence_to_int(b"A"), &[0, 20], 10, ctx);
        assert_eq!(block(&enc, 10, 0)[0], [1.0; 10]);

        // Before the start: the TAIL still lands. Clamping an `as usize` cast
        // instead would make this span empty and drop the base silently.
        let enc = encode_signal_kmer(&sequence_to_int(b"A"), &[-5, 4], 10, ctx);
        assert_eq!(block(&enc, 10, 0)[0][..4], [1.0; 4]);
        assert_eq!(block(&enc, 10, 0)[0][4..], [0.0; 6]);

        // Wholly outside, either side: nothing.
        for map in [[-20i64, -5], [12, 20]] {
            let enc = encode_signal_kmer(&sequence_to_int(b"A"), &map, 10, ctx);
            assert!(enc.iter().all(|&v| v == 0.0), "map {map:?} should not land");
        }
    }

    #[test]
    fn degenerate_inputs() {
        let ctx = KmerContext::new(0, 0);
        // Zero bases: the map is just the closing boundary.
        let enc = encode_signal_kmer(&[], &[0], 10, ctx);
        assert_eq!(enc.len(), 4 * 10);
        assert!(enc.iter().all(|&v| v == 0.0));
        // No map at all, and no signal.
        assert!(
            encode_signal_kmer(&[0], &[], 10, ctx)
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(encode_signal_kmer(&[0], &[0, 5], 0, ctx).is_empty());
    }

    #[test]
    fn into_reuses_and_clears_the_buffer() {
        let ctx = KmerContext::new(0, 0);
        let mut buf = vec![0.0f32; ctx.channels() * 5];
        encode_signal_kmer_into(&sequence_to_int(b"A"), &[0, 5], 5, ctx, &mut buf);
        assert_eq!(&buf[..5], &[1.0; 5]);

        // A second chunk must replace the first, not union with it.
        encode_signal_kmer_into(&sequence_to_int(b"C"), &[0, 5], 5, ctx, &mut buf);
        assert_eq!(
            buf,
            encode_signal_kmer(&sequence_to_int(b"C"), &[0, 5], 5, ctx)
        );
        assert!(buf[..5].iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "row-major")]
    fn into_rejects_a_mis_sized_buffer() {
        let ctx = KmerContext::new(1, 1);
        let mut buf = vec![0.0f32; 4 * 5]; // one block, not three
        encode_signal_kmer_into(&sequence_to_int(b"ACG"), &[0, 5], 5, ctx, &mut buf);
    }
}

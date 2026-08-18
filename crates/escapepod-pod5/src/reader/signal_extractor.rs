//! Thread-safe signal extractor for parallel per-read signal extraction.

use crate::arrow_ipc::RawSignalChunk;
use crate::error::Result;

/// Decode a read's compressed chunks in order, stopping once `max_samples`
/// samples have been produced.
///
/// The single definition of "a read's signal, optionally truncated" — every
/// reader path (single, bulk, extractor) funnels through here so the prefix
/// and full decodes cannot drift. Pass `usize::MAX` for a whole read.
pub(crate) fn decode_chunks(chunks: &[RawSignalChunk<'_>], max_samples: usize) -> Result<Vec<i16>> {
    use crate::compression::vbz::decompress_signal_prefix;

    let available: usize = chunks.iter().map(|c| c.samples as usize).sum();
    let mut result = Vec::with_capacity(available.min(max_samples));
    let mut remaining = max_samples;
    for chunk in chunks {
        if remaining == 0 {
            break;
        }
        let cs = chunk.samples as usize;
        let take = cs.min(remaining);
        // `decompress_signal_prefix(.., cs, cs)` is the full decode, so this
        // needs no special case for the last chunk.
        result.extend_from_slice(&decompress_signal_prefix(chunk.signal, cs, take)?);
        remaining -= take;
    }
    Ok(result)
}

/// Thread-safe signal extractor for parallel per-read signal extraction.
///
/// Holds an immutable reference to the memory-mapped signal table bytes and
/// a pre-parsed Arrow IPC footer. Because it contains only immutable data,
/// it is `Send + Sync` and can be shared across rayon threads.
pub struct SignalExtractor<'a> {
    pub(super) signal_bytes: &'a [u8],
    pub(super) footer: crate::arrow_ipc::ArrowIpcFooter,
}

impl<'a> SignalExtractor<'a> {
    /// Extract and decompress signal for a single read's signal rows.
    ///
    /// Thread-safe: no shared mutable state.
    pub fn get_signal(&self, signal_rows: &[u64]) -> Result<Vec<i16>> {
        self.get_signal_prefix(signal_rows, usize::MAX)
    }

    /// Like [`Self::get_signal`] but decodes at most the first `max_samples`
    /// samples — identical to `get_signal(..)[..max_samples]`, and shorter when
    /// the read is. Useful when a consumer (e.g. CNN adapter detection) only
    /// looks at a leading window of a potentially long read.
    ///
    /// The saving is in the SVB16 stage, and in whole 128 KiB ZSTD blocks for
    /// reads long enough to span several; see
    /// [`decompress_signal_prefix`](crate::compression::decompress_signal_prefix)
    /// for why a short read cannot do better than a full inflate.
    pub fn get_signal_prefix(&self, signal_rows: &[u64], max_samples: usize) -> Result<Vec<i16>> {
        let raw_chunks = self
            .footer
            .extract_signal_rows(signal_rows, self.signal_bytes)?;
        decode_chunks(&raw_chunks, max_samples)
    }
}

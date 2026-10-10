//! Thread-safe signal extractor for parallel per-read signal extraction.

use crate::arrow_ipc::RawSignalChunk;
use crate::error::{Error, Result};
use uuid::Uuid;

/// Check that every chunk fetched for one read carries that read's id.
///
/// `expected` is the requested read when the caller knows it; otherwise the
/// first chunk's id stands in, so chunks of one read still have to agree with
/// each other. `rows[i]` is the signal row `chunks[i]` was fetched from.
/// Sixteen-byte compares against a chunk the caller is about to decode, so the
/// cost is lost in the noise (see the PR for the bench).
pub(crate) fn verify_chunk_ids(
    chunks: &[RawSignalChunk<'_>],
    rows: &[u64],
    expected: Option<Uuid>,
    file: &str,
) -> Result<()> {
    let Some(first) = chunks.first() else {
        return Ok(());
    };
    let want: [u8; 16] = expected.map_or(first.read_id, |u| *u.as_bytes());
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.read_id != want {
            return Err(Error::SignalReadIdMismatch {
                file: file.to_string(),
                expected: Uuid::from_bytes(want),
                found: Uuid::from_bytes(chunk.read_id),
                row: rows.get(i).copied().unwrap_or(0),
            });
        }
    }
    Ok(())
}

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
        if chunk.uncompressed {
            result.extend_from_slice(&chunk.samples_i16()?[..take]);
        } else {
            result.extend_from_slice(&decompress_signal_prefix(chunk.signal, cs, take)?);
        }
        remaining -= take;
    }
    Ok(result)
}

/// Thread-safe signal extractor for parallel per-read signal extraction.
///
/// Holds an immutable reference to the memory-mapped signal table bytes and
/// a pre-parsed Arrow IPC footer. Because it contains only immutable data,
/// it is `Send + Sync` and can be shared across rayon threads.
///
/// The footer is a `Cow` so the common case borrows the one the `Reader`
/// already parsed and cached instead of walking every record batch header
/// again — see `Reader::signal_footer_for_bulk` for why that walk is the
/// expensive part.
pub struct SignalExtractor<'a> {
    pub(super) signal_bytes: &'a [u8],
    /// Where the bytes came from, for error messages.
    pub(super) source: String,
    pub(super) footer: std::borrow::Cow<'a, crate::arrow_ipc::ArrowIpcFooter>,
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
        self.get_signal_impl(None, signal_rows, max_samples)
    }

    /// [`Self::get_signal`] that also verifies each chunk's `read_id` against
    /// `read_id` and errors, naming file, read and row, on a mismatch.
    pub fn get_signal_checked(&self, read_id: Uuid, signal_rows: &[u64]) -> Result<Vec<i16>> {
        self.get_signal_impl(Some(read_id), signal_rows, usize::MAX)
    }

    /// [`Self::get_signal_prefix`] with the `read_id` check of
    /// [`Self::get_signal_checked`].
    pub fn get_signal_prefix_checked(
        &self,
        read_id: Uuid,
        signal_rows: &[u64],
        max_samples: usize,
    ) -> Result<Vec<i16>> {
        self.get_signal_impl(Some(read_id), signal_rows, max_samples)
    }

    fn get_signal_impl(
        &self,
        expected: Option<Uuid>,
        signal_rows: &[u64],
        max_samples: usize,
    ) -> Result<Vec<i16>> {
        let raw_chunks = self
            .footer
            .extract_signal_rows(signal_rows, self.signal_bytes)?;
        verify_chunk_ids(&raw_chunks, signal_rows, expected, &self.source)?;
        decode_chunks(&raw_chunks, max_samples)
    }
}

//! Thread-safe signal extractor for parallel per-read signal extraction.

use crate::error::Result;

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
        use crate::compression::vbz::decompress_signal;

        let raw_chunks = self
            .footer
            .extract_signal_rows(signal_rows, self.signal_bytes)?;
        let total_samples: usize = raw_chunks.iter().map(|c| c.samples as usize).sum();
        let mut result = Vec::with_capacity(total_samples);

        for chunk in &raw_chunks {
            let decompressed = decompress_signal(chunk.signal, chunk.samples as usize)?;
            result.extend_from_slice(&decompressed);
        }

        Ok(result)
    }

    /// Like [`Self::get_signal`] but decompresses only the first `max_samples`
    /// samples, decoding just the needed prefix of the chunk that crosses the
    /// boundary (the rest of the ZSTD stream is skipped). Identical to
    /// `get_signal(..)[..max_samples]`. Useful when a consumer (e.g. CNN adapter
    /// detection) only looks at a fixed leading window of a potentially long
    /// read.
    pub fn get_signal_prefix(&self, signal_rows: &[u64], max_samples: usize) -> Result<Vec<i16>> {
        use crate::compression::vbz::{decompress_signal, decompress_signal_prefix};

        let raw_chunks = self
            .footer
            .extract_signal_rows(signal_rows, self.signal_bytes)?;
        let mut result = Vec::with_capacity(max_samples);
        let mut remaining = max_samples;
        for chunk in &raw_chunks {
            if remaining == 0 {
                break;
            }
            let cs = chunk.samples as usize;
            let take = cs.min(remaining);
            let decoded = if take == cs {
                decompress_signal(chunk.signal, cs)?
            } else {
                decompress_signal_prefix(chunk.signal, cs, take)?
            };
            result.extend_from_slice(&decoded);
            remaining -= take;
        }
        Ok(result)
    }
}

//! Abstraction over the bytes backing a POD5 file.
//!
//! [`Reader`](super::Reader) never touches a file handle directly; it asks a
//! [`ByteSource`] for byte ranges. Locally that source is an mmap and every
//! range is a refcount bump over the mapping — no copy, same cost as the raw
//! slicing this replaces. The indirection exists so a source can also sit in
//! front of something that is *not* a mapped file (an object-storage GET, an
//! HTTP range request); see `remote.rs`.
//!
//! Ranges are returned as [`Bytes`] rather than `&[u8]` because a remote source
//! has nothing to borrow from — the bytes only exist once fetched. `Bytes` is
//! refcounted, `Send + Sync + 'static`, and derefs to `[u8]`, so callers that
//! only want a slice keep working unchanged.

use crate::error::{Error, Result};
use bytes::Bytes;
use std::ops::Range;
use std::sync::Arc;

/// A random-access source of bytes for a POD5 file.
///
/// Implementations must be cheap to clone-by-`Arc` and safe to call from many
/// rayon workers at once — `Reader` shares one source across threads.
pub trait ByteSource: Send + Sync {
    /// Total size of the underlying object in bytes.
    fn len(&self) -> u64;

    /// Whether the underlying object is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fetch `len` bytes starting at `offset`.
    ///
    /// Must fail rather than truncate when the request runs past the end of the
    /// object; POD5's structural offsets come from an untrusted footer, and a
    /// short read there would be silently misparsed.
    fn read_range(&self, offset: u64, len: u64) -> Result<Bytes>;

    /// Fetch several ranges at once.
    ///
    /// The default fetches them one at a time. Sources with per-request latency
    /// (object storage) should override this to issue the requests concurrently
    /// and coalesce adjacent ranges.
    fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        ranges
            .iter()
            .map(|r| self.read_range(r.start, r.end.saturating_sub(r.start)))
            .collect()
    }

    /// Hint that `[offset, offset + len)` will be read soon.
    ///
    /// Advisory and best-effort; the default does nothing.
    fn prefetch(&self, _offset: u64, _len: u64) {}

    /// Human-readable identification of this source, for error messages.
    fn describe(&self) -> String;
}

/// Whether a string names a remote object rather than a local path.
///
/// Recognises the schemes a [`RemoteSource`](super::remote::RemoteSource) can
/// address. `file://` is deliberately excluded so local files keep taking the
/// mmap path, and a bare Windows drive letter (`C:\…`) is not a URL.
///
/// Compiled unconditionally, including without the `remote` feature, so a
/// build that cannot service a URL can still recognise one and say *why* it is
/// refusing rather than reporting a missing path. This is the single source of
/// truth for the scheme list.
pub fn is_remote_url(input: &str) -> bool {
    const SCHEMES: [&str; 7] = [
        "s3://", "s3a://", "gs://", "az://", "abfs://", "http://", "https://",
    ];
    SCHEMES.iter().any(|s| input.starts_with(s))
}

/// Bounds check shared by every implementation, so an out-of-range structural
/// offset produces one consistent diagnostic instead of a panic or short read.
pub(crate) fn check_range(offset: u64, len: u64, total: u64, what: &str) -> Result<Range<u64>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFooter(format!("byte range overflows u64 in {what}")))?;
    if end > total {
        return Err(Error::InvalidFooter(format!(
            "byte range {offset}..{end} extends beyond {what} (length {total})"
        )));
    }
    Ok(offset..end)
}

/// Keeps the mapping alive for as long as any [`Bytes`] carved out of it.
///
/// `Bytes::from_owner` needs an owner that is `AsRef<[u8]>`; `Arc<Mmap>` is not,
/// so this thin wrapper supplies the impl while still allowing `MmapSource` to
/// retain its own handle for `madvise`.
struct MmapOwner(Arc<memmap2::Mmap>);

impl AsRef<[u8]> for MmapOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// A [`ByteSource`] over a memory-mapped local file.
///
/// `read_range` slices a `Bytes` view of the mapping, which is a refcount
/// increment — the mapped pages are never copied, so the local read path costs
/// exactly what direct `&mmap[a..b]` slicing did.
pub struct MmapSource {
    /// Retained for `madvise`; the readable bytes come from `whole`. Only the
    /// unix `prefetch` path touches it.
    #[cfg_attr(not(unix), allow(dead_code))]
    mmap: Arc<memmap2::Mmap>,
    whole: Bytes,
    label: String,
}

impl MmapSource {
    /// Wrap an existing mapping.
    pub fn new(mmap: memmap2::Mmap, label: impl Into<String>) -> Self {
        let mmap = Arc::new(mmap);
        let whole = Bytes::from_owner(MmapOwner(Arc::clone(&mmap)));
        Self {
            mmap,
            whole,
            label: label.into(),
        }
    }
}

impl ByteSource for MmapSource {
    fn len(&self) -> u64 {
        self.whole.len() as u64
    }

    fn read_range(&self, offset: u64, len: u64) -> Result<Bytes> {
        let r = check_range(offset, len, self.len(), "file")?;
        Ok(self.whole.slice(r.start as usize..r.end as usize))
    }

    fn prefetch(&self, offset: u64, len: u64) {
        #[cfg(unix)]
        {
            let start = (offset as usize).min(self.whole.len());
            let end = ((offset + len) as usize).min(self.whole.len());
            let _ = self
                .mmap
                .advise_range(memmap2::Advice::WillNeed, start, end - start);
        }
        #[cfg(not(unix))]
        {
            let _ = (offset, len);
        }
    }

    fn describe(&self) -> String {
        self.label.clone()
    }
}

/// A [`ByteSource`] over an in-memory buffer.
///
/// Used by tests and by callers that already hold a complete POD5 image.
pub struct MemorySource {
    data: Bytes,
    label: String,
}

impl MemorySource {
    /// Wrap an owned buffer.
    pub fn new(data: impl Into<Bytes>, label: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            label: label.into(),
        }
    }
}

impl ByteSource for MemorySource {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn read_range(&self, offset: u64, len: u64) -> Result<Bytes> {
        let r = check_range(offset, len, self.len(), "buffer")?;
        Ok(self.data.slice(r.start as usize..r.end as usize))
    }

    fn describe(&self) -> String {
        self.label.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_remote_schemes() {
        assert!(is_remote_url("s3://bucket/reads.pod5"));
        assert!(is_remote_url("gs://bucket/reads.pod5"));
        assert!(is_remote_url("https://example.org/reads.pod5"));
        assert!(!is_remote_url("/data/reads.pod5"));
        assert!(!is_remote_url("reads.pod5"));
        // Local files must keep taking the mmap path.
        assert!(!is_remote_url("file:///data/reads.pod5"));
    }

    #[test]
    fn memory_source_slices_and_bounds_check() {
        let src = MemorySource::new(Bytes::from_static(b"0123456789"), "test");
        assert_eq!(src.len(), 10);
        assert_eq!(&src.read_range(2, 3).unwrap()[..], b"234");
        assert_eq!(&src.read_range(10, 0).unwrap()[..], b"");
        assert!(src.read_range(8, 3).is_err());
        assert!(src.read_range(u64::MAX, 1).is_err());
    }

    #[test]
    fn read_ranges_matches_individual_reads() {
        let src = MemorySource::new(Bytes::from_static(b"abcdefghij"), "test");
        let got = src.read_ranges(&[0..2, 5..8]).unwrap();
        assert_eq!(&got[0][..], b"ab");
        assert_eq!(&got[1][..], b"fgh");
    }

    #[test]
    fn mmap_source_is_zero_copy() {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"0123456789").unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()).unwrap() };
        let base = mmap.as_ptr();
        let src = MmapSource::new(mmap, "tmp");

        let slice = src.read_range(4, 3).unwrap();
        assert_eq!(&slice[..], b"456");
        // The returned Bytes points into the mapping itself, not a copy.
        assert_eq!(slice.as_ptr(), unsafe { base.add(4) });
    }
}

//! Read UUID index for fast lookup by read ID.

use crate::types::Uuid;

/// Magic bytes for the `.p5i` sidecar index file.
pub(super) const P5I_MAGIC: &[u8; 4] = b"P5IX";
/// Current `.p5i` format version.
pub(super) const P5I_VERSION: u8 = 1;

pub struct ReadIndex {
    /// (uuid_bytes, batch_idx, row_idx) sorted by UUID for binary search.
    pub(super) entries: Vec<([u8; 16], u32, u32)>,
}

impl ReadIndex {
    /// Look up a UUID, returning `(batch_idx, row_idx)` if found.
    pub fn get(&self, uuid: &Uuid) -> Option<(usize, usize)> {
        let key = *uuid.as_bytes();
        self.entries
            .binary_search_by_key(&key, |&(k, _, _)| k)
            .ok()
            .map(|i| {
                let (_, batch, row) = self.entries[i];
                (batch as usize, row as usize)
            })
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

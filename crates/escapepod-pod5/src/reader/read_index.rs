//! Read UUID index for fast lookup by read ID.
//!
//! Persisted as the `batch_idx`/`row_idx` columns of the `.p5s` sidecar
//! (see [`crate::sidecar`]); this type is the in-memory, UUID-sorted form.

use crate::types::Uuid;

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

    /// The `(uuid bytes, batch, row)` entries, sorted by UUID.
    pub fn entries(&self) -> &[([u8; 16], u32, u32)] {
        &self.entries
    }

    /// Iterate over all read UUIDs, in UUID order.
    pub fn uuids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.entries
            .iter()
            .map(|&(bytes, _, _)| Uuid::from_bytes(bytes))
    }
}

//! Arrow IPC format parsing for raw byte-level operations.
//!
//! This module provides minimal parsing of Arrow IPC file format to enable
//! raw byte copying of record batches without full deserialization.

use crate::error::{Error, Result};
use rayon::prelude::*;

/// Magic bytes at start and end of Arrow IPC files.
const ARROW_MAGIC: &[u8; 6] = b"ARROW1";

/// Batches below which [`ArrowIpcFooter::read_row_counts`] stays serial.
///
/// The walk it parallelises is latency, not work, so the threshold is about
/// when a fan-out stops being free rather than when it starts paying: a POD5
/// with a handful of batches is a small file whose headers are likely on the
/// same pages anyway.
const ROW_COUNT_PARALLEL_MIN: usize = 16;

/// Whether `ESCAPEPOD_POD5_FOOTER_SERIAL` asks for the pre-parallel walk.
///
/// Read once. The answer cannot change within a process, and this sits on a
/// path that a reader hits per file — one that a `classify` run may open
/// dozens of.
fn footer_walk_forced_serial() -> bool {
    use std::sync::OnceLock;
    static FORCED: OnceLock<bool> = OnceLock::new();
    *FORCED.get_or_init(|| {
        matches!(
            std::env::var("ESCAPEPOD_POD5_FOOTER_SERIAL").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Read a little-endian `i32` at `off`, returning an error instead of panicking
/// if the 4-byte window falls outside `bytes`. Used throughout the hand-rolled
/// parser so that malformed/truncated input propagates `Error::InvalidArrowIpc`
/// rather than indexing out of bounds.
#[inline]
fn read_i32_le(bytes: &[u8], off: usize) -> Result<i32> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| Error::InvalidArrowIpc("i32 read offset overflow".into()))?;
    let slice = bytes
        .get(off..end)
        .ok_or_else(|| Error::InvalidArrowIpc("i32 read out of bounds".into()))?;
    Ok(i32::from_le_bytes(slice.try_into().unwrap()))
}

/// Read a little-endian `u32` at `off`, bounds-checked.
#[inline]
fn read_u32_le(bytes: &[u8], off: usize) -> Result<u32> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| Error::InvalidArrowIpc("u32 read offset overflow".into()))?;
    let slice = bytes
        .get(off..end)
        .ok_or_else(|| Error::InvalidArrowIpc("u32 read out of bounds".into()))?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

/// Read a little-endian `u16` at `off`, bounds-checked.
#[inline]
fn read_u16_le(bytes: &[u8], off: usize) -> Result<u16> {
    let end = off
        .checked_add(2)
        .ok_or_else(|| Error::InvalidArrowIpc("u16 read offset overflow".into()))?;
    let slice = bytes
        .get(off..end)
        .ok_or_else(|| Error::InvalidArrowIpc("u16 read out of bounds".into()))?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

/// Read a little-endian `i64` at `off`, bounds-checked.
#[inline]
fn read_i64_le(bytes: &[u8], off: usize) -> Result<i64> {
    let end = off
        .checked_add(8)
        .ok_or_else(|| Error::InvalidArrowIpc("i64 read offset overflow".into()))?;
    let slice = bytes
        .get(off..end)
        .ok_or_else(|| Error::InvalidArrowIpc("i64 read out of bounds".into()))?;
    Ok(i64::from_le_bytes(slice.try_into().unwrap()))
}

/// Borrow a `len`-byte slice starting at `off`, returning an error instead of
/// panicking if the window falls outside `bytes`.
#[inline]
fn slice_at(bytes: &[u8], off: usize, len: usize) -> Result<&[u8]> {
    let end = off
        .checked_add(len)
        .ok_or_else(|| Error::InvalidArrowIpc("slice offset overflow".into()))?;
    bytes
        .get(off..end)
        .ok_or_else(|| Error::InvalidArrowIpc("slice out of bounds".into()))
}

/// Resolve a FlatBuffer signed-offset subtraction (`base - soffset`) into a
/// `usize` position, erroring on negative/overflow results rather than wrapping.
#[inline]
fn fb_sub_offset(base: usize, soffset: i32) -> Result<usize> {
    let pos = (base as i64) - (soffset as i64);
    if pos < 0 {
        return Err(Error::InvalidArrowIpc(
            "Invalid negative FlatBuffer offset".into(),
        ));
    }
    Ok(pos as usize)
}

/// Location of a record batch within an Arrow IPC stream.
#[derive(Debug, Clone, Copy)]
pub struct BatchBlock {
    /// Byte offset from start of IPC stream.
    pub offset: i64,
    /// Size of the metadata section (includes padding).
    pub metadata_length: i32,
    /// Size of the body section (data buffers).
    pub body_length: i64,
    /// Number of rows in this batch (parsed from message header).
    pub row_count: u64,
}

impl BatchBlock {
    /// Total size of this batch in bytes (metadata + body).
    pub fn total_length(&self) -> i64 {
        self.metadata_length as i64 + self.body_length
    }

    /// Byte range for this batch within the IPC stream.
    pub fn byte_range(&self) -> std::ops::Range<usize> {
        let start = self.offset as usize;
        let end = start + self.total_length() as usize;
        start..end
    }
}

/// Parsed Arrow IPC footer with batch locations.
#[derive(Debug, Clone)]
pub struct ArrowIpcFooter {
    /// Locations of record batches (signal data).
    pub record_batches: Vec<BatchBlock>,
    /// Byte offset where the first batch starts (end of header/schema).
    pub batches_start_offset: usize,
    /// Byte offset where the last batch ends (start of footer).
    pub batches_end_offset: usize,
    /// Total number of rows across all batches.
    pub total_rows: u64,
    /// Prefix sum of row counts: `cumulative_rows[i]` is the total rows in
    /// batches `0..i`. Length `record_batches.len() + 1`. Enables O(log n)
    /// row→batch lookup via binary search.
    cumulative_rows: Vec<u64>,
}

impl ArrowIpcFooter {
    /// Parse the Arrow IPC footer from a complete IPC byte stream.
    ///
    /// The IPC file format is:
    /// ```text
    /// [ARROW1\0\0]           8 bytes magic (with padding at start)
    /// [schema message]       variable
    /// [record batches...]    variable
    /// [footer flatbuffer]    variable
    /// [footer length]        4 bytes (i32 LE)
    /// [ARROW1]               6 bytes magic (no padding at end)
    /// ```
    pub fn parse(ipc_bytes: &[u8]) -> Result<Self> {
        Self::parse_with_row_counts(ipc_bytes, None)
    }

    /// Like [`Self::parse`], but taking the per-batch row counts if a caller
    /// already knows them.
    ///
    /// Row count is the one field the IPC footer does not carry: the footer's
    /// `Block` records only offset, metadata length and body length, so
    /// recovering the count means reading every batch's own message header.
    /// That is one scattered touch per batch — measured at 15-24 ms each cold
    /// on BeeGFS, and the single largest removable cost in a scattered fetch.
    /// `known` short-circuits it; escapepod supplies it from the `.p5s`
    /// sidecar, which measured the same walk once (see
    /// `crate::sidecar::P5S_SIGNAL_BATCH_ROWS_KEY`).
    ///
    /// **`known` is checked, not believed.** It is used only if it has exactly
    /// one entry per batch the footer describes, and the first and last
    /// batches are then read for real and compared. Those two are the ones
    /// that matter: the last is the short one, so a geometry belonging to a
    /// file with a different read count disagrees there, and the first pins
    /// the stride. Any mismatch discards `known` and walks the batches, so a
    /// stale or corrupt cache costs time and never correctness.
    pub fn parse_with_row_counts(ipc_bytes: &[u8], known: Option<&[u64]>) -> Result<Self> {
        let len = ipc_bytes.len();

        // Need at least magic (8) + footer_len (4) + magic (6) = 18 bytes minimum
        if len < 18 {
            return Err(Error::InvalidArrowIpc("IPC stream too small".into()));
        }

        // Verify trailing magic (6 bytes at end, no padding)
        if slice_at(ipc_bytes, len - 6, 6)? != ARROW_MAGIC {
            return Err(Error::InvalidArrowIpc(
                "Missing trailing ARROW1 magic".into(),
            ));
        }

        // Read footer length (4 bytes immediately before trailing magic)
        let footer_len_offset = len - 6 - 4; // 6 = magic, 4 = footer_len
        let footer_len = read_i32_le(ipc_bytes, footer_len_offset)?;

        // Handle continuation indicator (negative means flatbuffer follows)
        let footer_len = if footer_len < 0 {
            // Read actual length after continuation marker
            let actual_len_offset = footer_len_offset + 4;
            if actual_len_offset + 4 > len - 6 {
                return Err(Error::InvalidArrowIpc("Invalid continuation marker".into()));
            }
            read_i32_le(ipc_bytes, actual_len_offset)?
        } else {
            footer_len
        };

        if footer_len <= 0 || footer_len as usize > len - 18 {
            return Err(Error::InvalidArrowIpc(format!(
                "Invalid footer length: {}",
                footer_len
            )));
        }

        // Locate footer FlatBuffer
        let footer_start = len - 6 - 4 - footer_len as usize;
        let footer_bytes = slice_at(ipc_bytes, footer_start, footer_len as usize)?;

        // Parse the footer FlatBuffer to extract batch blocks
        Self::parse_footer_flatbuffer(footer_bytes, ipc_bytes, known)
    }

    /// Parse the Footer FlatBuffer to extract record batch locations.
    fn parse_footer_flatbuffer(
        footer_bytes: &[u8],
        full_ipc: &[u8],
        known: Option<&[u64]>,
    ) -> Result<Self> {
        // FlatBuffer footer structure (simplified):
        // - Root table offset (4 bytes from start)
        // - Footer table with:
        //   - vtable offset
        //   - version (i16)
        //   - schema offset
        //   - dictionaries vector offset
        //   - recordBatches vector offset

        if footer_bytes.len() < 8 {
            return Err(Error::InvalidArrowIpc("Footer too small".into()));
        }

        // Read root table offset
        let root_offset = read_u32_le(footer_bytes, 0)? as usize;
        if root_offset >= footer_bytes.len() {
            return Err(Error::InvalidArrowIpc("Invalid root offset".into()));
        }

        let table_start = root_offset;

        // Read vtable offset (signed, relative to table_start)
        let vtable_soffset = read_i32_le(footer_bytes, table_start)?;
        let vtable_pos = fb_sub_offset(table_start, vtable_soffset)?;

        if vtable_pos + 4 > footer_bytes.len() {
            return Err(Error::InvalidArrowIpc("Invalid vtable position".into()));
        }

        // Read vtable size
        let vtable_size = read_u16_le(footer_bytes, vtable_pos)? as usize;

        if vtable_size < 10 {
            return Err(Error::InvalidArrowIpc("Vtable too small".into()));
        }

        // Footer vtable layout (after size and table_size):
        // offset 4: version (i16)
        // offset 6: schema offset
        // offset 8: dictionaries offset
        // offset 10: recordBatches offset

        // Get recordBatches vector offset (if vtable has it)
        let record_batches_field_offset = if vtable_size >= 12 {
            let offset_in_vtable = read_u16_le(footer_bytes, vtable_pos + 10)?;
            if offset_in_vtable > 0 {
                Some(offset_in_vtable as usize)
            } else {
                None
            }
        } else {
            None
        };

        let mut record_batches = Vec::new();

        if let Some(field_offset) = record_batches_field_offset {
            // Read vector offset from table
            let vec_offset_pos = table_start + field_offset;
            if vec_offset_pos + 4 <= footer_bytes.len() {
                let vec_offset = read_u32_le(footer_bytes, vec_offset_pos)? as usize;
                let vec_pos = vec_offset_pos + vec_offset;

                if vec_pos + 4 <= footer_bytes.len() {
                    // Read vector length
                    let vec_len = read_u32_le(footer_bytes, vec_pos)? as usize;

                    // Each Block is a struct with: offset(i64) + metaDataLength(i32) + padding(i32) + bodyLength(i64) = 24 bytes
                    // Actually in FlatBuffers, Block struct is: offset(8) + metaDataLength(4) + bodyLength(8) = 20 bytes
                    // But may have alignment padding
                    let block_size = 24; // Aligned size
                    let blocks_start = vec_pos + 4;

                    // Pass one: the block descriptors. Every one of them is a
                    // fixed-size struct inside the footer, which the tail read
                    // above already made resident, so this pass touches
                    // nothing scattered and nothing new.
                    for i in 0..vec_len {
                        let block_pos = blocks_start + i * block_size;
                        if block_pos + 20 > footer_bytes.len() {
                            break;
                        }

                        record_batches.push(BatchBlock {
                            offset: read_i64_le(footer_bytes, block_pos)?,
                            metadata_length: read_i32_le(footer_bytes, block_pos + 8)?,
                            // Skip 4 bytes padding
                            body_length: read_i64_le(footer_bytes, block_pos + 16)?,
                            // Filled in by pass two.
                            row_count: 0,
                        });
                    }

                    // Pass two: the row counts, the one field the footer does
                    // not carry. Supplied per batch where a caller knows them
                    // — which is the whole point, this read is the scattered
                    // one — and read for the rest, together rather than one
                    // after another.
                    let unknown: Vec<(usize, i64)> = record_batches
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| known.and_then(|k| k.get(*i)).is_none())
                        .map(|(i, b)| (i, b.offset))
                        .collect();
                    for (i, b) in record_batches.iter_mut().enumerate() {
                        if let Some(rows) = known.and_then(|k| k.get(i).copied()) {
                            b.row_count = rows;
                        }
                    }
                    for (i, rows) in Self::read_row_counts(full_ipc, &unknown)? {
                        record_batches[i].row_count = rows;
                    }
                }
            }
        }

        // Validate a supplied geometry before anything is derived from it, and
        // fall back to the real walk if it does not hold up. Two reads, not N:
        // the last batch is the short one, so a geometry from a file with a
        // different read count disagrees there, and the first pins the stride.
        if let Some(known) = known {
            let mut ok = known.len() == record_batches.len();
            if ok {
                for idx in [0usize, record_batches.len().saturating_sub(1)] {
                    let Some(block) = record_batches.get(idx) else {
                        continue;
                    };
                    let actual = Self::parse_batch_row_count(full_ipc, block.offset as usize)?;
                    if actual != block.row_count {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                tracing::warn!(
                    cached_batches = known.len(),
                    actual_batches = record_batches.len(),
                    "cached signal batch geometry does not match this file; \
                     ignoring it and reading the batch headers (rebuild with \
                     `escpod index`)"
                );
                return Self::parse_footer_flatbuffer(footer_bytes, full_ipc, None);
            }
        }

        // Compute offsets and total rows
        let batches_start_offset = record_batches
            .first()
            .map(|b| b.offset as usize)
            .unwrap_or(8);
        let batches_end_offset = record_batches
            .last()
            .map(|b| (b.offset + b.total_length()) as usize)
            .unwrap_or(batches_start_offset);
        let mut cumulative_rows = Vec::with_capacity(record_batches.len() + 1);
        cumulative_rows.push(0);
        let mut running = 0u64;
        for b in &record_batches {
            running += b.row_count;
            cumulative_rows.push(running);
        }
        let total_rows = running;

        Ok(ArrowIpcFooter {
            record_batches,
            batches_start_offset,
            batches_end_offset,
            total_rows,
            cumulative_rows,
        })
    }

    /// Read the row counts of the given `(index, offset)` batches, together.
    ///
    /// [`Self::parse_batch_row_count`] is a pure read of one message header at
    /// an offset the footer already gave us, and **every offset is in hand
    /// before the first header is read**. Nothing about this walk is
    /// sequential except that it was written as a loop.
    ///
    /// On a local file that distinction does not matter. On a network
    /// filesystem it is the whole cost: each header is a cold page fault
    /// costing a server round trip — 15-24 ms measured on BeeGFS — and a large
    /// POD5 has thousands of batches, so doing them one after another spends
    /// minutes accumulating a latency that could have overlapped. Measured
    /// against escapepod-rs#334's file set, opening one 6.3 GB POD5 and taking
    /// a signal extractor from it cost ~10 s of which ~97% was this loop
    /// blocked on faults at essentially zero CPU. That is the "slow to get
    /// going" stall, and it is not bandwidth: the walk moves a few KB per
    /// batch.
    ///
    /// So issue them together. The bytes read are identical, the row counts
    /// are identical, and the order they are *collected* in is fixed by the
    /// index each carries rather than by which thread finished first — this
    /// returns pairs, not a bare vector, so a caller cannot accidentally
    /// depend on completion order.
    ///
    /// Below [`ROW_COUNT_PARALLEL_MIN`] batches it stays serial: a handful of
    /// round trips is not worth a fan-out, and a small file is exactly where
    /// that overhead would show. `ESCAPEPOD_POD5_FOOTER_SERIAL=1` forces the
    /// serial path at any size, which is the A/B lever and the way back.
    fn read_row_counts(full_ipc: &[u8], wanted: &[(usize, i64)]) -> Result<Vec<(usize, u64)>> {
        let read_one = |&(i, off): &(usize, i64)| {
            Ok((i, Self::parse_batch_row_count(full_ipc, off as usize)?))
        };

        if wanted.len() < ROW_COUNT_PARALLEL_MIN || footer_walk_forced_serial() {
            return wanted.iter().map(read_one).collect();
        }
        wanted.par_iter().map(read_one).collect()
    }

    /// Parse the row count from a RecordBatch message at the given offset.
    /// The message format is:
    /// - [continuation: -1 as i32] (IPC v2) or [message_length: i32]
    /// - [metadata_length: i32] (if continuation)
    /// - [metadata FlatBuffer] - contains RecordBatch with length field
    fn parse_batch_row_count(ipc_bytes: &[u8], offset: usize) -> Result<u64> {
        if offset + 8 > ipc_bytes.len() {
            return Err(Error::InvalidArrowIpc("Batch offset out of bounds".into()));
        }

        // Read first 4 bytes - could be continuation marker (-1) or message length
        let first_word = read_i32_le(ipc_bytes, offset)?;

        let metadata_start = if first_word == -1 {
            // IPC v2 format: continuation marker followed by metadata length
            offset + 8 // Skip continuation (-1) and metadata_length
        } else {
            // IPC v1 format: first word is metadata length
            offset + 4
        };

        if metadata_start + 4 > ipc_bytes.len() {
            return Err(Error::InvalidArrowIpc(
                "Metadata offset out of bounds".into(),
            ));
        }

        // The metadata is a Message FlatBuffer. We need to navigate to RecordBatch.length.
        // Message table: version(i16), header_type(byte), header(union), bodyLength(i64)
        // The header union points to a RecordBatch table which has: length(i64), nodes, buffers
        let metadata = &ipc_bytes[metadata_start..];

        // Read root table offset
        if metadata.len() < 4 {
            return Err(Error::InvalidArrowIpc("Metadata too small".into()));
        }
        let root_offset = read_u32_le(metadata, 0)? as usize;
        if root_offset >= metadata.len() {
            return Err(Error::InvalidArrowIpc("Invalid message root offset".into()));
        }

        // Read vtable offset from root table
        let vtable_soffset = read_i32_le(metadata, root_offset)?;
        let vtable_pos = fb_sub_offset(root_offset, vtable_soffset)?;

        if vtable_pos + 10 > metadata.len() {
            return Err(Error::InvalidArrowIpc("Invalid message vtable".into()));
        }

        // Message vtable: size(2), table_size(2), version(2), header_type(2), header(2), bodyLength(2)
        // We need header offset at vtable_pos + 8
        let header_field_offset = read_u16_le(metadata, vtable_pos + 8)? as usize;

        if header_field_offset == 0 {
            return Err(Error::InvalidArrowIpc("No header in message".into()));
        }

        // Read header table offset (union value)
        let header_offset_pos = root_offset + header_field_offset;
        let header_offset = read_u32_le(metadata, header_offset_pos)? as usize;
        let header_table_pos = header_offset_pos + header_offset;

        // Now we're at the RecordBatch table
        // RecordBatch vtable: size(2), table_size(2), length(2), nodes(2), buffers(2)
        let rb_vtable_soffset = read_i32_le(metadata, header_table_pos)?;
        let rb_vtable_pos = fb_sub_offset(header_table_pos, rb_vtable_soffset)?;

        if rb_vtable_pos + 6 > metadata.len() {
            return Err(Error::InvalidArrowIpc(
                "RecordBatch vtable out of bounds".into(),
            ));
        }

        // Read length field offset (first field after size and table_size)
        let length_field_offset = read_u16_le(metadata, rb_vtable_pos + 4)? as usize;

        if length_field_offset == 0 {
            // No length field, assume 0 rows
            return Ok(0);
        }

        // Read the length value (i64)
        let length_pos = header_table_pos + length_field_offset;
        let length = read_i64_le(metadata, length_pos)?;
        Ok(length as u64)
    }

    /// Get the raw bytes for the IPC header (magic + schema message).
    /// These bytes should be written once at the start of the output.
    pub fn header_bytes<'a>(&self, ipc_bytes: &'a [u8]) -> &'a [u8] {
        &ipc_bytes[..self.batches_start_offset]
    }

    /// Get the raw bytes for all record batches.
    /// These bytes can be copied directly to the output.
    pub fn batches_bytes<'a>(&self, ipc_bytes: &'a [u8]) -> &'a [u8] {
        &ipc_bytes[self.batches_start_offset..self.batches_end_offset]
    }

    /// Get the raw bytes for a single record batch by index.
    /// Returns the slice containing the batch's metadata and body.
    pub fn batch_bytes<'a>(&self, batch_idx: usize, ipc_bytes: &'a [u8]) -> &'a [u8] {
        let batch = &self.record_batches[batch_idx];
        &ipc_bytes[batch.byte_range()]
    }

    /// Get the first signal row index for a given batch.
    /// Returns the cumulative row count before this batch.
    pub fn batch_first_row(&self, batch_idx: usize) -> u64 {
        self.cumulative_rows[batch_idx]
    }

    /// Find which batch contains a given signal row.
    /// Returns (batch_idx, row_within_batch). O(log n) via prefix-sum
    /// binary search.
    pub fn batch_for_row(&self, signal_row: u64) -> Option<(usize, u64)> {
        if signal_row >= self.total_rows {
            return None;
        }
        // Rightmost index with cumulative_rows[idx] <= signal_row.
        // `partition_point` returns the first index where the predicate
        // fails, so subtract 1 to get the containing batch.
        let idx = self.cumulative_rows.partition_point(|&c| c <= signal_row) - 1;
        Some((idx, signal_row - self.cumulative_rows[idx]))
    }

    /// Extract a single compressed signal chunk from raw IPC bytes.
    ///
    /// This bypasses Arrow deserialization entirely, parsing the IPC format
    /// directly to extract the signal bytes for a specific row.
    ///
    /// Signal table columns: read_id (FixedSizeBinary[16]), signal (LargeBinary), samples (UInt32)
    pub fn extract_signal_row<'a>(
        &self,
        signal_row: u64,
        ipc_bytes: &'a [u8],
    ) -> Result<RawSignalChunk<'a>> {
        let (batch_idx, local_row) = self.batch_for_row(signal_row).ok_or_else(|| {
            Error::InvalidState(format!("Signal row {} out of bounds", signal_row))
        })?;

        let batch = &self.record_batches[batch_idx];
        let batch_bytes = ipc_bytes
            .get(batch.byte_range())
            .ok_or_else(|| Error::InvalidArrowIpc("Batch range out of bounds".into()))?;
        let num_rows = batch.row_count as usize;

        // Parse the batch to extract buffers
        let parsed = ParsedBatch::parse(batch_bytes, batch.metadata_length as usize, num_rows)?;

        // Extract the signal data for this row
        let row = local_row as usize;

        // Read UUID (16 bytes at row offset)
        let read_id_offset = row * 16;
        let read_id_bytes: [u8; 16] = slice_at(parsed.read_id_data, read_id_offset, 16)?
            .try_into()
            .map_err(|_| Error::InvalidState("Invalid read_id length".into()))?;

        // Read signal offsets (i64 array, row and row+1)
        let offset_start = row * 8;
        let offset_end = (row + 1) * 8;
        let signal_start = read_i64_le(parsed.signal_offsets, offset_start)? as usize;
        let signal_end = read_i64_le(parsed.signal_offsets, offset_end)? as usize;

        let signal_bytes = parsed
            .signal_data
            .get(signal_start..signal_end)
            .ok_or_else(|| Error::InvalidState("signal data out of bounds".into()))?;

        // Read samples count (u32)
        let samples_offset = row * 4;
        let samples = read_u32_le(parsed.samples_data, samples_offset)?;

        Ok(RawSignalChunk {
            read_id: read_id_bytes,
            signal: signal_bytes,
            samples,
        })
    }

    /// Extract multiple signal rows efficiently, grouping by batch.
    pub fn extract_signal_rows<'a>(
        &self,
        signal_rows: &[u64],
        ipc_bytes: &'a [u8],
    ) -> Result<Vec<RawSignalChunk<'a>>> {
        // Group rows by batch for efficiency
        let mut batch_rows: std::collections::BTreeMap<usize, Vec<(usize, u64)>> =
            std::collections::BTreeMap::new();

        for (result_idx, &row) in signal_rows.iter().enumerate() {
            if let Some((batch_idx, local_row)) = self.batch_for_row(row) {
                batch_rows
                    .entry(batch_idx)
                    .or_default()
                    .push((result_idx, local_row));
            }
        }

        // Parse batches in parallel. Each batch is an independent IPC block
        // (`ParsedBatch::parse` only reads offset arrays into the mmap — no
        // decompression), and its rows scatter into disjoint `result_idx`
        // slots, so there is no cross-batch dependency. On a single-file
        // filter this is the phase that moves the bulk data, so parallelizing
        // here is what lets one file use more than one core. Nested under the
        // caller's per-file `par_iter`, the combined width is bounded by the
        // global rayon pool (cap it with the CLI `-t`).
        let batch_entries: Vec<(usize, Vec<(usize, u64)>)> = batch_rows.into_iter().collect();

        let per_batch: Vec<Vec<(usize, RawSignalChunk<'a>)>> = batch_entries
            .into_par_iter()
            .map(
                |(batch_idx, mut rows)| -> Result<Vec<(usize, RawSignalChunk<'a>)>> {
                    // Visit rows within a batch in ascending order. The
                    // grouping above only puts the *batches* in file order —
                    // the rows inside one are still in request order, and a
                    // request assembled by iterating a `HashSet<Uuid>` (which
                    // is what `reads_by_ids` and `find_signal_rows_by_ids`
                    // hand down) is in no order at all. A batch is megabytes
                    // of contiguous mmap, so hopping around inside it defeats
                    // kernel readahead for precisely the reason the batch
                    // grouping exists. The scatter into disjoint `result_idx`
                    // slots below already decouples output order from visit
                    // order, so this costs nothing but the sort.
                    rows.sort_unstable_by_key(|&(_, local_row)| local_row);

                    let batch = &self.record_batches[batch_idx];
                    let batch_bytes = ipc_bytes.get(batch.byte_range()).ok_or_else(|| {
                        Error::InvalidArrowIpc("Batch range out of bounds".into())
                    })?;
                    let num_rows = batch.row_count as usize;

                    let parsed =
                        ParsedBatch::parse(batch_bytes, batch.metadata_length as usize, num_rows)?;

                    let mut out = Vec::with_capacity(rows.len());
                    for (result_idx, local_row) in rows {
                        let row = local_row as usize;

                        // Extract read_id
                        let read_id_offset = row * 16;
                        let read_id_bytes: [u8; 16] =
                            slice_at(parsed.read_id_data, read_id_offset, 16)?
                                .try_into()
                                .map_err(|_| Error::InvalidState("Invalid read_id".into()))?;

                        // Extract signal
                        let offset_start = row * 8;
                        let offset_end = (row + 1) * 8;
                        let signal_start =
                            read_i64_le(parsed.signal_offsets, offset_start)? as usize;
                        let signal_end = read_i64_le(parsed.signal_offsets, offset_end)? as usize;
                        let signal_bytes = parsed
                            .signal_data
                            .get(signal_start..signal_end)
                            .ok_or_else(|| {
                                Error::InvalidState("signal data out of bounds".into())
                            })?;

                        // Extract samples
                        let samples_offset = row * 4;
                        let samples = read_u32_le(parsed.samples_data, samples_offset)?;

                        out.push((
                            result_idx,
                            RawSignalChunk {
                                read_id: read_id_bytes,
                                signal: signal_bytes,
                                samples,
                            },
                        ));
                    }
                    Ok(out)
                },
            )
            .collect::<Result<Vec<_>>>()?;

        // Scatter each batch's rows back into original request order. Slots are
        // disjoint across batches, so a plain sequential scatter is correct.
        let mut results: Vec<Option<RawSignalChunk<'a>>> = vec![None; signal_rows.len()];
        for chunk_list in per_batch {
            for (result_idx, chunk) in chunk_list {
                results[result_idx] = Some(chunk);
            }
        }

        // Convert to vec, filtering out any missing
        Ok(results.into_iter().flatten().collect())
    }
}

/// A raw signal chunk extracted directly from IPC bytes (no deserialization).
#[derive(Debug, Clone)]
pub struct RawSignalChunk<'a> {
    /// The read ID (16 bytes UUID).
    pub read_id: [u8; 16],
    /// The compressed VBZ signal data (borrowed from mmap).
    pub signal: &'a [u8],
    /// Number of samples in this chunk.
    pub samples: u32,
}

/// Parsed buffer locations from a signal batch.
struct ParsedBatch<'a> {
    read_id_data: &'a [u8],
    signal_offsets: &'a [u8],
    signal_data: &'a [u8],
    samples_data: &'a [u8],
}

impl<'a> ParsedBatch<'a> {
    /// Parse a signal table batch to extract buffer locations.
    ///
    /// Signal table schema: read_id (FixedSizeBinary[16]), signal (LargeBinary), samples (UInt32)
    /// Expected buffers:
    /// - 0: read_id validity (may be empty/null)
    /// - 1: read_id data (16 bytes * num_rows)
    /// - 2: signal validity (may be empty/null)
    /// - 3: signal offsets (8 bytes * (num_rows + 1))
    /// - 4: signal data (variable)
    /// - 5: samples validity (may be empty/null)
    /// - 6: samples data (4 bytes * num_rows)
    fn parse(batch_bytes: &'a [u8], _metadata_length: usize, num_rows: usize) -> Result<Self> {
        // Skip the message header to get to the body
        // Message format: [4 bytes: continuation or length][4 bytes: metadata_length if continuation]
        // Then metadata, then padding, then body

        let first_word = read_i32_le(batch_bytes, 0)?;
        let (metadata_start, actual_metadata_len): (usize, usize) = if first_word == -1 {
            // Continuation marker, actual length follows
            let len = read_i32_le(batch_bytes, 4)? as usize;
            (8, len)
        } else {
            (4, first_word.max(0) as usize)
        };

        // Body starts after metadata + padding to 8-byte boundary
        let metadata_end = metadata_start
            .checked_add(actual_metadata_len)
            .ok_or_else(|| Error::InvalidArrowIpc("Metadata length overflow".into()))?;
        let padded_metadata_end = (metadata_end + 7) & !7; // Round up to 8
        let body_start = padded_metadata_end;
        let body = batch_bytes
            .get(body_start..)
            .ok_or_else(|| Error::InvalidArrowIpc("Batch body out of bounds".into()))?;

        // Parse the metadata to get buffer offsets
        // The RecordBatch message has a buffers vector with offset and length for each buffer
        let metadata = slice_at(batch_bytes, metadata_start, actual_metadata_len)?;
        let buffer_infos = Self::parse_buffer_infos(metadata)?;

        // Buffer order is fixed by the schema (see the doc comment above), not
        // by size: an uncompressed batch whose signal data happens to be the
        // same length as the samples column (e.g. 2 i16 samples/row = 4
        // bytes/row) is ambiguous under a size match. Index by position.
        if buffer_infos.len() < 7 {
            return Err(Error::InvalidArrowIpc(format!(
                "Expected 7 signal-table buffers, found {}",
                buffer_infos.len()
            )));
        }

        let (read_id_off, read_id_len) = buffer_infos[1];
        let (signal_offsets_off, signal_offsets_len) = buffer_infos[3];
        let (signal_data_off, signal_data_len) = buffer_infos[4];
        let (samples_off, samples_len) = buffer_infos[6];

        let read_id_data = slice_at(body, read_id_off, read_id_len)?;
        let signal_offsets = slice_at(body, signal_offsets_off, signal_offsets_len)?;
        let signal_data = slice_at(body, signal_data_off, signal_data_len)?;
        let samples_data = slice_at(body, samples_off, samples_len)?;

        let read_id_size = 16 * num_rows;
        let signal_offsets_size = 8 * (num_rows + 1);
        let samples_size = 4 * num_rows;

        if read_id_data.len() != read_id_size
            || signal_offsets.len() != signal_offsets_size
            || samples_data.len() != samples_size
        {
            return Err(Error::InvalidArrowIpc(format!(
                "Unexpected signal-table buffer sizes. Found: read_id={} (want {}), offsets={} (want {}), samples={} (want {})",
                read_id_data.len(),
                read_id_size,
                signal_offsets.len(),
                signal_offsets_size,
                samples_data.len(),
                samples_size
            )));
        }

        Ok(ParsedBatch {
            read_id_data,
            signal_offsets,
            signal_data,
            samples_data,
        })
    }

    /// Parse buffer info (offset, length) from RecordBatch metadata.
    fn parse_buffer_infos(metadata: &[u8]) -> Result<Vec<(usize, usize)>> {
        if metadata.len() < 4 {
            return Err(Error::InvalidArrowIpc("Metadata too small".into()));
        }

        // Root table offset
        let root_offset = read_u32_le(metadata, 0)? as usize;
        if root_offset >= metadata.len() {
            return Err(Error::InvalidArrowIpc("Invalid root offset".into()));
        }

        // Navigate to Message -> header (RecordBatch) -> buffers
        let vtable_soffset = read_i32_le(metadata, root_offset)?;
        let vtable_pos = fb_sub_offset(root_offset, vtable_soffset)?;

        if vtable_pos + 10 > metadata.len() {
            return Err(Error::InvalidArrowIpc("Invalid vtable".into()));
        }

        // Message vtable: size(2), table_size(2), version(2), header_type(2), header(2), bodyLength(2)
        let header_field_offset = read_u16_le(metadata, vtable_pos + 8)? as usize;

        if header_field_offset == 0 {
            return Ok(Vec::new());
        }

        // Navigate to RecordBatch table
        let header_offset_pos = root_offset + header_field_offset;
        let header_offset = read_u32_le(metadata, header_offset_pos)? as usize;
        let rb_table_pos = header_offset_pos + header_offset;

        // RecordBatch vtable
        let rb_vtable_soffset = read_i32_le(metadata, rb_table_pos)?;
        let rb_vtable_pos = fb_sub_offset(rb_table_pos, rb_vtable_soffset)?;

        if rb_vtable_pos + 8 > metadata.len() {
            return Err(Error::InvalidArrowIpc(
                "RecordBatch vtable too small".into(),
            ));
        }

        let rb_vtable_size = read_u16_le(metadata, rb_vtable_pos)? as usize;

        // RecordBatch vtable: size(2), table_size(2), length(2), nodes(2), buffers(2)
        // buffers is at offset 8 in vtable
        if rb_vtable_size < 10 {
            return Ok(Vec::new());
        }

        let buffers_field_offset = read_u16_le(metadata, rb_vtable_pos + 8)? as usize;

        if buffers_field_offset == 0 {
            return Ok(Vec::new());
        }

        // Navigate to buffers vector
        let buffers_offset_pos = rb_table_pos + buffers_field_offset;
        if buffers_offset_pos + 4 > metadata.len() {
            return Ok(Vec::new());
        }

        let buffers_offset = read_u32_le(metadata, buffers_offset_pos)? as usize;
        let buffers_vec_pos = buffers_offset_pos + buffers_offset;

        if buffers_vec_pos + 4 > metadata.len() {
            return Ok(Vec::new());
        }

        let num_buffers = read_u32_le(metadata, buffers_vec_pos)? as usize;

        // Each Buffer struct is 16 bytes: offset(i64) + length(i64)
        let mut buffers = Vec::with_capacity(num_buffers.min(metadata.len() / 16 + 1));
        let buffers_data_start = buffers_vec_pos + 4;

        for i in 0..num_buffers {
            let buf_pos = buffers_data_start + i * 16;
            if buf_pos + 16 > metadata.len() {
                break;
            }

            let offset = read_i64_le(metadata, buf_pos)? as usize;
            let length = read_i64_le(metadata, buf_pos + 8)? as usize;
            buffers.push((offset, length));
        }

        Ok(buffers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_block_range() {
        let block = BatchBlock {
            offset: 100,
            metadata_length: 50,
            body_length: 200,
            row_count: 10,
        };
        assert_eq!(block.total_length(), 250);
        assert_eq!(block.byte_range(), 100..350);
    }

    /// A POD5 with more signal batches than [`ROW_COUNT_PARALLEL_MIN`], so
    /// the walk under test actually fans out.
    ///
    /// Written here rather than taken from `ext/`, because the fixtures there
    /// are a submodule CI does not check out — a test that needs one is a test
    /// that never runs (escapepod-rs, `.gitmodules` is empty).
    fn many_batch_pod5(dir: &std::path::Path) -> std::path::PathBuf {
        use crate::types::{EndReason, ReadData, RunInfoData};
        use crate::{Writer, WriterOptions};

        let path = dir.join("many_batches.pod5");
        let mut writer = Writer::create(
            &path,
            WriterOptions {
                // Two rows a batch, so 61 reads make 30 full batches and a
                // final short one — a geometry a shuffled count breaks.
                signal_batch_size: 2,
                ..Default::default()
            },
        )
        .expect("writer::create");

        let run_idx = writer
            .add_run_info(RunInfoData {
                acquisition_id: "row_counts".into(),
                acquisition_start_time: 1_609_459_200_000,
                adc_max: 2047,
                adc_min: -2048,
                sample_rate: 4_000,
                ..Default::default()
            })
            .expect("add_run_info");

        for i in 0..61u32 {
            let read = ReadData {
                read_id: crate::Uuid::new_v4(),
                read_number: i + 1,
                start_sample: i as u64 * 100,
                channel: 1,
                well: 1,
                pore_type: "not_set".into(),
                calibration_offset: 0.5,
                calibration_scale: 0.95,
                median_before: 200.0,
                end_reason: EndReason::SignalPositive,
                end_reason_forced: false,
                run_info_index: run_idx,
                num_minknow_events: 100,
                tracked_scaling_scale: 1.0,
                tracked_scaling_shift: 0.0,
                predicted_scaling_scale: 1.0,
                predicted_scaling_shift: 0.0,
                num_reads_since_mux_change: 0,
                time_since_mux_change: 0.0,
                num_samples: 100,
                open_pore_level: 220.0,
                expected_open_pore_level: 0.0,
                selected_read_level: 0.0,
                signal_rows: Vec::new(),
            };
            let signal: Vec<i16> = (0..100).map(|s| (s * 7 + i as i32) as i16).collect();
            writer.add_read(read, &signal).expect("add_read");
        }
        writer.finish().expect("finish");
        path
    }

    /// `ParsedBatch::parse` must locate signal-table buffers by their fixed
    /// schema position, not by matching their byte length: an uncompressed
    /// read with exactly 2 i16 samples/row is 4 bytes/row, the same width as
    /// the `samples: u32` column, so a size match cannot tell them apart.
    #[test]
    fn test_parse_uncompressed_batch_buffer_positions() {
        use crate::types::{EndReason, ReadData, RunInfoData};
        use crate::{Reader, Writer, WriterOptions};

        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("collide.pod5");

        let mut writer = Writer::create(
            &path,
            WriterOptions {
                signal_batch_size: 2,
                compress_signal: false,
                ..Default::default()
            },
        )
        .expect("writer::create");

        let run_idx = writer
            .add_run_info(RunInfoData {
                acquisition_id: "collide".into(),
                acquisition_start_time: 1_609_459_200_000,
                adc_max: 2047,
                adc_min: -2048,
                sample_rate: 4_000,
                ..Default::default()
            })
            .expect("add_run_info");

        let signals: [[i16; 2]; 2] = [[10, 20], [30, 40]];
        let mut read_ids = Vec::new();
        for (i, samples) in signals.iter().enumerate() {
            let read_id = crate::Uuid::new_v4();
            read_ids.push(read_id);
            let read = ReadData {
                read_id,
                read_number: i as u32 + 1,
                start_sample: i as u64 * 2,
                channel: 1,
                well: 1,
                pore_type: "not_set".into(),
                calibration_offset: 0.5,
                calibration_scale: 0.95,
                median_before: 200.0,
                end_reason: EndReason::SignalPositive,
                end_reason_forced: false,
                run_info_index: run_idx,
                num_minknow_events: 2,
                tracked_scaling_scale: 1.0,
                tracked_scaling_shift: 0.0,
                predicted_scaling_scale: 1.0,
                predicted_scaling_shift: 0.0,
                num_reads_since_mux_change: 0,
                time_since_mux_change: 0.0,
                num_samples: 2,
                open_pore_level: 220.0,
                expected_open_pore_level: 0.0,
                selected_read_level: 0.0,
                signal_rows: Vec::new(),
            };
            writer.add_read(read, samples).expect("add_read");
        }
        writer.finish().expect("finish");

        let reader = Reader::open(&path).expect("open");
        let signal_bytes = reader.signal_table_bytes().expect("signal bytes");
        let footer = ArrowIpcFooter::parse(signal_bytes).expect("parse footer");
        assert_eq!(footer.total_rows, 2);

        for (row, (expected_id, expected_samples)) in
            read_ids.iter().zip(signals.iter()).enumerate()
        {
            let chunk = footer
                .extract_signal_row(row as u64, signal_bytes)
                .unwrap_or_else(|e| panic!("row {row}: {e}"));
            assert_eq!(&chunk.read_id, expected_id.as_bytes(), "row {row} read_id");
            assert_eq!(chunk.samples, 2, "row {row} samples");
            let want: Vec<u8> = expected_samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            assert_eq!(chunk.signal, want.as_slice(), "row {row} signal bytes");
        }
    }

    /// The parallel walk must answer exactly what the serial one would, and
    /// must answer for the batch it was asked about.
    ///
    /// A fan-out does not risk a wrong row count — reading one message header
    /// is pure — it risks a wrong *pairing*. Results come back in completion
    /// order, so a collect that lost track of which batch it asked about would
    /// still build a perfectly plausible footer, with every batch carrying
    /// some other batch's row count and every cumulative offset after the
    /// first short batch wrong. Nothing about the shape of the answer would
    /// look off. So this asserts the mapping, not the multiset.
    #[test]
    fn parallel_row_counts_keep_their_batch() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = many_batch_pod5(tmp.path());

        let reader = crate::Reader::open(&path).expect("open");
        let signal_bytes = reader.signal_table_bytes().expect("signal bytes");
        let footer = ArrowIpcFooter::parse(signal_bytes).expect("parse");

        assert!(
            footer.record_batches.len() >= ROW_COUNT_PARALLEL_MIN,
            "fixture has {} batches, below the {ROW_COUNT_PARALLEL_MIN} that \
             makes the walk parallel — this test would prove nothing",
            footer.record_batches.len()
        );

        let wanted: Vec<(usize, i64)> = footer
            .record_batches
            .iter()
            .enumerate()
            .map(|(i, b)| (i, b.offset))
            .collect();
        let got = ArrowIpcFooter::read_row_counts(signal_bytes, &wanted).expect("row counts");
        assert_eq!(got.len(), wanted.len());

        let mut answered: Vec<Option<u64>> = vec![None; wanted.len()];
        for (i, rows) in got {
            assert!(answered[i].is_none(), "batch {i} answered twice");
            answered[i] = Some(rows);
        }
        for &(i, offset) in &wanted {
            let serial = ArrowIpcFooter::parse_batch_row_count(signal_bytes, offset as usize)
                .expect("serial row count");
            assert_eq!(
                answered[i],
                Some(serial),
                "batch {i} came back with another batch's row count"
            );
            assert_eq!(footer.record_batches[i].row_count, serial);
        }

        // The short last batch is what a shuffle would move, so say out loud
        // that the fixture has one.
        let last = footer.record_batches.last().expect("batches").row_count;
        let first = footer.record_batches[0].row_count;
        assert!(
            last < first,
            "fixture's last batch is not short ({last} vs {first}); a shuffled \
             geometry would be undetectable"
        );
    }

    #[test]
    fn test_parse_real_signal_table() {
        // Use bundled test data in data/drna
        let test_file = std::path::Path::new("../data/drna/yeast_trna_reads.pod5");
        if !test_file.exists() {
            eprintln!("Skipping test - test file not found at {:?}", test_file);
            return;
        }

        let reader = crate::Reader::open(test_file).expect("Failed to open test file");
        let signal_bytes = reader
            .signal_table_bytes()
            .expect("Failed to get signal bytes");

        // Should parse without error
        let footer = ArrowIpcFooter::parse(signal_bytes).expect("Failed to parse IPC footer");

        // Should have some batches
        assert!(!footer.record_batches.is_empty(), "No record batches found");

        // All offsets should be valid
        for batch in &footer.record_batches {
            assert!(batch.offset >= 0, "Negative offset");
            assert!(batch.metadata_length > 0, "Zero metadata length");
            assert!(batch.body_length >= 0, "Negative body length");
            assert!(
                batch.byte_range().end <= signal_bytes.len(),
                "Batch extends beyond signal table"
            );
        }

        eprintln!(
            "Parsed {} batches from {:.2} KB signal table, {} total rows",
            footer.record_batches.len(),
            signal_bytes.len() as f64 / 1024.0,
            footer.total_rows
        );
        eprintln!(
            "Header: 0..{}, Batches: {}..{}",
            footer.batches_start_offset, footer.batches_start_offset, footer.batches_end_offset
        );

        // Print batch details for debugging
        for (i, batch) in footer.record_batches.iter().enumerate() {
            eprintln!(
                "  Batch {}: offset={}, meta_len={}, body_len={}, rows={}, range={}..{}",
                i,
                batch.offset,
                batch.metadata_length,
                batch.body_length,
                batch.row_count,
                batch.byte_range().start,
                batch.byte_range().end
            );
        }

        // Verify total rows matches sum of batch row counts
        let sum_rows: u64 = footer.record_batches.iter().map(|b| b.row_count).sum();
        assert_eq!(footer.total_rows, sum_rows);
    }

    /// Locate a real POD5 signal-table IPC blob to use as the seed corpus for
    /// fuzz-style truncation tests. Returns `None` if no test file is present
    /// (so the test degrades to garbage-only input rather than failing).
    fn load_real_signal_bytes() -> Option<Vec<u8>> {
        for candidate in [
            "../data/drna/yeast_trna_reads.pod5",
            "../../ext/nanopore-dna-data/pod5/yeast_trna_reads.pod5",
        ] {
            let path = std::path::Path::new(candidate);
            if let Ok(reader) = crate::Reader::open(path)
                && let Ok(bytes) = reader.signal_table_bytes()
            {
                return Some(bytes.to_vec());
            }
        }
        None
    }

    /// The Arrow IPC parser is reachable from untrusted file bytes (and via the
    /// Python bindings). Feed it progressively truncated prefixes of a real
    /// signal table plus assorted garbage buffers, and assert that no input ever
    /// panics — every call must return `Ok` or `Err`, never unwind.
    #[test]
    fn test_parser_never_panics_on_malformed_input() {
        // Drive both the footer parser and the row-extraction path. The result
        // is ignored; we only care that the call completes without panicking.
        fn exercise(bytes: &[u8]) {
            if let Ok(footer) = ArrowIpcFooter::parse(bytes) {
                // If the footer parsed, hammer the downstream extraction paths
                // with both in-range and out-of-range rows.
                for row in [
                    0u64,
                    1,
                    footer.total_rows,
                    footer.total_rows.wrapping_add(1),
                ] {
                    let _ = footer.extract_signal_row(row, bytes);
                }
                let probe_rows: Vec<u64> = (0..footer.total_rows.min(8)).collect();
                let _ = footer.extract_signal_rows(&probe_rows, bytes);
            }
        }

        let run = |bytes: &[u8]| {
            let owned = bytes.to_vec();
            let result = std::panic::catch_unwind(move || exercise(&owned));
            assert!(
                result.is_ok(),
                "parser panicked on {}-byte input: {:02x?}",
                bytes.len(),
                &bytes[..bytes.len().min(32)],
            );
        };

        // 1. Every truncated prefix of a real signal table (header, schema,
        //    batch boundaries, footer — all get exercised mid-structure).
        if let Some(real) = load_real_signal_bytes() {
            let cap = real.len().min(4096);
            for len in 0..=cap {
                run(&real[..len]);
            }
            // Plus a few prefixes near the very end (footer region).
            for &tail in &[6usize, 10, 18, 24] {
                if real.len() > tail {
                    run(&real[..real.len() - tail]);
                }
            }
            // And single-byte corruptions sprinkled through the footer region.
            let footer_region = real.len().saturating_sub(64);
            for i in (footer_region..real.len()).step_by(3) {
                let mut corrupted = real.clone();
                corrupted[i] ^= 0xff;
                run(&corrupted);
            }
        }

        // 2. Pathological hand-built buffers: empty, all-zero, all-0xff, and
        //    a valid trailing magic with a garbage/oversized footer length.
        run(&[]);
        run(&[0u8; 17]);
        run(&[0u8; 18]);
        run(&[0xffu8; 64]);
        run(&[0x00u8; 256]);

        let mut fake = vec![0u8; 64];
        fake[58..64].copy_from_slice(ARROW_MAGIC); // trailing magic present
        // footer_len just before magic = huge value
        fake[54..58].copy_from_slice(&i32::MAX.to_le_bytes());
        run(&fake);

        // footer_len = -1 (continuation) with garbage actual length
        fake[54..58].copy_from_slice(&(-1i32).to_le_bytes());
        run(&fake);

        // 3. Deterministic pseudo-random garbage of assorted sizes.
        let mut state: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for &size in &[8usize, 16, 18, 32, 64, 128, 512, 2000] {
            for _ in 0..32 {
                let buf: Vec<u8> = (0..size).map(|_| (next() & 0xff) as u8).collect();
                run(&buf);
            }
        }
    }
}

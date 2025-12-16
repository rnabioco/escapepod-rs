//! Arrow IPC format parsing for raw byte-level operations.
//!
//! This module provides minimal parsing of Arrow IPC file format to enable
//! raw byte copying of record batches without full deserialization.

use crate::error::{Error, Result};

/// Magic bytes at start and end of Arrow IPC files.
const ARROW_MAGIC: &[u8; 6] = b"ARROW1";

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
#[derive(Debug)]
pub struct ArrowIpcFooter {
    /// Locations of record batches (signal data).
    pub record_batches: Vec<BatchBlock>,
    /// Byte offset where the first batch starts (end of header/schema).
    pub batches_start_offset: usize,
    /// Byte offset where the last batch ends (start of footer).
    pub batches_end_offset: usize,
    /// Total number of rows across all batches.
    pub total_rows: u64,
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
        let len = ipc_bytes.len();

        // Need at least magic (8) + footer_len (4) + magic (6) = 18 bytes minimum
        if len < 18 {
            return Err(Error::InvalidArrowIpc("IPC stream too small".into()));
        }

        // Verify trailing magic (6 bytes at end, no padding)
        if &ipc_bytes[len - 6..len] != ARROW_MAGIC {
            return Err(Error::InvalidArrowIpc("Missing trailing ARROW1 magic".into()));
        }

        // Read footer length (4 bytes immediately before trailing magic)
        let footer_len_offset = len - 6 - 4; // 6 = magic, 4 = footer_len
        let footer_len_bytes = &ipc_bytes[footer_len_offset..footer_len_offset + 4];
        let footer_len = i32::from_le_bytes(footer_len_bytes.try_into().unwrap());

        // Handle continuation indicator (negative means flatbuffer follows)
        let footer_len = if footer_len < 0 {
            // Read actual length after continuation marker
            let actual_len_offset = footer_len_offset + 4;
            if actual_len_offset + 4 > len - 6 {
                return Err(Error::InvalidArrowIpc("Invalid continuation marker".into()));
            }
            let actual_bytes = &ipc_bytes[actual_len_offset..actual_len_offset + 4];
            i32::from_le_bytes(actual_bytes.try_into().unwrap())
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
        let footer_bytes = &ipc_bytes[footer_start..footer_start + footer_len as usize];

        // Parse the footer FlatBuffer to extract batch blocks
        Self::parse_footer_flatbuffer(footer_bytes, ipc_bytes)
    }

    /// Parse the Footer FlatBuffer to extract record batch locations.
    fn parse_footer_flatbuffer(footer_bytes: &[u8], full_ipc: &[u8]) -> Result<Self> {
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
        let root_offset = u32::from_le_bytes(footer_bytes[0..4].try_into().unwrap()) as usize;
        if root_offset >= footer_bytes.len() {
            return Err(Error::InvalidArrowIpc("Invalid root offset".into()));
        }

        let table_start = root_offset;

        // Read vtable offset (signed, relative to table_start)
        let vtable_soffset =
            i32::from_le_bytes(footer_bytes[table_start..table_start + 4].try_into().unwrap());
        let vtable_pos = (table_start as i32 - vtable_soffset) as usize;

        if vtable_pos + 4 > footer_bytes.len() {
            return Err(Error::InvalidArrowIpc("Invalid vtable position".into()));
        }

        // Read vtable size
        let vtable_size =
            u16::from_le_bytes(footer_bytes[vtable_pos..vtable_pos + 2].try_into().unwrap())
                as usize;

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
            let offset_in_vtable =
                u16::from_le_bytes(footer_bytes[vtable_pos + 10..vtable_pos + 12].try_into().unwrap());
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
                let vec_offset = u32::from_le_bytes(
                    footer_bytes[vec_offset_pos..vec_offset_pos + 4].try_into().unwrap(),
                ) as usize;
                let vec_pos = vec_offset_pos + vec_offset;

                if vec_pos + 4 <= footer_bytes.len() {
                    // Read vector length
                    let vec_len = u32::from_le_bytes(
                        footer_bytes[vec_pos..vec_pos + 4].try_into().unwrap(),
                    ) as usize;

                    // Each Block is a struct with: offset(i64) + metaDataLength(i32) + padding(i32) + bodyLength(i64) = 24 bytes
                    // Actually in FlatBuffers, Block struct is: offset(8) + metaDataLength(4) + bodyLength(8) = 20 bytes
                    // But may have alignment padding
                    let block_size = 24; // Aligned size
                    let blocks_start = vec_pos + 4;

                    for i in 0..vec_len {
                        let block_pos = blocks_start + i * block_size;
                        if block_pos + 20 > footer_bytes.len() {
                            break;
                        }

                        let offset = i64::from_le_bytes(
                            footer_bytes[block_pos..block_pos + 8].try_into().unwrap(),
                        );
                        let metadata_length = i32::from_le_bytes(
                            footer_bytes[block_pos + 8..block_pos + 12].try_into().unwrap(),
                        );
                        // Skip 4 bytes padding
                        let body_length = i64::from_le_bytes(
                            footer_bytes[block_pos + 16..block_pos + 24].try_into().unwrap(),
                        );

                        // Parse row count from the batch message metadata
                        let row_count = Self::parse_batch_row_count(full_ipc, offset as usize)?;

                        record_batches.push(BatchBlock {
                            offset,
                            metadata_length,
                            body_length,
                            row_count,
                        });
                    }
                }
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
        let total_rows: u64 = record_batches.iter().map(|b| b.row_count).sum();

        Ok(ArrowIpcFooter {
            record_batches,
            batches_start_offset,
            batches_end_offset,
            total_rows,
        })
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
        let first_word = i32::from_le_bytes(ipc_bytes[offset..offset + 4].try_into().unwrap());

        let metadata_start = if first_word == -1 {
            // IPC v2 format: continuation marker followed by metadata length
            offset + 8 // Skip continuation (-1) and metadata_length
        } else {
            // IPC v1 format: first word is metadata length
            offset + 4
        };

        if metadata_start + 4 > ipc_bytes.len() {
            return Err(Error::InvalidArrowIpc("Metadata offset out of bounds".into()));
        }

        // The metadata is a Message FlatBuffer. We need to navigate to RecordBatch.length.
        // Message table: version(i16), header_type(byte), header(union), bodyLength(i64)
        // The header union points to a RecordBatch table which has: length(i64), nodes, buffers
        let metadata = &ipc_bytes[metadata_start..];

        // Read root table offset
        if metadata.len() < 4 {
            return Err(Error::InvalidArrowIpc("Metadata too small".into()));
        }
        let root_offset = u32::from_le_bytes(metadata[0..4].try_into().unwrap()) as usize;
        if root_offset >= metadata.len() {
            return Err(Error::InvalidArrowIpc("Invalid message root offset".into()));
        }

        // Read vtable offset from root table
        let vtable_soffset =
            i32::from_le_bytes(metadata[root_offset..root_offset + 4].try_into().unwrap());
        let vtable_pos = (root_offset as i32 - vtable_soffset) as usize;

        if vtable_pos + 10 > metadata.len() {
            return Err(Error::InvalidArrowIpc("Invalid message vtable".into()));
        }

        // Message vtable: size(2), table_size(2), version(2), header_type(2), header(2), bodyLength(2)
        // We need header offset at vtable_pos + 8
        let header_field_offset =
            u16::from_le_bytes(metadata[vtable_pos + 8..vtable_pos + 10].try_into().unwrap())
                as usize;

        if header_field_offset == 0 {
            return Err(Error::InvalidArrowIpc("No header in message".into()));
        }

        // Read header table offset (union value)
        let header_offset_pos = root_offset + header_field_offset;
        if header_offset_pos + 4 > metadata.len() {
            return Err(Error::InvalidArrowIpc("Header offset out of bounds".into()));
        }
        let header_offset =
            u32::from_le_bytes(metadata[header_offset_pos..header_offset_pos + 4].try_into().unwrap())
                as usize;
        let header_table_pos = header_offset_pos + header_offset;

        // Now we're at the RecordBatch table
        // RecordBatch vtable: size(2), table_size(2), length(2), nodes(2), buffers(2)
        if header_table_pos + 4 > metadata.len() {
            return Err(Error::InvalidArrowIpc("RecordBatch table out of bounds".into()));
        }

        let rb_vtable_soffset =
            i32::from_le_bytes(metadata[header_table_pos..header_table_pos + 4].try_into().unwrap());
        let rb_vtable_pos = (header_table_pos as i32 - rb_vtable_soffset) as usize;

        if rb_vtable_pos + 6 > metadata.len() {
            return Err(Error::InvalidArrowIpc("RecordBatch vtable out of bounds".into()));
        }

        // Read length field offset (first field after size and table_size)
        let length_field_offset =
            u16::from_le_bytes(metadata[rb_vtable_pos + 4..rb_vtable_pos + 6].try_into().unwrap())
                as usize;

        if length_field_offset == 0 {
            // No length field, assume 0 rows
            return Ok(0);
        }

        // Read the length value (i64)
        let length_pos = header_table_pos + length_field_offset;
        if length_pos + 8 > metadata.len() {
            return Err(Error::InvalidArrowIpc("Length field out of bounds".into()));
        }

        let length = i64::from_le_bytes(metadata[length_pos..length_pos + 8].try_into().unwrap());
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

    #[test]
    fn test_parse_real_signal_table() {
        // Use bundled test data in data/drna
        let test_file = std::path::Path::new("../data/drna/yeast_trna_reads.pod5");
        if !test_file.exists() {
            eprintln!("Skipping test - test file not found at {:?}", test_file);
            return;
        }

        let reader = crate::Reader::open(test_file).expect("Failed to open test file");
        let signal_bytes = reader.signal_table_bytes().expect("Failed to get signal bytes");

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
            footer.batches_start_offset,
            footer.batches_start_offset,
            footer.batches_end_offset
        );

        // Print batch details for debugging
        for (i, batch) in footer.record_batches.iter().enumerate() {
            eprintln!(
                "  Batch {}: offset={}, meta_len={}, body_len={}, rows={}, range={}..{}",
                i, batch.offset, batch.metadata_length, batch.body_length, batch.row_count,
                batch.byte_range().start, batch.byte_range().end
            );
        }

        // Verify total rows matches sum of batch row counts
        let sum_rows: u64 = footer.record_batches.iter().map(|b| b.row_count).sum();
        assert_eq!(footer.total_rows, sum_rows);
    }
}

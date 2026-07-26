//! POD5 file footer parsing using FlatBuffers.
//!
//! The footer contains metadata about the embedded Arrow IPC files
//! and is located at the end of the POD5 file.

use crate::error::{Error, Result};
use crate::types::{FOOTER_MAGIC, POD5_SIGNATURE, SECTION_MARKER_LENGTH};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

// FlatBuffer generated code would normally go here.
// For now, we'll parse the footer manually since the schema is simple.

/// Content type of an embedded file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    ReadsTable,
    SignalTable,
    RunInfoTable,
    ReadIdIndex,
    OtherIndex,
}

impl ContentType {
    fn from_i16(value: i16) -> Result<Self> {
        match value {
            0 => Ok(ContentType::ReadsTable),
            1 => Ok(ContentType::SignalTable),
            2 => Ok(ContentType::ReadIdIndex),
            3 => Ok(ContentType::OtherIndex),
            4 => Ok(ContentType::RunInfoTable),
            _ => Err(Error::InvalidFooter(format!(
                "Unknown content type: {}",
                value
            ))),
        }
    }
}

/// Format of an embedded file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    FeatherV2,
}

impl Format {
    fn from_i16(value: i16) -> Result<Self> {
        match value {
            0 => Ok(Format::FeatherV2),
            _ => Err(Error::InvalidFooter(format!("Unknown format: {}", value))),
        }
    }
}

/// Description of an embedded file within the POD5 container.
#[derive(Debug, Clone)]
pub struct EmbeddedFile {
    /// Offset from start of file to the embedded file.
    pub offset: i64,
    /// Length of the embedded file (excluding padding).
    pub length: i64,
    /// Format of the embedded file.
    #[allow(dead_code)]
    pub format: Format,
    /// Type of content in the embedded file.
    pub content_type: ContentType,
}

/// Parsed POD5 footer.
#[derive(Debug, Clone)]
pub struct Footer {
    /// Unique file identifier (UUID as string).
    pub file_identifier: String,
    /// Software that wrote this file.
    pub software: String,
    /// POD5 specification version.
    pub pod5_version: String,
    /// Embedded files (Arrow IPC tables).
    pub contents: Vec<EmbeddedFile>,
}

impl Footer {
    /// Find the embedded file with the given content type.
    pub fn find_content(&self, content_type: ContentType) -> Option<&EmbeddedFile> {
        self.contents
            .iter()
            .find(|f| f.content_type == content_type)
    }

    /// Get the reads table location.
    pub fn reads_table(&self) -> Option<&EmbeddedFile> {
        self.find_content(ContentType::ReadsTable)
    }

    /// Get the signal table location.
    pub fn signal_table(&self) -> Option<&EmbeddedFile> {
        self.find_content(ContentType::SignalTable)
    }

    /// Get the run info table location.
    pub fn run_info_table(&self) -> Option<&EmbeddedFile> {
        self.find_content(ContentType::RunInfoTable)
    }
}

/// Size of the fixed trailer at the very end of a POD5 file:
/// footer length (8) + section marker (16) + signature (8).
pub const TRAILER_LENGTH: usize = 8 + SECTION_MARKER_LENGTH + 8;

/// Smallest file that can possibly hold a footer: leading signature, the
/// trailer, the `FOOTER` magic, and a minimal FlatBuffer root.
const MIN_FOOTER_FILE_LENGTH: u64 = (8 + TRAILER_LENGTH + 8 + 4) as u64;

/// Locate the footer region — the `FOOTER\0\0` magic plus the FlatBuffer body —
/// from the file's fixed-size trailer alone.
///
/// This is the seam that lets a footer be parsed without the whole file in
/// hand: given the last [`TRAILER_LENGTH`] bytes and the object's total size,
/// it returns the absolute byte range a caller must fetch and hand to
/// [`parse_footer_region`]. Local readers slice that range out of the mmap;
/// remote readers turn it into a range request.
pub fn footer_body_range(trailer: &[u8], file_len: u64) -> Result<std::ops::Range<u64>> {
    if file_len < MIN_FOOTER_FILE_LENGTH {
        return Err(Error::InvalidFooter("File too small".to_string()));
    }
    if trailer.len() != TRAILER_LENGTH {
        return Err(Error::InvalidFooter(format!(
            "Footer trailer must be {TRAILER_LENGTH} bytes, got {}",
            trailer.len()
        )));
    }

    // Verify end signature
    if trailer[TRAILER_LENGTH - 8..] != POD5_SIGNATURE {
        return Err(Error::SignatureMismatch);
    }

    // Footer length sits immediately before the section marker and signature.
    let mut cursor = Cursor::new(&trailer[..8]);
    let footer_len = cursor.read_i64::<LittleEndian>()?;
    if footer_len < 0 {
        return Err(Error::InvalidFooter(format!(
            "Negative footer length: {footer_len}"
        )));
    }
    let footer_len = footer_len as u64;

    let footer_len_offset = file_len - TRAILER_LENGTH as u64;
    // The magic precedes the body, so the region is `footer_len + 8` bytes and
    // must fit between the leading file signature and the trailer. Checked
    // arithmetic here is what keeps a corrupt length from wrapping into a wild
    // slice.
    let region_len = footer_len + 8;
    let magic_start = footer_len_offset
        .checked_sub(region_len)
        .filter(|start| *start >= 8)
        .ok_or_else(|| {
            Error::InvalidFooter(format!(
                "Footer length {footer_len} does not fit in a {file_len}-byte file"
            ))
        })?;

    Ok(magic_start..footer_len_offset)
}

/// Parse a footer region that begins with the `FOOTER\0\0` magic — the range
/// reported by [`footer_body_range`].
pub fn parse_footer_region(region: &[u8]) -> Result<Footer> {
    if region.len() < 8 || region[..8] != FOOTER_MAGIC {
        return Err(Error::InvalidFooter("Missing FOOTER magic".to_string()));
    }
    parse_flatbuffer_footer(&region[8..])
}

/// Parse the footer from a complete POD5 file image.
///
/// The reader itself never has the whole file in hand — it goes through
/// [`footer_body_range`] + [`parse_footer_region`] so the same code path serves
/// local and remote sources. This whole-file form is retained as the oracle
/// those two are checked against.
///
/// The footer is located at the end of the file with the following structure:
/// - "FOOTER\0\0" magic (8 bytes)
/// - FlatBuffer data
/// - Footer length (8 bytes, little-endian i64)
/// - Section marker (16 bytes)
/// - Signature (8 bytes)
#[cfg(test)]
pub fn parse_footer(data: &[u8]) -> Result<Footer> {
    let file_len = data.len() as u64;
    if file_len < MIN_FOOTER_FILE_LENGTH {
        return Err(Error::InvalidFooter("File too small".to_string()));
    }

    let range = footer_body_range(&data[data.len() - TRAILER_LENGTH..], file_len)?;
    parse_footer_region(&data[range.start as usize..range.end as usize])
}

/// Parse the FlatBuffer-encoded footer data.
fn parse_flatbuffer_footer(data: &[u8]) -> Result<Footer> {
    // FlatBuffer root table offset is at the start
    if data.len() < 4 {
        return Err(Error::InvalidFooter("Footer too small".to_string()));
    }

    let root_offset = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let table_start = root_offset;

    if table_start >= data.len() {
        return Err(Error::InvalidFooter("Invalid root offset".to_string()));
    }

    // Read vtable offset - in FlatBuffers this is a signed offset that is SUBTRACTED
    // from the table position to find the vtable
    if table_start + 4 > data.len() {
        return Err(Error::InvalidFooter("Table out of bounds".to_string()));
    }
    let vtable_offset_bytes = &data[table_start..table_start + 4];
    let vtable_soffset = i32::from_le_bytes([
        vtable_offset_bytes[0],
        vtable_offset_bytes[1],
        vtable_offset_bytes[2],
        vtable_offset_bytes[3],
    ]);

    // vtable is at table_start - soffset (soffset is stored as signed but represents distance back)
    let vtable_start = if vtable_soffset >= 0 {
        table_start.checked_sub(vtable_soffset as usize)
    } else {
        // Negative soffset means vtable is AFTER the table (unusual but valid)
        table_start.checked_add((-vtable_soffset) as usize)
    };

    let vtable_start =
        vtable_start.ok_or_else(|| Error::InvalidFooter("Invalid vtable offset".to_string()))?;

    if vtable_start + 4 > data.len() {
        return Err(Error::InvalidFooter("Invalid vtable".to_string()));
    }

    // Read vtable size and table size
    let vtable_size = u16::from_le_bytes([data[vtable_start], data[vtable_start + 1]]) as usize;
    let _table_size = u16::from_le_bytes([data[vtable_start + 2], data[vtable_start + 3]]);

    // Helper to read field offset from vtable
    let read_field_offset = |field_idx: usize| -> Option<usize> {
        read_vtable_field_offset(data, vtable_start, vtable_size, table_start, field_idx)
    };

    // Helper to read string at offset
    let read_string = |offset: usize| -> Result<String> {
        if offset + 4 > data.len() {
            return Err(Error::InvalidFooter(
                "String offset out of bounds".to_string(),
            ));
        }
        let str_offset = offset
            + u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
        if str_offset + 4 > data.len() {
            return Err(Error::InvalidFooter(
                "String data out of bounds".to_string(),
            ));
        }
        let str_len = u32::from_le_bytes([
            data[str_offset],
            data[str_offset + 1],
            data[str_offset + 2],
            data[str_offset + 3],
        ]) as usize;
        if str_offset + 4 + str_len > data.len() {
            return Err(Error::InvalidFooter(
                "String content out of bounds".to_string(),
            ));
        }
        String::from_utf8(data[str_offset + 4..str_offset + 4 + str_len].to_vec())
            .map_err(|e| Error::InvalidFooter(format!("Invalid UTF-8 in string: {}", e)))
    };

    // Read fields: file_identifier (0), software (1), pod5_version (2), contents (3)
    let file_identifier = read_field_offset(0)
        .map(&read_string)
        .transpose()?
        .unwrap_or_default();

    let software = read_field_offset(1)
        .map(&read_string)
        .transpose()?
        .unwrap_or_default();

    let pod5_version = read_field_offset(2)
        .map(read_string)
        .transpose()?
        .unwrap_or_else(|| "1.0.0".to_string());

    // Read contents vector
    let contents = if let Some(contents_offset) = read_field_offset(3) {
        parse_embedded_files_vector(data, contents_offset)?
    } else {
        Vec::new()
    };

    Ok(Footer {
        file_identifier,
        software,
        pod5_version,
        contents,
    })
}

/// Parse the vector of EmbeddedFile entries.
fn parse_embedded_files_vector(data: &[u8], offset: usize) -> Result<Vec<EmbeddedFile>> {
    if offset + 4 > data.len() {
        return Err(Error::InvalidFooter(
            "Vector offset out of bounds".to_string(),
        ));
    }

    let vec_offset = offset
        + u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
    if vec_offset + 4 > data.len() {
        return Err(Error::InvalidFooter(
            "Vector data out of bounds".to_string(),
        ));
    }

    let vec_len = u32::from_le_bytes([
        data[vec_offset],
        data[vec_offset + 1],
        data[vec_offset + 2],
        data[vec_offset + 3],
    ]) as usize;

    let mut files = Vec::with_capacity(vec_len);
    for i in 0..vec_len {
        let entry_offset_pos = vec_offset + 4 + i * 4;
        if entry_offset_pos + 4 > data.len() {
            return Err(Error::InvalidFooter(
                "Entry offset out of bounds".to_string(),
            ));
        }
        let entry_offset = entry_offset_pos
            + u32::from_le_bytes([
                data[entry_offset_pos],
                data[entry_offset_pos + 1],
                data[entry_offset_pos + 2],
                data[entry_offset_pos + 3],
            ]) as usize;

        files.push(parse_embedded_file(data, entry_offset)?);
    }

    Ok(files)
}

/// Read a field offset from a FlatBuffer vtable.
///
/// Returns the absolute offset in `data` where the field value is located,
/// or None if the field is not present.
fn read_vtable_field_offset(
    data: &[u8],
    vtable_start: usize,
    vtable_size: usize,
    table_start: usize,
    field_idx: usize,
) -> Option<usize> {
    let vtable_entry_offset = vtable_start + 4 + field_idx * 2;
    if vtable_entry_offset + 2 > vtable_start + vtable_size {
        return None;
    }
    let offset =
        u16::from_le_bytes([data[vtable_entry_offset], data[vtable_entry_offset + 1]]) as usize;
    if offset == 0 {
        None
    } else {
        Some(table_start + offset)
    }
}

/// Parse a single EmbeddedFile entry.
fn parse_embedded_file(data: &[u8], table_start: usize) -> Result<EmbeddedFile> {
    if table_start + 4 > data.len() {
        return Err(Error::InvalidFooter(
            "EmbeddedFile table out of bounds".to_string(),
        ));
    }

    // Read vtable offset - subtract from table position
    let vtable_soffset = i32::from_le_bytes([
        data[table_start],
        data[table_start + 1],
        data[table_start + 2],
        data[table_start + 3],
    ]);

    let vtable_start = if vtable_soffset >= 0 {
        table_start.checked_sub(vtable_soffset as usize)
    } else {
        table_start.checked_add((-vtable_soffset) as usize)
    };

    let vtable_start = vtable_start
        .ok_or_else(|| Error::InvalidFooter("Invalid EmbeddedFile vtable offset".to_string()))?;

    if vtable_start + 4 > data.len() {
        return Err(Error::InvalidFooter(
            "EmbeddedFile vtable out of bounds".to_string(),
        ));
    }

    let vtable_size = u16::from_le_bytes([data[vtable_start], data[vtable_start + 1]]) as usize;

    // Helper to read field offset
    let read_field_offset = |field_idx: usize| -> Option<usize> {
        read_vtable_field_offset(data, vtable_start, vtable_size, table_start, field_idx)
    };

    // Read fields: offset (0), length (1), format (2), content_type (3)
    let offset = read_field_offset(0)
        .map(|o| {
            i64::from_le_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
                data[o + 4],
                data[o + 5],
                data[o + 6],
                data[o + 7],
            ])
        })
        .unwrap_or(0);

    let length = read_field_offset(1)
        .map(|o| {
            i64::from_le_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
                data[o + 4],
                data[o + 5],
                data[o + 6],
                data[o + 7],
            ])
        })
        .unwrap_or(0);

    let format_val = read_field_offset(2)
        .map(|o| i16::from_le_bytes([data[o], data[o + 1]]))
        .unwrap_or(0);

    let content_type_val = read_field_offset(3)
        .map(|o| i16::from_le_bytes([data[o], data[o + 1]]))
        .unwrap_or(0);

    Ok(EmbeddedFile {
        offset,
        length,
        format: Format::from_i16(format_val)?,
        content_type: ContentType::from_i16(content_type_val)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_from_i16() {
        assert_eq!(ContentType::from_i16(0).unwrap(), ContentType::ReadsTable);
        assert_eq!(ContentType::from_i16(1).unwrap(), ContentType::SignalTable);
        assert!(ContentType::from_i16(99).is_err());
    }

    #[test]
    fn test_format_from_i16() {
        assert_eq!(Format::from_i16(0).unwrap(), Format::FeatherV2);
        assert!(Format::from_i16(99).is_err());
    }

    /// Locate a real POD5 file to parse. Returns `None` when the bundled test
    /// data isn't present, so the suite degrades rather than fails.
    fn real_pod5() -> Option<Vec<u8>> {
        // Tests run with the crate directory as CWD, so the workspace-level
        // `data/` tree is two levels up.
        for candidate in [
            "../../data/drna/yeast_trna_reads.pod5",
            "../../ext/nanopore-dna-data/pod5/yeast_trna_reads.pod5",
        ] {
            if let Ok(bytes) = std::fs::read(candidate) {
                return Some(bytes);
            }
        }
        None
    }

    /// The seam the remote reader depends on: given only the fixed trailer and
    /// the object's size, `footer_body_range` must point at exactly the bytes
    /// `parse_footer` would have used from the whole file.
    #[test]
    fn tail_parse_matches_whole_file_parse() {
        let Some(data) = real_pod5() else {
            eprintln!("Skipping - no test POD5 available");
            return;
        };
        let file_len = data.len() as u64;

        let whole = parse_footer(&data).expect("whole-file parse failed");

        let trailer = &data[data.len() - TRAILER_LENGTH..];
        let range = footer_body_range(trailer, file_len).expect("footer_body_range failed");
        let tail = parse_footer_region(&data[range.start as usize..range.end as usize])
            .expect("region parse failed");

        assert_eq!(whole.file_identifier, tail.file_identifier);
        assert_eq!(whole.software, tail.software);
        assert_eq!(whole.pod5_version, tail.pod5_version);
        assert_eq!(whole.contents.len(), tail.contents.len());
        for (a, b) in whole.contents.iter().zip(&tail.contents) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.length, b.length);
            assert_eq!(a.content_type, b.content_type);
        }

        // The footer sits comfortably inside the tail the reader probes, which
        // is what keeps a remote open to one round trip for the footer.
        assert!(file_len - range.start < 64 * 1024);
    }

    /// A corrupt footer length must be rejected, not turned into a wild slice.
    /// These inputs reach the parser straight from untrusted file bytes.
    #[test]
    fn corrupt_trailers_are_rejected_without_panicking() {
        let file_len = 100_000u64;
        let mut trailer = vec![0u8; TRAILER_LENGTH];
        trailer[TRAILER_LENGTH - 8..].copy_from_slice(&POD5_SIGNATURE);

        // Footer longer than the file.
        trailer[..8].copy_from_slice(&(file_len as i64 * 2).to_le_bytes());
        assert!(footer_body_range(&trailer, file_len).is_err());

        // Negative footer length.
        trailer[..8].copy_from_slice(&(-1i64).to_le_bytes());
        assert!(footer_body_range(&trailer, file_len).is_err());

        // Footer length that would place the magic inside the leading signature.
        let overlap = file_len - TRAILER_LENGTH as u64 - 4;
        trailer[..8].copy_from_slice(&(overlap as i64).to_le_bytes());
        assert!(footer_body_range(&trailer, file_len).is_err());

        // Plausible length but a bad end signature.
        trailer[..8].copy_from_slice(&64i64.to_le_bytes());
        trailer[TRAILER_LENGTH - 1] = 0xff;
        assert!(matches!(
            footer_body_range(&trailer, file_len),
            Err(Error::SignatureMismatch)
        ));

        // Wrong trailer size, and a file too small to hold a footer at all.
        assert!(footer_body_range(&trailer[..4], file_len).is_err());
        assert!(footer_body_range(&trailer, 8).is_err());
    }

    #[test]
    fn region_without_magic_is_rejected() {
        assert!(parse_footer_region(b"NOTMAGIC____").is_err());
        assert!(parse_footer_region(b"").is_err());
    }
}

// SPDX-License-Identifier: MIT

//! Basecaller BAM tag decoding shared by signal-anchored commands.
//!
//! `escpod signal classify` and `escpod resquiggle` both start from dorado's
//! move-table tags (`mv`, plus the `ns`/`ts`/`sp` signal-trim integers);
//! decoding them lives here so the two cannot drift. What each command
//! builds *on top* differs deliberately and stays with the command: classify
//! maps query→signal in the Remora convention (`move_pos * stride + ts`,
//! frame-flipped through `ns`), resquiggle in the fishnet convention
//! (trim + RNA signal reversal).

use anyhow::{Result, bail};
use noodles_sam::alignment::RecordBuf;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::record_buf::data::field::Value;
use noodles_sam::alignment::record_buf::data::field::value::Array;

/// Decode the `mv` tag into `(stride, moves)`.
///
/// The first array element is the neural-network stride; the rest is the
/// move vector (1 = a new base starts). Accepts both the spec's UInt8 and
/// the Int8 some writers emit. Errors distinguish a missing tag, a
/// too-short array, and a zero stride so callers can report skips
/// precisely.
pub fn parse_mv_tag(record: &RecordBuf) -> Result<(usize, Vec<u8>)> {
    let mv_tag = Tag::new(b'm', b'v');
    let (stride, moves) = match record.data().get(&mv_tag) {
        Some(Value::Array(Array::UInt8(d))) => {
            if d.len() < 2 {
                bail!("mv tag too short (UInt8)");
            }
            (d[0] as usize, d[1..].to_vec())
        }
        Some(Value::Array(Array::Int8(d))) => {
            if d.len() < 2 {
                bail!("mv tag too short (Int8)");
            }
            (d[0] as usize, d[1..].iter().map(|&b| b as u8).collect())
        }
        _ => bail!("no mv tag"),
    };
    if stride == 0 {
        bail!("stride is 0");
    }
    Ok((stride, moves))
}

/// Extract an integer value from a BAM auxiliary tag (any int width).
pub fn int_tag(record: &RecordBuf, tag: Tag) -> Option<i64> {
    match record.data().get(&tag) {
        Some(Value::Int8(v)) => Some(*v as i64),
        Some(Value::UInt8(v)) => Some(*v as i64),
        Some(Value::Int16(v)) => Some(*v as i64),
        Some(Value::UInt16(v)) => Some(*v as i64),
        Some(Value::Int32(v)) => Some(*v as i64),
        Some(Value::UInt32(v)) => Some(*v as i64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_with_mv(values: Vec<u8>) -> RecordBuf {
        let mut record = RecordBuf::default();
        record
            .data_mut()
            .insert(Tag::new(b'm', b'v'), Value::Array(Array::UInt8(values)));
        record
    }

    #[test]
    fn test_parse_mv_tag() {
        let (stride, moves) = parse_mv_tag(&record_with_mv(vec![5, 1, 0, 1])).unwrap();
        assert_eq!(stride, 5);
        assert_eq!(moves, vec![1, 0, 1]);
    }

    #[test]
    fn test_parse_mv_tag_errors() {
        let err = parse_mv_tag(&RecordBuf::default()).unwrap_err();
        assert!(err.to_string().contains("no mv tag"));

        let err = parse_mv_tag(&record_with_mv(vec![5])).unwrap_err();
        assert!(err.to_string().contains("too short"));

        let err = parse_mv_tag(&record_with_mv(vec![0, 1, 0])).unwrap_err();
        assert!(err.to_string().contains("stride is 0"));
    }

    #[test]
    fn test_int_tag() {
        let mut record = RecordBuf::default();
        record
            .data_mut()
            .insert(Tag::new(b'n', b's'), Value::Int32(1234));
        assert_eq!(int_tag(&record, Tag::new(b'n', b's')), Some(1234));
        assert_eq!(int_tag(&record, Tag::new(b't', b's')), None);
    }
}

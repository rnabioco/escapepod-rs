//! Iterator over reads in a POD5 file.

use crate::arrow_helpers::BatchFieldExtractor;
use crate::error::{Error, Result};
use crate::types::ReadData;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::record_batch::RecordBatch;
use std::io::Cursor;

use super::Reader;

/// Iterator over reads in a POD5 file.
pub struct ReadIterator<'a> {
    #[allow(dead_code)]
    pub(super) pod5_reader: &'a Reader,
    pub(super) arrow_reader: ArrowFileReader<Cursor<&'a [u8]>>,
    pub(super) current_batch: Option<RecordBatch>,
    pub(super) batch_row: usize,
}

impl<'a> Iterator for ReadIterator<'a> {
    type Item = Result<ReadData>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we need a new batch
            let need_new_batch = match &self.current_batch {
                None => true,
                Some(batch) => self.batch_row >= batch.num_rows(),
            };

            if need_new_batch {
                match self.arrow_reader.next() {
                    Some(Ok(batch)) => {
                        self.current_batch = Some(batch);
                        self.batch_row = 0;
                    }
                    Some(Err(e)) => return Some(Err(Error::from(e))),
                    None => return None,
                }
            }

            // Extract read from current batch
            if let Some(batch) = &self.current_batch {
                let row = self.batch_row;
                self.batch_row += 1;
                return Some(Self::read_from_batch(batch, row));
            }
        }
    }
}

impl<'a> ReadIterator<'a> {
    fn read_from_batch(batch: &RecordBatch, row: usize) -> Result<ReadData> {
        extract_read_from_batch(batch, row, false)
    }
}

/// Extract a read from a record batch at the given row.
///
/// This is the shared implementation used by both `Reader::read_from_batch`
/// and `ReadIterator::read_from_batch`.
///
/// The `try_alternate_field_names` parameter controls whether to try alternate
/// field names for compatibility with different POD5 versions:
/// - `start_sample` vs `start`
/// - `predicted_scaling_open_pore_level` vs `open_pore_level`
pub(super) fn extract_read_from_batch(
    batch: &RecordBatch,
    row: usize,
    try_alternate_field_names: bool,
) -> Result<ReadData> {
    let ext = BatchFieldExtractor::new(batch, row);

    // Handle start_sample field name variations
    let start_sample = if try_alternate_field_names {
        ext.get_u64("start_sample")
            .or_else(|_| ext.get_u64("start"))?
    } else {
        ext.get_u64("start")?
    };

    // Handle open_pore_level field name variations
    let open_pore_level = if try_alternate_field_names {
        ext.get_f32("predicted_scaling_open_pore_level")
            .or_else(|_| ext.get_f32("open_pore_level"))
            .unwrap_or(0.0)
    } else {
        ext.get_f32("open_pore_level").unwrap_or(0.0)
    };

    // Get run_info index from dictionary key
    let run_info_index = ext
        .get_dict_index("run_info")
        .map(|idx| idx as u32)
        .unwrap_or(0);

    // Parse end_reason - use FromStr which returns Infallible
    let end_reason_str = ext.get_dict_string("end_reason").unwrap_or_default();
    let end_reason = end_reason_str.parse().unwrap_or_default();

    Ok(ReadData {
        read_id: ext.get_uuid("read_id")?,
        read_number: ext.get_u32("read_number")?,
        start_sample,
        channel: ext.get_u16("channel")?,
        well: ext.get_u8("well")?,
        pore_type: ext.get_dict_string("pore_type").unwrap_or_default().into(),
        calibration_offset: ext.get_f32("calibration_offset")?,
        calibration_scale: ext.get_f32("calibration_scale")?,
        median_before: ext.get_f32("median_before")?,
        end_reason,
        end_reason_forced: ext.get_bool("end_reason_forced")?,
        run_info_index,
        num_minknow_events: ext.get_u64("num_minknow_events")?,
        tracked_scaling_scale: ext.get_f32("tracked_scaling_scale").unwrap_or(1.0),
        tracked_scaling_shift: ext.get_f32("tracked_scaling_shift").unwrap_or(0.0),
        predicted_scaling_scale: ext.get_f32("predicted_scaling_scale").unwrap_or(1.0),
        predicted_scaling_shift: ext.get_f32("predicted_scaling_shift").unwrap_or(0.0),
        num_reads_since_mux_change: ext.get_u32("num_reads_since_mux_change").unwrap_or(0),
        time_since_mux_change: ext.get_f32("time_since_mux_change").unwrap_or(0.0),
        num_samples: ext.get_u64("num_samples")?,
        open_pore_level,
        expected_open_pore_level: ext.get_f32("expected_open_pore_level").unwrap_or(0.0),
        selected_read_level: ext.get_f32("selected_read_level").unwrap_or(0.0),
        signal_rows: ext.get_signal_rows()?,
    })
}

//! Reading a real, on-disk POD5 **V6** file.
//!
//! V6 (upstream 0.3.46) retypes the reads-table `channel` column from `uint16`
//! to `uint32`. Nothing about the container changes, so the only way to get
//! this wrong is in column resolution — but that is also the only way to get
//! it *catastrophically* wrong: a reader pinned to `uint16` rejects every file
//! written by pod5 0.3.46 and later.
//!
//! There is no v6 fixture to test against. Upstream's `test_data/` stops at
//! `multi_fast5_zip_v5.pod5`, and the newest `pod5` on PyPI is 0.3.44, which
//! predates V6 — so no installable tool can produce one. This module builds a
//! genuine one instead: write a normal V5 file, re-emit its reads table with
//! `channel` cast to `uint32` (and one value pushed past `u16::MAX`, which no
//! V5 file could hold), re-assemble the container around it, and open the
//! result through the ordinary [`Reader`]. That exercises mmap, footer parse,
//! Arrow IPC decode, and both read paths — not just a hand-built `RecordBatch`.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt16Array, UInt32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::ipc::writer::FileWriter as ArrowFileWriter;

use crate::footer::parse_footer;
use crate::types::{EndReason, ReadData, RunInfoData, Uuid};
use crate::utils::pod5_assembler::write_post_signal_sections;
use crate::utils::table_builders::SchemaMetadata;
use crate::writer::{Writer, WriterOptions};

use super::Reader;

fn run_info() -> RunInfoData {
    RunInfoData {
        acquisition_id: "acq-v6".to_string(),
        acquisition_start_time: 1_609_459_200_000,
        adc_max: 2047,
        adc_min: -2048,
        context_tags: HashMap::new(),
        experiment_name: "v6".to_string(),
        flow_cell_id: "FAK_V6".to_string(),
        flow_cell_product_code: "FLO-PRO114M".to_string(),
        protocol_name: "v6_protocol".to_string(),
        protocol_run_id: "protocol_v6".to_string(),
        protocol_start_time: 1_609_459_200_000,
        sample_id: "v6_sample".to_string(),
        sample_rate: 5_000,
        sequencing_kit: "SQK-RNA004".to_string(),
        sequencer_position: "PC24B000".to_string(),
        sequencer_position_type: "promethion".to_string(),
        software: "v6-fixture".to_string(),
        system_name: "v6_system".to_string(),
        system_type: "promethion".to_string(),
        tracking_id: HashMap::new(),
    }
}

fn read_data(run_info_index: u32, read_number: u32, channel: u32) -> ReadData {
    ReadData {
        read_id: Uuid::from_u128(u128::from(read_number)),
        read_number,
        start_sample: u64::from(read_number) * 1000,
        channel,
        well: 2,
        pore_type: "not_set".into(),
        calibration_offset: -220.0,
        calibration_scale: 0.19,
        median_before: 200.0,
        end_reason: EndReason::SignalPositive,
        end_reason_forced: false,
        run_info_index,
        num_minknow_events: 42,
        num_samples: 64,
        ..ReadData::default()
    }
}

/// Re-encode a V5 reads table as V6: same rows, `channel` widened to `uint32`,
/// with `wide_channels[i]` substituted for row `i` so the fixture carries
/// values the narrow column could not represent.
fn widen_reads_table(v5_bytes: &[u8], wide_channels: &[u32], meta: &SchemaMetadata) -> Vec<u8> {
    let reader = ArrowFileReader::try_new(Cursor::new(v5_bytes), None).expect("reads table");
    let v5_schema = reader.schema();

    let v6_schema = Arc::new(
        meta.apply(Schema::new(
            v5_schema
                .fields()
                .iter()
                .map(|f| {
                    if f.name() == "channel" {
                        assert_eq!(f.data_type(), &DataType::UInt16, "source must be V5");
                        Arc::new(Field::new("channel", DataType::UInt32, f.is_nullable()))
                    } else {
                        f.clone()
                    }
                })
                .collect::<Vec<_>>(),
        )),
    );

    let channel_idx = v5_schema.index_of("channel").expect("channel column");
    let mut out = Vec::new();
    {
        let mut writer = ArrowFileWriter::try_new(&mut out, &v6_schema).expect("v6 writer");
        let mut row_base = 0usize;
        for batch in reader {
            let batch = batch.expect("v5 batch");
            let narrow = batch
                .column(channel_idx)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .expect("uint16 channel");

            let mut wide = UInt32Builder::with_capacity(narrow.len());
            for row in 0..narrow.len() {
                wide.append_value(wide_channels[row_base + row]);
            }
            row_base += narrow.len();

            let mut columns = batch.columns().to_vec();
            columns[channel_idx] = Arc::new(wide.finish()) as ArrayRef;
            writer
                .write(&RecordBatch::try_new(v6_schema.clone(), columns).expect("v6 batch"))
                .expect("write v6 batch");
        }
        writer.finish().expect("finish v6 table");
    }
    out
}

/// Build a genuine V6 POD5 at `dst` from the V5 file at `src`.
fn transcode_v5_file_to_v6(
    src: &std::path::Path,
    dst: &std::path::Path,
    wide_channels: &[u32],
    run_infos: &[RunInfoData],
) {
    let bytes = std::fs::read(src).expect("read v5 file");
    let footer = parse_footer(&bytes).expect("parse v5 footer");
    assert_eq!(footer.pod5_version, crate::types::POD5_VERSION);

    let signal = footer.signal_table().expect("signal table");
    let signal_end = (signal.offset + signal.length) as usize;
    let reads = footer.reads_table().expect("reads table");
    let reads_slice = &bytes[reads.offset as usize..(reads.offset + reads.length) as usize];

    // Same identity, new version stamp — this file really is V6.
    let meta = SchemaMetadata {
        file_identifier: footer.file_identifier.clone(),
        software: footer.software.clone(),
        pod5_version: "0.3.46".to_string(),
    };
    let v6_reads = widen_reads_table(reads_slice, wide_channels, &meta);

    // The header is signature (8) + section marker (16); reuse both verbatim
    // along with the whole signal section, which V6 does not touch.
    let section_marker = Uuid::from_slice(&bytes[8..24]).expect("section marker");
    let mut out = std::io::BufWriter::new(std::fs::File::create(dst).expect("create v6 file"));
    std::io::Write::write_all(&mut out, &bytes[..signal_end]).expect("copy header + signal");
    write_post_signal_sections(
        &mut out,
        &section_marker,
        &meta,
        signal_end,
        run_infos,
        &v6_reads,
    )
    .expect("assemble v6 file");
}

#[test]
fn reader_opens_a_real_v6_file_with_channels_above_u16_max() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let v5_path = tmp.path().join("v5.pod5");
    let v6_path = tmp.path().join("v6.pod5");

    // A V5 file cannot hold these, so they are substituted during transcode —
    // which is exactly what makes the fixture prove something.
    let wide_channels: Vec<u32> = vec![1, 3000, 65_535, 70_000, 4_000_000_000];
    let infos = vec![run_info()];

    let mut writer = Writer::create(&v5_path, WriterOptions::default()).expect("create v5");
    let run_idx = writer.add_run_info(infos[0].clone()).expect("run info");
    for i in 0..wide_channels.len() {
        let read = read_data(run_idx, i as u32 + 1, 1);
        let signal: Vec<i16> = (0..64).map(|s| (s * 7 % 2048) as i16).collect();
        writer.add_read(read, &signal).expect("add read");
    }
    writer.finish().expect("finish v5");

    transcode_v5_file_to_v6(&v5_path, &v6_path, &wide_channels, &infos);

    let reader = Reader::open(&v6_path).expect("open v6 file");
    assert_eq!(reader.pod5_version(), "0.3.46");

    // Row path (`reads()` -> BatchFieldExtractor).
    let reads: Vec<ReadData> = reader
        .reads()
        .expect("reads")
        .collect::<crate::error::Result<Vec<_>>>()
        .expect("read rows");
    assert_eq!(
        reads.iter().map(|r| r.channel).collect::<Vec<_>>(),
        wide_channels
    );

    // Columnar path (`read_columns()` -> ReadsBatchView::append_columns).
    let cols = reader.read_columns().expect("read columns");
    assert_eq!(cols.channel, wide_channels);

    // Signal still decodes — the transcode must not have disturbed the
    // container around the reads table.
    let signal = reader.get_signal(&reads[3].signal_rows).expect("signal");
    assert_eq!(signal.len(), 64);
}

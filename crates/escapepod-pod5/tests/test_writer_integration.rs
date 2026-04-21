//! Integration tests exercising Writer paths not covered by the existing
//! escapepod-signal round-trip suite: multi-batch writes, block-level
//! compressed-signal passthrough, and writing without signal compression.

mod common;

use std::collections::HashSet;

use escapepod_pod5::{PredefinedDictionaries, Reader, Uuid, Writer, WriterOptions};
use tempfile::TempDir;

use common::{make_read, make_run_info, synth_signal, write_fixture};

#[test]
fn writer_flushes_multiple_read_batches() {
    // read_batch_size = 5 with 17 reads ⇒ at least 4 read batches.
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("multi_batch.pod5");
    let opts = WriterOptions {
        read_batch_size: 5,
        signal_batch_size: 3,
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&path, opts).expect("create");
    let run_idx = writer.add_run_info(make_run_info("multi_batch")).unwrap();
    let mut ids = Vec::new();
    for i in 0..17 {
        let read = make_read(run_idx, i as u32 + 1, 500);
        ids.push(read.read_id);
        writer
            .add_read(read, &synth_signal(500, 0xBEEF + i))
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = Reader::open(&path).unwrap();
    assert!(
        reader.read_batch_count().unwrap() >= 4,
        "expected ≥4 read batches, got {}",
        reader.read_batch_count().unwrap()
    );
    let seen: HashSet<Uuid> = reader
        .reads()
        .unwrap()
        .map(|r| r.unwrap().read_id)
        .collect();
    assert_eq!(seen, ids.into_iter().collect::<HashSet<_>>());
}

#[test]
fn writer_compressed_signal_passthrough() {
    // Build a source file, then rewrite it using add_read_with_compressed_signal
    // — the block-level path used by repack/filter/merge.
    let tmp = TempDir::new().expect("tempdir");
    let src = tmp.path().join("src.pod5");
    let dst = tmp.path().join("dst.pod5");
    let fx = write_fixture(&src, "src_acq", 10, 600);

    let reader = Reader::open(&src).unwrap();
    let mut writer = Writer::create(&dst, WriterOptions::default()).unwrap();
    for run_info in reader.run_infos() {
        writer.add_run_info(run_info.clone()).unwrap();
    }
    for read_result in reader.reads().unwrap() {
        let read = read_result.unwrap();
        let compressed = reader
            .get_compressed_signal_for_rows(&read.signal_rows)
            .unwrap();
        let rewritten = read.for_writing_same_run();
        writer
            .add_read_with_compressed_signal(rewritten, &compressed)
            .unwrap();
    }
    writer.finish().unwrap();

    // Confirm every read round-trips byte-identical signal.
    let reader_src = Reader::open(&src).unwrap();
    let reader_dst = Reader::open(&dst).unwrap();
    let src_reads: Vec<_> = reader_src
        .reads()
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let dst_reads: Vec<_> = reader_dst
        .reads()
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(src_reads.len(), dst_reads.len());
    assert_eq!(src_reads.len(), fx.read_ids.len());

    for (s, d) in src_reads.iter().zip(dst_reads.iter()) {
        let ss = reader_src.get_signal(&s.signal_rows).unwrap();
        let ds = reader_dst.get_signal(&d.signal_rows).unwrap();
        assert_eq!(ss, ds, "signal mismatch for {}", s.read_id);
    }
}

#[test]
fn writer_predefined_dictionaries_enforce_pore_types() {
    // Only `not_set` pore_type is allowed; any other must error.
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("dict.pod5");
    let opts = WriterOptions {
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(vec!["not_set".to_string()]),
            end_reasons: Some(vec![
                "unknown".to_string(),
                "mux_change".to_string(),
                "unblock_mux_change".to_string(),
                "data_service_unblock_mux_change".to_string(),
                "signal_positive".to_string(),
                "signal_negative".to_string(),
                "api_request".to_string(),
                "device_data_error".to_string(),
                "analysis_config_change".to_string(),
                "paused".to_string(),
            ]),
        }),
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&path, opts).unwrap();
    let run_idx = writer.add_run_info(make_run_info("dict")).unwrap();

    // First read uses the allowed pore_type.
    let allowed = make_read(run_idx, 1, 300);
    writer
        .add_read(allowed, &synth_signal(300, 0xD1C7))
        .unwrap();

    // Second read introduces a disallowed pore_type — must error.
    let mut bad = make_read(run_idx, 2, 300);
    bad.pore_type = "rna_pore".to_string();
    let err = writer.add_read(bad, &synth_signal(300, 0xBAD)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("pore") || msg.to_lowercase().contains("dict"),
        "unexpected error: {msg}"
    );
}

#[test]
fn writer_predefined_dictionaries_enforce_end_reasons() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("dict_er.pod5");
    // Deliberately omit EndReason::SignalPositive (what make_read uses by
    // default) to force a rejection.
    let opts = WriterOptions {
        predefined_dictionaries: Some(PredefinedDictionaries {
            pore_types: Some(vec!["not_set".to_string()]),
            end_reasons: Some(vec!["unknown".to_string()]),
        }),
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&path, opts).unwrap();
    let run_idx = writer.add_run_info(make_run_info("dict_er")).unwrap();
    let bad = make_read(run_idx, 1, 300);
    let err = writer
        .add_read(bad, &synth_signal(300, 0xAAAA))
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("end_reason") || msg.to_lowercase().contains("dict"),
        "unexpected error: {msg}"
    );
}

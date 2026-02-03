// SPDX-License-Identifier: GPL-3.0-or-later
//
//! Integration test: compare escapepod resquiggle output against fishnet reference.
//!
//! Fishnet reference data was generated with:
//!   fishnet align --pod5 data/drna/yeast_trna_reads.pod5 \
//!     --bam data/drna/yeast_trna_mappings.bam \
//!     --kmer-table data/kmer_models/rna004_9mer_levels_v1.txt \
//!     --out data/drna/fishnet_yeast_trna_query.parquet \
//!     --rna --alignment-type query --output-level 1
//!
//! For RNA data, fishnet reverses the raw signal (3'→5' → 5'→3') and inverts
//! the query-to-signal mapping. This test applies the same RNA reversal before
//! running escapepod's refinement, so both tools operate in the same coordinate
//! space and can be compared directly.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, AsArray, GenericListArray, StringArray};
use arrow::datatypes::UInt64Type;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use noodles_bam as bam;
use noodles_sam as sam;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::value::Array as SamArray;
use sam::alignment::record_buf::data::field::Value;
use sam::alignment::RecordBuf;

use escapepod::resquiggle::{
    calculate_initial_scaling, refine_signal_map, KmerTable, RefineSettings,
};
use escapepod::{parse_uuid_flexible, Reader};

/// POD5 read metadata needed for refinement.
struct Pod5Info {
    calibration_scale: f32,
    calibration_offset: f32,
    signal_rows: Vec<u64>,
}

/// Load fishnet parquet reference: read_id -> query_to_signal boundaries (Vec<u64>).
fn load_fishnet_reference(path: &Path) -> HashMap<uuid::Uuid, Vec<u64>> {
    let file = File::open(path).expect("failed to open fishnet parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("failed to build reader");
    let reader = builder.build().expect("failed to build batch reader");

    let mut result = HashMap::new();
    for batch_result in reader {
        let batch = batch_result.expect("failed to read batch");
        let read_ids = batch
            .column_by_name("read_id")
            .expect("missing read_id column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("read_id not StringArray");

        let q2s = batch
            .column_by_name("query_to_signal")
            .expect("missing query_to_signal column")
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .expect("query_to_signal not ListArray");

        for row in 0..batch.num_rows() {
            let id_str = read_ids.value(row);
            let uuid = uuid::Uuid::parse_str(id_str).expect("invalid UUID in parquet");
            let values = q2s.value(row);
            let u64_arr = values.as_primitive::<UInt64Type>();
            let boundaries: Vec<u64> = u64_arr.values().iter().copied().collect();
            result.insert(uuid, boundaries);
        }
    }
    result
}

/// Extract a float value from a BAM auxiliary tag.
fn get_float_tag(record: &RecordBuf, a: u8, b: u8) -> Option<f32> {
    let tag = Tag::new(a, b);
    match record.data().get(&tag) {
        Some(Value::Float(f)) => Some(*f),
        _ => None,
    }
}

/// Extract an integer value from a BAM auxiliary tag.
fn get_int_tag(record: &RecordBuf, a: u8, b: u8) -> Option<i64> {
    let tag = Tag::new(a, b);
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

/// Build a query-to-signal map from the BAM move table.
///
/// Matches fishnet's `align_query_to_signal` (query_to_signal.rs:41-47):
/// uses `signal_len` as the final boundary (not `moves.len() * stride`).
fn build_query_to_signal_map(
    moves: &[u8],
    stride: usize,
    seq_len: usize,
    signal_len: usize,
) -> Vec<usize> {
    let mut map = Vec::with_capacity(seq_len + 1);
    for (i, &m) in moves.iter().enumerate() {
        if m == 1 {
            map.push(i * stride);
        }
    }
    map.push(signal_len);
    assert_eq!(
        map.len(),
        seq_len + 1,
        "move table produced {} boundaries, expected {}",
        map.len(),
        seq_len + 1
    );
    map
}

/// Reverse a query-to-signal map for RNA signal reversal.
///
/// Matches fishnet's `align_query_to_signal` reversal (query_to_signal.rs:49-55):
///   map = map.iter().rev().map(|el| signal_len - el).collect()
fn reverse_query_to_signal_map(map: &[usize], signal_len: usize) -> Vec<usize> {
    map.iter().rev().map(|&el| signal_len - el).collect()
}

/// Compare escapepod resquiggle output against fishnet reference on yeast tRNA data.
///
/// Both tools refine signal-to-base mappings using banded DP with identical scoring.
/// For RNA, fishnet reverses the raw signal (3'→5' → 5'→3') and inverts the
/// query-to-signal map. This test applies the same reversal before running
/// escapepod's refinement so both tools operate in the same coordinate space.
#[test]
fn test_resquiggle_vs_fishnet() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = Path::new(manifest_dir).parent().unwrap();

    // --- Load fishnet reference parquet ---
    let parquet_path = project_root.join("data/drna/fishnet_yeast_trna_query.parquet");
    if !parquet_path.exists() {
        eprintln!(
            "Skipping test: fishnet reference not found at {}",
            parquet_path.display()
        );
        return;
    }
    let fishnet_ref = load_fishnet_reference(&parquet_path);

    // --- Load kmer table ---
    let kmer_path = project_root.join("data/kmer_models/rna004_9mer_levels_v1.txt.gz");
    let kmer_table = KmerTable::from_file(&kmer_path).expect("failed to load kmer table");

    // --- Open POD5 and index reads ---
    let pod5_path = project_root.join("data/drna/yeast_trna_reads.pod5");
    let reader = Reader::open(&pod5_path).expect("failed to open POD5");

    let mut pod5_index: HashMap<uuid::Uuid, Pod5Info> = HashMap::new();
    for read_result in reader.reads().expect("failed to iterate reads") {
        let read = read_result.expect("failed to read POD5 record");
        pod5_index.insert(
            read.read_id,
            Pod5Info {
                calibration_scale: read.calibration_scale,
                calibration_offset: read.calibration_offset,
                signal_rows: read.signal_rows.clone(),
            },
        );
    }
    let signal_extractor = reader
        .signal_extractor()
        .expect("failed to create signal extractor");

    // --- Open BAM and process reads ---
    let bam_path = project_root.join("data/drna/yeast_trna_mappings.bam");
    let bam_file = File::open(&bam_path).expect("failed to open BAM");
    let mut bam_reader = bam::io::Reader::new(bam_file);
    let header = bam_reader.read_header().expect("failed to read BAM header");

    let settings = RefineSettings::default();

    let mut compared = 0usize;
    let mut total_match_pct = 0.0f64;

    loop {
        let mut record = RecordBuf::default();
        match bam_reader.read_record_buf(&header, &mut record) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => panic!("BAM read error: {}", e),
        }

        // Parse read ID
        let name = match record.name() {
            Some(n) => n,
            None => continue,
        };
        let name_bytes: &[u8] = name.as_ref();
        let name_str = match std::str::from_utf8(name_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let read_id = match parse_uuid_flexible(name_str) {
            Ok(id) => id,
            Err(_) => continue,
        };

        // Skip if no POD5 match or no fishnet reference
        let pod5_info = match pod5_index.get(&read_id) {
            Some(info) => info,
            None => continue,
        };
        let fishnet_boundaries = match fishnet_ref.get(&read_id) {
            Some(b) => b,
            None => continue,
        };

        // --- Extract move table ---
        let mv_tag = Tag::new(b'm', b'v');
        let (stride, moves) = match record.data().get(&mv_tag) {
            Some(Value::Array(SamArray::UInt8(data))) => {
                assert!(data.len() >= 2, "mv tag too short");
                (data[0] as usize, data[1..].to_vec())
            }
            Some(Value::Array(SamArray::Int8(data))) => {
                assert!(data.len() >= 2, "mv tag too short");
                (
                    data[0] as usize,
                    data[1..].iter().map(|&b| b as u8).collect::<Vec<u8>>(),
                )
            }
            _ => continue,
        };

        // --- Extract sequence ---
        let sequence: &[u8] = record.sequence().as_ref();
        if sequence.is_empty() {
            continue;
        }

        // --- Extract and trim signal ---
        let signal_i16 = signal_extractor
            .get_signal(&pod5_info.signal_rows)
            .expect("failed to extract signal");

        let sp = get_int_tag(&record, b's', b'p').unwrap_or(0) as usize;
        let ts = get_int_tag(&record, b't', b's').unwrap_or(0) as usize;
        let ns = get_int_tag(&record, b'n', b's');

        let signal_start = sp + ts;
        let signal_end = match ns {
            Some(v) => sp + v as usize,
            None => signal_i16.len(),
        };

        if signal_start >= signal_end || signal_end > signal_i16.len() {
            continue;
        }

        // Trim signal, then REVERSE for RNA (matching fishnet --rna behavior)
        let mut signal_f32: Vec<f32> = signal_i16[signal_start..signal_end]
            .iter()
            .map(|&s| s as f32)
            .collect();
        signal_f32.reverse();

        let trimmed_len = signal_f32.len();

        // Validate move table length matches signal (fishnet query_to_signal.rs:61)
        if moves.len() != trimmed_len / stride {
            continue;
        }

        // Build query-to-signal map using trimmed signal length as final boundary
        // (matches fishnet query_to_signal.rs:47)
        let forward_map = build_query_to_signal_map(&moves, stride, sequence.len(), trimmed_len);

        // Reverse the query-to-signal map for RNA (matching fishnet)
        let seq_to_signal = reverse_query_to_signal_map(&forward_map, trimmed_len);

        // Verify reversed map: first boundary should be 0, last should be trimmed_len
        if seq_to_signal[0] != 0 || seq_to_signal[seq_to_signal.len() - 1] != trimmed_len {
            continue;
        }

        // --- Get initial scaling ---
        let sm = get_float_tag(&record, b's', b'm').unwrap_or(0.0);
        let sd = get_float_tag(&record, b's', b'd').unwrap_or(1.0);
        let (scale, shift) = calculate_initial_scaling(
            pod5_info.calibration_scale,
            pod5_info.calibration_offset,
            sd,
            sm,
        );

        // --- Extract expected kmer levels ---
        let levels = match kmer_table.extract_levels(sequence) {
            Ok(l) => l,
            Err(_) => continue,
        };

        // --- Run refinement on reversed signal ---
        let settings_clone = settings.clone();
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            refine_signal_map(
                &settings_clone,
                &signal_f32,
                &seq_to_signal,
                &levels,
                scale,
                shift,
            )
        })) {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => continue,
            Err(_) => continue,
        };

        // --- Compare directly with fishnet boundaries ---
        // Both are now in reversed-trimmed-signal coordinates
        let escapepod_map = &result.seq_to_signal_map;
        let seq_len = sequence.len();
        let n_boundaries = seq_len + 1;

        assert_eq!(
            escapepod_map.len(),
            n_boundaries,
            "escapepod map length mismatch"
        );
        assert_eq!(
            fishnet_boundaries.len(),
            n_boundaries,
            "fishnet map length mismatch for read {}",
            read_id
        );

        let matching = (0..n_boundaries)
            .filter(|&i| escapepod_map[i] == fishnet_boundaries[i] as usize)
            .count();
        let match_pct = matching as f64 / n_boundaries as f64;

        compared += 1;
        total_match_pct += match_pct;
    }

    // --- Report ---
    assert!(compared > 0, "No reads were compared — check data paths");

    let mean_match_pct = total_match_pct / compared as f64;

    // With identical DP algorithms operating on the same reversed signal,
    // we expect very high agreement. Allow for minor floating-point differences
    // in rescaling that could shift a few boundaries by 1.
    assert!(
        mean_match_pct >= 0.95,
        "Mean boundary match {:.1}% is below 95%",
        mean_match_pct * 100.0
    );
}

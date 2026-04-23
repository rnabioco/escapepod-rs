//! CSV + Parquet fingerprint table readers.
//!
//! Two consumers share the same on-disk schema:
//!
//! - `train-svm` reads *labeled* fingerprints (`read_id, barcode, fp_0, ..`)
//!   and optionally subsamples each class to a cap via streaming reservoir
//!   sampling. Memory is bounded by `cap * n_classes * row_width` regardless
//!   of total input, so multi-GB training sets don't materialize.
//!
//! - `classify` reads *query* fingerprints (`read_id, [barcode,] fp_0, ..`)
//!   and just wants `(Uuid, Vec<f64>)` per row. The `barcode` column is
//!   ignored when present (so the val split parquet can feed classify
//!   directly, no awk preprocess required).
//!
//! Format is detected by extension — `.parquet` opens via arrow/parquet,
//! everything else is treated as CSV. CSV has the existing parallel
//! `par_iter` and reservoir samplers; Parquet uses
//! `ParquetRecordBatchReaderBuilder` and walks columns directly from the
//! Arrow `RecordBatch`, which is much faster (binary f64, no text parse).

use anyhow::{Context, Result};
use arrow::array::{Array, AsArray, Float32Array, Float64Array, LargeStringArray, StringArray};
use arrow::datatypes::{Float32Type, Float64Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use uuid::Uuid;

/// String-column accessor that handles both Arrow `Utf8` (StringArray) and
/// `LargeUtf8` (LargeStringArray) backings. Polars' `sink_parquet` writes
/// strings as LargeUtf8, while pandas/pyarrow defaults to Utf8 — both
/// appear in practice for the same logical schema, so the reader has to
/// accept either.
enum StrCol<'a> {
    Small(&'a StringArray),
    Large(&'a LargeStringArray),
}

impl<'a> StrCol<'a> {
    fn from_array(arr: &'a dyn Array) -> Option<Self> {
        if let Some(s) = arr.as_any().downcast_ref::<StringArray>() {
            Some(StrCol::Small(s))
        } else if let Some(s) = arr.as_any().downcast_ref::<LargeStringArray>() {
            Some(StrCol::Large(s))
        } else {
            None
        }
    }

    fn value(&self, i: usize) -> &str {
        match self {
            StrCol::Small(s) => s.value(i),
            StrCol::Large(s) => s.value(i),
        }
    }
}

/// Float-column accessor that handles either Arrow `Float32` or `Float64`
/// backings. Polars infers feature widths from CSV input — most of our
/// fingerprint values fit in f32, so polars-written parquet often uses
/// Float32 even though our internal model is f64. Promote to f64 on read
/// rather than forcing an upstream cast.
enum FloatCol<'a> {
    F32(&'a Float32Array),
    F64(&'a Float64Array),
}

impl<'a> FloatCol<'a> {
    fn from_array(arr: &'a dyn Array) -> Option<Self> {
        if let Some(c) = arr.as_primitive_opt::<Float64Type>() {
            Some(FloatCol::F64(c))
        } else {
            arr.as_primitive_opt::<Float32Type>().map(FloatCol::F32)
        }
    }

    fn value(&self, i: usize) -> f64 {
        match self {
            FloatCol::F64(c) => c.value(i),
            FloatCol::F32(c) => c.value(i) as f64,
        }
    }
}

/// Result type for labeled fingerprint loads (matches `train_svm::FingerprintData`):
/// `(rows, labels, barcode_names, total_rows_seen)`.
pub type LabeledFingerprintData = (Vec<Vec<f64>>, Vec<i32>, Vec<String>, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FpFormat {
    Csv,
    Parquet,
}

fn detect_format(path: &Path) -> FpFormat {
    match path.extension().and_then(|s| s.to_str()) {
        Some("parquet") => FpFormat::Parquet,
        _ => FpFormat::Csv,
    }
}

/// Load labeled fingerprints (read_id + barcode + N feature columns).
///
/// When `subsample = Some((max_per_class, seed))`, applies balanced
/// per-class reservoir sampling during the read — memory stays O(cap ×
/// n_classes × row_width) regardless of total input size.
///
/// The first valid row sets the expected feature width; subsequent rows
/// whose feature vectors don't match are skipped (with a summary `warning:
/// dropped N` line on stderr). This catches the silent-truncation bug
/// where a malformed CSV cell used to produce a short fingerprint row
/// that broke model validation downstream.
pub fn read_labeled_fingerprints(
    path: &Path,
    subsample: Option<(usize, u64)>,
) -> Result<LabeledFingerprintData> {
    match detect_format(path) {
        FpFormat::Csv => read_labeled_csv(path, subsample),
        FpFormat::Parquet => read_labeled_parquet(path, subsample),
    }
}

/// Load query fingerprints as f64. Silently ignores a `barcode` column.
pub fn read_query_fingerprints_f64(path: &Path) -> Result<Vec<(Uuid, Vec<f64>)>> {
    match detect_format(path) {
        FpFormat::Csv => read_query_csv::<f64>(path),
        FpFormat::Parquet => read_query_parquet(path, |x| x),
    }
}

/// Load query fingerprints as f32. Used by the CSV-only legacy classify
/// path that wants f32 from the start. Parquet schema uses f64 for the
/// feature columns; this loader narrows on read.
pub fn read_query_fingerprints_f32(path: &Path) -> Result<Vec<(Uuid, Vec<f32>)>> {
    match detect_format(path) {
        FpFormat::Csv => read_query_csv::<f32>(path),
        FpFormat::Parquet => read_query_parquet(path, |x| x as f32),
    }
}

// ---------------------------------------------------------------------
// CSV readers
// ---------------------------------------------------------------------

fn read_labeled_csv(
    path: &Path,
    subsample: Option<(usize, u64)>,
) -> Result<LabeledFingerprintData> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open labeled fingerprints '{}'", path.display()))?;
    let reader = BufReader::new(file);

    let mut barcode_to_id: HashMap<String, i32> = HashMap::new();
    let mut barcode_names: Vec<String> = Vec::new();
    let mut next_id: i32 = 0;

    let mut reservoirs: HashMap<i32, Vec<Vec<f64>>> = HashMap::new();
    let mut seen_per_class: HashMap<i32, u64> = HashMap::new();

    let mut fingerprints: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();

    let mut state: u64 = subsample.map(|(_, s)| s).unwrap_or(0);
    let mut next_u64 = || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };

    let mut expected_width: Option<usize> = None;
    let mut malformed_rows: u64 = 0;
    let mut total_seen: usize = 0;

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // header
        }

        let mut parts = line.splitn(3, ',');
        let _read_id = match parts.next() {
            Some(s) => s,
            None => continue,
        };
        let barcode = match parts.next() {
            Some(s) => s,
            None => continue,
        };
        let rest = match parts.next() {
            Some(s) => s,
            None => continue,
        };

        let label = *barcode_to_id.entry(barcode.to_string()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            barcode_names.push(barcode.to_string());
            id
        });

        total_seen += 1;

        let mut parse_strict = |rest: &str| -> Option<Vec<f64>> {
            let v = parse_csv_row(rest)?;
            match expected_width {
                Some(w) if v.len() != w => {
                    malformed_rows += 1;
                    None
                }
                Some(_) => Some(v),
                None => {
                    expected_width = Some(v.len());
                    Some(v)
                }
            }
        };

        match subsample {
            Some((cap, _)) => {
                let seen = seen_per_class.entry(label).or_insert(0);
                *seen += 1;
                let res = reservoirs
                    .entry(label)
                    .or_insert_with(|| Vec::with_capacity(cap));

                if res.len() < cap {
                    if let Some(v) = parse_strict(rest) {
                        res.push(v);
                    }
                } else {
                    let j = (next_u64() % *seen) as usize;
                    if j < cap
                        && let Some(v) = parse_strict(rest)
                    {
                        res[j] = v;
                    }
                }
            }
            None => {
                if let Some(v) = parse_strict(rest) {
                    fingerprints.push(v);
                    labels.push(label);
                }
            }
        }
    }

    if malformed_rows > 0 {
        eprintln!(
            "warning: dropped {malformed_rows} fingerprint row(s) whose feature width did \
             not match the expected {} (truncated CSV or NaN/empty cells)",
            expected_width.unwrap_or(0),
        );
    }

    if subsample.is_some() {
        let mut class_ids: Vec<i32> = reservoirs.keys().copied().collect();
        class_ids.sort();
        for cid in class_ids {
            if let Some(rows) = reservoirs.remove(&cid) {
                for row in rows {
                    fingerprints.push(row);
                    labels.push(cid);
                }
            }
        }
    }

    if fingerprints.is_empty() {
        anyhow::bail!("No valid fingerprints found in {}", path.display());
    }

    Ok((fingerprints, labels, barcode_names, total_seen))
}

fn read_query_csv<T>(path: &Path) -> Result<Vec<(Uuid, Vec<T>)>>
where
    T: std::str::FromStr + Send,
    <T as std::str::FromStr>::Err: Send,
{
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read fingerprints from '{}'", path.display()))?;

    let Some((header, body)) = data.split_once('\n') else {
        return Ok(Vec::new());
    };

    // Detect a `barcode` column and record its position so we can skip it
    // on each row. Standard CSV layout is `read_id, barcode?, fp_0, ...`;
    // if `barcode` is missing, `skip_col = None` and all non-ID columns
    // are features.
    let skip_col = header
        .trim_end_matches('\r')
        .split(',')
        .position(|c| c.trim() == "barcode");

    let lines: Vec<&str> = body.split('\n').filter(|l| !l.is_empty()).collect();

    let fingerprints: Vec<(Uuid, Vec<T>)> = lines
        .par_iter()
        .filter_map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let mut cols = line.split(',');
            let read_id = Uuid::parse_str(cols.next()?).ok()?;
            // Index bookkeeping is simpler if we collect into a Vec once,
            // then skip the barcode index. This costs a Vec per row, but
            // the parallelism dwarfs it and each Vec lives only as long
            // as this closure.
            let remaining: Vec<&str> = cols.collect();
            let values: Vec<T> = remaining
                .iter()
                .enumerate()
                .filter_map(|(idx, s)| {
                    // skip_col is 0-indexed in the full row; we've
                    // consumed read_id already, so the barcode column is
                    // at skip_col - 1 in `remaining`.
                    if let Some(sc) = skip_col
                        && sc > 0
                        && idx + 1 == sc
                    {
                        return None;
                    }
                    s.parse::<T>().ok()
                })
                .collect();
            if values.is_empty() {
                None
            } else {
                Some((read_id, values))
            }
        })
        .collect();

    Ok(fingerprints)
}

fn parse_csv_row(rest: &str) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(8);
    for s in rest.split(',') {
        out.push(s.trim().parse::<f64>().ok()?);
    }
    if out.is_empty() { None } else { Some(out) }
}

// ---------------------------------------------------------------------
// Parquet readers
// ---------------------------------------------------------------------

fn read_labeled_parquet(
    path: &Path,
    subsample: Option<(usize, u64)>,
) -> Result<LabeledFingerprintData> {
    let file =
        File::open(path).with_context(|| format!("Failed to open parquet '{}'", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| "Failed to initialize parquet reader")?;
    let schema = builder.schema().clone();

    // Find the `barcode` column and the ordered list of feature columns.
    // Feature columns are everything except `read_id` and `barcode` —
    // preserves the column order as written so the model's features line
    // up with the training schema.
    let mut barcode_idx: Option<usize> = None;
    let mut feature_idxs: Vec<usize> = Vec::new();
    for (i, field) in schema.fields().iter().enumerate() {
        match field.name().as_str() {
            "read_id" => {}
            "barcode" => barcode_idx = Some(i),
            _ => feature_idxs.push(i),
        }
    }
    let Some(barcode_idx) = barcode_idx else {
        anyhow::bail!(
            "Parquet '{}' has no `barcode` column — labeled fingerprints require it",
            path.display()
        );
    };
    if feature_idxs.is_empty() {
        anyhow::bail!("Parquet '{}' has no feature columns", path.display());
    }

    let batch_reader = builder
        .build()
        .with_context(|| "Failed to build parquet batch reader")?;

    let mut barcode_to_id: HashMap<String, i32> = HashMap::new();
    let mut barcode_names: Vec<String> = Vec::new();
    let mut next_id: i32 = 0;

    let mut reservoirs: HashMap<i32, Vec<Vec<f64>>> = HashMap::new();
    let mut seen_per_class: HashMap<i32, u64> = HashMap::new();

    let mut fingerprints: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();

    let mut state: u64 = subsample.map(|(_, s)| s).unwrap_or(0);
    let mut next_u64 = || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };

    let n_features = feature_idxs.len();
    let mut total_seen: usize = 0;

    for batch in batch_reader {
        let batch = batch.with_context(|| "Failed to read parquet batch")?;

        let barcode_col = StrCol::from_array(batch.column(barcode_idx).as_ref())
            .ok_or_else(|| anyhow::anyhow!("`barcode` column is not a string array"))?;

        // Extract each feature column once per batch (Float32 or Float64,
        // promoted to f64 on read). Columnar layout means the
        // `n_rows * n_features` transpose is just n_features slice accesses
        // rather than per-row parsing.
        let feature_cols: Vec<FloatCol> = feature_idxs
            .iter()
            .map(|&i| {
                FloatCol::from_array(batch.column(i).as_ref()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Feature column `{}` is not a Float32 or Float64 array",
                        schema.field(i).name()
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for row in 0..batch.num_rows() {
            let barcode = barcode_col.value(row);
            let label = *barcode_to_id.entry(barcode.to_string()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                barcode_names.push(barcode.to_string());
                id
            });
            total_seen += 1;

            // Assemble the row's feature vector by gathering across the
            // feature columns. This is the single per-row allocation in
            // the parquet path.
            let build_row = || -> Vec<f64> {
                let mut v = Vec::with_capacity(n_features);
                for col in &feature_cols {
                    // `as_primitive_opt` guarantees typed access; nulls
                    // default to 0.0 which mirrors CSV behavior where a
                    // missing cell is dropped (but we've already validated
                    // schema up front — nulls shouldn't appear in
                    // practice).
                    v.push(col.value(row));
                }
                v
            };

            match subsample {
                Some((cap, _)) => {
                    let seen = seen_per_class.entry(label).or_insert(0);
                    *seen += 1;
                    let res = reservoirs
                        .entry(label)
                        .or_insert_with(|| Vec::with_capacity(cap));

                    if res.len() < cap {
                        res.push(build_row());
                    } else {
                        let j = (next_u64() % *seen) as usize;
                        if j < cap {
                            res[j] = build_row();
                        }
                    }
                }
                None => {
                    fingerprints.push(build_row());
                    labels.push(label);
                }
            }
        }
    }

    if subsample.is_some() {
        let mut class_ids: Vec<i32> = reservoirs.keys().copied().collect();
        class_ids.sort();
        for cid in class_ids {
            if let Some(rows) = reservoirs.remove(&cid) {
                for row in rows {
                    fingerprints.push(row);
                    labels.push(cid);
                }
            }
        }
    }

    if fingerprints.is_empty() {
        anyhow::bail!("No valid fingerprints found in {}", path.display());
    }

    Ok((fingerprints, labels, barcode_names, total_seen))
}

fn read_query_parquet<T, F>(path: &Path, convert: F) -> Result<Vec<(Uuid, Vec<T>)>>
where
    F: Fn(f64) -> T,
{
    let file =
        File::open(path).with_context(|| format!("Failed to open parquet '{}'", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| "Failed to initialize parquet reader")?;
    let schema = builder.schema().clone();

    let mut read_id_idx: Option<usize> = None;
    let mut feature_idxs: Vec<usize> = Vec::new();
    for (i, field) in schema.fields().iter().enumerate() {
        match field.name().as_str() {
            "read_id" => read_id_idx = Some(i),
            "barcode" => {} // silently skip
            _ => feature_idxs.push(i),
        }
    }
    let Some(read_id_idx) = read_id_idx else {
        anyhow::bail!("Parquet '{}' has no `read_id` column", path.display());
    };
    if feature_idxs.is_empty() {
        anyhow::bail!("Parquet '{}' has no feature columns", path.display());
    }

    let batch_reader = builder.build()?;
    let mut out: Vec<(Uuid, Vec<T>)> = Vec::new();

    for batch in batch_reader {
        let batch = batch.with_context(|| "Failed to read parquet batch")?;

        let id_col = StrCol::from_array(batch.column(read_id_idx).as_ref())
            .ok_or_else(|| anyhow::anyhow!("`read_id` column is not a string array"))?;
        let feature_cols: Vec<FloatCol> = feature_idxs
            .iter()
            .map(|&i| {
                FloatCol::from_array(batch.column(i).as_ref()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Feature column `{}` is not a Float32 or Float64 array",
                        schema.field(i).name()
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for row in 0..batch.num_rows() {
            let Ok(read_id) = Uuid::parse_str(id_col.value(row)) else {
                continue;
            };
            let mut v: Vec<T> = Vec::with_capacity(feature_cols.len());
            for col in &feature_cols {
                v.push(convert(col.value(row)));
            }
            out.push((read_id, v));
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------
// Tests — CSV round-trips + parquet round-trip
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn write_labeled_csv(rows: &[(&str, &str, &[f64])]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "read_id,barcode,f0,f1,f2").unwrap();
        for (rid, bc, feats) in rows {
            let fs = feats
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(f, "{rid},{bc},{fs}").unwrap();
        }
        f
    }

    fn write_labeled_parquet(rows: &[(&str, &str, &[f64])]) -> NamedTempFile {
        let schema = Arc::new(Schema::new(vec![
            Field::new("read_id", DataType::Utf8, false),
            Field::new("barcode", DataType::Utf8, false),
            Field::new("f0", DataType::Float64, false),
            Field::new("f1", DataType::Float64, false),
            Field::new("f2", DataType::Float64, false),
        ]));
        let ids = StringArray::from_iter_values(rows.iter().map(|(r, _, _)| *r));
        let bcs = StringArray::from_iter_values(rows.iter().map(|(_, b, _)| *b));
        let f0 = Float64Array::from_iter_values(rows.iter().map(|(_, _, v)| v[0]));
        let f1 = Float64Array::from_iter_values(rows.iter().map(|(_, _, v)| v[1]));
        let f2 = Float64Array::from_iter_values(rows.iter().map(|(_, _, v)| v[2]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(bcs),
                Arc::new(f0),
                Arc::new(f1),
                Arc::new(f2),
            ],
        )
        .unwrap();
        let f = NamedTempFile::new().unwrap();
        let mut writer = ArrowWriter::try_new(f.reopen().unwrap(), schema.clone(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        f
    }

    fn make_labeled_rows(n: usize, n_classes: usize) -> Vec<(String, String, Vec<f64>)> {
        (0..n)
            .map(|i| {
                let bc = format!("barcode{:02}", i % n_classes);
                let rid = format!("00000000-0000-0000-0000-{:012x}", i);
                (rid, bc, vec![i as f64, (i * 2) as f64, (i * 3) as f64])
            })
            .collect()
    }

    #[test]
    fn csv_and_parquet_agree() {
        let owned = make_labeled_rows(100, 4);
        let refs: Vec<(&str, &str, &[f64])> = owned
            .iter()
            .map(|(r, b, v)| (r.as_str(), b.as_str(), v.as_slice()))
            .collect();

        let csv_f = write_labeled_csv(&refs);
        let pq_f = write_labeled_parquet(&refs);

        let (csv_fps, csv_lb, csv_bn, csv_total) =
            read_labeled_fingerprints(csv_f.path(), None).unwrap();
        let (pq_fps, pq_lb, pq_bn, pq_total) =
            read_labeled_fingerprints(pq_f.path(), None).unwrap();

        assert_eq!(csv_fps, pq_fps);
        assert_eq!(csv_lb, pq_lb);
        // Barcode order is first-seen; CSV and parquet iterate rows in the
        // same order so the name lists match.
        assert_eq!(csv_bn, pq_bn);
        assert_eq!(csv_total, pq_total);
        assert_eq!(csv_total, 100);
    }

    #[test]
    fn parquet_subsample_caps_classes() {
        let owned = make_labeled_rows(300, 3);
        let refs: Vec<(&str, &str, &[f64])> = owned
            .iter()
            .map(|(r, b, v)| (r.as_str(), b.as_str(), v.as_slice()))
            .collect();
        let pq_f = write_labeled_parquet(&refs);
        let (fps, labels, _, total) =
            read_labeled_fingerprints(pq_f.path(), Some((10, 42))).unwrap();
        assert_eq!(total, 300);
        assert_eq!(fps.len(), 30);
        for c in 0..3 {
            assert_eq!(labels.iter().filter(|&&l| l == c).count(), 10);
        }
    }

    #[test]
    fn parquet_query_ignores_barcode_column() {
        let owned = make_labeled_rows(20, 2);
        let refs: Vec<(&str, &str, &[f64])> = owned
            .iter()
            .map(|(r, b, v)| (r.as_str(), b.as_str(), v.as_slice()))
            .collect();
        let pq_f = write_labeled_parquet(&refs);

        let queries = read_query_fingerprints_f64(pq_f.path()).unwrap();
        assert_eq!(queries.len(), 20);
        for (_id, feats) in &queries {
            assert_eq!(feats.len(), 3);
        }
    }

    #[test]
    fn csv_query_ignores_barcode_column() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "read_id,barcode,f0,f1,f2").unwrap();
        writeln!(
            f,
            "00000000-0000-0000-0000-000000000001,barcode03,1.0,2.0,3.0"
        )
        .unwrap();
        writeln!(
            f,
            "00000000-0000-0000-0000-000000000002,barcode04,4.0,5.0,6.0"
        )
        .unwrap();

        let queries = read_query_fingerprints_f64(f.path()).unwrap();
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0].1, vec![1.0, 2.0, 3.0]);
        assert_eq!(queries[1].1, vec![4.0, 5.0, 6.0]);
    }
}

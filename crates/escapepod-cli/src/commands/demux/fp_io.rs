//! CSV + Parquet fingerprint table readers.
//!
//! `classify` reads *query* fingerprints (`read_id, [barcode,] fp_0, ..`)
//! and just wants `(Uuid, Vec<f64>)` per row. The `barcode` column is
//! ignored when present (so the val split parquet can feed classify
//! directly, no awk preprocess required).
//!
//! Format is detected by extension — `.parquet` opens via arrow/parquet,
//! everything else is treated as CSV. CSV has the existing parallel
//! `par_iter`; Parquet uses
//! `ParquetRecordBatchReaderBuilder` and walks columns directly from the
//! Arrow `RecordBatch`, which is much faster (binary f64, no text parse).

use anyhow::{Context, Result};
use arrow::array::{Array, AsArray, Float32Array, Float64Array, LargeStringArray, StringArray};
use arrow::datatypes::{Float32Type, Float64Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;
use tracing::warn;
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
        } else {
            arr.as_any()
                .downcast_ref::<LargeStringArray>()
                .map(StrCol::Large)
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

    // Parse each row strictly: a cell that does not parse rejects the whole
    // row rather than silently shortening it. A short fingerprint is not a
    // recoverable condition downstream — the SVM/DTW path would happily
    // classify a truncated query and report a confident wrong barcode — so
    // it must never be produced. Mirrors `parse_csv_row` in the labeled
    // loader, which has the same guard.
    let parsed: Vec<Option<(Uuid, Vec<T>)>> = lines
        .par_iter()
        .map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let mut cols = line.split(',');
            let read_id = Uuid::parse_str(cols.next()?.trim()).ok()?;
            // `skip_col` is 0-indexed in the full row and we have already
            // consumed `read_id`, so the barcode sits at `skip_col - 1`
            // among the remaining columns.
            let mut values: Vec<T> = Vec::with_capacity(16);
            for (idx, s) in cols.enumerate() {
                if let Some(sc) = skip_col
                    && sc > 0
                    && idx + 1 == sc
                {
                    continue;
                }
                values.push(s.trim().parse::<T>().ok()?);
            }
            if values.is_empty() {
                None
            } else {
                Some((read_id, values))
            }
        })
        .collect();

    // The first successfully parsed row establishes the expected feature
    // width; anything that disagrees is dropped and counted rather than
    // being passed to the classifier.
    let expected_width = parsed
        .iter()
        .flatten()
        .map(|(_, v)| v.len())
        .next()
        .unwrap_or(0);

    let mut fingerprints = Vec::with_capacity(parsed.len());
    let mut malformed_rows = 0usize;
    for row in parsed {
        match row {
            Some((id, v)) if v.len() == expected_width => fingerprints.push((id, v)),
            _ => malformed_rows += 1,
        }
    }

    if malformed_rows > 0 {
        warn!(
            "dropped {malformed_rows} query fingerprint row(s) from '{}' that were \
             unparseable or whose feature width did not match the expected \
             {expected_width} (truncated CSV or NaN/empty cells)",
            path.display(),
        );
    }

    Ok(fingerprints)
}

// ---------------------------------------------------------------------
// Parquet readers
// ---------------------------------------------------------------------

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
        // Needs a `.parquet` suffix so `detect_format` reads it as parquet
        // rather than falling back to CSV (production files are `*.parquet`).
        let f = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();
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

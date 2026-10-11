//! Read a read-id → label table into sidecar-ready label columns.
//!
//! The table is whatever another tool wrote: a CSV or TSV with a header, one
//! column of read ids and one or more columns of labels. [`LabelTableBuilder`]
//! parses it straight into the form [`write_columns`](super::write_columns) and
//! [`write_collection_columns`](super::write_collection_columns) take for a
//! large run — [`ColumnValues::LabelCodesIndexed`]: one read-id → slot map
//! shared by every column, and per column a dictionary plus one `u32` code per
//! slot. A `HashMap<Uuid, String>` per column would hold a heap string per read
//! to say one of a handful of values, at tens of millions of reads.
//!
//! What this module deliberately does **not** do is decide what a read the
//! table never mentions should be called. A read absent from the table, and a
//! read with an empty cell, both come out as [`ABSENT_CODE`]; the caller fills
//! them (or leaves them absent). That decision needs the list of reads in the
//! POD5s, which a table parser has no business holding.
//!
//! Gzip is the caller's: this takes any [`Read`], so `MultiGzDecoder` (never
//! `GzDecoder` — it stops after the first member, and a truncated table would
//! silently turn real calls into "unassigned") goes around it.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;

use super::annotate::{ABSENT_CODE, ColumnValues, ColumnWrite};
use crate::error::{Error, Result};
use crate::sidecar::{MAX_ANNOTATION_LABELS, RESERVED_COLUMNS};
use crate::types::Uuid;
use crate::utils::parse_uuid_flexible;

/// How many header names an error lists before eliding the rest. A
/// WarpDemuX-shaped table has dozens of columns; naming all of them buries the
/// one the caller needs.
const HEADERS_SHOWN: usize = 30;

/// One label column of a [`LabelTable`].
#[derive(Debug, Clone)]
pub struct TableColumn {
    /// The sidecar column name it will be written under.
    pub name: String,
    /// Distinct labels, indexed by code. Verbatim from the table, trimmed.
    pub dictionary: Vec<String>,
    /// Per slot of [`LabelTable::index`]: an index into `dictionary`, or
    /// [`ABSENT_CODE`] for an empty cell. Always `index.len()` long.
    pub codes: Vec<u32>,
}

/// The parsed table: every read it names, and each requested column's label.
#[derive(Debug, Clone)]
pub struct LabelTable {
    /// Read id → slot, shared by every column. Owned rather than `Arc`'d so a
    /// caller can append slots (for reads the table does not name) before
    /// handing it on; [`into_column_writes`](Self::into_column_writes) wraps it.
    pub index: HashMap<Uuid, u32>,
    /// The requested columns, in the order they were asked for.
    pub columns: Vec<TableColumn>,
    /// Data rows read, counting repeats.
    pub rows: usize,
    /// Rows whose id had already appeared with exactly the same labels.
    pub identical_duplicates: usize,
}

impl LabelTable {
    /// Distinct reads the table names.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the table named no reads.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Turn the columns into [`ColumnWrite`]s sharing one `Arc`'d index, ready
    /// for [`write_columns`](super::write_columns) /
    /// [`write_collection_columns`](super::write_collection_columns).
    ///
    /// Every column's `codes` must be `index.len()` long: the writers read a
    /// slot with `get`, so a slot without a code is dropped without a word.
    pub fn into_column_writes(self) -> Vec<ColumnWrite> {
        debug_assert!(
            self.columns
                .iter()
                .all(|c| c.codes.len() == self.index.len())
        );
        let index = Arc::new(self.index);
        self.columns
            .into_iter()
            .map(|c| ColumnWrite {
                name: c.name,
                values: ColumnValues::LabelCodesIndexed {
                    dictionary: c.dictionary,
                    index: Arc::clone(&index),
                    codes: c.codes,
                },
            })
            .collect()
    }
}

/// Dictionary-encodes one column as it is read.
#[derive(Default)]
struct Interner {
    codes: HashMap<String, u32>,
    dictionary: Vec<String>,
}

impl Interner {
    /// The code for `label`, or `None` once the column would exceed what a
    /// sidecar annotation can hold.
    fn intern(&mut self, label: &str) -> Option<u32> {
        if let Some(&code) = self.codes.get(label) {
            return Some(code);
        }
        if self.dictionary.len() >= MAX_ANNOTATION_LABELS {
            return None;
        }
        let code = self.dictionary.len() as u32;
        self.codes.insert(label.to_string(), code);
        self.dictionary.push(label.to_string());
        Some(code)
    }
}

/// Accumulates one or more tables (shards of one run) into a [`LabelTable`].
///
/// ```
/// use escapepod_pod5::operations::LabelTableBuilder;
///
/// let table = "read_id,barcode\n\
///              a1b2c3d4-e5f6-7890-abcd-ef1234567890,BC01\n\
///              b2c3d4e5-f6a7-8901-bcde-f12345678901,\n";
/// let mut builder =
///     LabelTableBuilder::new("read_id", &[("barcode".into(), "barcode".into())])?;
/// builder.add_table(table.as_bytes(), "table.csv")?;
/// let table = builder.finish();
/// assert_eq!(table.len(), 2);
/// assert_eq!(table.columns[0].dictionary, ["BC01"]);
/// # Ok::<(), escapepod_pod5::Error>(())
/// ```
pub struct LabelTableBuilder {
    id_column: String,
    /// `(sidecar name, table column)`.
    columns: Vec<(String, String)>,
    index: HashMap<Uuid, u32>,
    interners: Vec<Interner>,
    codes: Vec<Vec<u32>>,
    rows: usize,
    identical_duplicates: usize,
}

impl LabelTableBuilder {
    /// Start a table read: `id_column` names the read-id column, and each
    /// `(name, column)` of `columns` maps a table column onto the sidecar
    /// column `name` it will be written as.
    ///
    /// The sidecar names are checked here — before a byte of any table is
    /// read — because a name the sidecar refuses (empty, one of
    /// [`RESERVED_COLUMNS`]) or two columns under one name would otherwise cost
    /// a parse of a multi-gigabyte table to find out.
    pub fn new(id_column: &str, columns: &[(String, String)]) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::Parse(
                "no label column requested; name at least one".to_string(),
            ));
        }
        for (i, (name, _)) in columns.iter().enumerate() {
            if name.is_empty() || RESERVED_COLUMNS.contains(&name.as_str()) {
                return Err(Error::Parse(format!(
                    "'{name}' is not a valid annotation name (empty, or reserved for the read \
                     index: {})",
                    RESERVED_COLUMNS.join(", ")
                )));
            }
            if columns[..i].iter().any(|(earlier, _)| earlier == name) {
                return Err(Error::Parse(format!(
                    "annotation '{name}' is requested more than once"
                )));
            }
        }
        Ok(Self {
            id_column: id_column.to_string(),
            columns: columns.to_vec(),
            index: HashMap::new(),
            interners: columns.iter().map(|_| Interner::default()).collect(),
            codes: columns.iter().map(|_| Vec::new()).collect(),
            rows: 0,
            identical_duplicates: 0,
        })
    }

    /// Read one table. `source` names it in errors (a path, usually).
    ///
    /// The delimiter is a tab if the header line contains one, else a comma. A
    /// leading UTF-8 byte-order mark is dropped. A column is found by exact
    /// header name first, then with one leading `#` trimmed from the header —
    /// WarpDemuX writes `#read_id`, and the default `read_id` should find it.
    /// Labels are taken verbatim apart from surrounding whitespace; an empty
    /// cell is "no label".
    ///
    /// A read that appears again with the same labels is counted; with
    /// different labels it is an error, here or across tables, since either
    /// way one of the two calls would silently win.
    pub fn add_table(&mut self, input: impl Read, source: &str) -> Result<()> {
        let mut reader = BufReader::new(input);

        // The header is read by hand, not by the csv reader, because the
        // delimiter has to be known before that reader is built.
        let mut head = Vec::new();
        let n = reader
            .read_until(b'\n', &mut head)
            .map_err(|e| Error::Parse(format!("{source}: cannot read the header: {e}")))?;
        if n == 0 {
            return Err(Error::Parse(format!(
                "{source}: empty table (no header row)"
            )));
        }
        if head.starts_with(&[0xEF, 0xBB, 0xBF]) {
            head.drain(..3);
        }
        let head = String::from_utf8(head)
            .map_err(|_| Error::Parse(format!("{source}: the header is not valid UTF-8")))?;
        let head = head.trim_end_matches(['\n', '\r']);
        let delimiter = if head.contains('\t') { b'\t' } else { b',' };

        let header: Vec<String> = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(false)
            .flexible(true)
            .from_reader(head.as_bytes())
            .records()
            .next()
            .transpose()
            .map_err(|e| Error::Parse(format!("{source}: bad header row: {e}")))?
            .map(|r| r.iter().map(|c| c.trim().to_string()).collect())
            .unwrap_or_default();

        let find = |name: &str| -> Result<usize> {
            header
                .iter()
                .position(|h| h == name)
                .or_else(|| {
                    header
                        .iter()
                        .position(|h| h.strip_prefix('#').is_some_and(|h| h.trim() == name))
                })
                .ok_or_else(|| {
                    let shown: Vec<String> = header
                        .iter()
                        .take(HEADERS_SHOWN)
                        .map(|h| format!("'{h}'"))
                        .collect();
                    let more = header.len().saturating_sub(HEADERS_SHOWN);
                    Error::Parse(format!(
                        "{source}: no column '{name}' in the header, which has: {}{}",
                        shown.join(", "),
                        if more > 0 {
                            format!(" (and {more} more)")
                        } else {
                            String::new()
                        }
                    ))
                })
        };
        let id_idx = find(&self.id_column)?;
        let col_idx: Vec<usize> = self
            .columns
            .iter()
            .map(|(_, column)| find(column))
            .collect::<Result<_>>()?;
        let header_id_cell = header[id_idx].clone();

        let mut records = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(false)
            .flexible(true)
            .from_reader(reader);
        let mut record = csv::StringRecord::new();
        let mut row_codes = vec![ABSENT_CODE; self.columns.len()];

        loop {
            match records.read_record(&mut record) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => return Err(Error::Parse(format!("{source}: {e}"))),
            }
            // The data reader started after the header line.
            let line = record.position().map_or(0, |p| p.line()) + 1;
            self.rows += 1;

            let cell = |idx: usize| -> Result<&str> {
                record.get(idx).map(str::trim).ok_or_else(|| {
                    Error::Parse(format!(
                        "{source}, line {line}: the row has {} cells, but column {} is needed",
                        record.len(),
                        idx + 1
                    ))
                })
            };

            let id_cell = cell(id_idx)?;
            // A header row in the middle of the data means two tables were
            // concatenated. Skipping it would work for exactly the tables that
            // agree on their columns, and give a wrong answer for any that do
            // not — the second one's labels would be read from the first's
            // layout. So it is refused, not skipped.
            if id_cell == header_id_cell
                || id_cell == self.id_column
                || id_cell
                    .strip_prefix('#')
                    .is_some_and(|c| c.trim() == self.id_column)
            {
                return Err(Error::Parse(format!(
                    "{source}, line {line}: a repeated header row — this looks like concatenated \
                     tables, which are not supported; pass each table separately (repeat \
                     --from-table)"
                )));
            }
            let uuid = parse_uuid_flexible(id_cell).map_err(|_| {
                Error::Parse(format!(
                    "{source}, line {line}: '{id_cell}' in column '{}' is not a read id (UUID)",
                    self.id_column
                ))
            })?;

            for (c, &idx) in col_idx.iter().enumerate() {
                let label = cell(idx)?;
                row_codes[c] = if label.is_empty() {
                    ABSENT_CODE
                } else {
                    self.interners[c].intern(label).ok_or_else(|| {
                        Error::Parse(format!(
                            "{source}, line {line}: column '{}' has more than \
                             {MAX_ANNOTATION_LABELS} distinct labels, which a sidecar annotation \
                             cannot hold — is it really a label column (and not a score or an \
                             id)?",
                            self.columns[c].1
                        ))
                    })?
                };
            }

            match self.index.get(&uuid) {
                Some(&slot) => {
                    let slot = slot as usize;
                    for (c, &code) in row_codes.iter().enumerate() {
                        let before = self.codes[c][slot];
                        if before != code {
                            let show = |code: u32| -> String {
                                if code == ABSENT_CODE {
                                    "(empty)".to_string()
                                } else {
                                    format!("'{}'", self.interners[c].dictionary[code as usize])
                                }
                            };
                            return Err(Error::Parse(format!(
                                "{source}, line {line}: read {uuid} appears twice with different \
                                 labels in column '{}' ({} then {})",
                                self.columns[c].1,
                                show(before),
                                show(code)
                            )));
                        }
                    }
                    self.identical_duplicates += 1;
                }
                None => {
                    let slot = u32::try_from(self.index.len()).map_err(|_| {
                        Error::Parse(format!("{source}: more than {} reads", u32::MAX))
                    })?;
                    self.index.insert(uuid, slot);
                    for (column_codes, &code) in self.codes.iter_mut().zip(&row_codes) {
                        column_codes.push(code);
                    }
                }
            }
        }
        Ok(())
    }

    /// Everything read so far, as one table.
    pub fn finish(self) -> LabelTable {
        LabelTable {
            index: self.index,
            columns: self
                .columns
                .into_iter()
                .zip(self.interners)
                .zip(self.codes)
                .map(|(((name, _), interner), codes)| TableColumn {
                    name,
                    dictionary: interner.dictionary,
                    codes,
                })
                .collect(),
            rows: self.rows,
            identical_duplicates: self.identical_duplicates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
    const B: &str = "b2c3d4e5-f6a7-8901-bcde-f12345678901";
    const C: &str = "c3d4e5f6-a7b8-9012-cdef-123456789012";

    fn cols(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    fn parse(text: &str, id: &str, pairs: &[(&str, &str)]) -> Result<LabelTable> {
        let mut b = LabelTableBuilder::new(id, &cols(pairs))?;
        b.add_table(text.as_bytes(), "t")?;
        Ok(b.finish())
    }

    /// `(read, label)` per read of column `c`, sorted, for comparing tables
    /// whose slot numbering is an implementation detail.
    fn labelled(t: &LabelTable, c: usize) -> Vec<(Uuid, Option<String>)> {
        let mut v: Vec<_> = t
            .index
            .iter()
            .map(|(id, &slot)| {
                let code = t.columns[c].codes[slot as usize];
                (
                    *id,
                    (code != ABSENT_CODE).then(|| t.columns[c].dictionary[code as usize].clone()),
                )
            })
            .collect();
        v.sort();
        v
    }

    #[test]
    fn comma_and_tab_tables_agree() {
        let csv = format!("read_id,barcode,score\n{A},4,0.9\n{B},-1,0.2\n{C},4,0.8\n");
        let tsv = csv.replace(',', "\t");
        let from_csv = parse(&csv, "read_id", &[("barcode", "barcode")]).unwrap();
        let from_tsv = parse(&tsv, "read_id", &[("barcode", "barcode")]).unwrap();
        assert_eq!(labelled(&from_csv, 0), labelled(&from_tsv, 0));
        assert_eq!(from_csv.len(), 3);
        // Bare ints, including -1, are labels like any other.
        let mut dictionary = from_csv.columns[0].dictionary.clone();
        dictionary.sort();
        assert_eq!(dictionary, ["-1", "4"]);
    }

    #[test]
    fn leading_hash_header_matches_plain_id_column() {
        // The shape WarpDemuX writes.
        let text = format!("#read_id,predicted_barcode,confidence_score\n{A},7,0.99\n");
        let t = parse(&text, "read_id", &[("barcode", "predicted_barcode")]).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t.columns[0].dictionary, ["7"]);

        // An exact match wins over the `#`-trimmed one.
        let text = format!("#read_id,read_id,x\nnot-an-id,{A},1\n");
        let t = parse(&text, "read_id", &[("x", "x")]).unwrap();
        assert_eq!(t.len(), 1);

        // And the `#` form can be asked for by its full name.
        let text = format!("#read_id,x\n{A},1\n");
        assert_eq!(parse(&text, "#read_id", &[("x", "x")]).unwrap().len(), 1);
    }

    #[test]
    fn bom_is_stripped() {
        let text = format!("\u{feff}read_id,barcode\n{A},BC01\n");
        let t = parse(&text, "read_id", &[("barcode", "barcode")]).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t.columns[0].dictionary, ["BC01"]);
    }

    #[test]
    fn missing_column_error_lists_headers() {
        let text = format!("read_id,predicted_barcode,confidence_score\n{A},7,0.9\n");
        let err = parse(&text, "read_id", &[("barcode", "barcode")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("no column 'barcode'"), "{err}");
        for header in ["'read_id'", "'predicted_barcode'", "'confidence_score'"] {
            assert!(err.contains(header), "{header} missing from: {err}");
        }
        let err = parse(&text, "id", &[("barcode", "predicted_barcode")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("no column 'id'"), "{err}");

        // A very wide header is elided rather than printed in full.
        let wide: Vec<String> = (0..100).map(|i| format!("c{i}")).collect();
        let text = format!("{}\n", wide.join(","));
        let err = parse(&text, "read_id", &[("barcode", "c1")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("and 70 more"), "{err}");
    }

    #[test]
    fn repeated_header_row_is_an_error_not_skipped() {
        let text = format!("read_id,barcode\n{A},BC01\nread_id,barcode\n{B},BC02\n");
        let err = parse(&text, "read_id", &[("barcode", "barcode")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("repeated header row"), "{err}");
        assert!(err.contains("line 3"), "{err}");
        assert!(err.contains("repeat --from-table"), "{err}");

        // The `#` spelling counts too, whichever header came first.
        let text = format!("#read_id,barcode\n{A},BC01\n#read_id,barcode\n");
        let err = parse(&text, "read_id", &[("barcode", "barcode")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("repeated header row"), "{err}");
    }

    #[test]
    fn non_uuid_id_names_its_line() {
        let text = format!("read_id,barcode\n{A},BC01\nnot-a-uuid,BC02\n");
        let err = parse(&text, "read_id", &[("barcode", "barcode")])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("line 3") && err.contains("not-a-uuid"),
            "{err}"
        );
    }

    #[test]
    fn conflicting_duplicate_ids_are_refused() {
        let text = format!("read_id,barcode\n{A},BC01\n{A},BC02\n");
        let err = parse(&text, "read_id", &[("barcode", "barcode")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("appears twice with different labels"), "{err}");
        assert!(err.contains("'BC01'") && err.contains("'BC02'"), "{err}");

        // A label against an empty cell is a disagreement too.
        let text = format!("read_id,barcode\n{A},BC01\n{A},\n");
        assert!(parse(&text, "read_id", &[("barcode", "barcode")]).is_err());

        // So is a disagreement between two shards of one run.
        let mut b = LabelTableBuilder::new("read_id", &cols(&[("barcode", "barcode")])).unwrap();
        b.add_table(format!("read_id,barcode\n{A},BC01\n").as_bytes(), "shard0")
            .unwrap();
        let err = b
            .add_table(format!("read_id,barcode\n{A},BC02\n").as_bytes(), "shard1")
            .unwrap_err()
            .to_string();
        assert!(err.contains("shard1"), "{err}");
    }

    #[test]
    fn identical_duplicates_are_counted() {
        let text = format!("read_id,barcode\n{A},BC01\n{B},BC02\n{A},BC01\n{A},BC01\n");
        let t = parse(&text, "read_id", &[("barcode", "barcode")]).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t.rows, 4);
        assert_eq!(t.identical_duplicates, 2);
        assert_eq!(t.columns[0].codes.len(), 2);
    }

    #[test]
    fn empty_cell_is_unassigned() {
        let text = format!("read_id,barcode\n{A},BC01\n{B},\n{C},  \n");
        let t = parse(&text, "read_id", &[("barcode", "barcode")]).unwrap();
        assert_eq!(t.len(), 3);
        let labelled = labelled(&t, 0);
        assert_eq!(labelled.iter().filter(|(_, l)| l.is_none()).count(), 2);
        // An unassigned read holds a slot (it is *in* the table) but no code.
        assert_eq!(t.columns[0].dictionary, ["BC01"]);
    }

    #[test]
    fn columns_share_one_index_and_code_len_matches() {
        let text = format!("read_id,ldx,fdx\n{A},1,x\n{B},2,\n{C},,y\n");
        let t = parse(&text, "read_id", &[("ldx", "ldx"), ("fdx", "fdx")]).unwrap();
        assert!(t.columns.iter().all(|c| c.codes.len() == t.len()));
        let writes = t.into_column_writes();
        assert_eq!(writes.len(), 2);
        let mut indexes = Vec::new();
        for w in &writes {
            match &w.values {
                ColumnValues::LabelCodesIndexed { index, codes, .. } => {
                    assert_eq!(codes.len(), index.len(), "{}", w.name);
                    indexes.push(Arc::clone(index));
                }
                other => panic!("expected LabelCodesIndexed, got {other:?}"),
            }
        }
        assert!(
            Arc::ptr_eq(&indexes[0], &indexes[1]),
            "columns must share one read-id map, not hold one each"
        );
        assert_eq!(writes[0].name, "ldx");
        assert_eq!(writes[1].name, "fdx");
    }

    #[test]
    fn more_labels_than_a_sidecar_holds_fails_during_parsing() {
        // `--column barcode=confidence_score` is the mistake this catches: a
        // continuous column has a distinct value per read.
        let mut text = String::from("read_id,score\n");
        for i in 0..=MAX_ANNOTATION_LABELS {
            let id = Uuid::from_u128(i as u128 + 1);
            text.push_str(&format!("{id},{i}\n"));
        }
        let err = parse(&text, "read_id", &[("barcode", "score")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("distinct labels"), "{err}");
        // Exactly the limit is fine.
        let mut text = String::from("read_id,score\n");
        for i in 0..MAX_ANNOTATION_LABELS {
            let id = Uuid::from_u128(i as u128 + 1);
            text.push_str(&format!("{id},{i}\n"));
        }
        assert!(parse(&text, "read_id", &[("barcode", "score")]).is_ok());
    }

    #[test]
    fn sidecar_names_are_checked_before_any_table_is_read() {
        for bad in ["read_id", "batch_idx", "row_idx", "member_idx", ""] {
            assert!(
                LabelTableBuilder::new("read_id", &cols(&[(bad, "x")])).is_err(),
                "{bad:?} accepted"
            );
        }
        let err = LabelTableBuilder::new("read_id", &cols(&[("a", "x"), ("a", "y")]))
            .err()
            .expect("duplicate names refused")
            .to_string();
        assert!(err.contains("more than once"), "{err}");
        assert!(LabelTableBuilder::new("read_id", &[]).is_err());
        // Two sidecar names may read one table column.
        assert!(LabelTableBuilder::new("read_id", &cols(&[("a", "x"), ("b", "x")])).is_ok());
    }

    #[test]
    fn ragged_rows_are_fine_when_the_needed_cells_exist() {
        // A trailing cell short is tolerated; a missing needed cell is not.
        let text = format!("read_id,barcode,extra\n{A},BC01\n{B},BC02,x,y\n");
        let t = parse(&text, "read_id", &[("barcode", "barcode")]).unwrap();
        assert_eq!(t.len(), 2);
        let text = format!("read_id,barcode\n{A}\n");
        let err = parse(&text, "read_id", &[("barcode", "barcode")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("line 2") && err.contains("column 2"), "{err}");
    }

    #[test]
    fn crlf_and_compact_uuids_are_accepted() {
        let compact = A.replace('-', "");
        let text = format!("read_id\tbarcode\r\n{compact}\tBC01\r\n");
        let t = parse(&text, "read_id", &[("barcode", "barcode")]).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t.columns[0].dictionary, ["BC01"]);
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(parse("", "read_id", &[("barcode", "barcode")]).is_err());
    }

    #[test]
    fn a_header_only_table_is_an_empty_table() {
        let t = parse("read_id,barcode\n", "read_id", &[("barcode", "barcode")]).unwrap();
        assert!(t.is_empty());
        assert_eq!(t.rows, 0);
    }
}

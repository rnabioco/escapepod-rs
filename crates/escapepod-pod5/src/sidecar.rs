//! The `.p5s` sidecar: per-read companion data for a POD5 file, stored as a
//! plain Arrow IPC (Feather v2) table.
//!
//! One sidecar per POD5 (`reads.pod5` → `reads.pod5.p5s`) holds one row per
//! read, sorted by read UUID:
//!
//! ```text
//! read_id (FixedSizeBinary 16) | batch_idx (u32) | row_idx (u32) | <annotation columns…>
//! ```
//!
//! `batch_idx`/`row_idx` locate the read in the POD5 reads table — this *is*
//! the read index (successor of the retired `.p5i` format). Every additional
//! column is a named **annotation**: a dictionary-encoded utf8 label per
//! read (e.g. `barcode` from demultiplexing), null where unassigned. The
//! index columns are a rebuildable cache; annotation columns are data
//! products that exist nowhere else once the CSV that produced them is gone.
//!
//! The POD5 file itself is never modified — raw sequencer output stays
//! byte-identical and checksummable. And because the sidecar is an ordinary
//! Arrow file, any Arrow reader can consume it directly, no escapepod
//! required:
//!
//! ```python
//! import pyarrow.ipc as ipc
//! table = ipc.open_file("reads.pod5.p5s").read_all()
//! ```
//!
//! Binding to the POD5 lives in the Arrow schema metadata
//! ([`P5S_FILE_ID_KEY`], [`P5S_POD5_SIZE_KEY`], [`P5S_VERSION_KEY`]) and is
//! validated from the IPC footer before any record batch is decoded, so a
//! sidecar copied next to the wrong file — or left behind after the POD5
//! was replaced — fails loudly. Writes are atomic (temp file + rename) and
//! column-preserving: rebuilding the index keeps annotations, annotating
//! keeps everything else, and a crash cannot lose the previous sidecar.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, FixedSizeBinaryBuilder, Int32Builder, StringArray,
    UInt32Array,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::ipc::CompressionType;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::ipc::writer::{FileWriter as ArrowFileWriter, IpcWriteOptions};
use arrow::record_batch::RecordBatch;

use crate::error::{Error, Result};
use crate::types::Uuid;
use crate::writer::atomic::AtomicFile;

/// Schema-metadata key for the sidecar format version.
pub const P5S_VERSION_KEY: &str = "escapepod:p5s_version";
/// Schema-metadata key holding the experimental-design table (JSON).
pub const P5S_DESIGN_KEY: &str = "escapepod:design";
/// Schema-metadata key binding the sidecar to the POD5's `file_identifier`.
pub const P5S_FILE_ID_KEY: &str = "escapepod:file_identifier";
/// Schema-metadata key binding the sidecar to the POD5's byte size.
pub const P5S_POD5_SIZE_KEY: &str = "escapepod:pod5_size";
/// Current sidecar format version (stored under [`P5S_VERSION_KEY`]).
pub const P5S_VERSION: &str = "1";
/// Default annotation column name (what `escpod annotate` writes).
pub const DEFAULT_ANNOTATION_NAME: &str = "barcode";

/// Column names reserved for the read index; everything else is an
/// annotation.
pub const RESERVED_COLUMNS: [&str; 3] = ["read_id", "batch_idx", "row_idx"];

/// Sidecar path for a POD5 file: `.p5s` appended to the full filename
/// (`reads.pod5` → `reads.pod5.p5s`), mirroring the samtools convention
/// (`reads.bam` → `reads.bam.bai`).
pub fn sidecar_path(pod5_path: impl AsRef<Path>) -> PathBuf {
    let mut s = pod5_path.as_ref().as_os_str().to_owned();
    s.push(".p5s");
    PathBuf::from(s)
}

/// The POD5 identity a sidecar is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pod5Identity {
    /// The POD5's footer `file_identifier`.
    pub file_id: Uuid,
    /// The POD5's byte size at binding time.
    pub size: u64,
}

/// A named read → label annotation.
#[derive(Debug, Clone)]
pub struct AnnotationSection {
    name: String,
    labels: Vec<String>,
    /// (uuid bytes, label index), sorted by UUID for binary search.
    entries: Vec<([u8; 16], u16)>,
}

impl AnnotationSection {
    /// Build an annotation from `(read, label)` pairs, deduplicating labels
    /// into a sorted table. Empty labels are rejected — unassigned reads are
    /// represented by absence, not by a sentinel value.
    pub fn from_pairs<'a>(
        name: &str,
        pairs: impl IntoIterator<Item = (Uuid, &'a str)>,
    ) -> Result<Self> {
        if name.is_empty() || RESERVED_COLUMNS.contains(&name) {
            return Err(Error::Parse(format!(
                "'{name}' is not a valid annotation name"
            )));
        }
        let pairs: Vec<([u8; 16], &str)> = pairs
            .into_iter()
            .map(|(uuid, label)| {
                if label.is_empty() {
                    return Err(Error::Parse(format!(
                        "empty label for read {uuid} (omit unassigned reads instead)"
                    )));
                }
                Ok((*uuid.as_bytes(), label))
            })
            .collect::<Result<_>>()?;

        let labels: Vec<&str> = pairs
            .iter()
            .map(|&(_, label)| label)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if labels.len() > usize::from(u16::MAX) {
            return Err(Error::Parse(format!(
                "{} distinct labels exceed the annotation limit of 65535",
                labels.len()
            )));
        }
        let label_id: HashMap<&str, u16> =
            labels.iter().zip(0u16..).map(|(&l, i)| (l, i)).collect();
        let mut entries: Vec<([u8; 16], u16)> = pairs
            .iter()
            .map(|&(uuid_bytes, label)| (uuid_bytes, label_id[label]))
            .collect();
        entries.sort_unstable_by_key(|e| e.0);
        entries.dedup_by_key(|e| e.0);

        Ok(Self {
            name: name.to_string(),
            labels: labels.into_iter().map(str::to_string).collect(),
            entries,
        })
    }

    /// The annotation name (e.g. `"barcode"`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The distinct labels, in label-table (sorted) order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Number of annotated reads.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any reads are annotated.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the label for a read.
    pub fn get(&self, uuid: &Uuid) -> Option<&str> {
        self.label_idx(uuid.as_bytes())
            .map(|i| self.labels[usize::from(i)].as_str())
    }

    /// Iterate over `(read, label)` pairs in UUID order.
    pub fn iter(&self) -> impl Iterator<Item = (Uuid, &str)> + '_ {
        self.entries.iter().map(|&(bytes, idx)| {
            (
                Uuid::from_bytes(bytes),
                self.labels[usize::from(idx)].as_str(),
            )
        })
    }

    /// Collect into the owned map shape `demux split` and `subset` consume.
    pub fn to_map(&self) -> HashMap<Uuid, String> {
        self.iter().map(|(id, l)| (id, l.to_string())).collect()
    }

    fn label_idx(&self, key: &[u8; 16]) -> Option<u16> {
        self.entries
            .binary_search_by_key(key, |&(k, _)| k)
            .ok()
            .map(|i| self.entries[i].1)
    }
}

/// An experimental-design table: maps combinations of annotation labels
/// (key columns) to experimental variables (value columns) — e.g.
/// `(ldx, edx) → condition`. Stored as JSON in the sidecar's schema
/// metadata under [`P5S_DESIGN_KEY`]; its value columns are materialized
/// as ordinary (derived) annotation columns via
/// [`Sidecar::derive_design_columns`], so consumers never need join logic.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Design {
    /// Annotation names whose per-read labels form the lookup key.
    pub key_columns: Vec<String>,
    /// Experimental variable names; each is materialized as a derived
    /// annotation column.
    pub value_columns: Vec<String>,
    /// Rows aligned to `key_columns` followed by `value_columns`. An empty
    /// value cell means "no assignment for this variable in this row".
    pub rows: Vec<Vec<String>>,
}

impl Design {
    /// Structural validation, independent of any sidecar contents.
    fn validate(&self) -> Result<()> {
        if self.key_columns.is_empty() {
            return Err(Error::Parse("design has no key columns".to_string()));
        }
        if self.value_columns.is_empty() {
            return Err(Error::Parse("design has no value columns".to_string()));
        }
        let mut seen = BTreeSet::new();
        for name in self.key_columns.iter().chain(&self.value_columns) {
            if name.is_empty() || RESERVED_COLUMNS.contains(&name.as_str()) {
                return Err(Error::Parse(format!(
                    "'{name}' is not a valid design column name"
                )));
            }
            if !seen.insert(name.as_str()) {
                return Err(Error::Parse(format!("duplicate design column '{name}'")));
            }
        }
        let width = self.key_columns.len() + self.value_columns.len();
        let mut combos = BTreeSet::new();
        for (i, row) in self.rows.iter().enumerate() {
            if row.len() != width {
                return Err(Error::Parse(format!(
                    "design row {} has {} cells, expected {width}",
                    i + 1,
                    row.len()
                )));
            }
            let key = &row[..self.key_columns.len()];
            if key.iter().any(String::is_empty) {
                return Err(Error::Parse(format!(
                    "design row {} has an empty key cell",
                    i + 1
                )));
            }
            if !combos.insert(key.to_vec()) {
                return Err(Error::Parse(format!(
                    "duplicate design key combination {:?}",
                    key
                )));
            }
        }
        Ok(())
    }
}

/// An in-memory sidecar: the read index plus any number of annotations,
/// and optionally an experimental-design table.
#[derive(Debug, Clone, Default)]
pub struct Sidecar {
    /// (uuid bytes, batch, row), sorted by UUID — one entry per read.
    entries: Vec<([u8; 16], u32, u32)>,
    /// Name-sorted annotations; entries are subsets of `entries`.
    annotations: Vec<AnnotationSection>,
    /// Experimental design, if one has been recorded.
    design: Option<Design>,
}

impl Sidecar {
    /// Build a sidecar from the read index entries (one per read in the
    /// POD5, any order).
    pub fn new(mut entries: Vec<([u8; 16], u32, u32)>) -> Self {
        entries.sort_unstable_by_key(|e| e.0);
        Self {
            entries,
            annotations: Vec::new(),
            design: None,
        }
    }

    /// The read-index entries, sorted by UUID.
    pub fn entries(&self) -> &[([u8; 16], u32, u32)] {
        &self.entries
    }

    /// Number of reads covered by the sidecar.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the sidecar covers no reads.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The annotation with the given name, if present.
    pub fn annotation(&self, name: &str) -> Option<&AnnotationSection> {
        self.annotations.iter().find(|a| a.name == name)
    }

    /// All annotations, in name order.
    pub fn annotations(&self) -> &[AnnotationSection] {
        &self.annotations
    }

    /// The annotation names, in order.
    pub fn annotation_names(&self) -> Vec<&str> {
        self.annotations.iter().map(|a| a.name.as_str()).collect()
    }

    /// Replace (or add) an annotation, keyed by its name. Annotation entries
    /// for reads not in the index are dropped at write time.
    pub fn set_annotation(&mut self, annotation: AnnotationSection) {
        match self
            .annotations
            .iter_mut()
            .find(|a| a.name == annotation.name)
        {
            Some(slot) => *slot = annotation,
            None => self.annotations.push(annotation),
        }
        self.annotations.sort_by(|a, b| a.name.cmp(&b.name));
    }

    /// Replace the read-index entries (one per read, any order).
    pub fn set_entries(&mut self, mut entries: Vec<([u8; 16], u32, u32)>) {
        entries.sort_unstable_by_key(|e| e.0);
        self.entries = entries;
    }

    /// Remove an annotation by name; returns whether it was present.
    pub fn remove_annotation(&mut self, name: &str) -> bool {
        let before = self.annotations.len();
        self.annotations.retain(|a| a.name != name);
        self.annotations.len() != before
    }

    /// The experimental design, if one has been recorded.
    pub fn design(&self) -> Option<&Design> {
        self.design.as_ref()
    }

    /// Record (or replace) the experimental design. Key columns must name
    /// existing annotations; value columns must not collide with the key
    /// columns or any non-derived annotation semantics — a same-named
    /// annotation column is treated as the derived materialization and
    /// overwritten by [`Self::derive_design_columns`].
    pub fn set_design(&mut self, design: Design) -> Result<()> {
        design.validate()?;
        for key in &design.key_columns {
            if self.annotation(key).is_none() {
                return Err(Error::Parse(format!(
                    "design key column '{key}' is not an annotation in the sidecar"
                )));
            }
        }
        self.design = Some(design);
        Ok(())
    }

    /// Remove the design and its derived annotation columns; returns
    /// whether one was present.
    pub fn remove_design(&mut self) -> bool {
        match self.design.take() {
            Some(design) => {
                for name in &design.value_columns {
                    self.remove_annotation(name);
                }
                true
            }
            None => false,
        }
    }

    /// Materialize the design's value columns as derived annotations: each
    /// read whose key-annotation labels exactly match a design row gets that
    /// row's values. Reads missing any key label, or matching no row, are
    /// left unassigned; empty value cells assign nothing for that variable.
    ///
    /// Returns `(value column, reads assigned)` per column. No-op without a
    /// design. Called automatically when a design is written and when a key
    /// annotation is replaced, so the derived columns cannot go stale.
    pub fn derive_design_columns(&mut self) -> Result<Vec<(String, usize)>> {
        let Some(design) = self.design.clone() else {
            return Ok(Vec::new());
        };
        let n_keys = design.key_columns.len();

        let mut derived: Vec<(String, Vec<(Uuid, String)>)> = design
            .value_columns
            .iter()
            .map(|name| (name.clone(), Vec::new()))
            .collect();
        {
            let mut lookup: HashMap<Vec<&str>, &[String]> = HashMap::new();
            for row in &design.rows {
                let (key, values) = row.split_at(n_keys);
                lookup.insert(key.iter().map(String::as_str).collect(), values);
            }
            let key_sections: Vec<&AnnotationSection> = design
                .key_columns
                .iter()
                .map(|name| {
                    self.annotation(name).ok_or_else(|| {
                        Error::Parse(format!(
                            "design key column '{name}' is not an annotation in the sidecar"
                        ))
                    })
                })
                .collect::<Result<_>>()?;

            for &(uuid_bytes, _, _) in &self.entries {
                let uuid = Uuid::from_bytes(uuid_bytes);
                let key: Option<Vec<&str>> = key_sections.iter().map(|s| s.get(&uuid)).collect();
                let Some(key) = key else { continue };
                let Some(values) = lookup.get(&key) else {
                    continue;
                };
                for (slot, value) in derived.iter_mut().zip(values.iter()) {
                    if !value.is_empty() {
                        slot.1.push((uuid, value.clone()));
                    }
                }
            }
        }

        let mut counts = Vec::with_capacity(derived.len());
        for (name, pairs) in derived {
            counts.push((name.clone(), pairs.len()));
            self.set_annotation(AnnotationSection::from_pairs(
                &name,
                pairs.iter().map(|(uuid, label)| (*uuid, label.as_str())),
            )?)
        }
        Ok(counts)
    }
}

/// Load the sidecar at `p5s_path`, validating it against the POD5's
/// identity. `Ok(None)` when no sidecar exists; an error when one exists
/// but is malformed, or bound to a different / since-replaced POD5.
///
/// Identity is checked from the IPC footer's schema metadata before any
/// record batch is decoded.
pub fn read_sidecar_file(
    p5s_path: impl AsRef<Path>,
    expect: &Pod5Identity,
) -> Result<Option<Sidecar>> {
    let p5s_path = p5s_path.as_ref();
    let file = match File::open(p5s_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::from(e)),
    };

    let reader = ArrowFileReader::try_new(file, None).map_err(|e| {
        Error::Parse(format!(
            "{} is not a readable .p5s sidecar (legacy .p5i or corrupt?): {e}; \
             delete it and rebuild with `escpod index` / `escpod annotate`",
            p5s_path.display()
        ))
    })?;
    let schema = reader.schema();
    let metadata = schema.metadata();

    match metadata.get(P5S_VERSION_KEY).map(String::as_str) {
        Some(P5S_VERSION) => {}
        Some(other) => {
            return Err(Error::Parse(format!(
                ".p5s version {other} unsupported (expected {P5S_VERSION})"
            )));
        }
        None => {
            return Err(Error::Parse(format!(
                "{} has no {P5S_VERSION_KEY} metadata; not an escapepod sidecar",
                p5s_path.display()
            )));
        }
    }
    let stored_file_id = metadata
        .get(P5S_FILE_ID_KEY)
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or_else(|| Error::Parse(format!("{} has no valid file id", p5s_path.display())))?;
    let stored_size: u64 = metadata
        .get(P5S_POD5_SIZE_KEY)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Parse(format!("{} has no valid pod5 size", p5s_path.display())))?;
    if stored_file_id != expect.file_id || stored_size != expect.size {
        return Err(Error::Parse(format!(
            "{} does not match this POD5 file (stale or copied from another); \
             rebuild with `escpod index` / `escpod annotate`",
            p5s_path.display()
        )));
    }

    let read_id_idx = schema.index_of("read_id")?;
    let batch_idx_idx = schema.index_of("batch_idx")?;
    let row_idx_idx = schema.index_of("row_idx")?;
    let annotation_columns: Vec<(usize, String)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| !RESERVED_COLUMNS.contains(&f.name().as_str()))
        .map(|(i, f)| (i, f.name().clone()))
        .collect();

    let mut entries: Vec<([u8; 16], u32, u32)> = Vec::new();
    let mut annotation_pairs: HashMap<String, Vec<(Uuid, String)>> = annotation_columns
        .iter()
        .map(|(_, name)| (name.clone(), Vec::new()))
        .collect();

    for batch in reader {
        let batch = batch.map_err(|e| Error::Parse(format!("corrupt .p5s: {e}")))?;
        let ids = batch
            .column(read_id_idx)
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeBinaryArray>()
            .filter(|a| a.value_length() == 16)
            .ok_or_else(|| {
                Error::Parse(".p5s read_id column is not FixedSizeBinary(16)".to_string())
            })?;
        let batches = downcast_u32(&batch, batch_idx_idx, "batch_idx")?;
        let rows = downcast_u32(&batch, row_idx_idx, "row_idx")?;

        for i in 0..batch.num_rows() {
            let uuid_bytes: [u8; 16] = ids.value(i).try_into().unwrap();
            entries.push((uuid_bytes, batches.value(i), rows.value(i)));
        }

        for (col, name) in &annotation_columns {
            let dict = batch
                .column(*col)
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| {
                    Error::Parse(format!(
                        ".p5s annotation column '{name}' is not dictionary-encoded utf8"
                    ))
                })?;
            let values = dict
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::Parse(format!(
                        ".p5s annotation column '{name}' values are not utf8"
                    ))
                })?;
            let pairs = annotation_pairs.get_mut(name).expect("initialized above");
            for i in 0..batch.num_rows() {
                if dict.is_null(i) {
                    continue;
                }
                let key = usize::try_from(dict.keys().value(i))
                    .map_err(|_| Error::Parse(format!("negative dictionary key in '{name}'")))?;
                if key >= values.len() {
                    return Err(Error::Parse(format!(
                        "dictionary key out of range in '{name}'"
                    )));
                }
                let uuid_bytes: [u8; 16] = ids.value(i).try_into().unwrap();
                pairs.push((Uuid::from_bytes(uuid_bytes), values.value(key).to_string()));
            }
        }
    }

    let mut sidecar = Sidecar::new(entries);
    for (_, name) in &annotation_columns {
        let pairs = &annotation_pairs[name];
        sidecar.set_annotation(AnnotationSection::from_pairs(
            name,
            pairs.iter().map(|(u, l)| (*u, l.as_str())),
        )?);
    }
    if let Some(json) = metadata.get(P5S_DESIGN_KEY) {
        let design: Design = serde_json::from_str(json)
            .map_err(|e| Error::Parse(format!("invalid design metadata in .p5s: {e}")))?;
        design.validate()?;
        sidecar.design = Some(design);
    }
    Ok(Some(sidecar))
}

/// Atomically write a sidecar, bound to `identity`. The destination is
/// either the previous sidecar or the complete new one — never a torn mix.
pub fn write_sidecar_file(
    p5s_path: impl AsRef<Path>,
    identity: &Pod5Identity,
    sidecar: &Sidecar,
) -> Result<()> {
    let mut metadata = HashMap::new();
    metadata.insert(P5S_VERSION_KEY.to_string(), P5S_VERSION.to_string());
    metadata.insert(P5S_FILE_ID_KEY.to_string(), identity.file_id.to_string());
    metadata.insert(P5S_POD5_SIZE_KEY.to_string(), identity.size.to_string());
    if let Some(design) = &sidecar.design {
        let json = serde_json::to_string(design)
            .map_err(|e| Error::Parse(format!("design serialization failed: {e}")))?;
        metadata.insert(P5S_DESIGN_KEY.to_string(), json);
    }

    let mut fields = vec![
        // Same physical type + extension marker as POD5's own read_id.
        Field::new("read_id", DataType::FixedSizeBinary(16), false).with_metadata(HashMap::from([
            (
                "ARROW:extension:name".to_string(),
                "minknow.uuid".to_string(),
            ),
        ])),
        Field::new("batch_idx", DataType::UInt32, false),
        Field::new("row_idx", DataType::UInt32, false),
    ];
    for annotation in &sidecar.annotations {
        fields.push(Field::new(
            annotation.name(),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        ));
    }
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));

    let n = sidecar.entries.len();
    let mut id_builder = FixedSizeBinaryBuilder::with_capacity(n, 16);
    for (uuid_bytes, _, _) in &sidecar.entries {
        id_builder.append_value(uuid_bytes).map_err(Error::Arrow)?;
    }
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(id_builder.finish()),
        Arc::new(UInt32Array::from_iter_values(
            sidecar.entries.iter().map(|&(_, batch, _)| batch),
        )),
        Arc::new(UInt32Array::from_iter_values(
            sidecar.entries.iter().map(|&(_, _, row)| row),
        )),
    ];
    for annotation in &sidecar.annotations {
        let mut keys = Int32Builder::with_capacity(n);
        for (uuid_bytes, _, _) in &sidecar.entries {
            match annotation.label_idx(uuid_bytes) {
                Some(idx) => keys.append_value(i32::from(idx)),
                None => keys.append_null(),
            }
        }
        let values = Arc::new(StringArray::from_iter_values(annotation.labels().iter()));
        let dict = DictionaryArray::<Int32Type>::try_new(keys.finish(), values)?;
        columns.push(Arc::new(dict));
    }
    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let options = IpcWriteOptions::default()
        .try_with_compression(Some(CompressionType::ZSTD))
        .map_err(Error::Arrow)?;
    let mut buf = Vec::new();
    {
        let mut writer = ArrowFileWriter::try_new_with_options(&mut buf, &schema, options)?;
        writer.write(&batch)?;
        writer.finish()?;
    }

    let atomic = AtomicFile::new(p5s_path.as_ref())?;
    std::fs::write(atomic.temp_path()?, &buf)?;
    atomic.commit()
}

fn downcast_u32<'a>(batch: &'a RecordBatch, idx: usize, name: &str) -> Result<&'a UInt32Array> {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Parse(format!(".p5s column '{name}' is not u32")))
}

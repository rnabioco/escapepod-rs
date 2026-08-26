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
//! column is a named **annotation** in one of two kinds, null where the read
//! has no value:
//!
//! * **labels** — dictionary-encoded utf8, e.g. `barcode` from demultiplexing
//!   ([`AnnotationSection`]);
//! * **scores** — `Float32`, e.g. `crf_logp` from `demux --ref-scores`
//!   ([`ScoreSection`]).
//!
//! A column's Arrow type is what says which kind it is, so a reader dispatches
//! on the schema rather than on a convention. The index columns are a
//! rebuildable cache; both kinds of annotation column are data products that
//! exist nowhere else once the CSV that produced them is gone — which is the
//! whole reason scores had to become a column at all, since sidecar-only demux
//! (`--annotate` with no `-d`) writes no CSV to keep them in.
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
//! was replaced — fails loudly. Separately, [`SidecarProvenance`] records what
//! the sidecar was built *from*, so that failure can name a file rather than
//! only report that two UUIDs differ; it is descriptive and never compared.
//! The binding is file-level, and a locator inside a bound sidecar is still
//! confirmed against the row's own `read_id` before use — see
//! `arrow_helpers::verify_index_row`. Writes are atomic (temp file + rename) and
//! column-preserving: rebuilding the index keeps annotations, annotating
//! keeps everything else, and a crash cannot lose the previous sidecar.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, FixedSizeBinaryBuilder, Float32Array, Int32Builder,
    StringArray, UInt32Array,
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

// ---------------------------------------------------------------------------
// Provenance — descriptive, never a gate
//
// Identity is `file_identifier` + `pod5_size` and nothing else. The keys below
// exist so that a sidecar which *fails* that check can say what it was built
// from: without them the error knows only that the UUIDs differ, which is
// precisely the moment the operator needs a filename. They are never compared
// against anything — matching them would break every legitimate rename — and
// every one is optional on read, so a sidecar written before they existed
// still loads and an older escpod ignores them. That is why adding them is not
// a version bump.
//
// Deliberately absent: a write timestamp. The sidecar file's own mtime already
// records it, and duplicating the filesystem would have cost the format crate
// a hard `chrono` dependency it does not otherwise carry.
// ---------------------------------------------------------------------------

/// Schema-metadata key recording the POD5's file name at write time.
///
/// Base name only. The path is deliberately not recorded: it would go stale on
/// any legitimate move and leak directory layout, while the base name is the
/// part an operator recognises.
pub const P5S_SOURCE_NAME_KEY: &str = "escapepod:source_name";
/// Schema-metadata key recording how many reads the index covers.
pub const P5S_READ_COUNT_KEY: &str = "escapepod:read_count";
/// Schema-metadata key recording what wrote the sidecar.
pub const P5S_WRITER_KEY: &str = "escapepod:writer";

// ---------------------------------------------------------------------------
// Signal batch geometry — a cache for the POD5's own layout
// ---------------------------------------------------------------------------

/// Schema-metadata key holding the signal table's per-batch row counts,
/// run-length encoded (see [`encode_batch_rows`]).
///
/// This is the row count of every record batch in the POD5's *signal* table,
/// which is the one part of that table's Arrow IPC footer that the footer
/// itself does not carry. Recovering it means reading each batch's message
/// header — one scattered touch per batch, and on a network filesystem that
/// measured 15-24 ms *each* cold, 27-39% of a cold scattered fetch and a flat
/// ~5 s per process on a 33 GB file with 8866 batches. It is also pure
/// function of a file that never changes, so paying for it more than once is
/// waste. `escpod index` walks it once and records the answer here.
///
/// Two things it deliberately is **not**:
///
/// * It is not an *assumption* about the stride. The obvious cheap fix is to
///   read batch 0, assume every batch matches, and derive the rest — which the
///   official `pod5` library and dorado do, and which `Reader::nonuniform_signal_batch`
///   exists to catch them out on. Storing the measured counts costs a few
///   bytes and is exact for a non-uniform file too, so escapepod does not have
///   to make that bet at all.
/// * It is not a change to POD5. The sidecar caches what the POD5 already
///   says; the POD5 is never read differently because of it, and never
///   written. A reader that finds this key still spot-checks it against the
///   file (see `ArrowIpcFooter::parse_with_row_counts`) and falls back to the
///   full walk if it does not fit.
///
/// Optional on read, like the provenance keys and for the same reason: a
/// sidecar written before it existed still loads, and an older escpod ignores
/// it. Adding it is therefore not a version bump.
pub const P5S_SIGNAL_BATCH_ROWS_KEY: &str = "escapepod:signal_batch_rows";

/// Cap on the encoded geometry, so a pathological file cannot bloat a sidecar.
///
/// Only a file whose batch row counts genuinely alternate reaches this — a
/// uniform file encodes to about 15 bytes however many batches it has. Past
/// the cap the key is simply omitted and readers walk the footer as before.
const MAX_ENCODED_BATCH_ROWS: usize = 1 << 20;

/// Run-length encode per-batch row counts as `"<runs>x<rows>,…"`.
///
/// A conformant POD5 has one run: every batch but the last holds exactly the
/// writer's `signal_batch_size`, so 8866 batches encode as `"8865x100,1x37"`.
/// A non-uniform file degrades gracefully to one term per run rather than
/// being rejected — the point is to record what is there, not to insist on
/// what should be.
pub fn encode_batch_rows(counts: &[u64]) -> Option<String> {
    if counts.is_empty() {
        return None;
    }
    let mut out = String::new();
    let mut run_value = counts[0];
    let mut run_len = 0u64;
    for &c in counts {
        if c == run_value {
            run_len += 1;
        } else {
            if !out.is_empty() {
                out.push(',');
            }
            out.push_str(&format!("{run_len}x{run_value}"));
            run_value = c;
            run_len = 1;
        }
        if out.len() > MAX_ENCODED_BATCH_ROWS {
            return None;
        }
    }
    if !out.is_empty() {
        out.push(',');
    }
    out.push_str(&format!("{run_len}x{run_value}"));
    (out.len() <= MAX_ENCODED_BATCH_ROWS).then_some(out)
}

/// Decode [`encode_batch_rows`]. `None` on anything malformed — the geometry
/// is a cache, so a value that does not parse means "walk the footer", never
/// an error that fails the open.
pub fn decode_batch_rows(s: &str) -> Option<Vec<u64>> {
    let mut out = Vec::new();
    for term in s.split(',') {
        let (n, value) = term.split_once('x')?;
        let n: u64 = n.parse().ok()?;
        let value: u64 = value.parse().ok()?;
        // A run of zero batches is meaningless, and an enormous one is a
        // malformed value trying to make us allocate.
        if n == 0 || out.len() as u64 + n > u32::MAX as u64 {
            return None;
        }
        out.extend(std::iter::repeat_n(value, n as usize));
    }
    (!out.is_empty()).then_some(out)
}

/// What a sidecar says about its own origin.
///
/// Every field is optional: sidecars written before these keys existed carry
/// none of them, and that is not an error. Used only to make messages and
/// `escpod inspect summary` informative — never to accept or reject.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SidecarProvenance {
    /// The POD5's base name when the sidecar was written.
    pub source_name: Option<String>,
    /// Reads covered by the index when the sidecar was written.
    pub read_count: Option<u64>,
    /// The crate and version that wrote it, e.g. `escapepod-pod5 0.12.0`.
    pub writer: Option<String>,
}

impl SidecarProvenance {
    /// Render as a trailing clause for an error or report, or `None` when the
    /// sidecar predates these keys and there is nothing to say.
    pub fn describe(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(name) = &self.source_name {
            parts.push(format!("from \"{name}\""));
        }
        if let Some(n) = self.read_count {
            parts.push(format!("{n} reads"));
        }
        if let Some(w) = &self.writer {
            parts.push(format!("written by {w}"));
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

/// Sidecar format version for a file whose annotations are all label columns.
///
/// Still what gets written whenever no numeric column is present, which is
/// every sidecar `escpod annotate` produces and every demux run without
/// `--ref-scores`. See [`P5S_VERSION_SCORES`] for why the bump is gated on
/// content rather than applied to every write.
pub const P5S_VERSION: &str = "1";

/// Sidecar format version for a file that carries at least one numeric
/// (`Float32`) column.
///
/// Written **only** when one is present. An escpod that predates score columns
/// rejects any column it cannot downcast to a dictionary, with a message about
/// the column not being "dictionary-encoded utf8" — accurate but not
/// actionable. Declaring the version instead makes that failure say what it is,
/// while a version bump on every write would have made older escpods reject
/// barcode-only sidecars they can read perfectly well.
pub const P5S_VERSION_SCORES: &str = "2";
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

/// A named read → number annotation.
///
/// The numeric counterpart of [`AnnotationSection`], for per-read quantities
/// rather than labels — `crf_logp` from `demux --ref-scores` is the one that
/// motivated it. Dictionary-encoding those would be absurd: a continuous score
/// over a million reads is a million distinct "labels", well past the
/// [`AnnotationSection`] limit of 65535, and the dictionary would be larger
/// than the data it indexes.
///
/// Like a label annotation, an unscored read is represented by **absence**
/// rather than a sentinel, which is why `NaN` is refused on the way in: it is
/// the one `f32` that already means "no value" and would make the two
/// representations ambiguous. Infinities are allowed — `log P = -inf` is a real
/// answer, not a missing one.
#[derive(Debug, Clone)]
pub struct ScoreSection {
    name: String,
    /// (uuid bytes, value), sorted by UUID for binary search.
    entries: Vec<([u8; 16], f32)>,
}

impl ScoreSection {
    /// Build a score column from `(read, value)` pairs.
    pub fn from_pairs(name: &str, pairs: impl IntoIterator<Item = (Uuid, f32)>) -> Result<Self> {
        if name.is_empty() || RESERVED_COLUMNS.contains(&name) {
            return Err(Error::Parse(format!(
                "'{name}' is not a valid annotation name"
            )));
        }
        let mut entries: Vec<([u8; 16], f32)> = pairs
            .into_iter()
            .map(|(uuid, value)| {
                if value.is_nan() {
                    return Err(Error::Parse(format!(
                        "NaN score for read {uuid} in '{name}' (omit unscored reads instead)"
                    )));
                }
                Ok((*uuid.as_bytes(), value))
            })
            .collect::<Result<_>>()?;
        entries.sort_unstable_by_key(|e| e.0);
        entries.dedup_by_key(|e| e.0);

        Ok(Self {
            name: name.to_string(),
            entries,
        })
    }

    /// The column name (e.g. `"crf_logp"`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of scored reads.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any reads are scored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a read's score.
    pub fn get(&self, uuid: &Uuid) -> Option<f32> {
        self.entries
            .binary_search_by_key(uuid.as_bytes(), |&(k, _)| k)
            .ok()
            .map(|i| self.entries[i].1)
    }

    /// Iterate over `(read, score)` pairs in UUID order.
    pub fn iter(&self) -> impl Iterator<Item = (Uuid, f32)> + '_ {
        self.entries
            .iter()
            .map(|&(bytes, v)| (Uuid::from_bytes(bytes), v))
    }

    /// Collect into an owned map.
    pub fn to_map(&self) -> HashMap<Uuid, f32> {
        self.iter().collect()
    }

    fn value_of(&self, key: &[u8; 16]) -> Option<f32> {
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
    /// Name-sorted numeric columns; entries are subsets of `entries`.
    scores: Vec<ScoreSection>,
    /// Experimental design, if one has been recorded.
    design: Option<Design>,
    /// What the file said about its own origin. Read-only: rewriting a sidecar
    /// re-stamps this from the file being written, never from here.
    provenance: SidecarProvenance,
    /// Cached per-batch row counts of the POD5's signal table, if recorded.
    ///
    /// Unlike [`Self::provenance`] this **is** carried through a
    /// read-modify-write: it describes the POD5, which identity binding
    /// already proves has not changed, so an `annotate` that does not look at
    /// the signal table has no reason to discard it. Only `escpod index`
    /// re-measures it.
    signal_batch_rows: Option<Vec<u64>>,
}

impl Sidecar {
    /// What this sidecar recorded about its origin when it was written.
    ///
    /// Empty for a sidecar written before the provenance keys existed, and for
    /// one built in memory rather than loaded.
    pub fn provenance(&self) -> &SidecarProvenance {
        &self.provenance
    }

    /// Build a sidecar from the read index entries (one per read in the
    /// POD5, any order).
    pub fn new(mut entries: Vec<([u8; 16], u32, u32)>) -> Self {
        entries.sort_unstable_by_key(|e| e.0);
        Self {
            entries,
            annotations: Vec::new(),
            scores: Vec::new(),
            design: None,
            provenance: SidecarProvenance::default(),
            signal_batch_rows: None,
        }
    }

    /// The cached signal-table batch row counts, if this sidecar records them.
    ///
    /// See [`P5S_SIGNAL_BATCH_ROWS_KEY`] for why they are worth caching and
    /// what they are not.
    pub fn signal_batch_rows(&self) -> Option<&[u64]> {
        self.signal_batch_rows.as_deref()
    }

    /// Record the signal-table batch row counts, as measured from the POD5.
    ///
    /// Only a caller that has actually walked the signal footer should call
    /// this — the value is trusted (subject to the reader's spot check) on
    /// every later open, so a guess here is worse than nothing.
    pub fn set_signal_batch_rows(&mut self, counts: Vec<u64>) {
        self.signal_batch_rows = (!counts.is_empty()).then_some(counts);
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
    ///
    /// A name is one Arrow column, so this also displaces a **score** column of
    /// the same name — the alternative is an in-memory sidecar that cannot be
    /// written. Writing a label column over a numeric one is a deliberate act;
    /// it is not something the demux or annotate paths can do by accident,
    /// since they choose both the names and the kinds.
    pub fn set_annotation(&mut self, annotation: AnnotationSection) {
        self.scores.retain(|s| s.name != annotation.name);
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

    /// The score column with the given name, if present.
    pub fn score(&self, name: &str) -> Option<&ScoreSection> {
        self.scores.iter().find(|s| s.name == name)
    }

    /// All score columns, in name order.
    pub fn scores(&self) -> &[ScoreSection] {
        &self.scores
    }

    /// The score column names, in order.
    pub fn score_names(&self) -> Vec<&str> {
        self.scores.iter().map(|s| s.name.as_str()).collect()
    }

    /// Replace (or add) a score column, keyed by its name. Displaces a
    /// same-named annotation — see [`Self::set_annotation`].
    pub fn set_score(&mut self, score: ScoreSection) {
        self.annotations.retain(|a| a.name != score.name);
        match self.scores.iter_mut().find(|s| s.name == score.name) {
            Some(slot) => *slot = score,
            None => self.scores.push(score),
        }
        self.scores.sort_by(|a, b| a.name.cmp(&b.name));
    }

    /// Remove a score column by name; returns whether it was present.
    pub fn remove_score(&mut self, name: &str) -> bool {
        let before = self.scores.len();
        self.scores.retain(|s| s.name != name);
        self.scores.len() != before
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
        Some(P5S_VERSION | P5S_VERSION_SCORES) => {}
        Some(other) => {
            return Err(Error::Parse(format!(
                ".p5s version {other} unsupported (this escpod reads \
                 {P5S_VERSION} and {P5S_VERSION_SCORES})"
            )));
        }
        None => {
            return Err(Error::Parse(format!(
                "{} has no {P5S_VERSION_KEY} metadata; not an escapepod sidecar",
                p5s_path.display()
            )));
        }
    }
    // Read before the gate, not after: the mismatch message is the one place
    // provenance earns its keep, and by then we have already returned.
    let provenance = SidecarProvenance {
        source_name: metadata.get(P5S_SOURCE_NAME_KEY).cloned(),
        read_count: metadata
            .get(P5S_READ_COUNT_KEY)
            .and_then(|s| s.parse().ok()),
        writer: metadata.get(P5S_WRITER_KEY).cloned(),
    };

    let stored_file_id = metadata
        .get(P5S_FILE_ID_KEY)
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or_else(|| Error::Parse(format!("{} has no valid file id", p5s_path.display())))?;
    let stored_size: u64 = metadata
        .get(P5S_POD5_SIZE_KEY)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Parse(format!("{} has no valid pod5 size", p5s_path.display())))?;
    if stored_file_id != expect.file_id || stored_size != expect.size {
        let origin = provenance
            .describe()
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        return Err(Error::Parse(format!(
            "{} does not match this POD5 file (stale or copied from another){}; \
             rebuild with `escpod index` / `escpod annotate`",
            p5s_path.display(),
            origin
        )));
    }

    let read_id_idx = schema.index_of("read_id")?;
    let batch_idx_idx = schema.index_of("batch_idx")?;
    let row_idx_idx = schema.index_of("row_idx")?;
    // A column's Arrow type is what says which kind it is; the version in the
    // metadata only says whether numeric ones are expected at all.
    let mut annotation_columns: Vec<(usize, String)> = Vec::new();
    let mut score_columns: Vec<(usize, String)> = Vec::new();
    for (i, f) in schema.fields().iter().enumerate() {
        if RESERVED_COLUMNS.contains(&f.name().as_str()) {
            continue;
        }
        match f.data_type() {
            DataType::Dictionary(_, _) => annotation_columns.push((i, f.name().clone())),
            DataType::Float32 => score_columns.push((i, f.name().clone())),
            other => {
                return Err(Error::Parse(format!(
                    ".p5s column '{}' is {other}; expected a dictionary-encoded \
                     utf8 label column or a float32 score column",
                    f.name()
                )));
            }
        }
    }

    let mut entries: Vec<([u8; 16], u32, u32)> = Vec::new();
    let mut annotation_pairs: HashMap<String, Vec<(Uuid, String)>> = annotation_columns
        .iter()
        .map(|(_, name)| (name.clone(), Vec::new()))
        .collect();
    let mut score_pairs: HashMap<String, Vec<(Uuid, f32)>> = score_columns
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

        for (col, name) in &score_columns {
            let values = batch
                .column(*col)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    Error::Parse(format!(".p5s score column '{name}' is not float32"))
                })?;
            let pairs = score_pairs.get_mut(name).expect("initialized above");
            for i in 0..batch.num_rows() {
                if values.is_null(i) {
                    continue;
                }
                let uuid_bytes: [u8; 16] = ids.value(i).try_into().unwrap();
                pairs.push((Uuid::from_bytes(uuid_bytes), values.value(i)));
            }
        }
    }

    let mut sidecar = Sidecar::new(entries);
    sidecar.provenance = provenance;
    for (_, name) in &annotation_columns {
        let pairs = &annotation_pairs[name];
        sidecar.set_annotation(AnnotationSection::from_pairs(
            name,
            pairs.iter().map(|(u, l)| (*u, l.as_str())),
        )?);
    }
    for (_, name) in &score_columns {
        sidecar.set_score(ScoreSection::from_pairs(
            name,
            score_pairs[name].iter().copied(),
        )?);
    }
    if let Some(json) = metadata.get(P5S_DESIGN_KEY) {
        let design: Design = serde_json::from_str(json)
            .map_err(|e| Error::Parse(format!("invalid design metadata in .p5s: {e}")))?;
        design.validate()?;
        sidecar.design = Some(design);
    }
    // A geometry that will not parse is dropped, not fatal: it is a cache of
    // something the POD5 still holds, so the cost of ignoring it is a slower
    // open rather than a wrong answer.
    sidecar.signal_batch_rows = metadata
        .get(P5S_SIGNAL_BATCH_ROWS_KEY)
        .and_then(|s| decode_batch_rows(s));
    Ok(Some(sidecar))
}

/// The parts of a sidecar that live in its schema metadata, read **without
/// decoding any record batch**.
///
/// [`read_sidecar_file`] materialises every column, which for a multi-million
/// read index is far more work than a caller that only wants the signal
/// geometry should pay. Arrow's file reader loads the footer eagerly and the
/// batches lazily, so this is a footer read and nothing more.
///
/// Identity is validated exactly as in [`read_sidecar_file`] — a sidecar bound
/// to another POD5 is an error here too, never a silent `None`.
pub fn read_sidecar_metadata(
    p5s_path: impl AsRef<Path>,
    expect: &Pod5Identity,
) -> Result<Option<SidecarMetadata>> {
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

    let provenance = SidecarProvenance {
        source_name: metadata.get(P5S_SOURCE_NAME_KEY).cloned(),
        read_count: metadata
            .get(P5S_READ_COUNT_KEY)
            .and_then(|s| s.parse().ok()),
        writer: metadata.get(P5S_WRITER_KEY).cloned(),
    };
    let stored_file_id = metadata
        .get(P5S_FILE_ID_KEY)
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or_else(|| Error::Parse(format!("{} has no valid file id", p5s_path.display())))?;
    let stored_size: u64 = metadata
        .get(P5S_POD5_SIZE_KEY)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Parse(format!("{} has no valid pod5 size", p5s_path.display())))?;
    if stored_file_id != expect.file_id || stored_size != expect.size {
        let origin = provenance
            .describe()
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        return Err(Error::Parse(format!(
            "{} does not match this POD5 file (stale or copied from another){}; \
             rebuild with `escpod index` / `escpod annotate`",
            p5s_path.display(),
            origin
        )));
    }

    Ok(Some(SidecarMetadata {
        provenance,
        signal_batch_rows: metadata
            .get(P5S_SIGNAL_BATCH_ROWS_KEY)
            .and_then(|s| decode_batch_rows(s)),
    }))
}

/// What [`read_sidecar_metadata`] recovers from a sidecar's schema metadata.
#[derive(Debug, Clone, Default)]
pub struct SidecarMetadata {
    /// What the sidecar says about its own origin.
    pub provenance: SidecarProvenance,
    /// Cached per-batch row counts of the POD5's signal table, if recorded.
    pub signal_batch_rows: Option<Vec<u64>>,
}

/// Atomically write a sidecar, bound to `identity`. The destination is
/// either the previous sidecar or the complete new one — never a torn mix.
pub fn write_sidecar_file(
    p5s_path: impl AsRef<Path>,
    identity: &Pod5Identity,
    sidecar: &Sidecar,
) -> Result<()> {
    let mut metadata = HashMap::new();
    let version = if sidecar.scores.is_empty() {
        P5S_VERSION
    } else {
        P5S_VERSION_SCORES
    };
    metadata.insert(P5S_VERSION_KEY.to_string(), version.to_string());
    metadata.insert(P5S_FILE_ID_KEY.to_string(), identity.file_id.to_string());
    metadata.insert(P5S_POD5_SIZE_KEY.to_string(), identity.size.to_string());

    // Provenance is re-derived from what is being written, never carried over
    // from `sidecar.provenance` — a read-modify-write of a sidecar whose POD5
    // was renamed should record the name it has now.
    //
    // The POD5's name comes from the sidecar's own path: `sidecar_path` only
    // ever appends `.p5s`, so stripping it back off is exact, and it saves
    // every caller from threading the source path through a second argument.
    if let Some(source) = p5s_path
        .as_ref()
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(".p5s"))
    {
        metadata.insert(P5S_SOURCE_NAME_KEY.to_string(), source.to_string());
    }
    metadata.insert(
        P5S_READ_COUNT_KEY.to_string(),
        sidecar.entries.len().to_string(),
    );
    metadata.insert(
        P5S_WRITER_KEY.to_string(),
        concat!("escapepod-pod5 ", env!("CARGO_PKG_VERSION")).to_string(),
    );
    if let Some(design) = &sidecar.design {
        let json = serde_json::to_string(design)
            .map_err(|e| Error::Parse(format!("design serialization failed: {e}")))?;
        metadata.insert(P5S_DESIGN_KEY.to_string(), json);
    }
    // Carried from the struct rather than re-derived, unlike provenance above:
    // this describes the POD5, not this write, and identity binding already
    // proves the POD5 is the same one.
    if let Some(counts) = &sidecar.signal_batch_rows
        && let Some(encoded) = encode_batch_rows(counts)
    {
        metadata.insert(P5S_SIGNAL_BATCH_ROWS_KEY.to_string(), encoded);
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
    for score in &sidecar.scores {
        if sidecar.annotations.iter().any(|a| a.name == score.name) {
            return Err(Error::Parse(format!(
                "'{}' is both a label and a score column; a name is one column",
                score.name
            )));
        }
        fields.push(Field::new(score.name(), DataType::Float32, true));
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
    for score in &sidecar.scores {
        // Null, not a sentinel, for a read this column has no value for —
        // every f32 is a possible score.
        columns.push(Arc::new(Float32Array::from_iter(
            sidecar
                .entries
                .iter()
                .map(|(uuid_bytes, _, _)| score.value_of(uuid_bytes)),
        )));
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

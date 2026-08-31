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

/// Sidecar format version for a **collection** sidecar: one `.p5s` covering
/// every POD5 under a directory instead of a single file.
///
/// Unlike [`P5S_VERSION_SCORES`], this one is not content-gated on a column
/// type — it marks a different *shape*. A collection has no single POD5 to
/// bind to, so it carries a member table ([`P5S_MEMBERS_KEY`]) in place of
/// [`P5S_FILE_ID_KEY`]/[`P5S_POD5_SIZE_KEY`] and one extra index column
/// (`member_idx`). An escpod that predates it must not try to read that as a
/// per-file sidecar, and the version is what stops it.
pub const P5S_VERSION_COLLECTION: &str = "3";

/// Schema-metadata key holding a collection sidecar's member table (JSON).
///
/// Its presence — not the version alone — is what makes a file a collection,
/// so the two readers can tell each other's files apart and say so, rather
/// than failing on a missing column further in.
pub const P5S_MEMBERS_KEY: &str = "escapepod:members";
/// Default annotation column name (what `escpod annotate` writes).
pub const DEFAULT_ANNOTATION_NAME: &str = "barcode";

/// Column names reserved for the read index; everything else is an
/// annotation.
///
/// `member_idx` is only ever *written* by a collection sidecar, but it is
/// reserved in both shapes: the naming rules for an annotation should not
/// depend on which shape it happens to land in, and a per-file annotation
/// called `member_idx` would become unreadable the moment its file joined a
/// collection.
pub const RESERVED_COLUMNS: [&str; 4] = ["read_id", "batch_idx", "row_idx", "member_idx"];

/// Sidecar path for a POD5 file: `.p5s` appended to the full filename
/// (`reads.pod5` → `reads.pod5.p5s`), mirroring the samtools convention
/// (`reads.bam` → `reads.bam.bai`).
pub fn sidecar_path(pod5_path: impl AsRef<Path>) -> PathBuf {
    let mut s = pod5_path.as_ref().as_os_str().to_owned();
    s.push(".p5s");
    PathBuf::from(s)
}

/// One row of a per-file sidecar's read index: `(read UUID bytes, batch, row)`.
pub type IndexEntry = ([u8; 16], u32, u32);

/// One row of a collection sidecar's read index: `(read UUID bytes, member,
/// batch, row)`, where `member` indexes [`CollectionSidecar::members`].
pub type CollectionEntry = ([u8; 16], u32, u32, u32);

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
    entries: Vec<IndexEntry>,
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
    pub fn new(mut entries: Vec<IndexEntry>) -> Self {
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
    ///
    /// An **empty** `counts` is "I could not measure", not "there are no
    /// batches", and leaves any existing value alone. The distinction is not
    /// academic: `Reader::measure_signal_batch_rows` returns an empty vec for
    /// every failure it has — no signal table, a slice past EOF, an
    /// unparseable footer — so treating empty as a value let a single failed
    /// re-measure erase a geometry an earlier run had recorded correctly, and
    /// silently, since the sidecar stays valid and merely gets slow again.
    /// Nothing is lost by declining to record: the POD5 still holds the
    /// answer.
    pub fn set_signal_batch_rows(&mut self, counts: Vec<u64>) {
        if counts.is_empty() {
            return;
        }
        self.signal_batch_rows = Some(counts);
    }

    /// Forget any recorded signal-table batch geometry.
    ///
    /// Separate from [`Self::set_signal_batch_rows`] so that discarding the
    /// cache is something a caller has to say, rather than something an empty
    /// vec does by accident.
    pub fn clear_signal_batch_rows(&mut self) {
        self.signal_batch_rows = None;
    }

    /// The read-index entries, sorted by UUID.
    pub fn entries(&self) -> &[IndexEntry] {
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
    pub fn set_entries(&mut self, mut entries: Vec<IndexEntry>) {
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

/// The `.p5s` version gate, shared by [`read_sidecar_file`] and
/// [`read_sidecar_metadata`] so the two can never disagree about whether a
/// sidecar is loadable.
fn check_version(metadata: &HashMap<String, String>, p5s_path: &Path) -> Result<()> {
    // Checked before the version, so the message names the shape rather than a
    // number. A collection genuinely is a `.p5s` this build reads — just not
    // through this reader — and "version 3 unsupported" would say the opposite.
    if metadata.contains_key(P5S_MEMBERS_KEY) {
        return Err(Error::Parse(format!(
            "{} is a collection sidecar covering a directory of POD5 files, \
             not a per-file sidecar; read it with `escpod annotate --list <dir>`",
            p5s_path.display()
        )));
    }
    match metadata.get(P5S_VERSION_KEY).map(String::as_str) {
        Some(P5S_VERSION | P5S_VERSION_SCORES) => Ok(()),
        Some(other) => Err(Error::Parse(format!(
            ".p5s version {other} unsupported (this escpod reads \
             {P5S_VERSION} and {P5S_VERSION_SCORES})"
        ))),
        None => Err(Error::Parse(format!(
            "{} has no {P5S_VERSION_KEY} metadata; not an escapepod sidecar",
            p5s_path.display()
        ))),
    }
}

/// The `.p5s` identity gate, shared by every per-file reader so none of them
/// can drift on what "belongs to this POD5" means — or on the message it
/// produces, which is the one place [`SidecarProvenance`] earns its keep.
///
/// Returns the provenance it had to read anyway, so a caller that wants it
/// does not parse the same three keys a second time.
fn check_identity(
    metadata: &HashMap<String, String>,
    p5s_path: &Path,
    expect: &Pod5Identity,
) -> Result<SidecarProvenance> {
    // Read before the gate, not after: by the time the mismatch is known we
    // have already returned.
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
    Ok(provenance)
}

/// Why a sidecar could not be loaded — the distinction that decides whether
/// replacing it is repair or destruction.
///
/// A sidecar mixes a rebuildable cache with data products that exist nowhere
/// else: a demux run that took hours leaves its barcode and score columns here
/// and nowhere else. So "could not load it, start fresh" is only ever right
/// when there is nothing to lose ([`Self::Absent`]) or when what would be lost
/// describes a different file ([`Self::Foreign`]). Every other failure —
/// a truncated file, a transient read error, a version this build does not
/// know — must stop the write, because the annotations are probably still
/// there and a rebuild would replace them with an empty column set.
#[derive(Debug)]
pub enum SidecarLoad {
    /// Loaded and bound to this POD5.
    Loaded(Box<Sidecar>),
    /// No sidecar exists. Nothing to lose.
    Absent,
    /// Exists, but is bound to a different (or since-replaced) POD5. Its
    /// contents describe another file, so replacing it loses nothing that
    /// applies here.
    Foreign(Error),
    /// Exists and belongs here as far as anyone can tell, but could not be
    /// read. **Never** a reason to replace it.
    Unreadable(Error),
}

/// Load the sidecar at `p5s_path`, classifying a failure rather than
/// flattening it into one error.
///
/// Use this instead of [`read_sidecar_file`] anywhere the next step is a
/// *write*. `read_sidecar_file` cannot tell a caller whether an error means
/// "this belongs to another file" or "this is your data and I could not read
/// it", and a caller that guesses wrong turns a bad read into permanent loss.
pub fn load_sidecar_for_write(p5s_path: impl AsRef<Path>, expect: &Pod5Identity) -> SidecarLoad {
    let p5s_path = p5s_path.as_ref();
    if !p5s_path.exists() {
        return SidecarLoad::Absent;
    }
    match read_sidecar_file(p5s_path, expect) {
        Ok(Some(sc)) => SidecarLoad::Loaded(Box::new(sc)),
        Ok(None) => SidecarLoad::Absent,
        Err(e) => {
            // Identity is the one failure that proves the contents are not
            // about this POD5. `read_sidecar_metadata` re-reads the binding
            // without decoding a batch, so a mismatch is separable from a
            // sidecar that merely would not parse.
            match read_sidecar_metadata(p5s_path, expect) {
                Err(inner) if is_identity_mismatch(&inner) => SidecarLoad::Foreign(e),
                _ => SidecarLoad::Unreadable(e),
            }
        }
    }
}

/// Whether an error from the sidecar readers is the identity gate refusing a
/// sidecar bound to another POD5.
fn is_identity_mismatch(e: &Error) -> bool {
    matches!(e, Error::Parse(msg) if msg.contains("does not match this POD5 file"))
}

/// Load the sidecar at `p5s_path`, validating it against the POD5's
/// identity. `Ok(None)` when no sidecar exists; an error when one exists
/// but is malformed, or bound to a different / since-replaced POD5.
///
/// Callers that intend to **write** the sidecar back should use
/// [`load_sidecar_for_write`] instead, which says *why* a load failed — the
/// difference between a sidecar worth replacing and one worth protecting.
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

    check_version(metadata, p5s_path)?;
    let provenance = check_identity(metadata, p5s_path, expect)?;

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

    let mut entries: Vec<IndexEntry> = Vec::new();
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

    // The same version gate `read_sidecar_file` applies, and it must stay the
    // same. A caller that checks cheaply here and then rewrites the sidecar —
    // `escpod index` completing a missing cache — would otherwise be told the
    // file is fine, and only discover on the rewrite that it cannot be loaded,
    // by which point the rewrite has already replaced annotations that took
    // hours to compute. The two readers agreeing about "is this loadable" is
    // what makes check-then-write safe.
    check_version(metadata, p5s_path)?;

    let provenance = check_identity(metadata, p5s_path, expect)?;

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
///
/// This overwrites unconditionally. A caller doing a read-modify-write should
/// prefer [`write_sidecar_file_checked`], which additionally refuses to
/// discard an update that landed in between.
pub fn write_sidecar_file(
    p5s_path: impl AsRef<Path>,
    identity: &Pod5Identity,
    sidecar: &Sidecar,
) -> Result<()> {
    write_sidecar_file_checked(p5s_path, identity, sidecar, None)
}

/// [`write_sidecar_file`], refusing the write if the destination changed since
/// `expect_unchanged` was taken.
///
/// The guard exists because a sidecar is not only a cache: a `demux --annotate`
/// run that took hours leaves its barcode and score columns here and nowhere
/// else. See [`SidecarStamp`] for the race this closes and the one it does not.
pub fn write_sidecar_file_checked(
    p5s_path: impl AsRef<Path>,
    identity: &Pod5Identity,
    sidecar: &Sidecar,
    expect_unchanged: Option<&SidecarStamp>,
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
    fields.extend(value_fields(&sidecar.annotations, &sidecar.scores)?);
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
    columns.extend(build_value_columns(
        n,
        || sidecar.entries.iter().map(|(uuid_bytes, _, _)| uuid_bytes),
        &sidecar.annotations,
        &sidecar.scores,
    )?);
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
    if let Some(expected) = expect_unchanged {
        // Last check before the rename. See `SidecarStamp`.
        expected.verify(p5s_path.as_ref())?;
    }
    atomic.commit()
}

/// A cheap fingerprint of a sidecar file, taken when it is read and checked
/// again immediately before it is overwritten.
///
/// The atomic write makes a sidecar update all-or-nothing, which prevents a
/// *torn* file — but says nothing about a **lost update**. The realistic case
/// is not corruption, it is two writers: a `demux --annotate` run that takes
/// hours is still going when someone runs `escpod index` on the same file to
/// speed it up. Index reads the sidecar as it is now, demux finishes and
/// writes its barcodes, index finishes and renames over them. Nothing errors,
/// nothing is corrupt, and hours of classification are gone.
///
/// Comparing size and mtime across the read-modify-write closes the window a
/// caller actually has. It is not a lock and does not pretend to be: two
/// writers that interleave inside the stat granularity can still race, and the
/// remedy for that is not to run two writers. What it does guarantee is that
/// an update which visibly landed in between is never silently discarded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarStamp {
    len: u64,
    modified: Option<std::time::SystemTime>,
}

impl SidecarStamp {
    /// Fingerprint the sidecar at `p5s_path`, or `None` if it does not exist.
    ///
    /// Take this *before* the slow part of a read-modify-write (the index
    /// scan, the geometry walk), so the window it covers is the whole
    /// operation rather than its last instant.
    pub fn of(p5s_path: impl AsRef<Path>) -> Option<Self> {
        let meta = std::fs::metadata(p5s_path.as_ref()).ok()?;
        Some(Self {
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }

    /// Fail if the sidecar no longer matches this fingerprint.
    fn verify(&self, p5s_path: &Path) -> Result<()> {
        let now = Self::of(p5s_path);
        if now.as_ref() == Some(self) {
            return Ok(());
        }
        Err(Error::Parse(format!(
            "{} changed while it was being updated — another process (a demux \
             run finishing, or a concurrent `escpod index`) wrote it in the \
             meantime. Refusing to overwrite, because that write may hold \
             annotations or scores that exist nowhere else. Re-run this \
             command.",
            p5s_path.display()
        )))
    }
}

fn downcast_u32<'a>(batch: &'a RecordBatch, idx: usize, name: &str) -> Result<&'a UInt32Array> {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Parse(format!(".p5s column '{name}' is not u32")))
}

// ---------------------------------------------------------------------------
// Value columns, shared by both sidecar shapes
// ---------------------------------------------------------------------------

/// The `Field`s for a sidecar's annotation and score columns, in the order
/// [`build_value_columns`] produces them.
///
/// Shared by the per-file and collection writers so the two cannot drift on
/// what a column of each kind looks like — nor on the rule that a name is one
/// column, which is enforced here rather than separately in each.
fn value_fields(annotations: &[AnnotationSection], scores: &[ScoreSection]) -> Result<Vec<Field>> {
    let mut fields = Vec::with_capacity(annotations.len() + scores.len());
    for annotation in annotations {
        fields.push(Field::new(
            annotation.name(),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        ));
    }
    for score in scores {
        if annotations.iter().any(|a| a.name == score.name) {
            return Err(Error::Parse(format!(
                "'{}' is both a label and a score column; a name is one column",
                score.name
            )));
        }
        fields.push(Field::new(score.name(), DataType::Float32, true));
    }
    Ok(fields)
}

/// The value columns themselves, aligned to [`value_fields`].
///
/// `ids` yields the row keys and is called once per column rather than
/// materialised into a slice: a collection over fifty files indexes tens of
/// millions of reads, and copying their UUIDs out to hand this a `&[[u8; 16]]`
/// would cost more than every column it builds.
fn build_value_columns<'a, F, I>(
    n: usize,
    ids: F,
    annotations: &[AnnotationSection],
    scores: &[ScoreSection],
) -> Result<Vec<ArrayRef>>
where
    F: Fn() -> I,
    I: Iterator<Item = &'a [u8; 16]>,
{
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(annotations.len() + scores.len());
    for annotation in annotations {
        let mut keys = Int32Builder::with_capacity(n);
        for uuid_bytes in ids() {
            match annotation.label_idx(uuid_bytes) {
                Some(idx) => keys.append_value(i32::from(idx)),
                None => keys.append_null(),
            }
        }
        let values = Arc::new(StringArray::from_iter_values(annotation.labels().iter()));
        let dict = DictionaryArray::<Int32Type>::try_new(keys.finish(), values)?;
        columns.push(Arc::new(dict));
    }
    for score in scores {
        // Null, not a sentinel, for a read this column has no value for —
        // every f32 is a possible score.
        columns.push(Arc::new(Float32Array::from_iter(
            ids().map(|uuid_bytes| score.value_of(uuid_bytes)),
        )));
    }
    Ok(columns)
}

// ---------------------------------------------------------------------------
// Collection sidecars
// ---------------------------------------------------------------------------

/// Collection-sidecar path for a directory of POD5 files: `.p5s` appended to
/// the directory's own path (`run1/pod5` → `run1/pod5.p5s`), so it sits
/// *beside* the directory rather than among the files it describes.
///
/// The same rule as [`sidecar_path`] — append `.p5s` to the path you name —
/// which is also why a directory's collection can never collide with a
/// member's own sidecar: those always end in `.pod5.p5s`, inside.
///
/// Two normalisations, both so the destination depends on the directory and
/// not on how it was typed:
///
/// * a trailing separator is dropped first, since appending to `pod5/` would
///   otherwise produce the hidden file `pod5/.p5s` *inside* the directory;
/// * a path with no file name at all (`.`, `..`, `/`) is canonicalised first,
///   because `.` + `.p5s` is the file `..p5s`, which is nobody's intent.
pub fn collection_sidecar_path(dir: impl AsRef<Path>) -> PathBuf {
    let dir = dir.as_ref();
    let normalized: PathBuf = dir.components().collect();
    let base = if normalized.file_name().is_some() {
        normalized
    } else {
        dir.canonicalize().unwrap_or(normalized)
    };
    sidecar_path(base)
}

/// One POD5 covered by a [`CollectionSidecar`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarMember {
    /// The POD5's path relative to the directory the collection covers, so a
    /// collection survives that directory being moved or renamed. Falls back
    /// to the bare file name for a member that lies outside it.
    pub name: String,
    /// The POD5's footer `file_identifier`.
    pub file_id: Uuid,
    /// The POD5's byte size when the collection was written.
    pub size: u64,
    /// How many of this member's reads the collection indexes.
    pub reads: u64,
}

impl SidecarMember {
    /// The identity a per-file reader checks this member against.
    pub fn identity(&self) -> Pod5Identity {
        Pod5Identity {
            file_id: self.file_id,
            size: self.size,
        }
    }
}

/// Serialisation shape for [`SidecarMember`].
///
/// A mirror rather than a derive on the type itself because `uuid` is built
/// here without its `serde` feature, and turning that on for one JSON blob
/// would change a workspace-wide dependency.
#[derive(serde::Serialize, serde::Deserialize)]
struct MemberJson {
    name: String,
    file_id: String,
    size: u64,
    reads: u64,
}

/// One `.p5s` covering every POD5 in a directory.
///
/// Same file format, same annotation and score columns, same Arrow-readable
/// layout — what differs is what a row is bound to. A per-file [`Sidecar`]
/// locates a read by `(batch_idx, row_idx)` in the single POD5 named in its
/// schema metadata; a collection adds `member_idx` and carries a member table
/// ([`P5S_MEMBERS_KEY`]) naming the file each locator is into.
///
/// It exists because a fifty-file run produced fifty sidecars, and the
/// annotations in them — a demux run's barcodes and scores — are one result,
/// not fifty. Read UUIDs are globally unique, so a single set of columns
/// covers every member with no join.
///
/// It **complements** the per-file sidecars rather than replacing them: those
/// also cache the read index and the signal batch geometry that
/// `Reader::open` uses, both of which are per-POD5 by nature. `demux
/// --annotate` writes both.
///
/// Two things a collection deliberately does not carry:
///
/// * an experimental design — `escpod annotate --design` targets member
///   sidecars, where the derived columns it materialises belong;
/// * a signal batch geometry — that is a property of one POD5's signal table,
///   and belongs in that file's own sidecar.
#[derive(Debug, Clone, Default)]
pub struct CollectionSidecar {
    /// The POD5 files covered, in the order `member_idx` indexes.
    members: Vec<SidecarMember>,
    /// (uuid bytes, member, batch, row), sorted by UUID.
    entries: Vec<CollectionEntry>,
    /// Name-sorted annotations; entries are subsets of `entries`.
    annotations: Vec<AnnotationSection>,
    /// Name-sorted numeric columns; entries are subsets of `entries`.
    scores: Vec<ScoreSection>,
}

impl CollectionSidecar {
    /// Build a collection from its members and their read index entries (any
    /// order; `member` indexes into `members`).
    ///
    /// Entries naming a member that does not exist are dropped rather than
    /// rejected — a caller assembling this from N files should not be able to
    /// write a locator that points nowhere. Duplicate UUIDs keep the first
    /// occurrence in member order, which is what a file listed twice produces.
    pub fn new(members: Vec<SidecarMember>, mut entries: Vec<CollectionEntry>) -> Self {
        let n_members = members.len() as u32;
        entries.retain(|&(_, member, _, _)| member < n_members);
        entries.sort_unstable_by_key(|e| (e.0, e.1));
        entries.dedup_by_key(|e| e.0);
        Self {
            members,
            entries,
            annotations: Vec::new(),
            scores: Vec::new(),
        }
    }

    /// The POD5 files covered, in `member_idx` order.
    pub fn members(&self) -> &[SidecarMember] {
        &self.members
    }

    /// The index entries `(uuid, member_idx, batch_idx, row_idx)`, sorted by
    /// UUID.
    pub fn entries(&self) -> &[CollectionEntry] {
        &self.entries
    }

    /// Total reads indexed across every member.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the collection indexes no reads.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Locate a read: which member holds it, and where in that member's reads
    /// table.
    pub fn locate(&self, uuid: &Uuid) -> Option<(&SidecarMember, u32, u32)> {
        let i = self
            .entries
            .binary_search_by_key(uuid.as_bytes(), |&(k, _, _, _)| k)
            .ok()?;
        let (_, member, batch, row) = self.entries[i];
        Some((self.members.get(member as usize)?, batch, row))
    }

    /// The annotation with the given name, if present.
    pub fn annotation(&self, name: &str) -> Option<&AnnotationSection> {
        self.annotations.iter().find(|a| a.name == name)
    }

    /// All annotations, in name order.
    pub fn annotations(&self) -> &[AnnotationSection] {
        &self.annotations
    }

    /// Replace (or add) an annotation, keyed by its name. Displaces a score
    /// column of the same name — see [`Sidecar::set_annotation`].
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

    /// Remove an annotation by name; returns whether it was present.
    pub fn remove_annotation(&mut self, name: &str) -> bool {
        let before = self.annotations.len();
        self.annotations.retain(|a| a.name != name);
        self.annotations.len() != before
    }

    /// The score column with the given name, if present.
    pub fn score(&self, name: &str) -> Option<&ScoreSection> {
        self.scores.iter().find(|s| s.name == name)
    }

    /// All score columns, in name order.
    pub fn scores(&self) -> &[ScoreSection] {
        &self.scores
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

    /// Replace the members and index entries, keeping the annotation and score
    /// columns.
    ///
    /// This is a re-annotation of the same directory: the file set is
    /// authoritative and the columns are data. A column keeps only the reads
    /// still indexed — [`build_value_columns`] looks every row up by UUID, so
    /// a read that left the collection simply stops appearing.
    pub fn set_index(&mut self, members: Vec<SidecarMember>, entries: Vec<CollectionEntry>) {
        let rebuilt = Self::new(members, entries);
        self.members = rebuilt.members;
        self.entries = rebuilt.entries;
    }

    /// The member matching `identity`, if this collection covers it.
    ///
    /// Identity, not name: a collection is *found* by where it sits, but the
    /// question a caller has is "do this file's reads appear in here", which
    /// the recorded `file_id` + `size` answer and a path never could.
    pub fn member_of(&self, identity: &Pod5Identity) -> Option<(u32, &SidecarMember)> {
        self.members
            .iter()
            .position(|m| m.identity() == *identity)
            .map(|i| (i as u32, &self.members[i]))
    }

    /// Project the collection down to one member: the per-file [`Sidecar`]
    /// that POD5 would have had if the run had written one.
    ///
    /// This is what lets every existing consumer — `demux split --sidecar`,
    /// `filter --annotation`, `view --include`, the Python `Reader` — read a
    /// collection without knowing one exists. They ask a POD5 for its columns;
    /// whether the answer came from a file of its own or from the directory's
    /// collection is a question about storage, not about the read.
    ///
    /// `None` when no member matches `identity`, and that **is** the identity
    /// gate. A collection has no file-level binding to check — it is bound to
    /// N files, not one — so the check moves down a level: a member whose
    /// `file_id` and `size` do not match the POD5 in hand is not that file,
    /// and none of its rows are ever returned for it.
    pub fn view_for(&self, identity: &Pod5Identity) -> Option<Sidecar> {
        let (member_idx, _) = self.member_of(identity)?;
        let entries: Vec<IndexEntry> = self
            .entries
            .iter()
            .filter(|&&(_, m, _, _)| m == member_idx)
            .map(|&(id, _, batch, row)| (id, batch, row))
            .collect();

        let mut view = Sidecar::new(entries);
        // Each column is restricted to this member's reads rather than carried
        // across whole: one that still named every read in the directory would
        // make `escpod view` on a single POD5 report labels for reads that file
        // does not hold.
        for annotation in &self.annotations {
            let pairs: Vec<(Uuid, &str)> = view
                .entries()
                .iter()
                .filter_map(|&(id, _, _)| {
                    let uuid = Uuid::from_bytes(id);
                    annotation.get(&uuid).map(|label| (uuid, label))
                })
                .collect();
            // Name and labels both came out of a sidecar that already parsed,
            // so a failure here would be a bug in `from_pairs` rather than
            // anything about the file; dropping the column beats panicking.
            if let Ok(section) = AnnotationSection::from_pairs(annotation.name(), pairs) {
                view.set_annotation(section);
            }
        }
        for score in &self.scores {
            let pairs: Vec<(Uuid, f32)> = view
                .entries()
                .iter()
                .filter_map(|&(id, _, _)| {
                    let uuid = Uuid::from_bytes(id);
                    score.get(&uuid).map(|v| (uuid, v))
                })
                .collect();
            if let Ok(section) = ScoreSection::from_pairs(score.name(), pairs) {
                view.set_score(section);
            }
        }
        Some(view)
    }
}

/// Read the collection sidecar at `p5s_path`. `Ok(None)` when none exists.
///
/// There is no identity argument: a collection is bound to N POD5 files, and
/// each member carries its own [`Pod5Identity`] for a caller that opens it to
/// check ([`SidecarMember::identity`]). Verifying all of them here would mean
/// stat-ing and opening every member just to read one column.
pub fn read_collection_file(p5s_path: impl AsRef<Path>) -> Result<Option<CollectionSidecar>> {
    let p5s_path = p5s_path.as_ref();
    let file = match File::open(p5s_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::from(e)),
    };
    let reader = ArrowFileReader::try_new(file, None).map_err(|e| {
        Error::Parse(format!(
            "{} is not a readable .p5s sidecar: {e}",
            p5s_path.display()
        ))
    })?;
    let schema = reader.schema();
    let metadata = schema.metadata();

    // The mirror of `check_version`'s collection guard: each reader names the
    // other's file rather than reporting a missing column or a bad version.
    let Some(members_json) = metadata.get(P5S_MEMBERS_KEY) else {
        return Err(Error::Parse(format!(
            "{} is a per-file sidecar for a single POD5, not a collection",
            p5s_path.display()
        )));
    };
    match metadata.get(P5S_VERSION_KEY).map(String::as_str) {
        Some(P5S_VERSION_COLLECTION) => {}
        Some(other) => {
            return Err(Error::Parse(format!(
                ".p5s collection version {other} unsupported \
                 (this escpod reads {P5S_VERSION_COLLECTION})"
            )));
        }
        None => {
            return Err(Error::Parse(format!(
                "{} has no {P5S_VERSION_KEY} metadata; not an escapepod sidecar",
                p5s_path.display()
            )));
        }
    }

    let members: Vec<SidecarMember> = serde_json::from_str::<Vec<MemberJson>>(members_json)
        .map_err(|e| {
            Error::Parse(format!(
                "{}: unreadable member table: {e}",
                p5s_path.display()
            ))
        })?
        .into_iter()
        .map(|m| {
            let file_id = Uuid::parse_str(&m.file_id).map_err(|e| {
                Error::Parse(format!(
                    "{}: member '{}' has an invalid file id: {e}",
                    p5s_path.display(),
                    m.name
                ))
            })?;
            Ok(SidecarMember {
                name: m.name,
                file_id,
                size: m.size,
                reads: m.reads,
            })
        })
        .collect::<Result<_>>()?;

    let read_id_idx = schema.index_of("read_id")?;
    let member_idx_idx = schema.index_of("member_idx")?;
    let batch_idx_idx = schema.index_of("batch_idx")?;
    let row_idx_idx = schema.index_of("row_idx")?;
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

    let mut entries: Vec<CollectionEntry> = Vec::new();
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
            .ok_or_else(|| {
                Error::Parse(".p5s read_id column is not FixedSizeBinary(16)".to_string())
            })?;
        let members_col = downcast_u32(&batch, member_idx_idx, "member_idx")?;
        let batches = downcast_u32(&batch, batch_idx_idx, "batch_idx")?;
        let rows = downcast_u32(&batch, row_idx_idx, "row_idx")?;

        let base = entries.len();
        entries.reserve(batch.num_rows());
        for i in 0..batch.num_rows() {
            let mut key = [0u8; 16];
            key.copy_from_slice(ids.value(i));
            entries.push((key, members_col.value(i), batches.value(i), rows.value(i)));
        }

        for (idx, name) in &annotation_columns {
            let dict = batch
                .column(*idx)
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| {
                    Error::Parse(format!(
                        ".p5s column '{name}' is not dictionary-encoded utf8"
                    ))
                })?;
            let values = dict
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::Parse(format!(
                        ".p5s column '{name}' has non-utf8 dictionary values"
                    ))
                })?;
            let out = annotation_pairs.get_mut(name).expect("column registered");
            for i in 0..batch.num_rows() {
                if dict.is_null(i) {
                    continue;
                }
                let key = dict.keys().value(i) as usize;
                out.push((
                    Uuid::from_bytes(entries[base + i].0),
                    values.value(key).to_string(),
                ));
            }
        }
        for (idx, name) in &score_columns {
            let values = batch
                .column(*idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::Parse(format!(".p5s column '{name}' is not float32")))?;
            let out = score_pairs.get_mut(name).expect("column registered");
            for i in 0..batch.num_rows() {
                if values.is_null(i) {
                    continue;
                }
                out.push((Uuid::from_bytes(entries[base + i].0), values.value(i)));
            }
        }
    }

    let mut sidecar = CollectionSidecar::new(members, entries);
    for (_, name) in &annotation_columns {
        let pairs = annotation_pairs.remove(name).expect("column registered");
        sidecar.set_annotation(AnnotationSection::from_pairs(
            name,
            pairs.iter().map(|(id, label)| (*id, label.as_str())),
        )?);
    }
    for (_, name) in &score_columns {
        let pairs = score_pairs.remove(name).expect("column registered");
        sidecar.set_score(ScoreSection::from_pairs(name, pairs)?);
    }
    Ok(Some(sidecar))
}

/// Atomically write a collection sidecar. The destination is either the
/// previous file or the complete new one — never a torn mix.
pub fn write_collection_file(
    p5s_path: impl AsRef<Path>,
    sidecar: &CollectionSidecar,
) -> Result<()> {
    write_collection_file_checked(p5s_path, sidecar, None)
}

/// [`write_collection_file`], refusing the write if the destination changed
/// since `expect_unchanged` was taken. See [`SidecarStamp`].
pub fn write_collection_file_checked(
    p5s_path: impl AsRef<Path>,
    sidecar: &CollectionSidecar,
    expect_unchanged: Option<&SidecarStamp>,
) -> Result<()> {
    let members: Vec<MemberJson> = sidecar
        .members
        .iter()
        .map(|m| MemberJson {
            name: m.name.clone(),
            file_id: m.file_id.to_string(),
            size: m.size,
            reads: m.reads,
        })
        .collect();
    let members_json = serde_json::to_string(&members)
        .map_err(|e| Error::Parse(format!("member table serialization failed: {e}")))?;

    let mut metadata = HashMap::new();
    // Not content-gated the way `P5S_VERSION_SCORES` is: the member table is a
    // different shape, not a column type, so every collection declares 3.
    metadata.insert(
        P5S_VERSION_KEY.to_string(),
        P5S_VERSION_COLLECTION.to_string(),
    );
    metadata.insert(P5S_MEMBERS_KEY.to_string(), members_json);
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

    let mut fields = vec![
        Field::new("read_id", DataType::FixedSizeBinary(16), false).with_metadata(HashMap::from([
            (
                "ARROW:extension:name".to_string(),
                "minknow.uuid".to_string(),
            ),
        ])),
        Field::new("member_idx", DataType::UInt32, false),
        Field::new("batch_idx", DataType::UInt32, false),
        Field::new("row_idx", DataType::UInt32, false),
    ];
    fields.extend(value_fields(&sidecar.annotations, &sidecar.scores)?);
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));

    let n = sidecar.entries.len();
    let mut id_builder = FixedSizeBinaryBuilder::with_capacity(n, 16);
    for (uuid_bytes, _, _, _) in &sidecar.entries {
        id_builder.append_value(uuid_bytes).map_err(Error::Arrow)?;
    }
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(id_builder.finish()),
        Arc::new(UInt32Array::from_iter_values(
            sidecar.entries.iter().map(|&(_, member, _, _)| member),
        )),
        Arc::new(UInt32Array::from_iter_values(
            sidecar.entries.iter().map(|&(_, _, batch, _)| batch),
        )),
        Arc::new(UInt32Array::from_iter_values(
            sidecar.entries.iter().map(|&(_, _, _, row)| row),
        )),
    ];
    columns.extend(build_value_columns(
        n,
        || {
            sidecar
                .entries
                .iter()
                .map(|(uuid_bytes, _, _, _)| uuid_bytes)
        },
        &sidecar.annotations,
        &sidecar.scores,
    )?);
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
    if let Some(expected) = expect_unchanged {
        expected.verify(p5s_path.as_ref())?;
    }
    atomic.commit()
}

/// Read **only** the index columns of a per-file sidecar, validating identity
/// and version exactly as [`read_sidecar_file`] does.
///
/// The same answer as `read_sidecar_file(..).entries()`, without decoding the
/// annotation columns — which for an annotated sidecar is most of the work and
/// all of the allocation: a label column materialises one `String` per read on
/// the way in, and a demux run leaves five columns behind. Every caller that
/// wants a read index and nothing else (`Reader::read_index`, assembling a
/// [`CollectionSidecar`] out of N members) was paying for columns it discarded.
///
/// Arrow's IPC file reader takes the projection up front and decompresses only
/// the buffers it is asked for, so this is genuinely less work rather than the
/// same work filtered afterwards.
pub fn read_sidecar_entries(
    p5s_path: impl AsRef<Path>,
    expect: &Pod5Identity,
) -> Result<Option<Vec<IndexEntry>>> {
    let p5s_path = p5s_path.as_ref();
    let file = match File::open(p5s_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::from(e)),
    };
    let probe = ArrowFileReader::try_new(file, None).map_err(|e| {
        Error::Parse(format!(
            "{} is not a readable .p5s sidecar (legacy .p5i or corrupt?): {e}; \
             delete it and rebuild with `escpod index` / `escpod annotate`",
            p5s_path.display()
        ))
    })?;
    let schema = probe.schema();
    check_version(schema.metadata(), p5s_path)?;
    check_identity(schema.metadata(), p5s_path, expect)?;

    let projection = vec![
        schema.index_of("read_id")?,
        schema.index_of("batch_idx")?,
        schema.index_of("row_idx")?,
    ];
    drop(probe);

    let file = File::open(p5s_path)?;
    let reader = ArrowFileReader::try_new(file, Some(projection))
        .map_err(|e| Error::Parse(format!("corrupt .p5s: {e}")))?;
    // Positions in the *projected* schema, which need not be 0/1/2 — arrow is
    // free to return the selected fields in original-schema order, and looking
    // them up again costs nothing next to being silently wrong about it.
    let projected = reader.schema();
    let read_id_idx = projected.index_of("read_id")?;
    let batch_idx_idx = projected.index_of("batch_idx")?;
    let row_idx_idx = projected.index_of("row_idx")?;

    let mut entries: Vec<IndexEntry> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| Error::Parse(format!("corrupt .p5s: {e}")))?;
        let ids = batch
            .column(read_id_idx)
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeBinaryArray>()
            .ok_or_else(|| {
                Error::Parse(".p5s read_id column is not FixedSizeBinary(16)".to_string())
            })?;
        let batches = downcast_u32(&batch, batch_idx_idx, "batch_idx")?;
        let rows = downcast_u32(&batch, row_idx_idx, "row_idx")?;
        entries.reserve(batch.num_rows());
        for i in 0..batch.num_rows() {
            let mut key = [0u8; 16];
            key.copy_from_slice(ids.value(i));
            entries.push((key, batches.value(i), rows.value(i)));
        }
    }
    entries.sort_unstable_by_key(|e| e.0);
    Ok(Some(entries))
}

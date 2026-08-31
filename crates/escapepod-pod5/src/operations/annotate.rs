//! High-level annotation operations over the `.p5s` sidecar.
//!
//! [`write_annotation`] records a read → label mapping (e.g. demux barcode
//! assignments) in the sidecar next to a POD5 file; [`read_annotation`]
//! loads one back. The POD5 itself is never touched — see [`crate::sidecar`]
//! for the format and the identity binding that keeps a sidecar from ever
//! describing the wrong file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::reader::Reader;
use crate::sidecar::{
    self, AnnotationSection, CollectionSidecar, DEFAULT_ANNOTATION_NAME, Design, ScoreSection,
    Sidecar, SidecarMember, sidecar_path,
};
use crate::types::Uuid;

/// Options for [`write_annotation`].
#[derive(Debug, Clone)]
pub struct AnnotateOptions {
    /// Annotation (column) name to write.
    pub name: String,
    /// Replace a sidecar that is bound to a *different* POD5 — one left
    /// behind by a file that has since been replaced, whose contents describe
    /// something else. Without this, a valid sidecar is merged into (other
    /// annotations and the index are preserved) and a stale one is an error.
    ///
    /// This is **not** a licence to replace a sidecar that merely failed to
    /// load. A truncated file, or a version this build does not know, is
    /// refused either way: the annotations are probably still there, and a
    /// rebuild would replace them with an empty column set.
    pub overwrite: bool,
}

impl Default for AnnotateOptions {
    fn default() -> Self {
        Self {
            name: DEFAULT_ANNOTATION_NAME.to_string(),
            overwrite: false,
        }
    }
}

/// Outcome of [`write_annotation`].
#[derive(Debug, Clone)]
pub struct AnnotateResult {
    /// Reads present in the POD5 file.
    pub total_reads: usize,
    /// Reads that received a label (present in the file AND in the mapping,
    /// with a non-empty label).
    pub assigned_reads: usize,
    /// Distinct labels in the written annotation.
    pub labels: usize,
    /// Annotations in the sidecar after the write.
    pub annotations_in_sidecar: usize,
    /// Where the sidecar was written.
    pub sidecar_path: PathBuf,
}

/// Write (or update) an annotation in the `.p5s` sidecar next to a POD5.
///
/// The mapping is intersected with the reads actually present in the file;
/// entries for other reads, and entries with empty labels, are ignored. An
/// existing valid sidecar is merged into — its read index and other
/// annotations survive; an annotation with the same name is replaced.
pub fn write_annotation(
    pod5_path: impl AsRef<Path>,
    assignments: &HashMap<Uuid, String>,
    options: &AnnotateOptions,
) -> Result<AnnotateResult> {
    let column = ColumnWrite {
        name: options.name.clone(),
        values: ColumnValues::Labels(assignments.clone()),
    };
    let out = write_columns(pod5_path, std::slice::from_ref(&column), options.overwrite)?;
    let (assigned_reads, labels) = out
        .columns
        .first()
        .map_or((0, 0), |c| (c.assigned, c.labels));
    Ok(AnnotateResult {
        total_reads: out.total_reads,
        assigned_reads,
        labels,
        annotations_in_sidecar: out.columns_in_sidecar,
        sidecar_path: out.sidecar_path,
    })
}

/// One column to write into a sidecar.
#[derive(Debug, Clone)]
pub struct ColumnWrite {
    /// Column name, e.g. `barcode` or `crf_logp`.
    pub name: String,
    /// Its per-read values, and by their type which kind of column it is.
    pub values: ColumnValues,
}

/// The two kinds of annotation column a sidecar can hold.
#[derive(Debug, Clone)]
pub enum ColumnValues {
    /// Dictionary-encoded utf8 labels. Empty labels are dropped — an
    /// unassigned read is represented by absence.
    Labels(HashMap<Uuid, String>),
    /// The same column, supplied already interned: `codes` maps a read to an
    /// index into `dictionary`, and a code with no entry in `dictionary` is
    /// dropped like an empty label.
    ///
    /// Identical on disk to [`Labels`](Self::Labels) — the section is
    /// dictionary-encoded either way — and it exists for the caller's memory,
    /// not the file's. A demultiplexing run holds one entry per read until every
    /// input's sidecar is written, and a `String` per read to say `nbc07` is a
    /// 32-byte heap chunk plus a 24-byte struct for one of sixteen values: about
    /// 4 GB at 57 M reads, against 1.1 GB of `u32` codes and a dictionary that
    /// fits in a cache line's worth of pointers. Build the codes and hand them
    /// over; do not materialize the strings to call `Labels`.
    LabelCodes {
        /// Distinct labels, indexed by code.
        dictionary: Vec<String>,
        /// Per-read index into `dictionary`.
        codes: HashMap<Uuid, u32>,
    },
    /// `Float32` scores. `NaN` is refused rather than dropped, because unlike
    /// an empty label it is a value someone may have meant.
    Scores(HashMap<Uuid, f32>),
}

/// What one column write did.
#[derive(Debug, Clone)]
pub struct ColumnStat {
    pub name: String,
    /// Reads that got a value (present in the file AND in the mapping).
    pub assigned: usize,
    /// Distinct labels; 0 for a score column, which has no dictionary.
    pub labels: usize,
}

/// Outcome of [`write_columns`].
#[derive(Debug, Clone)]
pub struct ColumnsResult {
    /// Reads present in the POD5 file.
    pub total_reads: usize,
    /// Per column, in the order given.
    pub columns: Vec<ColumnStat>,
    /// Annotation columns of both kinds in the sidecar after the write.
    pub columns_in_sidecar: usize,
    /// Where the sidecar was written.
    pub sidecar_path: PathBuf,
}

/// Write several columns into the `.p5s` sidecar in **one** read-modify-write.
///
/// [`write_annotation`] is this with a single label column. Batching matters
/// because a scored demux run produces five columns at once (a barcode, the
/// lattice's preferred reference, and three numbers): writing them one at a
/// time would re-read, re-decode and re-encode the whole sidecar five times,
/// and leave four intermediate states on disk that describe a run that never
/// happened.
///
/// Each mapping is intersected with the reads actually present in the file. An
/// existing valid sidecar is merged into — its read index and other columns
/// survive; a column of the same name is replaced, whichever kind it was.
pub fn write_columns(
    pod5_path: impl AsRef<Path>,
    columns: &[ColumnWrite],
    overwrite: bool,
) -> Result<ColumnsResult> {
    let pod5_path = pod5_path.as_ref();
    let reader = Reader::open(pod5_path)?;
    let identity = reader.sidecar_identity()?;
    let p5s_path = sidecar_path(pod5_path);

    // Reuse a valid existing sidecar (its index entries are bound to this
    // exact POD5); otherwise build the index entries by scanning.
    //
    // `overwrite` licenses replacing a sidecar that will not load — but it was
    // meant for the *stale* case, a sidecar left behind by a POD5 that has
    // since been replaced. Extending it to every read failure means a
    // truncated file or a version this build does not know is answered by
    // silently discarding barcode and score columns that a demux run spent
    // hours on. `--force` is a statement about a sidecar that belongs to
    // another file, not a licence to destroy this one's data on any error.
    let stamp = sidecar::SidecarStamp::of(&p5s_path);
    let mut sc = match sidecar::load_sidecar_for_write(&p5s_path, &identity) {
        sidecar::SidecarLoad::Loaded(existing) => *existing,
        sidecar::SidecarLoad::Absent => {
            Sidecar::new(reader.build_read_index_from_scan()?.entries().to_vec())
        }
        sidecar::SidecarLoad::Foreign(e) if overwrite => {
            tracing::warn!(
                "{e} — replacing; any annotations or scores it held described \
                 another file and are discarded"
            );
            Sidecar::new(reader.build_read_index_from_scan()?.entries().to_vec())
        }
        sidecar::SidecarLoad::Foreign(e) | sidecar::SidecarLoad::Unreadable(e) => return Err(e),
    };

    let mut stats = Vec::with_capacity(columns.len());
    let mut touched_key = false;
    for column in columns {
        if let Some(design) = sc.design()
            && design.value_columns.contains(&column.name)
        {
            return Err(Error::Parse(format!(
                "annotation '{}' is derived from the experimental design; \
                 update the design instead",
                column.name
            )));
        }

        let stat = match &column.values {
            ColumnValues::Labels(assignments) => {
                let annotation = AnnotationSection::from_pairs(
                    &column.name,
                    sc.entries().iter().filter_map(|&(uuid_bytes, _, _)| {
                        let uuid = Uuid::from_bytes(uuid_bytes);
                        assignments
                            .get(&uuid)
                            .filter(|label| !label.is_empty())
                            .map(|label| (uuid, label.as_str()))
                    }),
                )?;
                let stat = ColumnStat {
                    name: column.name.clone(),
                    assigned: annotation.len(),
                    labels: annotation.labels().len(),
                };
                sc.set_annotation(annotation);
                stat
            }
            ColumnValues::LabelCodes { dictionary, codes } => {
                let annotation = AnnotationSection::from_pairs(
                    &column.name,
                    sc.entries().iter().filter_map(|&(uuid_bytes, _, _)| {
                        let uuid = Uuid::from_bytes(uuid_bytes);
                        let label = dictionary.get(*codes.get(&uuid)? as usize)?;
                        (!label.is_empty()).then_some((uuid, label.as_str()))
                    }),
                )?;
                let stat = ColumnStat {
                    name: column.name.clone(),
                    assigned: annotation.len(),
                    labels: annotation.labels().len(),
                };
                sc.set_annotation(annotation);
                stat
            }
            ColumnValues::Scores(values) => {
                let score = ScoreSection::from_pairs(
                    &column.name,
                    sc.entries().iter().filter_map(|&(uuid_bytes, _, _)| {
                        let uuid = Uuid::from_bytes(uuid_bytes);
                        values.get(&uuid).map(|&v| (uuid, v))
                    }),
                )?;
                let stat = ColumnStat {
                    name: column.name.clone(),
                    assigned: score.len(),
                    labels: 0,
                };
                sc.set_score(score);
                stat
            }
        };
        touched_key |= sc
            .design()
            .is_some_and(|d| d.key_columns.contains(&column.name));
        stats.push(stat);
    }

    // Rewriting a design key column changes what each read's key combination
    // resolves to — keep the derived columns in sync. Once, after all the
    // columns are in, rather than per column.
    if touched_key {
        sc.derive_design_columns()?;
    }
    reader.write_sidecar(&p5s_path, &mut sc, stamp.as_ref())?;

    Ok(ColumnsResult {
        total_reads: sc.len(),
        columns: stats,
        columns_in_sidecar: sc.annotations().len() + sc.scores().len(),
        sidecar_path: p5s_path,
    })
}

/// Options for [`write_design`].
#[derive(Debug, Clone, Default)]
pub struct DesignOptions {
    /// Explicit key columns. When `None`, every CSV column whose name
    /// matches an existing annotation is a key; the rest are experimental
    /// variables.
    pub keys: Option<Vec<String>>,
    /// Replace a sidecar bound to a *different* POD5 (annotations are lost —
    /// a design needs its key annotations, so this is mostly useful for
    /// recovery). As in [`AnnotateOptions::overwrite`], a sidecar that simply
    /// would not load is refused rather than replaced.
    pub overwrite: bool,
}

/// Outcome of [`write_design`].
#[derive(Debug, Clone)]
pub struct DesignResult {
    /// Key (annotation) columns, in CSV order.
    pub key_columns: Vec<String>,
    /// Experimental-variable columns, in CSV order.
    pub value_columns: Vec<String>,
    /// Rows in the design table.
    pub design_rows: usize,
    /// `(value column, reads assigned)` for each materialized column.
    pub derived: Vec<(String, usize)>,
    /// Where the sidecar was written.
    pub sidecar_path: PathBuf,
}

/// Record an experimental-design table in the `.p5s` sidecar and
/// materialize its variables as derived annotation columns.
///
/// The CSV has one column per design key (an existing annotation name, e.g.
/// `barcode` — or several, for combination designs like `ldx,edx`) and one
/// column per experimental variable (e.g. `condition`). Each read whose
/// key-annotation labels match a row gets that row's variable values.
pub fn write_design(
    pod5_path: impl AsRef<Path>,
    design_csv: impl AsRef<Path>,
    options: &DesignOptions,
) -> Result<DesignResult> {
    let pod5_path = pod5_path.as_ref();
    let reader = Reader::open(pod5_path)?;
    let identity = reader.sidecar_identity()?;
    let p5s_path = sidecar_path(pod5_path);

    // Same classification as `write_columns` — see there for why `overwrite`
    // may not stand in for "any read failure".
    let stamp = sidecar::SidecarStamp::of(&p5s_path);
    let mut sc = match sidecar::load_sidecar_for_write(&p5s_path, &identity) {
        sidecar::SidecarLoad::Loaded(existing) => *existing,
        sidecar::SidecarLoad::Absent => {
            Sidecar::new(reader.build_read_index_from_scan()?.entries().to_vec())
        }
        sidecar::SidecarLoad::Foreign(e) if options.overwrite => {
            tracing::warn!(
                "{e} — replacing; any annotations or scores it held described \
                 another file and are discarded"
            );
            Sidecar::new(reader.build_read_index_from_scan()?.entries().to_vec())
        }
        sidecar::SidecarLoad::Foreign(e) | sidecar::SidecarLoad::Unreadable(e) => return Err(e),
    };

    let design = parse_design_csv(design_csv.as_ref(), options.keys.as_deref(), &sc)?;
    let result_columns = (design.key_columns.clone(), design.value_columns.clone());
    let design_rows = design.rows.len();

    sc.set_design(design)?;
    let derived = sc.derive_design_columns()?;
    reader.write_sidecar(&p5s_path, &mut sc, stamp.as_ref())?;

    Ok(DesignResult {
        key_columns: result_columns.0,
        value_columns: result_columns.1,
        design_rows,
        derived,
        sidecar_path: p5s_path,
    })
}

/// Remove an annotation from the `.p5s` sidecar. Returns whether it was
/// present. Design key and value columns are refused — remove or update the
/// design instead, which keeps it the single source of truth.
pub fn remove_annotation(pod5_path: impl AsRef<Path>, name: &str) -> Result<bool> {
    let pod5_path = pod5_path.as_ref();
    let reader = Reader::open(pod5_path)?;
    let identity = reader.sidecar_identity()?;
    let p5s_path = sidecar_path(pod5_path);

    let stamp = sidecar::SidecarStamp::of(&p5s_path);
    let mut sc = sidecar::read_sidecar_file(&p5s_path, &identity)?
        .ok_or_else(|| Error::Parse(format!("no sidecar at {}", p5s_path.display())))?;
    if let Some(design) = sc.design() {
        if design.key_columns.contains(&name.to_string()) {
            return Err(Error::Parse(format!(
                "'{name}' is a design key column; remove the design first"
            )));
        }
        if design.value_columns.contains(&name.to_string()) {
            return Err(Error::Parse(format!(
                "'{name}' is derived from the experimental design; \
                 remove or update the design instead"
            )));
        }
    }
    let removed = sc.remove_annotation(name);
    if removed {
        reader.write_sidecar(&p5s_path, &mut sc, stamp.as_ref())?;
    }
    Ok(removed)
}

/// Remove the experimental design (and its derived columns) from the
/// `.p5s` sidecar. Returns whether one was present.
pub fn remove_design(pod5_path: impl AsRef<Path>) -> Result<bool> {
    let pod5_path = pod5_path.as_ref();
    let reader = Reader::open(pod5_path)?;
    let identity = reader.sidecar_identity()?;
    let p5s_path = sidecar_path(pod5_path);

    let stamp = sidecar::SidecarStamp::of(&p5s_path);
    let mut sc = sidecar::read_sidecar_file(&p5s_path, &identity)?
        .ok_or_else(|| Error::Parse(format!("no sidecar at {}", p5s_path.display())))?;
    let removed = sc.remove_design();
    if removed {
        reader.write_sidecar(&p5s_path, &mut sc, stamp.as_ref())?;
    }
    Ok(removed)
}

/// Read the experimental design recorded in the `.p5s` sidecar.
pub fn read_design(pod5_path: impl AsRef<Path>) -> Result<Design> {
    let pod5_path = pod5_path.as_ref();
    let reader = Reader::open(pod5_path)?;
    let identity = reader.sidecar_identity()?;
    let p5s_path = sidecar_path(pod5_path);

    let sc = sidecar::read_sidecar_file(&p5s_path, &identity)?.ok_or_else(|| {
        Error::Parse(format!(
            "no sidecar at {}; create one with `escpod annotate`",
            p5s_path.display()
        ))
    })?;
    sc.design().cloned().ok_or_else(|| {
        Error::Parse(format!(
            "{} has no experimental design; record one with `escpod annotate --design`",
            p5s_path.display()
        ))
    })
}

/// Parse a design CSV, splitting header columns into keys and variables.
fn parse_design_csv(path: &Path, keys: Option<&[String]>, sc: &Sidecar) -> Result<Design> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|e| Error::Parse(format!("cannot read design CSV: {e}")))?;
    let headers: Vec<String> = csv_reader
        .headers()
        .map_err(|e| Error::Parse(format!("design CSV header error: {e}")))?
        .iter()
        .map(str::to_string)
        .collect();

    let is_key: Vec<bool> = match keys {
        Some(keys) => {
            for key in keys {
                if !headers.iter().any(|h| h == key) {
                    return Err(Error::Parse(format!(
                        "--keys column '{key}' is not in the design CSV header"
                    )));
                }
            }
            headers.iter().map(|h| keys.contains(h)).collect()
        }
        None => headers.iter().map(|h| sc.annotation(h).is_some()).collect(),
    };
    let key_columns: Vec<String> = headers
        .iter()
        .zip(&is_key)
        .filter(|&(_, &k)| k)
        .map(|(h, _)| h.clone())
        .collect();
    let value_columns: Vec<String> = headers
        .iter()
        .zip(&is_key)
        .filter(|&(_, &k)| !k)
        .map(|(h, _)| h.clone())
        .collect();
    if key_columns.is_empty() {
        return Err(Error::Parse(
            "no design CSV column matches an annotation in the sidecar; \
             annotate first, or name the key columns with --keys"
                .to_string(),
        ));
    }

    // Reorder each row to keys-then-values, the layout `Design` stores.
    let order: Vec<usize> = (0..headers.len())
        .filter(|&i| is_key[i])
        .chain((0..headers.len()).filter(|&i| !is_key[i]))
        .collect();
    let mut rows = Vec::new();
    for (line, record) in csv_reader.records().enumerate() {
        let record =
            record.map_err(|e| Error::Parse(format!("design CSV line {}: {e}", line + 2)))?;
        if record.len() != headers.len() {
            return Err(Error::Parse(format!(
                "design CSV line {} has {} cells, expected {}",
                line + 2,
                record.len(),
                headers.len()
            )));
        }
        rows.push(
            order
                .iter()
                .map(|&i| record.get(i).unwrap_or_default().to_string())
                .collect(),
        );
    }

    Ok(Design {
        key_columns,
        value_columns,
        rows,
    })
}

/// One sidecar column, of whichever kind it turned out to be stored as.
#[derive(Debug, Clone)]
pub enum SidecarColumn {
    Labels(AnnotationSection),
    Scores(ScoreSection),
}

impl SidecarColumn {
    pub fn name(&self) -> &str {
        match self {
            Self::Labels(a) => a.name(),
            Self::Scores(s) => s.name(),
        }
    }
}

/// The sidecar data describing one POD5, and the file(s) it came from.
struct Resolved {
    sidecar: Sidecar,
    /// What an error should name — a caller told "no column 'barcode' in X"
    /// needs X to be every place that was actually looked at.
    source: String,
}

/// Find the sidecar data describing `pod5_path`, in both of the places it can
/// live: the file's own `.p5s`, and the collection `.p5s` beside the directory
/// the file sits in.
///
/// A POD5 in a demultiplexed directory has its labels in the collection and
/// nothing but an index — or nothing at all — of its own, because a run's
/// barcodes are one result and writing them into fifty files as well as the
/// collection would be fifty copies to keep in step. Every reader below goes
/// through here so that `escpod view`, `filter --annotation`, `demux split
/// --sidecar` and the Python `Reader` need not know which shape produced the
/// answer.
///
/// The two are **merged**, per column, with the file's own sidecar winning:
/// it is bound to this exact POD5 by `file_identifier` + size, which is the
/// stronger claim. Merging rather than choosing is what lets `escpod index`'s
/// index-only sidecar sit next to an annotated collection without either one
/// hiding the other.
fn resolve_sidecar(pod5_path: &Path) -> Result<Resolved> {
    let describe = || {
        let own = sidecar_path(pod5_path);
        match pod5_path.parent().map(sidecar::collection_sidecar_path) {
            Some(c) => format!("{} or {}", own.display(), c.display()),
            None => own.display().to_string(),
        }
    };
    resolve_sidecar_opt(pod5_path)?.ok_or_else(|| {
        Error::Parse(format!(
            "no sidecar at {}; create one with `escpod annotate`",
            describe()
        ))
    })
}

/// The sidecar data describing `pod5_path`, or `None` if neither shape holds
/// any.
///
/// For a caller that is *listing* what is there — the Python `Reader`'s
/// `annotation_names()` and `score_names()`, `escpod inspect summary` — where
/// "no sidecar" is an ordinary answer rather than a failure.
pub fn read_sidecar_for(pod5_path: impl AsRef<Path>) -> Result<Option<Sidecar>> {
    Ok(resolve_sidecar_opt(pod5_path.as_ref())?.map(|r| r.sidecar))
}

fn resolve_sidecar_opt(pod5_path: &Path) -> Result<Option<Resolved>> {
    let reader = Reader::open(pod5_path)?;
    let identity = reader.sidecar_identity()?;
    let p5s_path = sidecar_path(pod5_path);
    let own = sidecar::read_sidecar_file(&p5s_path, &identity)?;

    let collection_path = pod5_path.parent().map(sidecar::collection_sidecar_path);
    let from_collection = collection_path.as_deref().and_then(|path| {
        match sidecar::read_collection_file(path) {
            Ok(Some(collection)) => collection.view_for(&identity),
            Ok(None) => None,
            // A collection that will not parse is not fatal here. It is one of
            // two candidate sources, and the POD5's own sidecar may well answer
            // the question; if neither does, the error below names both paths
            // and the operator has what they need. Failing the whole read
            // instead would let one damaged collection break `escpod view` for
            // every file in the directory.
            Err(e) => {
                tracing::debug!("{}: {e}", path.display());
                None
            }
        }
    });

    Ok(match (own, from_collection) {
        (Some(mut own), Some(view)) => {
            for annotation in view.annotations() {
                if own.annotation(annotation.name()).is_none()
                    && own.score(annotation.name()).is_none()
                {
                    own.set_annotation(annotation.clone());
                }
            }
            for score in view.scores() {
                if own.annotation(score.name()).is_none() && own.score(score.name()).is_none() {
                    own.set_score(score.clone());
                }
            }
            let source = match collection_path.as_deref() {
                Some(c) => format!("{} or {}", p5s_path.display(), c.display()),
                None => p5s_path.display().to_string(),
            };
            Some(Resolved {
                sidecar: own,
                source,
            })
        }
        (Some(own), None) => Some(Resolved {
            sidecar: own,
            source: p5s_path.display().to_string(),
        }),
        (None, Some(view)) => Some(Resolved {
            sidecar: view,
            source: collection_path.as_deref().map_or_else(
                || p5s_path.display().to_string(),
                |c| c.display().to_string(),
            ),
        }),
        (None, None) => None,
    })
}

/// Read several named columns from the sidecar in one pass, whichever kind each
/// one is.
///
/// For a consumer that just wants a column by name — `escpod view --include`
/// being the one — and should not have to know in advance whether it holds
/// labels or numbers. Also one sidecar read for all of them rather than one
/// per name, and one error listing everything available rather than a list per
/// kind that omits the column the caller actually asked for.
pub fn read_columns(
    pod5_path: impl AsRef<Path>,
    names: &[impl AsRef<str>],
) -> Result<Vec<SidecarColumn>> {
    // Asking for no columns needs no sidecar. Without this, `escpod view` with
    // only built-in fields — the overwhelmingly common case — starts demanding
    // a `.p5s` that has nothing to do with what it was asked for, which is how
    // this arrived: the loop it replaced never touched the sidecar when the
    // name list was empty, because there was nothing to iterate.
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let Resolved {
        sidecar: sc,
        source,
    } = resolve_sidecar(pod5_path.as_ref())?;

    names
        .iter()
        .map(|name| {
            let name = name.as_ref();
            if let Some(a) = sc.annotation(name) {
                return Ok(SidecarColumn::Labels(a.clone()));
            }
            if let Some(s) = sc.score(name) {
                return Ok(SidecarColumn::Scores(s.clone()));
            }
            let mut available = sc.annotation_names();
            available.extend(sc.score_names());
            available.sort_unstable();
            Err(Error::Parse(format!(
                "no column '{name}' in {source} (available: {})",
                join_or_none(&available)
            )))
        })
        .collect()
}

/// Read a score column from the `.p5s` sidecar next to a POD5.
///
/// Always by name, unlike [`read_annotation`]'s `None`: score columns arrive in
/// groups (a scored demux writes three at once), so "the only one" is not a
/// useful default and guessing between `crf_logp` and `crf_margin` would be
/// worse than asking.
pub fn read_score(pod5_path: impl AsRef<Path>, name: &str) -> Result<ScoreSection> {
    let Resolved {
        sidecar: sc,
        source,
    } = resolve_sidecar(pod5_path.as_ref())?;

    sc.score(name).cloned().ok_or_else(|| {
        // A name that is present as a label column is the likely mistake, so
        // say that rather than only listing what is available.
        if sc.annotation(name).is_some() {
            return Error::Parse(format!(
                "'{name}' in {source} is a label column, not a score column"
            ));
        }
        Error::Parse(format!(
            "no score column '{name}' in {source} (available: {})",
            join_or_none(&sc.score_names())
        ))
    })
}

/// Read an annotation from the `.p5s` sidecar next to a POD5.
///
/// With `name = None`, the sidecar must contain exactly one annotation;
/// otherwise the ambiguity (or absence) is an error that lists what is
/// available.
pub fn read_annotation(
    pod5_path: impl AsRef<Path>,
    name: Option<&str>,
) -> Result<AnnotationSection> {
    let Resolved {
        sidecar: sc,
        source,
    } = resolve_sidecar(pod5_path.as_ref())?;

    match name {
        Some(n) => sc.annotation(n).cloned().ok_or_else(|| {
            Error::Parse(format!(
                "no annotation '{n}' in {source} (available: {})",
                join_or_none(&sc.annotation_names())
            ))
        }),
        None => match sc.annotations() {
            [] => Err(Error::Parse(format!(
                "{source} has no annotations (index only)"
            ))),
            [single] => Ok(single.clone()),
            many => Err(Error::Parse(format!(
                "{source} has multiple annotations ({}); specify one by name",
                many.iter()
                    .map(AnnotationSection::name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        },
    }
}

fn join_or_none(names: &[&str]) -> String {
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

/// Outcome of [`write_collection_columns`].
#[derive(Debug, Clone)]
pub struct CollectionResult {
    /// POD5 files covered.
    pub members: usize,
    /// Reads indexed across every member.
    pub total_reads: usize,
    /// Per column, in the order given.
    pub columns: Vec<ColumnStat>,
    /// Annotation columns of both kinds in the collection after the write.
    pub columns_in_sidecar: usize,
    /// Where the collection sidecar was written.
    pub sidecar_path: PathBuf,
}

/// Write several columns into a **collection** `.p5s` — one sidecar covering
/// every POD5 in `members`, rather than one per file.
///
/// The collection counterpart of [`write_columns`], and deliberately not a
/// replacement for it: a per-file sidecar also caches the read index and the
/// signal batch geometry `Reader::open` uses, both of which are properties of
/// one POD5. `demux --annotate` writes the per-file sidecars first and then
/// this, so the members' own indexes are already on disk and assembling the
/// collection costs a projected read per file rather than a reads-table scan.
///
/// Each mapping is intersected with the reads actually present, exactly as
/// [`write_columns`] does — one global assignment map serves every member with
/// no provenance tracking, because read UUIDs are unique across files.
///
/// `overwrite` licenses replacing an existing file at `collection_path` that
/// cannot be read as a collection. As in [`write_columns`], it is a statement
/// about a file that does not belong here — a per-file sidecar, or a version
/// this build does not know — and not a licence to discard a readable
/// collection's other columns, which are merged into either way.
pub fn write_collection_columns(
    collection_path: impl AsRef<Path>,
    members: &[PathBuf],
    columns: &[ColumnWrite],
    overwrite: bool,
) -> Result<CollectionResult> {
    let collection_path = collection_path.as_ref();
    let stamp = sidecar::SidecarStamp::of(collection_path);

    // The directory the collection covers: `collection_sidecar_path` only ever
    // appends `.p5s`, so stripping it back off is exact. Member names are
    // recorded relative to it, which is what lets the whole directory move.
    let base: Option<PathBuf> = collection_path
        .to_str()
        .and_then(|s| s.strip_suffix(".p5s"))
        .map(PathBuf::from);

    let mut existing = match sidecar::read_collection_file(collection_path) {
        Ok(Some(cs)) => cs,
        Ok(None) => CollectionSidecar::default(),
        Err(e) if overwrite => {
            tracing::warn!("{e} — replacing; any annotations or scores it held are discarded");
            CollectionSidecar::default()
        }
        Err(e) => return Err(e),
    };

    let mut member_table = Vec::with_capacity(members.len());
    let mut entries: Vec<([u8; 16], u32, u32, u32)> = Vec::new();
    for (i, path) in members.iter().enumerate() {
        let reader = Reader::open(path)?;
        let identity = reader.sidecar_identity()?;
        let member_p5s = sidecar::sidecar_path(path);

        // The member's own sidecar, if it has a usable one — that is the whole
        // read index, already built. A sidecar that will not load here is not
        // fatal and not repaired: it is a cache of something the POD5 still
        // holds, so fall back to the scan and leave whatever is there alone.
        let member_entries = match sidecar::read_sidecar_entries(&member_p5s, &identity) {
            Ok(Some(e)) => e,
            Ok(None) => reader.build_read_index_from_scan()?.entries().to_vec(),
            Err(e) => {
                tracing::debug!(
                    "{}: {e} — scanning the reads table for this member instead",
                    member_p5s.display()
                );
                reader.build_read_index_from_scan()?.entries().to_vec()
            }
        };

        let name = base
            .as_deref()
            .and_then(|b| path.strip_prefix(b).ok())
            .or_else(|| path.file_name().map(Path::new))
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .into_owned();
        member_table.push(SidecarMember {
            name,
            file_id: identity.file_id,
            size: identity.size,
            reads: member_entries.len() as u64,
        });

        let member_idx = i as u32;
        entries.reserve(member_entries.len());
        entries.extend(
            member_entries
                .into_iter()
                .map(|(id, batch, row)| (id, member_idx, batch, row)),
        );
    }

    existing.set_index(member_table, entries);

    let mut stats = Vec::with_capacity(columns.len());
    for column in columns {
        let stat = match &column.values {
            ColumnValues::Labels(assignments) => {
                let annotation = AnnotationSection::from_pairs(
                    &column.name,
                    existing.entries().iter().filter_map(|&(uuid_bytes, ..)| {
                        let uuid = Uuid::from_bytes(uuid_bytes);
                        assignments
                            .get(&uuid)
                            .filter(|label| !label.is_empty())
                            .map(|label| (uuid, label.as_str()))
                    }),
                )?;
                let stat = ColumnStat {
                    name: column.name.clone(),
                    assigned: annotation.len(),
                    labels: annotation.labels().len(),
                };
                existing.set_annotation(annotation);
                stat
            }
            ColumnValues::LabelCodes { dictionary, codes } => {
                let annotation = AnnotationSection::from_pairs(
                    &column.name,
                    existing.entries().iter().filter_map(|&(uuid_bytes, ..)| {
                        let uuid = Uuid::from_bytes(uuid_bytes);
                        let label = dictionary.get(*codes.get(&uuid)? as usize)?;
                        (!label.is_empty()).then_some((uuid, label.as_str()))
                    }),
                )?;
                let stat = ColumnStat {
                    name: column.name.clone(),
                    assigned: annotation.len(),
                    labels: annotation.labels().len(),
                };
                existing.set_annotation(annotation);
                stat
            }
            ColumnValues::Scores(values) => {
                let score = ScoreSection::from_pairs(
                    &column.name,
                    existing.entries().iter().filter_map(|&(uuid_bytes, ..)| {
                        let uuid = Uuid::from_bytes(uuid_bytes);
                        values.get(&uuid).map(|&v| (uuid, v))
                    }),
                )?;
                let stat = ColumnStat {
                    name: column.name.clone(),
                    assigned: score.len(),
                    labels: 0,
                };
                existing.set_score(score);
                stat
            }
        };
        stats.push(stat);
    }

    sidecar::write_collection_file_checked(collection_path, &existing, stamp.as_ref())?;

    Ok(CollectionResult {
        members: existing.members().len(),
        total_reads: existing.len(),
        columns: stats,
        columns_in_sidecar: existing.annotations().len() + existing.scores().len(),
        sidecar_path: collection_path.to_path_buf(),
    })
}

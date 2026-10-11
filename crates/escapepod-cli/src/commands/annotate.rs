//! Record per-read annotations (e.g. demux barcode assignments) in the
//! `.p5s` sidecar next to POD5 files. The POD5 files themselves are never
//! modified; see `escapepod_pod5::sidecar` for the format.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use tracing::{info, warn};

use escapepod_signal::Uuid;
use escapepod_signal::operations::{
    ABSENT_CODE, ColumnStat, ColumnValues, ColumnWrite, DesignOptions, LabelTable,
    LabelTableBuilder, parse_barcode_mapping, remove_annotation, remove_design,
    write_collection_columns, write_columns, write_design,
};
use escapepod_signal::pod5::sidecar::{MAX_ANNOTATION_LABELS, UNCLASSIFIED_LABEL};

use crate::style;
use crate::util::{collect_pod5_inputs, open_maybe_gzip};

/// Parsed `escpod sidecar annotate` arguments (one action per invocation).
///
/// The one definition behind both spellings: `escpod sidecar annotate` and the
/// hidden `escpod annotate` nest this same struct, so the two cannot drift
/// apart (see [`crate::commands::sidecar`]).
#[derive(clap::Args)]
#[command(after_help = "\
Writes a read → label mapping into the .p5s sidecar next to each POD5 file.
The POD5 itself is never modified. The sidecar is a plain Arrow IPC table
(read_id, batch_idx, row_idx, plus one dictionary column per annotation) —
readable directly with pyarrow/polars — and is bound to its POD5 by file
identifier + size, so a stale or misplaced sidecar fails loudly. Existing
sidecar contents (the read index, other annotations) are preserved.

Examples:
  escpod sidecar annotate -a demux.csv input.pod5        Annotate with demux assignments
  escpod sidecar annotate -a demux.csv data_dir/         Annotate a directory recursively
  escpod sidecar annotate -a samples.csv --name sample input.pod5
  escpod sidecar annotate data_dir/ --from-table calls.csv.gz --column barcode=predicted_barcode
                                                 Label reads from another tool's
                                                 read-to-label table (CSV/TSV, gzip ok)
  escpod sidecar annotate --design samplesheet.csv input.pod5
                                                 Map barcode (combinations) to
                                                 experimental conditions
")]
#[command(group(
    clap::ArgGroup::new("action")
        .required(true)
        .args([
            "assignments",
            "design",
            "from_table",
            "list",
            "remove",
            "remove_design"
        ])
))]
pub struct AnnotateArgs {
    /// Input POD5 file(s) or directory
    #[arg(required = true, value_name = "FILES")]
    pub inputs: Vec<PathBuf>,

    /// CSV mapping reads to labels (read_id + barcode/predicted_barcode
    /// columns — the output of `demux classify` or `demux basecall --barcodes`)
    #[arg(short = 'a', long, value_name = "CSV")]
    pub assignments: Option<PathBuf>,

    /// Table mapping reads to labels, from another tool (CSV or TSV, one
    /// header row; gzip is detected). Needs --column. Reads the table does
    /// not label get --unassigned-label. Repeat for the shards of one run
    #[arg(
        long,
        value_name = "TABLE",
        num_args = 1..,
        action = clap::ArgAction::Append,
        requires = "columns",
        conflicts_with_all = ["assignments", "name"]
    )]
    pub from_table: Vec<PathBuf>,

    /// With --from-table: the table's read-id column (default: read_id; a
    /// leading '#' in the header, as WarpDemuX writes, is ignored)
    #[arg(
        long,
        value_name = "COL",
        requires = "from_table",
        conflicts_with_all = TABLE_ONLY_CONFLICTS
    )]
    pub id_column: Option<String>,

    /// With --from-table: write sidecar column NAME from table column COL,
    /// verbatim (e.g. barcode=predicted_barcode). Repeat for several
    #[arg(
        long = "column",
        value_name = "NAME=COL",
        requires = "from_table",
        conflicts_with_all = TABLE_ONLY_CONFLICTS,
        action = clap::ArgAction::Append,
        value_parser = parse_column_spec
    )]
    pub columns: Vec<(String, String)>,

    /// With --from-table: label for reads the table does not label (absent
    /// from it, or an empty cell) [default: unclassified]. Pass '' to leave
    /// them unlabelled
    #[arg(
        long,
        value_name = "LABEL",
        requires = "from_table",
        conflicts_with_all = TABLE_ONLY_CONFLICTS
    )]
    pub unassigned_label: Option<String>,

    /// Experimental-design CSV: one column per key annotation (e.g.
    /// `barcode`, or `ldx,edx` for combinations) plus one column per
    /// experimental variable (e.g. `condition`). Stored in the sidecar
    /// and materialized as derived per-read columns
    #[arg(long, value_name = "CSV")]
    pub design: Option<PathBuf>,

    /// With --design: which CSV columns are the keys (default: every
    /// column whose name matches an existing annotation)
    #[arg(long, value_name = "COLS", requires = "design", value_delimiter = ',')]
    pub keys: Option<Vec<String>>,

    /// Annotation name (the sidecar column to write, with -a; --from-table
    /// names its columns with --column instead)
    #[arg(long, default_value = escapepod_signal::pod5::sidecar::DEFAULT_ANNOTATION_NAME)]
    pub name: String,

    /// List each file's sidecar contents (index, annotations, design)
    #[arg(long)]
    pub list: bool,

    /// Remove an annotation by name (design columns are refused —
    /// remove or update the design instead)
    #[arg(long, value_name = "NAME")]
    pub remove: Option<String>,

    /// Remove the experimental design and its derived columns
    #[arg(long)]
    pub remove_design: bool,

    /// Replace a sidecar bound to a different POD5 instead of erroring
    #[arg(long)]
    pub force: bool,

    /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,
}

/// The other actions, for the options that only mean something with
/// `--from-table`. `requires` alone is not enough: clap does not report a
/// required argument as missing when it conflicts with one that was given, so
/// `-a x.csv --column barcode=y` would otherwise parse and quietly ignore
/// `--column`.
const TABLE_ONLY_CONFLICTS: [&str; 5] =
    ["assignments", "design", "list", "remove", "remove_design"];

/// `clap` value parser for `--column NAME=COL`.
pub fn parse_column_spec(s: &str) -> Result<(String, String), String> {
    let (name, column) = s.split_once('=').ok_or_else(|| {
        format!("expected NAME=COLUMN (e.g. barcode=predicted_barcode), got '{s}'")
    })?;
    let (name, column) = (name.trim(), column.trim());
    if name.is_empty() || column.is_empty() {
        return Err(format!(
            "both sides of NAME=COLUMN must be non-empty, got '{s}'"
        ));
    }
    Ok((name.to_string(), column.to_string()))
}

/// Annotate one or more POD5 files: write a read → barcode mapping (`-a`,
/// the CSV format `demux split` consumes), label reads from an external table
/// (`--from-table`), or record an experimental-design table (`--design`), or
/// inspect/prune the sidecar (`--list`, `--remove`, `--remove-design`).
pub fn run(cmd: AnnotateArgs) -> anyhow::Result<()> {
    // Captured before expansion flattens directories away: a directory may
    // also have a collection sidecar beside it, and `--list` should say so
    // rather than printing only the members' own.
    let dirs: Vec<PathBuf> = cmd.inputs.iter().filter(|p| p.is_dir()).cloned().collect();
    let files = collect_pod5_inputs(&cmd.inputs)?;
    if cmd.list {
        return run_list(&files, &dirs);
    }
    if let Some(name) = cmd.remove {
        return run_remove(&files, &name);
    }
    if cmd.remove_design {
        return run_remove_design(&files);
    }
    if !cmd.from_table.is_empty() {
        return run_from_table(
            &files,
            &dirs,
            &FromTable {
                tables: cmd.from_table,
                id_column: cmd.id_column.unwrap_or_else(|| "read_id".to_string()),
                columns: cmd.columns,
                fill: fill_label(cmd.unassigned_label),
            },
            cmd.force,
        );
    }
    match (cmd.assignments, cmd.design) {
        (Some(assignments), None) => {
            run_assignments(&files, &dirs, assignments, cmd.name, cmd.force)
        }
        (None, Some(design)) => run_design(&files, design, cmd.keys, cmd.force),
        _ => unreachable!("clap enforces exactly one action"),
    }
}

/// Print each file's sidecar contents (stdout — this is the command's data),
/// preceded by the collection sidecar of any directory that has one.
fn run_list(files: &[PathBuf], dirs: &[PathBuf]) -> anyhow::Result<()> {
    use escapepod_signal::pod5::sidecar::{
        Pod5Identity, collection_sidecar_path, read_collection_file, read_sidecar_file,
        sidecar_path,
    };

    // Which POD5s a listed collection already accounts for, by identity. A
    // member with no sidecar of its own is *covered*, not unannotated, and
    // reporting "no sidecar" for each of fifty of them — directly under the
    // collection that holds their labels — says the opposite of what is true.
    let mut covered: Vec<(Pod5Identity, PathBuf)> = Vec::new();
    for dir in dirs {
        let p5s_path = collection_sidecar_path(dir);
        match read_collection_file(&p5s_path) {
            Ok(Some(collection)) => {
                println!(
                    "{}: {} reads indexed across {} files",
                    p5s_path.display(),
                    collection.len(),
                    collection.members().len()
                );
                for annotation in collection.annotations() {
                    println!(
                        "  {}: {} labels, {} reads",
                        annotation.name(),
                        annotation.labels().len(),
                        annotation.len()
                    );
                }
                for score in collection.scores() {
                    println!("  {}: {} reads scored", score.name(), score.len());
                }
                covered.extend(
                    collection
                        .members()
                        .iter()
                        .map(|m| (m.identity(), p5s_path.clone())),
                );
            }
            // Nothing there is the ordinary case for a directory that has
            // never been demultiplexed — not worth a line of output.
            Ok(None) => {}
            Err(e) => println!("{}: {e}", p5s_path.display()),
        }
    }

    for pod5_path in files {
        let p5s_path = sidecar_path(pod5_path);
        let reader = escapepod_signal::Reader::open(pod5_path)?;
        let identity = reader.sidecar_identity()?;
        let sidecar = match read_sidecar_file(&p5s_path, &identity) {
            Ok(Some(s)) => s,
            Ok(None) => {
                match covered.iter().find(|(id, _)| *id == identity) {
                    Some((_, collection)) => println!(
                        "{}: covered by {}",
                        pod5_path.display(),
                        collection.display()
                    ),
                    None => println!("{}: no sidecar", pod5_path.display()),
                }
                continue;
            }
            Err(e) => {
                println!("{}: {e}", p5s_path.display());
                continue;
            }
        };
        println!("{}: {} reads indexed", p5s_path.display(), sidecar.len());
        let derived: Vec<&String> = sidecar
            .design()
            .map(|d| d.value_columns.iter().collect())
            .unwrap_or_default();
        for annotation in sidecar.annotations() {
            println!(
                "  {}: {} labels, {} reads{}",
                annotation.name(),
                annotation.labels().len(),
                annotation.len(),
                if derived.iter().any(|d| *d == annotation.name()) {
                    " (derived)"
                } else {
                    ""
                }
            );
        }
        if let Some(design) = sidecar.design() {
            println!(
                "  design: [{}] -> [{}], {} rows",
                design.key_columns.join(","),
                design.value_columns.join(","),
                design.rows.len()
            );
        }
    }
    Ok(())
}

fn run_remove(files: &[PathBuf], name: &str) -> anyhow::Result<()> {
    for pod5_path in files {
        let removed = remove_annotation(pod5_path, name)
            .map_err(|e| anyhow::anyhow!("{}: {e}", pod5_path.display()))?;
        if removed {
            info!(
                "{} annotation '{}' from {}",
                style::action("removed"),
                name,
                style::path(pod5_path.display())
            );
        } else {
            info!(
                "no annotation '{}' in {} — nothing removed",
                name,
                style::path(pod5_path.display())
            );
        }
    }
    Ok(())
}

fn run_remove_design(files: &[PathBuf]) -> anyhow::Result<()> {
    for pod5_path in files {
        let removed = remove_design(pod5_path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", pod5_path.display()))?;
        if removed {
            info!(
                "{} design (and derived columns) from {}",
                style::action("removed"),
                style::path(pod5_path.display())
            );
        } else {
            info!(
                "no design in {} — nothing removed",
                style::path(pod5_path.display())
            );
        }
    }
    Ok(())
}

fn run_assignments(
    files: &[PathBuf],
    dirs: &[PathBuf],
    assignments: PathBuf,
    name: String,
    force: bool,
) -> anyhow::Result<()> {
    let mapping = parse_barcode_mapping(&assignments)?;
    let n_assignments = mapping.len();
    info!(
        "loaded {} assignments from {}",
        style::count(n_assignments),
        style::path(assignments.display()),
    );

    // Assignments are intersected with each file's own reads, so a CSV from
    // the wrong run drops every row. Writing that would replace the column of
    // the same name — a good one, on a re-annotation — with an empty one, and
    // only *then* say so. Look first: across *all* inputs "none matched" is
    // conclusive, where per file it is only suspicious (one demux CSV
    // legitimately spans many POD5s and covers none of some of them).
    let mut matched = 0;
    if !mapping.is_empty() {
        let scans = scan_inputs(
            files,
            |id| mapping.get(id).is_some_and(|label| !label.is_empty()),
            false,
        )?;
        matched = scans.iter().map(|s| s.matched).sum();
        if matched == 0 {
            anyhow::bail!(
                "none of the {n_assignments} assignments in {} match any read in the {} input \
                 file(s) — this is almost always the wrong CSV for these POD5s. Nothing was \
                 written.",
                assignments.display(),
                files.len(),
            );
        }
        warn_unmatched_files(files, &scans, n_assignments, "assignments");
    }

    let columns = vec![ColumnWrite {
        name,
        values: ColumnValues::Labels(mapping),
    }];
    let annotated = write_sidecars(dirs, files, &columns, force)?;

    info!(
        "{} file(s) annotated — {} reads matched {} assignments",
        style::count(annotated),
        style::count(matched),
        style::count(n_assignments),
    );
    Ok(())
}

/// What `--from-table` was asked to do.
struct FromTable {
    tables: Vec<PathBuf>,
    id_column: String,
    /// `(sidecar column, table column)`.
    columns: Vec<(String, String)>,
    /// The label for a read the table does not label; `None` leaves it absent.
    fill: Option<String>,
}

/// The fill label `--unassigned-label` asks for: [`UNCLASSIFIED_LABEL`] unless
/// given, and nothing at all when given empty — the sidecar's own way of saying
/// a read has no label is absence, so `''` is the way to ask for exactly that.
fn fill_label(arg: Option<String>) -> Option<String> {
    match arg {
        None => Some(UNCLASSIFIED_LABEL.to_string()),
        Some(label) => {
            let label = label.trim();
            (!label.is_empty()).then(|| label.to_string())
        }
    }
}

/// Label reads from external read → label table(s), through the same writers
/// `demux --annotate` uses — so the collection sidecar for a directory, the
/// footer UUID/size binding and in-place replacement of a named column are
/// theirs, not reimplemented here.
fn run_from_table(
    files: &[PathBuf],
    dirs: &[PathBuf],
    cmd: &FromTable,
    force: bool,
) -> anyhow::Result<()> {
    // Names first (inside `new`): a reserved or repeated sidecar column name
    // should cost nothing, not a parse of a multi-gigabyte table.
    let mut builder = LabelTableBuilder::new(&cmd.id_column, &cmd.columns)?;
    let t0 = Instant::now();
    for path in &cmd.tables {
        let reader = open_maybe_gzip(path)?;
        builder.add_table(reader, &path.display().to_string())?;
    }
    let mut table = builder.finish();
    info!(
        "read {} rows from {} table(s), naming {} reads{} in {:.1}s",
        style::count(table.rows),
        style::count(cmd.tables.len()),
        style::count(table.len()),
        match table.identical_duplicates {
            0 => String::new(),
            n => format!(" ({} repeated rows with the same labels)", style::count(n)),
        },
        t0.elapsed().as_secs_f64(),
    );

    // Pre-flight, before a byte is written. With a fill label every read
    // ends up "assigned", so the writers' own "nothing matched" signal can no
    // longer fire — and an in-place replace would overwrite a good column with
    // all-`unclassified` on the strength of the wrong table.
    let seen: Vec<AtomicU8> = (0..table.len()).map(|_| AtomicU8::new(0)).collect();
    let mut scans = scan_inputs(
        files,
        |id| match table.index.get(id) {
            Some(&slot) => {
                seen[slot as usize].store(1, Ordering::Relaxed);
                true
            }
            None => false,
        },
        cmd.fill.is_some(),
    )?;
    let total: usize = scans.iter().map(|s| s.total).sum();
    let matched: usize = scans.iter().map(|s| s.matched).sum();
    if matched == 0 {
        anyhow::bail!(
            "none of the {} reads named in {} match any of the {} reads in the {} input \
             file(s) — this is almost always the table of a different run, or the wrong \
             --id-column (currently '{}'). Nothing was written.",
            table.len(),
            cmd.tables
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            total,
            files.len(),
            cmd.id_column,
        );
    }
    warn_unmatched_files(files, &scans, table.len(), "table rows");

    let unmatched_rows = seen
        .iter()
        .filter(|s| s.load(Ordering::Relaxed) == 0)
        .count();
    // Per column, matched reads whose cell was empty — counted before the fill
    // turns those into a label and they can no longer be told apart.
    let empty_cells: Vec<usize> = table
        .columns
        .iter()
        .map(|c| {
            c.codes
                .iter()
                .zip(&seen)
                .filter(|&(&code, s)| code == ABSENT_CODE && s.load(Ordering::Relaxed) == 1)
                .count()
        })
        .collect();
    let labels: Vec<usize> = table.columns.iter().map(|c| c.dictionary.len()).collect();
    info!(
        "{} of {} reads in {} POD5 file(s) are in the table; {} table rows matched no read",
        style::count(matched),
        style::count(total),
        style::count(files.len()),
        style::count(unmatched_rows),
    );

    let absent_from_table = match &cmd.fill {
        Some(label) => {
            let missing = scans
                .iter_mut()
                .flat_map(|s| std::mem::take(&mut s.missing));
            apply_fill(&mut table, missing, label)?
        }
        None => 0,
    };
    for (i, column) in table.columns.iter().enumerate() {
        match &cmd.fill {
            Some(label) => info!(
                "'{}': {} labels in the table; {} reads labelled '{label}' ({} absent from the \
                 table, {} with an empty cell)",
                column.name,
                style::count(labels[i]),
                style::count(absent_from_table + empty_cells[i]),
                style::count(absent_from_table),
                style::count(empty_cells[i]),
            ),
            None => info!(
                "'{}': {} labels in the table; {} matched reads have an empty cell and stay \
                 unlabelled, as do the reads the table does not name",
                column.name,
                style::count(labels[i]),
                style::count(empty_cells[i]),
            ),
        }
    }

    let columns = table.into_column_writes();
    let annotated = write_sidecars(dirs, files, &columns, force)?;
    info!("{} file(s) annotated", style::count(annotated));
    Ok(())
}

/// What the pre-flight learned about one input file.
struct FileScan {
    /// Reads in the file.
    total: usize,
    /// Reads the source (table or CSV) has something for.
    matched: usize,
    /// The reads it does not — only collected when a fill needs them.
    missing: Vec<Uuid>,
}

/// Count, per input file, the reads `matches` accepts, in parallel.
///
/// Reads the POD5's read ids only (`Reader::read_ids` projects one column), so
/// this is the cheap half of a look before writing; the writers do their own
/// pass afterwards. Results come back in `files` order.
fn scan_inputs(
    files: &[PathBuf],
    matches: impl Fn(&Uuid) -> bool + Sync,
    keep_missing: bool,
) -> anyhow::Result<Vec<FileScan>> {
    files
        .par_iter()
        .map(|path| {
            let ids = escapepod_signal::Reader::open(path)
                .and_then(|reader| reader.read_ids())
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
            let mut scan = FileScan {
                total: ids.len(),
                matched: 0,
                missing: Vec::new(),
            };
            for id in ids {
                if matches(&id) {
                    scan.matched += 1;
                } else if keep_missing {
                    scan.missing.push(id);
                }
            }
            Ok(scan)
        })
        .collect()
}

/// A file none of the source's reads are in is only suspicious on its own — one
/// table spans many POD5s and legitimately covers none of some — so this warns
/// and the caller decides about the whole run.
fn warn_unmatched_files(files: &[PathBuf], scans: &[FileScan], rows: usize, what: &str) {
    for (path, scan) in files.iter().zip(scans) {
        if scan.matched == 0 {
            warn!(
                "{} — none of the {} {what} match any read in this file",
                style::path(path.display()),
                style::count(rows),
            );
        }
    }
}

/// Give every read the table leaves unlabelled the label `label`, in every
/// column: reads in `missing` (in the inputs, not in the table) get a slot
/// appended, and an empty cell becomes the label. Returns how many slots it
/// appended.
///
/// Fill lives here and not in the writers because only the CLI knows what the
/// inputs hold and what the user wants an unlabelled read called; the writers
/// stay "a read with no code has no label". The one thing it must keep true is
/// `codes.len() == index.len()` for every column — the writers read a slot with
/// `get`, so a read whose slot has no code is dropped without a word.
fn apply_fill(
    table: &mut LabelTable,
    missing: impl IntoIterator<Item = Uuid>,
    label: &str,
) -> anyhow::Result<usize> {
    let mut fill_codes = Vec::with_capacity(table.columns.len());
    for column in &mut table.columns {
        let code = match column.dictionary.iter().position(|l| l == label) {
            Some(i) => i as u32,
            None => {
                // The sidecar refuses the 65 536th label at write time; say so
                // before anything is written.
                if column.dictionary.len() >= MAX_ANNOTATION_LABELS {
                    anyhow::bail!(
                        "column '{}' already has {MAX_ANNOTATION_LABELS} distinct labels, so \
                         there is no room for the fill label '{label}'",
                        column.name
                    );
                }
                column.dictionary.push(label.to_string());
                (column.dictionary.len() - 1) as u32
            }
        };
        for cell in &mut column.codes {
            if *cell == ABSENT_CODE {
                *cell = code;
            }
        }
        fill_codes.push(code);
    }

    let before = table.index.len();
    for id in missing {
        if table.index.contains_key(&id) {
            continue;
        }
        let slot = u32::try_from(table.index.len())
            .map_err(|_| anyhow::anyhow!("more reads than a sidecar slot can number"))?;
        table.index.insert(id, slot);
        for (column, &code) in table.columns.iter_mut().zip(&fill_codes) {
            column.codes.push(code);
        }
    }
    Ok(table.index.len() - before)
}

/// Write `columns` into the sidecars for `files`: one collection sidecar per
/// directory argument over whichever inputs live beneath it, and a sidecar of
/// its own for every input that was named as a file. Returns how many input
/// files were annotated.
///
/// A directory argument gets one collection sidecar covering every POD5
/// beneath it, and those files get no sidecar of their own: a run's labels
/// are one result, and writing them into fifty files as well as the
/// collection would be fifty copies to keep in step. Readers look in both
/// places (see `operations::read_annotation`), so nothing downstream has to
/// know which shape holds the answer.
fn write_sidecars(
    dirs: &[PathBuf],
    files: &[PathBuf],
    columns: &[ColumnWrite],
    force: bool,
) -> anyhow::Result<usize> {
    let covered = write_collections(dirs, files, columns)?;

    let annotated = AtomicUsize::new(0);
    let errors: Vec<anyhow::Error> = files
        .par_iter()
        .filter(|pod5_path| !covered.contains(pod5_path.as_path()))
        .filter_map(|pod5_path| {
            let t0 = Instant::now();
            // One read-modify-write for every column, and no per-file copy of
            // the (potentially tens-of-millions-entry) values.
            let result = match write_columns(pod5_path, columns, force) {
                Ok(r) => r,
                Err(e) => {
                    return Some(anyhow::Error::from(e).context(pod5_path.display().to_string()));
                }
            };
            info!(
                "{} {} — {} in {:.1}s",
                style::action("wrote"),
                style::path(result.sidecar_path.display()),
                describe_columns(&result.columns, result.total_reads),
                t0.elapsed().as_secs_f64(),
            );
            annotated.fetch_add(1, Ordering::Relaxed);
            None
        })
        .collect();

    if let Some(first_err) = errors.into_iter().next() {
        return Err(first_err);
    }
    Ok(annotated.load(Ordering::Relaxed) + covered.len())
}

/// `barcode: 90 of 100 reads across 5 labels` for each column written.
fn describe_columns(stats: &[ColumnStat], total_reads: usize) -> String {
    stats
        .iter()
        .map(|c| {
            format!(
                "{}: {} of {} reads across {} labels",
                c.name,
                style::count(c.assigned),
                style::count(total_reads),
                style::count(c.labels),
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Write one collection sidecar per directory argument, over whichever inputs
/// live beneath it.
///
/// Returns the input files it covered — those get no per-file sidecar.
fn write_collections(
    dirs: &[PathBuf],
    files: &[PathBuf],
    columns: &[ColumnWrite],
) -> anyhow::Result<HashSet<PathBuf>> {
    use escapepod_signal::pod5::sidecar::collection_sidecar_path;

    let mut covered = HashSet::new();
    for dir in dirs {
        let members: Vec<PathBuf> = files
            .iter()
            .filter(|p| p.starts_with(dir))
            .cloned()
            .collect();
        if members.is_empty() {
            continue;
        }
        let path = collection_sidecar_path(dir);
        let t0 = Instant::now();
        let result = write_collection_columns(&path, &members, columns, false)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        covered.extend(members);
        info!(
            "{} {} — {}, {} files in one sidecar in {:.1}s",
            style::action("wrote"),
            style::path(result.sidecar_path.display()),
            describe_columns(&result.columns, result.total_reads),
            style::count(result.members),
            t0.elapsed().as_secs_f64(),
        );
    }
    Ok(covered)
}

fn run_design(
    files: &[PathBuf],
    design: PathBuf,
    keys: Option<Vec<String>>,
    force: bool,
) -> anyhow::Result<()> {
    let options = DesignOptions {
        keys,
        overwrite: force,
    };
    let annotated = AtomicUsize::new(0);

    let errors: Vec<anyhow::Error> = files
        .par_iter()
        .filter_map(|pod5_path| {
            let t0 = Instant::now();
            let result = match write_design(pod5_path, &design, &options) {
                Ok(r) => r,
                Err(e) => {
                    return Some(anyhow::Error::from(e).context(pod5_path.display().to_string()));
                }
            };
            let derived = result
                .derived
                .iter()
                .map(|(name, count)| format!("{name}={count}"))
                .collect::<Vec<_>>()
                .join(", ");
            info!(
                "{} {} — design [{}] → [{}], {} rows; derived reads: {} in {:.1}s",
                style::action("wrote"),
                style::path(result.sidecar_path.display()),
                result.key_columns.join(","),
                result.value_columns.join(","),
                style::count(result.design_rows),
                derived,
                t0.elapsed().as_secs_f64(),
            );
            annotated.fetch_add(1, Ordering::Relaxed);
            None
        })
        .collect();

    if let Some(first_err) = errors.into_iter().next() {
        return Err(first_err);
    }

    info!(
        "{} file(s) annotated with design",
        style::count(annotated.load(Ordering::Relaxed))
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uuid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn table(text: &str, columns: &[(&str, &str)]) -> LabelTable {
        let columns: Vec<(String, String)> = columns
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let mut builder = LabelTableBuilder::new("read_id", &columns).unwrap();
        builder.add_table(text.as_bytes(), "t").unwrap();
        builder.finish()
    }

    fn label_of(t: &LabelTable, c: usize, id: Uuid) -> Option<&str> {
        let slot = *t.index.get(&id)? as usize;
        let code = t.columns[c].codes[slot];
        (code != ABSENT_CODE).then(|| t.columns[c].dictionary[code as usize].as_str())
    }

    #[test]
    fn fill_label_defaults_to_unclassified_and_empty_disables_it() {
        assert_eq!(fill_label(None).as_deref(), Some("unclassified"));
        assert_eq!(fill_label(Some(String::new())), None);
        assert_eq!(fill_label(Some("  ".into())), None);
        assert_eq!(fill_label(Some(" none ".into())).as_deref(), Some("none"));
    }

    #[test]
    fn column_spec_needs_both_sides() {
        assert_eq!(
            parse_column_spec("barcode=predicted_barcode"),
            Ok(("barcode".to_string(), "predicted_barcode".to_string()))
        );
        for bad in ["barcode", "=x", "x=", " = "] {
            assert!(parse_column_spec(bad).is_err(), "{bad:?} accepted");
        }
    }

    /// The writers read a slot with `get`, so a column shorter than the index
    /// drops reads silently — the one invariant `apply_fill` must keep.
    #[test]
    fn fill_keeps_every_column_as_long_as_the_index() {
        let a = uuid(1);
        let b = uuid(2);
        let text = format!("read_id,ldx,fdx\n{a},1,x\n{b},,y\n");
        let mut t = table(&text, &[("ldx", "ldx"), ("fdx", "fdx")]);
        let missing = [uuid(10), uuid(11), uuid(10)];
        let appended = apply_fill(&mut t, missing, "unclassified").unwrap();

        assert_eq!(appended, 2, "a repeated missing read gets one slot");
        assert_eq!(t.len(), 4);
        for column in &t.columns {
            assert_eq!(column.codes.len(), t.index.len(), "{}", column.name);
            assert!(
                !column.codes.contains(&ABSENT_CODE),
                "{} still has an absent cell",
                column.name
            );
        }
        assert_eq!(label_of(&t, 0, a), Some("1"));
        assert_eq!(label_of(&t, 0, b), Some("unclassified"), "empty cell");
        assert_eq!(label_of(&t, 0, uuid(10)), Some("unclassified"), "absent");
        assert_eq!(label_of(&t, 1, b), Some("y"), "other column untouched");
        assert_eq!(label_of(&t, 1, uuid(11)), Some("unclassified"));
    }

    #[test]
    fn fill_reuses_a_label_the_table_already_has() {
        let text = format!("read_id,bc\n{},unclassified\n{},4\n", uuid(1), uuid(2));
        let mut t = table(&text, &[("barcode", "bc")]);
        apply_fill(&mut t, [uuid(3)], "unclassified").unwrap();
        assert_eq!(t.columns[0].dictionary.len(), 2, "no duplicate label");
        assert_eq!(label_of(&t, 0, uuid(3)), Some("unclassified"));
    }

    #[test]
    fn fill_refuses_when_the_label_would_not_fit() {
        let mut text = String::from("read_id,bc\n");
        for i in 0..MAX_ANNOTATION_LABELS {
            text.push_str(&format!("{},{i}\n", uuid(i as u128 + 1)));
        }
        let mut t = table(&text, &[("barcode", "bc")]);
        let err = apply_fill(&mut t, [uuid(u128::MAX)], "unclassified")
            .unwrap_err()
            .to_string();
        assert!(err.contains("no room for the fill label"), "{err}");
    }
}

//! Record per-read annotations (e.g. demux barcode assignments) in the
//! `.p5s` sidecar next to POD5 files. The POD5 files themselves are never
//! modified; see `escapepod_pod5::sidecar` for the format.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use tracing::{info, warn};

use escapepod_signal::operations::{
    AnnotateOptions, DesignOptions, parse_barcode_mapping, remove_annotation, remove_design,
    write_annotation, write_design,
};

use crate::style;
use crate::util::collect_pod5_inputs;

/// Parsed `escpod annotate` arguments (one action per invocation).
pub struct AnnotateCmd {
    pub inputs: Vec<PathBuf>,
    pub assignments: Option<PathBuf>,
    pub design: Option<PathBuf>,
    pub keys: Option<Vec<String>>,
    pub name: String,
    pub list: bool,
    pub remove: Option<String>,
    pub remove_design: bool,
    pub force: bool,
}

/// Annotate one or more POD5 files: write a read → barcode mapping (`-a`,
/// the CSV format `demux split` consumes) or an experimental-design table
/// (`--design`), or inspect/prune the sidecar (`--list`, `--remove`,
/// `--remove-design`).
pub fn run(cmd: AnnotateCmd) -> anyhow::Result<()> {
    let files = collect_pod5_inputs(&cmd.inputs)?;
    if cmd.list {
        return run_list(&files);
    }
    if let Some(name) = cmd.remove {
        return run_remove(&files, &name);
    }
    if cmd.remove_design {
        return run_remove_design(&files);
    }
    match (cmd.assignments, cmd.design) {
        (Some(assignments), None) => run_assignments(&files, assignments, cmd.name, cmd.force),
        (None, Some(design)) => run_design(&files, design, cmd.keys, cmd.force),
        _ => unreachable!("clap enforces exactly one action"),
    }
}

/// Print each file's sidecar contents (stdout — this is the command's data).
fn run_list(files: &[PathBuf]) -> anyhow::Result<()> {
    use escapepod_signal::pod5::sidecar::{read_sidecar_file, sidecar_path};

    for pod5_path in files {
        let p5s_path = sidecar_path(pod5_path);
        let reader = escapepod_signal::Reader::open(pod5_path)?;
        let sidecar = match read_sidecar_file(&p5s_path, &reader.sidecar_identity()?) {
            Ok(Some(s)) => s,
            Ok(None) => {
                println!("{}: no sidecar", pod5_path.display());
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
    assignments: PathBuf,
    name: String,
    force: bool,
) -> anyhow::Result<()> {
    let mapping = parse_barcode_mapping(&assignments)?;
    info!(
        "loaded {} assignments from {}",
        style::count(mapping.len()),
        style::path(assignments.display()),
    );

    let options = AnnotateOptions {
        name,
        overwrite: force,
    };
    let annotated = AtomicUsize::new(0);
    // Assignments are intersected with each file's own reads, so a CSV from
    // the wrong run drops every row and still writes a valid (empty) column.
    // Track the overlap so that failure is reported rather than inferred from
    // an "0 of N assigned" line nobody reads — and which `-q` hides entirely.
    let assigned_total = AtomicUsize::new(0);

    let errors: Vec<anyhow::Error> = files
        .par_iter()
        .filter_map(|pod5_path| {
            let t0 = Instant::now();
            let result = match write_annotation(pod5_path, &mapping, &options) {
                Ok(r) => r,
                Err(e) => {
                    return Some(anyhow::Error::from(e).context(pod5_path.display().to_string()));
                }
            };
            assigned_total.fetch_add(result.assigned_reads, Ordering::Relaxed);
            // Per file this is only suspicious: one demux CSV legitimately
            // spans many POD5s and covers none of some of them. Across *all*
            // inputs it is conclusive, which is why the error is raised below.
            if result.assigned_reads == 0 {
                warn!(
                    "{} — none of the {} assignments match any read in this file",
                    style::path(pod5_path.display()),
                    style::count(mapping.len()),
                );
            }
            info!(
                "{} {} — {} of {} reads assigned across {} labels in {:.1}s",
                style::action("wrote"),
                style::path(result.sidecar_path.display()),
                style::count(result.assigned_reads),
                style::count(result.total_reads),
                style::count(result.labels),
                t0.elapsed().as_secs_f64(),
            );
            annotated.fetch_add(1, Ordering::Relaxed);
            None
        })
        .collect();

    if let Some(first_err) = errors.into_iter().next() {
        return Err(first_err);
    }

    let assigned_total = assigned_total.load(Ordering::Relaxed);
    if assigned_total == 0 && !mapping.is_empty() {
        anyhow::bail!(
            "none of the {} assignments in {} match any read in the {} input file(s) — \
             this is almost always the wrong CSV for these POD5s. The '{}' column was \
             written but is empty; drop it with `escpod annotate --remove {}`.",
            mapping.len(),
            assignments.display(),
            files.len(),
            options.name,
            options.name,
        );
    }

    info!(
        "{} file(s) annotated — {} of {} assignments matched",
        style::count(annotated.load(Ordering::Relaxed)),
        style::count(assigned_total),
        style::count(mapping.len()),
    );
    Ok(())
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

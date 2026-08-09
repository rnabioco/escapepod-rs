//! Record per-read annotations (e.g. demux barcode assignments) in the
//! `.p5s` sidecar next to POD5 files. The POD5 files themselves are never
//! modified; see `escapepod_pod5::sidecar` for the format.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use tracing::info;

use escapepod_signal::operations::{
    AnnotateOptions, DesignOptions, parse_barcode_mapping, write_annotation, write_design,
};

use crate::style;
use crate::util::collect_pod5_inputs;

/// Annotate one or more POD5 files: either a read → barcode mapping (`-a`,
/// the CSV format `demux split` consumes) or an experimental-design table
/// (`--design`, mapping barcode combinations to conditions).
pub fn run(
    inputs: Vec<PathBuf>,
    assignments: Option<PathBuf>,
    design: Option<PathBuf>,
    keys: Option<Vec<String>>,
    name: String,
    force: bool,
) -> anyhow::Result<()> {
    let files = collect_pod5_inputs(&inputs)?;
    match (assignments, design) {
        (Some(assignments), None) => run_assignments(&files, assignments, name, force),
        (None, Some(design)) => run_design(&files, design, keys, force),
        _ => unreachable!("clap enforces exactly one of -a/--design"),
    }
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

    info!(
        "{} file(s) annotated",
        style::count(annotated.load(Ordering::Relaxed))
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

//! Build the read-index columns of `.p5s` sidecars for fast read lookup.
//!
//! Reached as `escpod sidecar index`; the top-level `escpod index` is a hidden,
//! deprecated forwarder that parses the same [`IndexArgs`] (see
//! [`crate::commands::sidecar`]).

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use tracing::{debug, info, warn};

use crate::style;
use crate::util::collect_pod5_inputs;
use escapepod_signal::pod5::sidecar::sidecar_path;

/// Parsed `escpod sidecar index` arguments.
///
/// The one definition behind both spellings: `escpod sidecar index` and the
/// hidden `escpod index` nest this same struct, so the two cannot drift apart.
#[derive(clap::Args)]
#[command(after_help = "\
Both caches are rebuildable from the POD5, which is never modified. The read
index gives O(log n) lookup by read ID; the signal batch geometry saves a
scattered read of every signal batch header, which is paid on every process
start otherwise and grows with the file.

Worth running once on a networked node before submitting jobs against a large
file — the same reason `escpod demux models fetch` exists.

Examples:
  escpod sidecar index input.pod5           Index one file
  escpod sidecar index *.pod5               Index all POD5 files
  escpod sidecar index data_dir/            Index directory recursively
  escpod sidecar index input.pod5 --force   Rebuild an existing sidecar

A sidecar that already holds both caches is skipped; one missing either (e.g.
written by `demux --annotate`, which records the index but not the geometry) is
completed in place, preserving annotations and scores.
")]
pub struct IndexArgs {
    /// Input POD5 file(s) or directory
    #[arg(required = true, value_name = "FILES")]
    pub inputs: Vec<PathBuf>,

    /// Rebuild existing .p5s sidecars (annotations are preserved)
    #[arg(short, long)]
    pub force: bool,

    /// Number of threads for parallel processing (default: 16, or all available CPUs if fewer)
    #[arg(short = 't', long, visible_short_alias = 'j', value_name = "N")]
    pub threads: Option<usize>,
}

/// Build the `.p5s` read index for one or more POD5 files.
///
/// The index maps each read UUID to its location in the reads table,
/// enabling O(log n) binary-search lookup instead of a full-table scan.
/// The sidecar is
/// written next to the POD5 file by appending `.p5s` to the full
/// filename (e.g. `reads.pod5` → `reads.pod5.p5s`); annotations already
/// in the sidecar are preserved.
///
/// `args.threads` is not read here: `main` sizes the rayon pool from it before
/// dispatch (see `crate::requested_threads`).
pub fn run(args: IndexArgs) -> anyhow::Result<()> {
    let IndexArgs {
        inputs,
        force,
        threads: _,
    } = args;
    let files = collect_pod5_inputs(&inputs)?;

    let total = files.len();
    info!("building read indexes for {} file(s)", style::count(total),);

    let indexed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);

    let errors: Vec<anyhow::Error> = files
        .par_iter()
        .filter_map(|pod5_path| {
            let p5s_path = sidecar_path(pod5_path);

            let t0 = Instant::now();
            let reader = match escapepod_signal::Reader::open(pod5_path) {
                Ok(r) => r,
                Err(e) => return Some(anyhow::Error::from(e)),
            };

            // An existing sidecar is skipped without --force only if it already
            // holds *everything this command writes* — both the read index and
            // the signal batch geometry. Asking merely "does a sidecar load?"
            // was right when the index was all there was, and became wrong the
            // moment the geometry joined it: a sidecar created by
            // `demux --annotate` or `escpod sidecar annotate` has an index and no
            // geometry, so the obvious remedy — run `escpod sidecar index` — would
            // report "already indexed" and do nothing, leaving every later open
            // to re-walk the batch headers.
            //
            // A sidecar that does not load falls through to the rebuild below,
            // which is where the decision actually belongs: only one bound to a
            // *different* POD5 is replaced, and one that merely would not be
            // read is an error rather than a fresh start. Answering "already
            // exists" here for a sidecar the reader refuses to load would be a
            // dead end either way.
            if p5s_path.exists() && !force {
                let identity = match reader.sidecar_identity() {
                    Ok(id) => id,
                    Err(e) => return Some(anyhow::Error::from(e)),
                };
                // Metadata only — the geometry lives in the schema, so this
                // does not decode the (potentially multi-million row) index
                // just to decide whether to rebuild it.
                match escapepod_signal::pod5::sidecar::read_sidecar_metadata(&p5s_path, &identity) {
                    Ok(Some(meta)) if meta.signal_batch_rows.is_some() => {
                        warn!(
                            "skipping {} (already has both caches, use --force to rebuild)",
                            style::path(p5s_path.display()),
                        );
                        skipped.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                    Ok(Some(_)) => {
                        // Complete it in place rather than rebuilding. The
                        // index is already correct — only a cache is missing —
                        // and a rebuild would re-scan the reads table and
                        // rewrite every column, including barcode and score
                        // columns a demux run spent hours producing. Adding one
                        // metadata key cannot lose them; passing them through a
                        // rebuild can.
                        match reader.complete_sidecar_geometry(&p5s_path) {
                            Ok(true) => {
                                info!(
                                    "{} {} — added signal batch geometry in {:.1}s",
                                    style::action("completed"),
                                    style::path(p5s_path.display()),
                                    t0.elapsed().as_secs_f64(),
                                );
                                indexed.fetch_add(1, Ordering::Relaxed);
                            }
                            // Raced with another writer that got there first.
                            Ok(false) => {
                                skipped.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => return Some(anyhow::Error::from(e)),
                        }
                        return None;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // Deliberately not a warning. This read only decides
                        // whether the rebuild can be skipped; it cannot tell a
                        // sidecar that belongs to another POD5 from one that
                        // merely would not parse, and `build_and_write_index`
                        // can. Warning "rebuilding, annotations discarded" here
                        // would be a lie on the second case, where nothing is
                        // discarded and the rebuild refuses outright — and a
                        // duplicate on the first, which warns for itself.
                        debug!("{e} — deferring to the rebuild to classify");
                    }
                }
            }
            let count = match reader.build_and_write_index(&p5s_path) {
                Ok(c) => c,
                Err(e) => return Some(anyhow::Error::from(e)),
            };
            let elapsed = t0.elapsed();

            info!(
                "{} {} — {} reads in {:.1}s",
                style::action("wrote"),
                style::path(p5s_path.display()),
                style::count(count),
                elapsed.as_secs_f64(),
            );
            indexed.fetch_add(1, Ordering::Relaxed);
            None
        })
        .collect();

    if let Some(first_err) = errors.into_iter().next() {
        return Err(first_err);
    }

    info!(
        "{} indexed, {} skipped",
        style::count(indexed.load(Ordering::Relaxed)),
        style::count(skipped.load(Ordering::Relaxed)),
    );

    Ok(())
}

//! Build the read-index columns of `.p5s` sidecars for fast read lookup.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use tracing::{debug, info, warn};

use crate::style;
use crate::util::collect_pod5_inputs;
use escapepod_signal::pod5::sidecar::sidecar_path;

/// Build the `.p5s` read index for one or more POD5 files.
///
/// The index maps each read UUID to its location in the reads table,
/// enabling O(log n) binary-search lookup instead of a full-table scan.
/// The sidecar is
/// written next to the POD5 file by appending `.p5s` to the full
/// filename (e.g. `reads.pod5` → `reads.pod5.p5s`); annotations already
/// in the sidecar are preserved.
pub fn run(inputs: Vec<PathBuf>, force: bool) -> anyhow::Result<()> {
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
            // `demux --annotate` or `escpod annotate` has an index and no
            // geometry, so the obvious remedy — run `escpod index` — would
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

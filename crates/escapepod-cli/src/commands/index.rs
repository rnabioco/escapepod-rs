//! Build the read-index columns of `.p5s` sidecars for fast read lookup.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use tracing::{info, warn};

use crate::style;
use crate::util::collect_pod5_inputs;
use escapepod_signal::pod5::sidecar::sidecar_path;

/// Build the `.p5s` read index for one or more POD5 files.
///
/// The index maps each read UUID to its location in the reads table,
/// enabling O(1) lookup instead of a full-table scan. The sidecar is
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

            if p5s_path.exists() && !force {
                warn!(
                    "skipping {} (already exists, use --force to rebuild)",
                    style::path(p5s_path.display()),
                );
                skipped.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            let t0 = Instant::now();
            let reader = match escapepod_signal::Reader::open(pod5_path) {
                Ok(r) => r,
                Err(e) => return Some(anyhow::Error::from(e)),
            };
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

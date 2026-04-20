//! Build `.p5i` sidecar indexes for fast read lookup in POD5 files.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::style;
use crate::util::collect_pod5_inputs;

/// Build `.p5i` read index for one or more POD5 files.
///
/// The index maps each read UUID to its location in the reads table,
/// enabling O(1) lookup instead of a full-table scan. The sidecar is
/// written next to the POD5 file by appending `.p5i` to the full
/// filename (e.g. `reads.pod5` → `reads.pod5.p5i`).
pub fn run(inputs: Vec<PathBuf>, force: bool, threads: Option<usize>) -> anyhow::Result<()> {
    // Configure rayon thread pool if threads specified
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // Ignore error if pool already initialized
    }

    let files = collect_pod5_inputs(&inputs)?;

    let total = files.len();
    eprintln!(
        "{} Building read indexes for {} file(s)...",
        style::action("Index:"),
        style::count(total),
    );

    let indexed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);

    let errors: Vec<anyhow::Error> = files
        .par_iter()
        .filter_map(|pod5_path| {
            let p5i_path = {
                let mut s = pod5_path.as_os_str().to_owned();
                s.push(".p5i");
                PathBuf::from(s)
            };

            if p5i_path.exists() && !force {
                eprintln!(
                    "  {} {} (already exists, use --force to overwrite)",
                    style::info("skip"),
                    style::path(p5i_path.display()),
                );
                skipped.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            let t0 = Instant::now();
            let reader = match escapepod_signal::Reader::open(pod5_path) {
                Ok(r) => r,
                Err(e) => return Some(anyhow::Error::from(e)),
            };
            let count = match reader.build_and_write_index(&p5i_path) {
                Ok(c) => c,
                Err(e) => return Some(anyhow::Error::from(e)),
            };
            let elapsed = t0.elapsed();

            eprintln!(
                "  {} {} — {} reads in {:.1}s",
                style::action("wrote"),
                style::path(p5i_path.display()),
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

    eprintln!(
        "{} {} indexed, {} skipped",
        style::action("Done:"),
        style::count(indexed.load(Ordering::Relaxed)),
        style::count(skipped.load(Ordering::Relaxed)),
    );

    Ok(())
}

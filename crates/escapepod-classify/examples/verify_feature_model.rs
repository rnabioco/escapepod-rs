// SPDX-License-Identifier: MIT

//! Check a real `feature_model` charging bundle against reference
//! probabilities, on real weights and a real corpus.
//!
//! The golden-vector test (`tests/charging_fnn_parity.rs`) pins the rules on
//! 19 fixture reads, a 2 k-parameter graph and the full four-statistic grid.
//! That is the right shape for CI and the wrong shape for believing a shipped
//! model: the models actually worth shipping use a **subset** feature set
//! (two statistics of four, so `select_columns` is not the identity), 33
//! offsets rather than 25, a graph two orders of magnitude larger, and a
//! corpus where a fifth of the bases are unresolved. This runs the same code
//! on those.
//!
//! Reference side: `scripts/dump_feature_model_reference.py`, which follows
//! the bundle's declared contract with onnxruntime and imports nothing from
//! the training package — so an agreement here is evidence that the bundle
//! describes itself sufficiently, not just that two copies of one library
//! agree.
//!
//! ```bash
//! python scripts/dump_feature_model_reference.py \
//!     --bundle <bundle dir> --corpus <prefix>_F.npy --n 4096 --out /tmp/ref
//! cargo run --release --example verify_feature_model --features fnn-onnx -- \
//!     <bundle dir> /tmp/ref
//! ```
//!
//! Exits non-zero if the worst absolute probability difference exceeds
//! `--tol` (default 1e-5). It prints the distribution either way: a bare
//! "N differ" cannot tell a wrong rule from a numerical tie, and the first
//! version of the CRF decode check made exactly that mistake.

use escapepod_classify::{ChargingBundle, ChargingScorer};
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{}: not a whole f32 array",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f64(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 8,
        0,
        "{}: not a whole f64 array",
        path.display()
    );
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let bundle_dir = PathBuf::from(args.next().expect("usage: <bundle dir> <ref prefix> [tol]"));
    let prefix = args.next().expect("usage: <bundle dir> <ref prefix> [tol]");
    let tol: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(1e-5);

    let bundle = ChargingBundle::load(&bundle_dir)?;
    let net = match &bundle.scorer {
        ChargingScorer::FeatureNn(n) => n,
        other => anyhow::bail!("{} carries a {} scorer", bundle_dir.display(), other.kind()),
    };
    println!(
        "bundle {}{}: {} selected columns, {} value channels x {} offsets",
        bundle.model_id,
        bundle
            .model_version
            .as_deref()
            .map(|v| format!(" v{v}"))
            .unwrap_or_default(),
        bundle.columns.len(),
        net.n_value_channels(),
        net.n_offsets(),
    );

    let grid = read_f32(Path::new(&format!("{prefix}.grid.f32")));
    let want = read_f64(Path::new(&format!("{prefix}.p.f64")));
    let n_cols = bundle.offsets.len() * escapepod_classify::FEAT_STATS.len();
    assert_eq!(
        grid.len(),
        want.len() * n_cols,
        "grid is {} floats but {} rows x {n_cols} columns is {}",
        grid.len(),
        want.len(),
        want.len() * n_cols
    );

    let mut devs: Vec<f64> = Vec::with_capacity(want.len());
    let mut worst = (0usize, 0.0f64);
    for (row, chunk) in grid.chunks_exact(n_cols).enumerate() {
        // The whole runtime path: the bundle's own column selection, then
        // fold + standardise + mask + graph + softmax.
        let cols = bundle.select_columns(chunk);
        let p = net.predict(&cols)?[1];
        let d = (p - want[row]).abs();
        if d > worst.1 {
            worst = (row, d);
        }
        devs.push(d);
    }
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| devs[((devs.len() - 1) as f64 * f) as usize];
    println!(
        "{} rows | median {:e} | p99 {:e} | max {:e} (row {})",
        devs.len(),
        q(0.5),
        q(0.99),
        worst.1,
        worst.0
    );
    let n_over = devs.iter().filter(|&&d| d > tol).count();
    println!("{n_over} of {} rows exceed tol {tol:e}", devs.len());
    if n_over > 0 {
        anyhow::bail!("probabilities disagree beyond {tol:e}");
    }
    println!("OK");
    Ok(())
}

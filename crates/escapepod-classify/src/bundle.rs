// SPDX-License-Identifier: MIT

//! The charging-classifier model bundle: weights + the full feature recipe.
//!
//! `escpod signal classify` reads the recipe from the bundle's `metadata.json`
//! rather than taking flags: a caller that computes the features differently
//! gets a **wrong answer, not an error**, so the definition travels with the
//! weights (rnabioco/escapepod-rs#204). The k-mer table is pinned by sha256
//! for the same reason — the residual is defined relative to it, so a
//! swapped table is a different feature space and an invalid model.
//!
//! Bundle layout (format `escapepod-charging-classifier/1`, the GBM
//! variant; emitted by escapepod-models' `build_charging_bundle.py`):
//!
//! ```text
//! <bundle dir>/
//!   metadata.json     — the contract below
//!   <gbm file>        — GbmModel JSON (scripts/export_gbm_model.py)
//!   <kmer table>      — tab-separated k-mer levels (optionally .gz)
//! ```
//!
//! `metadata.json` fields consumed here (unknown fields are ignored, so the
//! bundle can carry provenance/metrics blocks freely):
//!
//! ```json
//! {
//!   "format": "escapepod-charging-classifier/1",
//!   "model": {"id": "...", "version": "..."},
//!   "classes": ["uncharged", "charged"],
//!   "gbm": {"file": "model.gbm.json", "sha256": "..."},
//!   "anchor": {"motif": "CCAGGC", "motif_offset": 3,
//!              "common_arm": "GGCTTCTTCTTGCTCTT"},
//!   "features": {"offsets": [-8, ..., 16],
//!                "stats": ["dwell", "mean", "std", "resid"],
//!                "order": ["b-8_dwell", ...]},
//!   "kmer_table": {"file": "levels.txt.gz", "sha256": "...",
//!                  "center_idx": null},
//!   "operating_point": {"probability": 0.784, "cl": 200, "source": "..."}
//! }
//! ```
//!
//! `features.order` is the model's exact input column order (names
//! `b<+/-offset>_<stat>`); `n_features` of the GBM must equal its length.
//! `kmer_table` is required whenever a `resid` column is present.
//! `operating_point` is the recommended call threshold derived from the
//! cross-experiment evaluation — consumers should read it rather than
//! assume the legacy 200.

use crate::features::FEAT_STATS;
use anyhow::{Context, Result, bail};
use escapepod_demux::{GbmModel, load_gbm_model};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct MetaFile {
    format: String,
    model: ModelBlock,
    classes: Vec<String>,
    gbm: FileRef,
    anchor: AnchorBlock,
    features: FeaturesBlock,
    #[serde(default)]
    kmer_table: Option<KmerTableBlock>,
    #[serde(default)]
    operating_point: Option<OperatingPoint>,
}

#[derive(Debug, Deserialize)]
struct ModelBlock {
    id: String,
    #[serde(default)]
    version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FileRef {
    file: String,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnchorBlock {
    pub motif: String,
    pub motif_offset: usize,
    pub common_arm: String,
}

#[derive(Debug, Deserialize)]
struct FeaturesBlock {
    offsets: Vec<i32>,
    stats: Vec<String>,
    order: Vec<String>,
    #[serde(default)]
    resolution: ResolutionBlock,
}

/// How each offset's signal span is found — the half of the feature contract
/// that is not the offsets themselves. `escapepod-models` writes this as
/// `features.resolution`.
///
/// Asking the aligner and walking the query give different spans on precisely
/// the reads that matter, because bwa stops at the adduct. A bundle scored
/// under the wrong one gets a confident wrong answer, not an error.
///
/// Absent means the aligner path: bundles built before the counting anchor
/// existed (`charging_cnn_rna004@v0.1.0`, 2026-08-10) genuinely used it.
#[derive(Debug, Default, Deserialize)]
struct ResolutionBlock {
    #[serde(default)]
    count_arm_bases: u32,
}

#[derive(Debug, Deserialize)]
struct KmerTableBlock {
    file: String,
    sha256: String,
    #[serde(default)]
    center_idx: Option<usize>,
}

/// Recommended call threshold shipped with the model (derived from the
/// cross-experiment evaluation, not the legacy hard-coded 200).
#[derive(Debug, Clone, Deserialize)]
pub struct OperatingPoint {
    /// Probability threshold for calling the positive class.
    pub probability: f64,
    /// The same threshold on the `cl` scale (`round(p * 255)`), if recorded.
    #[serde(default)]
    pub cl: Option<u8>,
    /// Where the threshold came from.
    #[serde(default)]
    pub source: Option<String>,
}

/// The k-mer level model the residual feature is defined against.
#[derive(Debug)]
pub struct KmerLevels {
    pub map: HashMap<String, f64>,
    pub k: usize,
    /// Which base of the k-mer a level belongs to (default `k / 2`).
    pub center_idx: usize,
}

/// A loaded, hash-verified charging-classifier bundle.
#[derive(Debug)]
pub struct ChargingBundle {
    pub dir: PathBuf,
    pub model_id: String,
    pub model_version: Option<String>,
    /// Class names in probability order; the `cl` tag encodes
    /// `P(classes[1])`.
    pub classes: [String; 2],
    pub gbm: GbmModel,
    pub anchor: AnchorBlock,
    /// Feature offsets relative to the junction, recipe order.
    pub offsets: Vec<i32>,
    /// How each offset's signal span is found (`features.resolution`).
    pub span_mode: crate::anchor::SpanMode,
    /// Model input columns as `(offset index, stat index)` into the
    /// canonical `offsets × FEAT_STATS` grid, in `features.order` order.
    pub columns: Vec<(usize, usize)>,
    /// Present iff the recipe uses `resid` columns.
    pub kmer: Option<KmerLevels>,
    pub operating_point: Option<OperatingPoint>,
}

/// sha256 of a file, lowercase hex.
fn sha256_file(path: &Path) -> Result<String> {
    use std::fmt::Write as _;
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read {} for hashing", path.display()))?;
    let digest = Sha256::digest(&bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        write!(s, "{:02x}", b).expect("writing to a String cannot fail");
    }
    Ok(s)
}

fn verify_sha256(path: &Path, expected: &str, what: &str) -> Result<()> {
    let got = sha256_file(path)?;
    if !got.eq_ignore_ascii_case(expected) {
        bail!(
            "{} checksum mismatch for {}: expected {}, got {} — the bundle is \
             corrupt or its {} was swapped (which would silently change the \
             feature space)",
            what,
            path.display(),
            expected,
            got,
            what
        );
    }
    Ok(())
}

/// Parse a feature-column name (`b<signed offset>_<stat>`) into indices
/// over `offsets` × [`FEAT_STATS`].
fn parse_column(name: &str, offsets: &[i32]) -> Result<(usize, usize)> {
    let rest = name
        .strip_prefix('b')
        .with_context(|| format!("feature name {:?} does not start with 'b'", name))?;
    let (off_str, stat) = rest
        .split_once('_')
        .with_context(|| format!("feature name {:?} has no '_<stat>' suffix", name))?;
    let off: i32 = off_str
        .parse()
        .with_context(|| format!("feature name {:?}: bad offset {:?}", name, off_str))?;
    let oi = offsets
        .iter()
        .position(|&o| o == off)
        .with_context(|| format!("feature {:?}: offset {} not in recipe offsets", name, off))?;
    let si = FEAT_STATS
        .iter()
        .position(|&s| s == stat)
        .with_context(|| format!("feature {:?}: unknown stat {:?}", name, stat))?;
    Ok((oi, si))
}

impl ChargingBundle {
    /// Load a bundle from its directory (or a direct `metadata.json` path),
    /// verifying every pinned checksum.
    pub fn load(path: &Path) -> Result<Self> {
        let (dir, meta_path) = if path.is_dir() {
            (path.to_path_buf(), path.join("metadata.json"))
        } else {
            (
                path.parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from(".")),
                path.to_path_buf(),
            )
        };
        let text = std::fs::read_to_string(&meta_path)
            .with_context(|| format!("cannot read bundle metadata {}", meta_path.display()))?;
        let meta: MetaFile = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse bundle metadata {}", meta_path.display()))?;

        if meta.format != "escapepod-charging-classifier/1" {
            bail!(
                "unsupported bundle format {:?} (expected escapepod-charging-classifier/1)",
                meta.format
            );
        }
        let [c0, c1]: [String; 2] = meta
            .classes
            .clone()
            .try_into()
            .map_err(|c: Vec<String>| anyhow::anyhow!("expected 2 classes, got {}", c.len()))?;

        let gbm_path = dir.join(&meta.gbm.file);
        verify_sha256(&gbm_path, &meta.gbm.sha256, "GBM model")?;
        let gbm = load_gbm_model(&gbm_path)?;
        if gbm.n_classes != 2 {
            bail!("charging GBM must have 2 classes, has {}", gbm.n_classes);
        }
        if gbm.n_features != meta.features.order.len() {
            bail!(
                "GBM expects {} features but the recipe orders {} columns",
                gbm.n_features,
                meta.features.order.len()
            );
        }

        for s in &meta.features.stats {
            if !FEAT_STATS.contains(&s.as_str()) {
                bail!("recipe stat {:?} is not one of {:?}", s, FEAT_STATS);
            }
        }
        let columns = meta
            .features
            .order
            .iter()
            .map(|n| parse_column(n, &meta.features.offsets))
            .collect::<Result<Vec<_>>>()?;

        let needs_resid = columns.iter().any(|&(_, si)| FEAT_STATS[si] == "resid");
        let kmer = match (&meta.kmer_table, needs_resid) {
            (Some(kt), _) => {
                let table_path = dir.join(&kt.file);
                verify_sha256(&table_path, &kt.sha256, "k-mer table")?;
                let (map, k) = escapepod_signal::resquiggle::load_kmer_table(&table_path)?;
                let center_idx = kt.center_idx.unwrap_or(k / 2);
                if center_idx >= k {
                    bail!(
                        "kmer_table center_idx {} out of range for k={}",
                        center_idx,
                        k
                    );
                }
                Some(KmerLevels { map, k, center_idx })
            }
            (None, true) => bail!(
                "the recipe has resid columns but the bundle pins no kmer_table — \
                 the residual would be undefined"
            ),
            (None, false) => None,
        };

        Ok(ChargingBundle {
            dir,
            model_id: meta.model.id,
            model_version: meta.model.version,
            classes: [c0, c1],
            gbm,
            anchor: meta.anchor,
            offsets: meta.features.offsets,
            span_mode: crate::anchor::SpanMode::from_arm_bases(
                meta.features.resolution.count_arm_bases,
            ),
            columns,
            kmer,
            operating_point: meta.operating_point,
        })
    }

    /// Select the model's input columns (as `f64`, NaN preserved) from the
    /// canonical `offsets × FEAT_STATS` feature grid.
    pub fn select_columns(&self, grid: &[f32]) -> Vec<f64> {
        self.columns
            .iter()
            .map(|&(oi, si)| grid[oi * FEAT_STATS.len() + si] as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_column() {
        let offsets: Vec<i32> = (-8..=16).collect();
        assert_eq!(parse_column("b-8_dwell", &offsets).unwrap(), (0, 0));
        assert_eq!(parse_column("b+0_mean", &offsets).unwrap(), (8, 1));
        assert_eq!(parse_column("b+16_resid", &offsets).unwrap(), (24, 3));
        assert!(parse_column("b+17_mean", &offsets).is_err()); // not in recipe
        assert!(parse_column("b+1_bogus", &offsets).is_err());
        assert!(parse_column("x+1_mean", &offsets).is_err());
    }
}

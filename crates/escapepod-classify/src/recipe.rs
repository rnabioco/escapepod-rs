// SPDX-License-Identifier: MIT

//! The feature recipe: what the model's input space *is*, independent of
//! where it was read from.
//!
//! A charging bundle's `metadata.json` carries a great deal — weights, class
//! names, checksums, the operating point, provenance — but the part that
//! defines the **feature space** is only three things: which base offsets to
//! measure, how each offset's signal span is resolved, and the k-mer level
//! table the residual is taken against. Everything else in the bundle is
//! about scoring the features or vouching for them.
//!
//! Those three travelled as fields of [`ChargingBundle`], which made
//! [`crate::feature_grid`] take a whole bundle. That is the wrong dependency:
//! computing features is what the corpus builder does — it has offsets, a
//! span mode and a k-mer table, but no weights, no operating point and no
//! model at all, because the model does not exist yet. It was forced either
//! to own a bundle it had no use for or to reimplement the grid, and this
//! project has already paid for a second, divergent feature definition once.
//!
//! [`FeatureRecipe`] is a **borrowed view**, so producing one from a loaded
//! bundle is free and there is no second copy of the recipe to drift from the
//! first. [`ChargingBundle`] stays the thing that parses and verifies
//! `metadata.json`; this is only about what consumption needs.
//!
//! [`ChargingBundle`]: crate::ChargingBundle

use crate::anchor::SpanMode;
use std::collections::HashMap;

/// The k-mer level model the `resid` statistic is defined against.
///
/// Pinned by sha256 wherever it is shipped: the residual is *relative* to
/// these levels, so a swapped table is a different feature space rather than
/// a differently-calibrated one.
#[derive(Debug)]
pub struct KmerLevels {
    pub map: HashMap<String, f64>,
    pub k: usize,
    /// Which base of the k-mer a level belongs to (default `k / 2`).
    pub center_idx: usize,
}

/// Everything [`crate::feature_grid`] needs, and nothing else.
///
/// Cheap to build and `Copy`, so callers make one per batch (or per call)
/// rather than threading three arguments through.
#[derive(Debug, Clone, Copy)]
pub struct FeatureRecipe<'a> {
    /// Base offsets relative to the junction, in recipe order. The grid is
    /// `offsets × `[`crate::FEAT_STATS`], offsets-outer.
    pub offsets: &'a [i32],
    /// How each offset's signal span is found.
    pub span_mode: SpanMode,
    /// Present iff the recipe uses `resid` columns; without it the residual
    /// stat is `NaN` for every offset.
    pub kmer: Option<&'a KmerLevels>,
}

impl<'a> FeatureRecipe<'a> {
    /// Build a recipe from its parts (the corpus-builder path, where there is
    /// no bundle to borrow from).
    pub fn new(offsets: &'a [i32], span_mode: SpanMode, kmer: Option<&'a KmerLevels>) -> Self {
        Self {
            offsets,
            span_mode,
            kmer,
        }
    }
}

impl<'a> From<&'a crate::bundle::FeatureSpace> for FeatureRecipe<'a> {
    /// Borrow a loaded bundle's feature space. Deliberately from the space
    /// rather than from the bundle: a windowed bundle has no columns at all,
    /// and a `From` that has to invent an empty recipe for it is a conversion
    /// that cannot fail where the underlying question can. Use
    /// [`crate::ChargingBundle::recipe`], which pairs the space with the k-mer
    /// levels and says so when there is none.
    fn from(space: &'a crate::bundle::FeatureSpace) -> Self {
        Self {
            offsets: &space.offsets,
            span_mode: space.span_mode,
            kmer: None,
        }
    }
}

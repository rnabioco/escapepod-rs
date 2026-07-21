//! Native gradient-boosted tree-ensemble barcode classifier.
//!
//! This is the runtime counterpart to a `sklearn.HistGradientBoostingClassifier`
//! trained on warpdemux-compat fingerprints. The production tRNA model
//! `WDX4_tRNA_rna004_v1_0` is itself gradient-boosted (`fpt_boost`), so a tree
//! ensemble on the raw fingerprint is the most faithful reproduction of that
//! head — and unlike the DTW-SVM path it needs no DTW, no reference bank, and no
//! GPU. Inference is a plain per-tree walk summed across boosting iterations,
//! followed by softmax over the per-class raw scores.
//!
//! escpod can't run these models through ONNX — `tract` has no `ai.onnx.ml`
//! `TreeEnsembleClassifier` op — so we carry a small JSON dump of the fitted
//! trees (`scripts/export_gbm_model.py`) and walk them directly here. The JSON
//! is auto-detected by [`crate::load_any_model`] via its `model_type: "gbm"`
//! discriminator.
//!
//! The decision head (argmax + top1−top2 margin + per-class threshold gate) is
//! shared verbatim with the SVM path through [`crate::process_probabilities`].

use crate::probability::{ProbabilityResult, process_probabilities, softmax};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// A single node in a gradient-boosted decision tree.
///
/// Mirrors one entry of sklearn's `TreePredictor.nodes` structured array. Split
/// rule matches sklearn's `_predict_one_from_raw_data`: `x[feature] <= threshold`
/// descends to `left`, otherwise `right`; a `NaN` feature follows `missing_left`.
/// Leaf nodes carry the boosted `value` (learning-rate shrinkage already baked
/// in by sklearn at fit time — it must NOT be re-applied here).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbmNode {
    /// Split feature index. Unused for leaves (conventionally `-1`).
    pub feature: i32,
    /// Split threshold (`x[feature] <= threshold` ⇒ left).
    pub threshold: f64,
    /// Index of the left child within the owning tree's node array.
    pub left: u32,
    /// Index of the right child within the owning tree's node array.
    pub right: u32,
    /// Where a `NaN` feature value is routed (`true` ⇒ left child).
    #[serde(default)]
    pub missing_left: bool,
    /// Leaf prediction (raw boosted contribution). Meaningful iff `leaf`.
    pub value: f64,
    /// Whether this node is a leaf.
    pub leaf: bool,
}

/// One boosted decision tree: a flat, preorder-indexed node array. Serializes
/// transparently as a bare JSON array of [`GbmNode`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GbmTree {
    pub nodes: Vec<GbmNode>,
}

/// A gradient-boosted tree-ensemble barcode classifier, deserialized from the
/// JSON emitted by `scripts/export_gbm_model.py`.
///
/// Multiclass only: every boosting iteration contributes one tree per class
/// (`n_trees_per_iteration == n_classes`), and the per-class raw scores are
/// combined with softmax — matching sklearn's multinomial `HistGradientBoosting`
/// head. Binary models (1 tree/iter + sigmoid) are intentionally unsupported.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbmModel {
    /// Schema discriminator; always `"gbm"`. Lets [`crate::load_any_model`]
    /// route a GBM JSON before the SVM/WarpDemux branches (a GBM JSON also
    /// carries `label_mapper`, so this must win).
    pub model_type: String,

    /// Number of barcode classes (= per-iteration tree count).
    pub n_classes: usize,

    /// Feature dimension expected of every query fingerprint.
    pub n_features: usize,

    /// Per-class initial raw score (`clf._baseline_prediction`), length
    /// `n_classes`. Raw scores start here before the trees are summed in.
    pub baseline: Vec<f64>,

    /// Boosted trees, indexed `[iteration][class]`. Every inner vector has
    /// length `n_classes`.
    pub trees: Vec<Vec<GbmTree>>,

    /// Maps class index (`0..n_classes`) to barcode ID. Same convention as
    /// [`crate::DtwSvmModel`] and upstream WarpDemuX `fpt_base`.
    pub label_mapper: HashMap<usize, i32>,

    /// Optional per-class confidence-margin thresholds (indexed by class).
    /// Predictions whose top1−top2 margin falls below their class threshold are
    /// rejected (returned as barcode `-1`).
    #[serde(default)]
    pub thresholds: Option<Vec<f64>>,

    /// Optional barcode IDs in class order (informational; not required for
    /// inference, which uses `label_mapper`).
    #[serde(default)]
    pub classes: Option<Vec<i32>>,
}

impl GbmModel {
    /// Number of boosting iterations.
    pub fn n_iterations(&self) -> usize {
        self.trees.len()
    }

    /// Validate structural invariants. Returns `Err(msg)` on the first problem.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != "gbm" {
            return Err(format!(
                "model_type is '{}', expected 'gbm'",
                self.model_type
            ));
        }
        if self.n_classes < 2 {
            return Err(format!(
                "n_classes is {}; GBM head requires >= 2 classes",
                self.n_classes
            ));
        }
        if self.baseline.len() != self.n_classes {
            return Err(format!(
                "baseline has {} entries, expected n_classes={}",
                self.baseline.len(),
                self.n_classes
            ));
        }
        if self.trees.is_empty() {
            return Err("trees is empty (no boosting iterations)".to_string());
        }
        for (it, iter_trees) in self.trees.iter().enumerate() {
            if iter_trees.len() != self.n_classes {
                return Err(format!(
                    "iteration {} has {} trees, expected n_classes={} \
                     (binary/sigmoid models are unsupported)",
                    it,
                    iter_trees.len(),
                    self.n_classes
                ));
            }
            for (cls, tree) in iter_trees.iter().enumerate() {
                if tree.nodes.is_empty() {
                    return Err(format!("tree [{}][{}] has no nodes", it, cls));
                }
                let n_nodes = tree.nodes.len();
                for (ni, node) in tree.nodes.iter().enumerate() {
                    if node.leaf {
                        continue;
                    }
                    if node.feature < 0 || (node.feature as usize) >= self.n_features {
                        return Err(format!(
                            "tree [{}][{}] node {} splits on feature {} out of range \
                             (n_features={})",
                            it, cls, ni, node.feature, self.n_features
                        ));
                    }
                    if node.left as usize >= n_nodes || node.right as usize >= n_nodes {
                        return Err(format!(
                            "tree [{}][{}] node {} has child index out of range \
                             (left={}, right={}, n_nodes={})",
                            it, cls, ni, node.left, node.right, n_nodes
                        ));
                    }
                }
            }
        }
        // Every class index must map to a barcode so output isn't all -1.
        for cls in 0..self.n_classes {
            if !self.label_mapper.contains_key(&cls) {
                return Err(format!("label_mapper is missing class index {}", cls));
            }
        }
        if let Some(t) = &self.thresholds
            && t.len() != self.n_classes
        {
            return Err(format!(
                "thresholds has {} entries, expected n_classes={}",
                t.len(),
                self.n_classes
            ));
        }
        Ok(())
    }
}

/// `flags` bit: this node is a leaf (its `thresh_or_value` is the leaf value).
const FLAG_LEAF: u16 = 1;
/// `flags` bit: a `NaN` feature routes to the left child.
const FLAG_MISSING_LEFT: u16 = 2;

/// Cache-compact node for the compiled inference arena — 16 bytes vs the
/// 40-byte serde [`GbmNode`] (f64 + i32 + 2×u32 + 2×bool, padded). Shrinking the
/// node 2.5× keeps the whole ensemble in L2: the wdx4 model's ~47k nodes drop
/// from ~1.9 MB (AoS, thrashes a 1–1.25 MB L2) to ~0.75 MB. `feature`/`left`/
/// `right` are tree-relative — every HistGradientBoosting tree here has ≤61
/// nodes, far inside u16 — so the children stay 2 bytes even though all trees
/// share one flat arena. For a leaf, `thresh_or_value` carries the boosted leaf
/// value; otherwise it is the split threshold.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CompactNode {
    /// Split threshold (internal) or boosted leaf value (`FLAG_LEAF`).
    thresh_or_value: f64,
    /// Tree-relative index of the left child (`x[feature] <= threshold`).
    left: u16,
    /// Tree-relative index of the right child.
    right: u16,
    /// Split feature index. Unused for leaves.
    feature: u16,
    /// Bit flags: [`FLAG_LEAF`], [`FLAG_MISSING_LEFT`].
    flags: u16,
}

/// A [`GbmModel`] lowered into a single flat, cache-compact node arena for the
/// hot path. Built once per [`GbmPredictor`] and shared (read-only) across all
/// rayon workers.
///
/// Trees are laid out **grouped by class**: all of class 0's per-iteration
/// trees, then class 1's, … . Inference accumulates one class at a time into a
/// scalar register (`acc`), so each class's contiguous ~190 KB sub-arena is
/// streamed once rather than striding across all classes every boosting
/// iteration (the old `[iteration][class]` walk touched four scattered trees per
/// inner step).
#[derive(Debug)]
struct CompiledGbm {
    /// All trees' nodes, concatenated class-major then iteration. Leaves are
    /// made **self-absorbing** (`left == right == own index`, threshold `+INF`,
    /// NaN-left) so the branch-free batched walk can step a fixed number of
    /// times and let early leaves spin in place; their boosted value lives in
    /// [`Self::values`], not the node.
    nodes: Vec<CompactNode>,
    /// Leaf value per arena slot (0.0 for internal nodes), indexed in lockstep
    /// with [`Self::nodes`]. Split out of the node so a self-absorbing leaf can
    /// keep `+INF` in `thresh_or_value` for the batched walk.
    values: Vec<f64>,
    /// `class_roots[k]` = arena offset of each of class `k`'s tree roots.
    class_roots: Vec<Vec<u32>>,
    /// `tree_depth[k][t]` = longest root→leaf edge count of class `k`'s tree
    /// `t`. The batched walk steps exactly this many times (every leaf is
    /// reached; shallower paths self-absorb the remainder).
    tree_depth: Vec<Vec<u16>>,
    /// Per-class initial raw score (copied from `GbmModel::baseline`).
    baseline: Vec<f64>,
}

/// Longest root→leaf edge count of the (tree-relative) subtree at `i`.
fn subtree_depth(nodes: &[GbmNode], i: usize) -> u16 {
    let n = &nodes[i];
    if n.leaf {
        0
    } else {
        1 + subtree_depth(nodes, n.left as usize).max(subtree_depth(nodes, n.right as usize))
    }
}

impl CompiledGbm {
    /// Lower a validated model into the flat arena. Relies on
    /// [`GbmModel::validate`] having already bounds-checked child/feature
    /// indices, so the walk needs no re-validation.
    fn from_model(m: &GbmModel) -> Self {
        let cap = m.trees.iter().flatten().map(|t| t.nodes.len()).sum();
        let mut nodes: Vec<CompactNode> = Vec::with_capacity(cap);
        let mut values: Vec<f64> = Vec::with_capacity(cap);
        let mut class_roots: Vec<Vec<u32>> = vec![Vec::with_capacity(m.trees.len()); m.n_classes];
        let mut tree_depth: Vec<Vec<u16>> = vec![Vec::with_capacity(m.trees.len()); m.n_classes];
        for (k, (roots, depths)) in class_roots
            .iter_mut()
            .zip(tree_depth.iter_mut())
            .enumerate()
        {
            for iter_trees in &m.trees {
                let tree = &iter_trees[k];
                roots.push(nodes.len() as u32);
                depths.push(subtree_depth(&tree.nodes, 0));
                for (rel, node) in tree.nodes.iter().enumerate() {
                    if node.leaf {
                        // Self-absorbing leaf: `<= +INF` (and NaN-left) routes to
                        // itself, so the fixed-step batched walk parks here.
                        nodes.push(CompactNode {
                            thresh_or_value: f64::INFINITY,
                            left: rel as u16,
                            right: rel as u16,
                            feature: 0,
                            flags: FLAG_LEAF | FLAG_MISSING_LEFT,
                        });
                        values.push(node.value);
                    } else {
                        let f = if node.missing_left {
                            FLAG_MISSING_LEFT
                        } else {
                            0
                        };
                        nodes.push(CompactNode {
                            thresh_or_value: node.threshold,
                            left: node.left as u16,
                            right: node.right as u16,
                            feature: node.feature as u16,
                            flags: f,
                        });
                        values.push(0.0);
                    }
                }
            }
        }
        Self {
            nodes,
            values,
            class_roots,
            tree_depth,
            baseline: m.baseline.clone(),
        }
    }

    /// Walk one tree (rooted at arena offset `root`) and return its leaf value.
    /// Applies [`GbmNode`]'s split rule verbatim — same `<=`/`NaN` routing on the
    /// same f64 thresholds — so results are bit-identical to a direct serde walk.
    #[inline]
    fn leaf_value(&self, root: usize, x: &[f64]) -> f64 {
        let nodes = &self.nodes;
        let mut i = root;
        loop {
            let n = &nodes[i];
            if n.flags & FLAG_LEAF != 0 {
                return self.values[i];
            }
            let v = x[n.feature as usize];
            let go_left = if v.is_nan() {
                n.flags & FLAG_MISSING_LEFT != 0
            } else {
                v <= n.thresh_or_value
            };
            i = root + if go_left { n.left } else { n.right } as usize;
        }
    }
}

/// Inference wrapper around a [`GbmModel`].
///
/// Lowers the model into a cache-compact [`CompiledGbm`] arena at construction
/// (see its docs for the layout). There is no per-read mutable state to
/// amortize (no DTW workspace, no coupling matrices), so the predictor is
/// trivially `Sync` and the CLI classify loop is a plain
/// `par_iter().map(predict)`.
#[derive(Debug)]
pub struct GbmPredictor<'a> {
    model: &'a GbmModel,
    compiled: CompiledGbm,
}

impl<'a> GbmPredictor<'a> {
    /// Build a predictor over a (already validated) model.
    pub fn new(model: &'a GbmModel) -> Self {
        Self {
            compiled: CompiledGbm::from_model(model),
            model,
        }
    }

    /// Classify one fingerprint.
    ///
    /// Returns the per-class probability vector (softmax over the boosted raw
    /// scores) alongside the gated [`ProbabilityResult`]. Errors if the
    /// fingerprint's length doesn't match the model's `n_features`.
    pub fn predict(&self, fingerprint: &[f64]) -> Result<(Vec<f64>, ProbabilityResult), String> {
        let m = self.model;
        if fingerprint.len() != m.n_features {
            return Err(format!(
                "fingerprint has {} features, model expects {}",
                fingerprint.len(),
                m.n_features
            ));
        }

        // raw[k] = baseline[k] + Σ_iter tree[iter][k].leaf_value(x), accumulated
        // class-by-class so each class's contiguous sub-arena streams once.
        let c = &self.compiled;
        let mut raw = c.baseline.clone();
        for (k, roots) in c.class_roots.iter().enumerate() {
            let mut acc = 0.0f64;
            for &root in roots {
                acc += c.leaf_value(root as usize, fingerprint);
            }
            raw[k] += acc;
        }

        let probs = softmax(&raw);
        let result = process_probabilities(&probs, &m.label_mapper, m.thresholds.as_deref());
        Ok((probs, result))
    }

    /// Classify many fingerprints, walking `BATCH` reads through each tree in
    /// branch-free lockstep.
    ///
    /// The single-read [`predict`](Self::predict) walk is latency-bound: every
    /// node's address depends on the previous node's comparison (a serial
    /// pointer chase), and the data-dependent branch mispredicts, so the core
    /// stalls (~0.7 IPC measured). Here `BATCH` independent reads descend the
    /// same tree together: the `for lane` body issues `BATCH` independent loads,
    /// letting the out-of-order engine overlap one lane's L2 latency with
    /// another's work. Leaves are self-absorbing (built in
    /// [`CompiledGbm::from_model`]) so the walk runs a fixed `tree_depth` steps
    /// with no per-lane termination branch.
    ///
    /// Results are bit-identical to per-read [`predict`](Self::predict): same f64
    /// thresholds, same `<=`/NaN routing, same accumulation order. The `< BATCH`
    /// tail falls back to scalar `predict`.
    pub fn predict_many(
        &self,
        fingerprints: &[&[f64]],
    ) -> Result<Vec<(Vec<f64>, ProbabilityResult)>, String> {
        // K=8 measured the sweet spot on Cascade Lake (rna): ~2.6× over scalar.
        // K=4 under-hides latency; K≥16 spills the per-lane `idx`/`xs` arrays.
        self.predict_many_k::<8>(fingerprints)
    }

    /// [`predict_many`](Self::predict_many) with an explicit lane count `K` (the
    /// number of reads walked in lockstep). Exposed for batch-width tuning;
    /// `predict_many` picks the measured sweet spot.
    pub fn predict_many_k<const K: usize>(
        &self,
        fingerprints: &[&[f64]],
    ) -> Result<Vec<(Vec<f64>, ProbabilityResult)>, String> {
        let m = self.model;
        let c = &self.compiled;
        for fp in fingerprints {
            if fp.len() != m.n_features {
                return Err(format!(
                    "fingerprint has {} features, model expects {}",
                    fp.len(),
                    m.n_features
                ));
            }
        }

        let mut out: Vec<(Vec<f64>, ProbabilityResult)> = Vec::with_capacity(fingerprints.len());
        let mut chunks = fingerprints.chunks_exact(K);
        for ch in &mut chunks {
            let xs: [&[f64]; K] = ch.try_into().expect("chunks_exact yields exactly K");
            // raw[lane] starts at the per-class baseline and accumulates trees.
            let mut raw: [Vec<f64>; K] = std::array::from_fn(|_| c.baseline.clone());
            for (k, (roots, depths)) in c.class_roots.iter().zip(&c.tree_depth).enumerate() {
                let mut acc = [0.0f64; K];
                for (&root, &depth) in roots.iter().zip(depths) {
                    let root = root as usize;
                    let mut idx = [root; K];
                    for _ in 0..depth {
                        for lane in 0..K {
                            let n = &c.nodes[idx[lane]];
                            let v = xs[lane][n.feature as usize];
                            let go_left = if v.is_nan() {
                                n.flags & FLAG_MISSING_LEFT != 0
                            } else {
                                v <= n.thresh_or_value
                            };
                            idx[lane] = root + if go_left { n.left } else { n.right } as usize;
                        }
                    }
                    for lane in 0..K {
                        acc[lane] += c.values[idx[lane]];
                    }
                }
                for lane in 0..K {
                    raw[lane][k] += acc[lane];
                }
            }
            for raw_lane in raw {
                let probs = softmax(&raw_lane);
                let result =
                    process_probabilities(&probs, &m.label_mapper, m.thresholds.as_deref());
                out.push((probs, result));
            }
        }
        for &fp in chunks.remainder() {
            out.push(self.predict(fp)?);
        }
        Ok(out)
    }
}

/// Load a [`GbmModel`] from JSON and validate it.
pub fn load_gbm_model(path: &Path) -> Result<GbmModel, anyhow::Error> {
    use anyhow::Context;

    let file = File::open(path)
        .with_context(|| format!("Failed to open model file '{}'", path.display()))?;
    let reader = BufReader::new(file);
    let model: GbmModel =
        serde_json::from_reader(reader).with_context(|| "Failed to parse GBM model JSON")?;
    model
        .validate()
        .map_err(|e| anyhow::anyhow!("Invalid GbmModel: {}", e))?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny 2-class, 2-iteration model. Each tree is a single stump on
    /// feature 0: `x[0] <= 0.5` ⇒ left leaf, else right leaf. Class 0's stumps
    /// favor small x, class 1's favor large x.
    fn toy_model() -> GbmModel {
        let stump = |left_val: f64, right_val: f64| GbmTree {
            nodes: vec![
                GbmNode {
                    feature: 0,
                    threshold: 0.5,
                    left: 1,
                    right: 2,
                    missing_left: true,
                    value: 0.0,
                    leaf: false,
                },
                GbmNode {
                    feature: -1,
                    threshold: 0.0,
                    left: 0,
                    right: 0,
                    missing_left: false,
                    value: left_val,
                    leaf: true,
                },
                GbmNode {
                    feature: -1,
                    threshold: 0.0,
                    left: 0,
                    right: 0,
                    missing_left: false,
                    value: right_val,
                    leaf: true,
                },
            ],
        };
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 3);
        label_mapper.insert(1, 7);
        GbmModel {
            model_type: "gbm".to_string(),
            n_classes: 2,
            n_features: 1,
            baseline: vec![0.0, 0.0],
            // Per iteration: [class0 stump, class1 stump].
            trees: vec![
                vec![stump(1.0, -1.0), stump(-1.0, 1.0)],
                vec![stump(1.0, -1.0), stump(-1.0, 1.0)],
            ],
            label_mapper,
            thresholds: None,
            classes: Some(vec![3, 7]),
        }
    }

    #[test]
    fn validates_ok() {
        toy_model().validate().expect("toy model should validate");
    }

    #[test]
    fn predicts_low_feature_as_class0() {
        let m = toy_model();
        let p = GbmPredictor::new(&m);
        // x[0]=0.0 ≤ 0.5: class0 raw = +2, class1 raw = -2 → class 0 (barcode 3).
        let (probs, res) = p.predict(&[0.0]).unwrap();
        assert_eq!(res.predicted_index, 0);
        assert_eq!(res.predicted_barcode, 3);
        assert!(probs[0] > probs[1]);
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predicts_high_feature_as_class1() {
        let m = toy_model();
        let p = GbmPredictor::new(&m);
        // x[0]=1.0 > 0.5: class0 raw = -2, class1 raw = +2 → class 1 (barcode 7).
        let (_probs, res) = p.predict(&[1.0]).unwrap();
        assert_eq!(res.predicted_index, 1);
        assert_eq!(res.predicted_barcode, 7);
    }

    #[test]
    fn nan_follows_missing_left() {
        let m = toy_model();
        let p = GbmPredictor::new(&m);
        // NaN routes left (missing_left=true) → same as the low-feature case.
        let (_probs, res) = p.predict(&[f64::NAN]).unwrap();
        assert_eq!(res.predicted_index, 0);
    }

    #[test]
    fn rejects_wrong_feature_count() {
        let m = toy_model();
        let p = GbmPredictor::new(&m);
        assert!(p.predict(&[0.0, 1.0]).is_err());
    }

    #[test]
    fn threshold_gate_rejects_low_margin() {
        let mut m = toy_model();
        // Zero out the trees so both classes tie (margin 0), then demand a
        // positive margin → rejection (barcode -1).
        m.baseline = vec![0.0, 0.0];
        m.trees = vec![vec![
            GbmTree {
                nodes: vec![GbmNode {
                    feature: -1,
                    threshold: 0.0,
                    left: 0,
                    right: 0,
                    missing_left: false,
                    value: 0.0,
                    leaf: true,
                }],
            },
            GbmTree {
                nodes: vec![GbmNode {
                    feature: -1,
                    threshold: 0.0,
                    left: 0,
                    right: 0,
                    missing_left: false,
                    value: 0.0,
                    leaf: true,
                }],
            },
        ]];
        m.thresholds = Some(vec![0.5, 0.5]);
        let p = GbmPredictor::new(&m);
        let (_probs, res) = p.predict(&[0.0]).unwrap();
        assert!(!res.is_confident);
        assert_eq!(res.predicted_barcode, -1);
    }

    #[test]
    fn json_round_trips() {
        let m = toy_model();
        let json = serde_json::to_string(&m).unwrap();
        // Tree serializes as a bare array (transparent newtype).
        assert!(json.contains("\"trees\":[[["));
        let back: GbmModel = serde_json::from_str(&json).unwrap();
        back.validate().unwrap();
        assert_eq!(back.n_iterations(), 2);
    }
}

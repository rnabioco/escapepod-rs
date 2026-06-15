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

impl GbmTree {
    /// Walk the tree on a raw feature vector and return its leaf value.
    ///
    /// `x.len()` is assumed validated against the model's `n_features` by the
    /// caller (see [`GbmPredictor::predict`]); feature indices are validated by
    /// [`GbmModel::validate`], so this hot path does no bounds re-checking
    /// beyond what the slice index already guarantees.
    #[inline]
    fn leaf_value(&self, x: &[f64]) -> f64 {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.leaf {
                return node.value;
            }
            let v = x[node.feature as usize];
            let go_left = if v.is_nan() {
                node.missing_left
            } else {
                v <= node.threshold
            };
            idx = if go_left {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }
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

/// Inference wrapper around a [`GbmModel`].
///
/// Holds only a borrow of the model — there is no per-read state to amortize
/// (no DTW workspace, no coupling matrices), so the predictor is trivially
/// `Sync` and the CLI classify loop is a plain `par_iter().map(predict)`.
#[derive(Debug)]
pub struct GbmPredictor<'a> {
    model: &'a GbmModel,
}

impl<'a> GbmPredictor<'a> {
    /// Build a predictor over a (already validated) model.
    pub fn new(model: &'a GbmModel) -> Self {
        Self { model }
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

        // raw[k] = baseline[k] + Σ_iter tree[iter][k].leaf_value(x)
        let mut raw = m.baseline.clone();
        for iter_trees in &m.trees {
            for (k, tree) in iter_trees.iter().enumerate() {
                raw[k] += tree.leaf_value(fingerprint);
            }
        }

        let probs = softmax(&raw);
        let result = process_probabilities(&probs, &m.label_mapper, m.thresholds.as_deref());
        Ok((probs, result))
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

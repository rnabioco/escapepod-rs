//! OvO (one-vs-one) SVM predictor: decision function, Platt scaling, and
//! multiclass probability coupling.

use crate::model::DtwSvmModel;
use crate::probability::{ProbabilityResult, process_probabilities};

use super::kernel::distances_to_kernel_into;
use super::workspace::SvmWorkspace;

// Re-export SvmModel as an alias for DtwSvmModel for backwards compatibility
pub type SvmModel = DtwSvmModel;

/// SVM predictor for One-vs-One multiclass classification.
///
/// Implements the decision function for sklearn's SVC with precomputed kernel.
///
/// Precomputes two index tables at construction so the per-read hot loop
/// doesn't rebuild a `HashMap<i32, usize>` on every call:
/// - `training_class[i]` — class index for each training sample `i`
/// - `sv_class[sv_local]` — class index for each support vector
pub struct SvmPredictor<'a> {
    model: &'a DtwSvmModel,
    /// Class index (0..n_classes) for each training sample, or None if the
    /// sample's label isn't in `model.classes` (malformed model — treated as
    /// "not in any class" to match the old HashMap::get behavior).
    training_class: Vec<Option<usize>>,
    /// Class index for each support vector, indexed by the support vector's
    /// local index (i.e. `sv_class[k]` is the class of `support_indices[k]`).
    sv_class: Vec<Option<usize>>,
    /// `f32` copy of `model.training_fingerprints`, converted once at
    /// construction. The DTW distance loop runs in `f32`; without this the
    /// per-read hot loop would re-cast all (n_train × fp_len) values from `f64`
    /// on every read. Constant across reads, so it belongs on the predictor.
    training_f32: Vec<Vec<f32>>,
    /// When every training fingerprint has the same length (the usual case),
    /// this is `Some(len)` and `training_blocks` holds the SoA-packed bank for
    /// the lane-parallel DTW fast path. `None` for ragged training sets, which
    /// fall back to the scalar per-fingerprint loop.
    train_uniform_len: Option<usize>,
    /// SoA-packed training bank for [`dtw_distances_batch_unconstrained`].
    /// Empty unless `train_uniform_len.is_some()`.
    training_blocks: Vec<f32>,
}

impl<'a> SvmPredictor<'a> {
    pub fn new(model: &'a SvmModel) -> Self {
        // One-time label → class-index lookup. Hot loops below index directly
        // into training_class / sv_class, so the HashMap never escapes `new`.
        let label_to_class: std::collections::HashMap<i32, usize> = model
            .classes
            .iter()
            .enumerate()
            .map(|(idx, &label)| (label, idx))
            .collect();

        let training_class: Vec<Option<usize>> = model
            .training_labels
            .iter()
            .map(|label| label_to_class.get(label).copied())
            .collect();

        let sv_class: Vec<Option<usize>> = model
            .support_indices
            .iter()
            .map(|&gidx| {
                model
                    .training_labels
                    .get(gidx)
                    .and_then(|label| label_to_class.get(label).copied())
            })
            .collect();

        let training_f32: Vec<Vec<f32>> = model
            .training_fingerprints
            .iter()
            .map(|fp| fp.iter().map(|&x| x as f32).collect())
            .collect();

        // If every training fingerprint is the same length, pre-pack the bank
        // into the SoA block layout for the lane-parallel DTW fast path.
        let train_uniform_len = match training_f32.first() {
            Some(first)
                if training_f32.iter().all(|fp| fp.len() == first.len()) && !first.is_empty() =>
            {
                Some(first.len())
            }
            _ => None,
        };
        let training_blocks = match train_uniform_len {
            Some(len) => escapepod_signal::dtw::pack_training_blocks(&training_f32, len),
            None => Vec::new(),
        };

        Self {
            model,
            training_class,
            sv_class,
            training_f32,
            train_uniform_len,
            training_blocks,
        }
    }

    /// Compute class scores using kernel-weighted voting.
    ///
    /// For each class, sum the kernel similarities to all training samples of that class.
    /// This provides a soft voting mechanism where closer samples contribute more.
    ///
    /// # Arguments
    ///
    /// * `kernel_values` - Kernel values K(query, x_i) for all training samples
    ///
    /// # Returns
    ///
    /// Score for each class (higher = more similar to that class)
    pub fn kernel_weighted_scores(&self, kernel_values: &[f64]) -> Vec<f64> {
        let n_classes = self.model.n_classes;
        let mut class_scores = vec![0.0; n_classes];
        let mut class_counts = vec![0usize; n_classes];

        // Accumulate kernel-weighted votes for each class. The class index
        // for each training sample was pre-resolved in `new()`.
        for (i, class_opt) in self.training_class.iter().enumerate() {
            if let Some(class_idx) = *class_opt {
                class_scores[class_idx] += kernel_values[i];
                class_counts[class_idx] += 1;
            }
        }

        // Normalize by class size to avoid bias toward larger classes
        for (score, count) in class_scores.iter_mut().zip(class_counts.iter()) {
            if *count > 0 {
                *score /= *count as f64;
            }
        }

        class_scores
    }

    /// Workspace-backed variant. Reuses `scores` / `counts` buffers across
    /// calls so hot-loop callers don't allocate two `Vec`s per prediction.
    #[inline]
    fn kernel_weighted_scores_into(
        &self,
        kernel_values: &[f64],
        scores: &mut Vec<f64>,
        counts: &mut Vec<usize>,
    ) {
        let n_classes = self.model.n_classes;
        scores.clear();
        scores.resize(n_classes, 0.0);
        counts.clear();
        counts.resize(n_classes, 0);

        for (i, class_opt) in self.training_class.iter().enumerate() {
            if let Some(class_idx) = *class_opt {
                scores[class_idx] += kernel_values[i];
                counts[class_idx] += 1;
            }
        }

        for (score, count) in scores.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                *score /= *count as f64;
            }
        }
    }

    /// Compute decision function for a query fingerprint.
    ///
    /// For OvO classification, this computes decision values for each pair of classes.
    /// Uses kernel-weighted voting if model.use_kernel_weighted is true.
    ///
    /// # Arguments
    ///
    /// * `kernel_values` - Kernel values K(query, x_i) for all training samples
    ///
    /// # Returns
    ///
    /// Decision values for each class pair
    pub fn decision_function(&self, kernel_values: &[f64]) -> Vec<f64> {
        let n_classes = self.model.n_classes;
        let n_pairs = n_classes * (n_classes - 1) / 2;

        // Use kernel-weighted voting if specified in model
        if self.model.use_kernel_weighted {
            let class_scores = self.kernel_weighted_scores(kernel_values);
            let mut decisions = vec![0.0; n_pairs];
            let mut pair_idx = 0;
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    decisions[pair_idx] = class_scores[i] - class_scores[j];
                    pair_idx += 1;
                }
            }
            return decisions;
        }

        // Use real SVM dual coefficients (libsvm OvO layout)
        //
        // For pairwise classifier (class_i, class_j) with i < j:
        //   - SVs of class i: use dual_coef[j-1][sv_idx]
        //   - SVs of class j: use dual_coef[i][sv_idx]
        //   - Other SVs: don't contribute
        //
        // `sv_class[sv_local_idx]` is precomputed in `new()`, so the inner
        // loop touches only Vec indexing — no HashMap probe per SV per pair.
        let mut decisions = vec![0.0; n_pairs];
        let sv_class = self.sv_class.as_slice();
        let support_indices = self.model.support_indices.as_slice();
        let intercept = self.model.intercept.as_slice();
        let dual_coef = self.model.dual_coef.as_slice();

        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                // Indexed once per pair; inner loop reads contiguous slices
                // and drops the prior `.get().copied().unwrap_or(0.0)` guard.
                let coef_i = dual_coef[j - 1].as_slice();
                let coef_j = dual_coef[i].as_slice();
                let mut sum = intercept[pair_idx];

                for (sv_local_idx, &sv_global_idx) in support_indices.iter().enumerate() {
                    let coef = match sv_class[sv_local_idx] {
                        Some(c) if c == i => coef_i[sv_local_idx],
                        Some(c) if c == j => coef_j[sv_local_idx],
                        _ => 0.0,
                    };
                    sum += coef * kernel_values[sv_global_idx];
                }

                decisions[pair_idx] = sum;
                pair_idx += 1;
            }
        }

        decisions
    }

    /// Workspace-backed variant of [`Self::decision_function`]. Reuses
    /// `ws.class_scores` / `ws.class_counts` for the kernel-weighted path so
    /// the inner hot loop avoids two `Vec<f64>` allocations per call.
    fn decision_function_into(
        &self,
        kernel_values: &[f64],
        ws: &mut SvmWorkspace,
        decisions: &mut Vec<f64>,
    ) {
        let n_classes = self.model.n_classes;
        let n_pairs = n_classes * (n_classes - 1) / 2;
        decisions.clear();
        decisions.resize(n_pairs, 0.0);

        if self.model.use_kernel_weighted {
            self.kernel_weighted_scores_into(
                kernel_values,
                &mut ws.class_scores,
                &mut ws.class_counts,
            );
            let class_scores = ws.class_scores.as_slice();
            let mut pair_idx = 0;
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    decisions[pair_idx] = class_scores[i] - class_scores[j];
                    pair_idx += 1;
                }
            }
            return;
        }

        let sv_class = self.sv_class.as_slice();
        let support_indices = self.model.support_indices.as_slice();
        let intercept = self.model.intercept.as_slice();
        let dual_coef = self.model.dual_coef.as_slice();

        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let coef_i = dual_coef[j - 1].as_slice();
                let coef_j = dual_coef[i].as_slice();
                let mut sum = intercept[pair_idx];

                for (sv_local_idx, &sv_global_idx) in support_indices.iter().enumerate() {
                    let coef = match sv_class[sv_local_idx] {
                        Some(c) if c == i => coef_i[sv_local_idx],
                        Some(c) if c == j => coef_j[sv_local_idx],
                        _ => 0.0,
                    };
                    sum += coef * kernel_values[sv_global_idx];
                }

                decisions[pair_idx] = sum;
                pair_idx += 1;
            }
        }
    }

    /// Convert OvO decision values to class votes.
    ///
    /// Each binary classifier votes for one of its two classes.
    /// The class with the most votes wins.
    ///
    /// # Arguments
    ///
    /// * `decisions` - Decision values from `decision_function`
    ///
    /// # Returns
    ///
    /// Vote count for each class
    pub fn ovo_votes(&self, decisions: &[f64]) -> Vec<i32> {
        let n_classes = self.model.n_classes;
        let mut votes = vec![0i32; n_classes];

        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                if decisions[pair_idx] > 0.0 {
                    votes[i] += 1;
                } else {
                    votes[j] += 1;
                }
                pair_idx += 1;
            }
        }

        votes
    }

    /// Convert decision values to probabilities using sigmoid + OvO voting.
    ///
    /// If Platt scaling parameters are available, uses those for calibration.
    /// Otherwise, uses a simple sigmoid and normalization.
    ///
    /// # Arguments
    ///
    /// * `decisions` - Decision values from `decision_function`
    ///
    /// # Returns
    ///
    /// Probability distribution over classes
    pub fn decision_to_probabilities(&self, decisions: &[f64]) -> Vec<f64> {
        let mut ws = SvmWorkspace::new();
        self.decision_to_probabilities_with(decisions, &mut ws)
    }

    pub(super) fn decision_to_probabilities_with(
        &self,
        decisions: &[f64],
        ws: &mut SvmWorkspace,
    ) -> Vec<f64> {
        let n_classes = self.model.n_classes;

        // Use Platt scaling if available
        if let (Some(prob_a), Some(prob_b)) = (&self.model.prob_a, &self.model.prob_b) {
            // Platt scaling: P = 1 / (1 + exp(A * f + B))
            ws.pair_probs.clear();
            ws.pair_probs.extend(
                decisions
                    .iter()
                    .zip(prob_a.iter().zip(prob_b.iter()))
                    .map(|(&f, (&a, &b))| 1.0 / (1.0 + (a * f + b).exp())),
            );

            // Aggregate pairwise probabilities to class probabilities
            // Using the coupling method from sklearn
            return self.couple_probabilities_with(ws);
        }

        // Fallback: simple sigmoid + voting (reuses ws.class_scores).
        ws.class_scores.clear();
        ws.class_scores.resize(n_classes, 0.0);
        let scores = ws.class_scores.as_mut_slice();

        let mut pair_idx = 0;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let prob_i = 1.0 / (1.0 + (-decisions[pair_idx]).exp());
                scores[i] += prob_i;
                scores[j] += 1.0 - prob_i;
                pair_idx += 1;
            }
        }

        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            scores.iter_mut().for_each(|v| *v /= sum);
        }

        ws.class_scores.clone()
    }

    /// Couple pairwise probabilities to class probabilities.
    ///
    /// Reads `ws.pair_probs` as input and reuses flat row-major `ws.r` / `ws.q`
    /// matrices (length `k * k`) plus `ws.p` / `ws.qp` vectors. Returns an
    /// owned copy of `ws.p` so the caller can keep the result independently of
    /// the workspace.
    ///
    /// Implements libsvm's `multiclass_probability` (Wu et al. 2004); matches
    /// sklearn's internal probability coupling exactly.
    #[allow(clippy::needless_range_loop)]
    fn couple_probabilities_with(&self, ws: &mut SvmWorkspace) -> Vec<f64> {
        let k = self.model.n_classes;
        let kk = k * k;

        // Flat k×k row-major layout for r and q. We hand them to slice views
        // below so the inner loops see `&[f64]` / `&mut [f64]` and can get
        // vectorized by the autovectorizer.
        ws.r.clear();
        ws.r.resize(kk, 0.0);
        let r = ws.r.as_mut_slice();

        let mut pair_idx = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                let v = ws.pair_probs[pair_idx].clamp(1e-7, 1.0 - 1e-7);
                r[i * k + j] = v;
                r[j * k + i] = 1.0 - v;
                pair_idx += 1;
            }
        }

        // Build Q matrix: Q[t][t] = sum_{j!=t} r[j][t]^2
        //                  Q[t][j] = -r[j][t] * r[t][j]  (j != t)
        ws.q.clear();
        ws.q.resize(kk, 0.0);
        let q = ws.q.as_mut_slice();
        for t in 0..k {
            let mut diag = 0.0;
            for j in 0..k {
                if j != t {
                    let rjt = r[j * k + t];
                    diag += rjt * rjt;
                    q[t * k + j] = -rjt * r[t * k + j];
                }
            }
            q[t * k + t] = diag;
        }

        // Initialize uniform probabilities
        ws.p.clear();
        ws.p.resize(k, 1.0 / k as f64);
        ws.qp.clear();
        ws.qp.resize(k, 0.0);
        let p = ws.p.as_mut_slice();
        let qp = ws.qp.as_mut_slice();
        let eps = 0.005 / k as f64;
        let max_iter = 100.max(k);

        for _ in 0..max_iter {
            // Compute Qp and pQp
            let mut p_qp = 0.0;
            for t in 0..k {
                let mut acc = 0.0;
                let q_row = &q[t * k..t * k + k];
                for j in 0..k {
                    acc += q_row[j] * p[j];
                }
                qp[t] = acc;
                p_qp += p[t] * acc;
            }

            // Check convergence
            let mut max_error = 0.0_f64;
            for t in 0..k {
                let e = (qp[t] - p_qp).abs();
                if e > max_error {
                    max_error = e;
                }
            }
            if max_error < eps {
                break;
            }

            // Update each p[t]
            for t in 0..k {
                let q_tt = q[t * k + t];
                let diff = (-qp[t] + p_qp) / q_tt;
                p[t] += diff;
                p_qp = (p_qp + diff * (diff * q_tt + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
                let scale = 1.0 / (1.0 + diff);
                let q_row = &q[t * k..t * k + k];
                for j in 0..k {
                    qp[j] = (qp[j] + diff * q_row[j]) * scale;
                    p[j] *= scale;
                }
            }
        }

        ws.p.clone()
    }

    /// Classify a query fingerprint.
    ///
    /// Full prediction pipeline:
    /// 1. Compute DTW distances
    /// 2. Convert to kernel
    /// 3. Compute decision function
    /// 4. Convert to probabilities
    /// 5. Process to get prediction and confidence
    ///
    /// # Arguments
    ///
    /// * `query` - Query fingerprint
    ///
    /// # Returns
    ///
    /// Tuple of (probabilities, prediction result)
    pub fn predict(&self, query: &[f64]) -> (Vec<f64>, ProbabilityResult) {
        // For a single predict, lazy growth costs less than pre-reserving
        // k×k coupling matrices. Hot-loop callers should share a workspace
        // via `predict_with_workspace`.
        let mut ws = SvmWorkspace::new();
        self.predict_with_workspace(query, &mut ws)
    }

    /// Workspace-backed variant of [`Self::predict`]. Reuses all scratch
    /// buffers across calls; prefer this in a tight `par_iter` by creating
    /// one workspace per rayon worker via
    /// `map_init(|| SvmWorkspace::for_model(predictor.model()), …)`.
    pub fn predict_with_workspace(
        &self,
        query: &[f64],
        ws: &mut SvmWorkspace,
    ) -> (Vec<f64>, ProbabilityResult) {
        // 1) DTW distances → ws.distances. Use the predictor's precomputed f32
        // training set (constant across reads) and the workspace's shared DTW
        // buffers, so a read scoring against n_train fingerprints neither
        // re-casts the training set nor allocates per DTW call.
        ws.query_f32.clear();
        ws.query_f32.extend(query.iter().map(|&x| x as f32));
        match self.train_uniform_len {
            // Uniform-length bank → score DTW_LANES training fingerprints per
            // SIMD batch. Handles both the unconstrained case (window=None,
            // penalty=0) and the real WarpDemuX config (Sakoe-Chiba band +
            // warping penalty); `dtw_distances_batch` fast-paths the former.
            Some(len) => {
                escapepod_signal::dtw::dtw_distances_batch(
                    &ws.query_f32,
                    &self.training_blocks,
                    len,
                    self.training_f32.len(),
                    self.model.window,
                    self.model.penalty,
                    &mut ws.distances,
                    &mut ws.dtw_batch,
                );
            }
            // Fallback: ragged bank (training fingerprints of differing length)
            // → score one fingerprint at a time, reusing the row buffers.
            None => {
                super::kernel::compute_distances_f32_into(
                    &ws.query_f32,
                    &self.training_f32,
                    self.model.window,
                    self.model.penalty,
                    &mut ws.distances,
                    &mut ws.dtw,
                );
            }
        }

        // 2) Kernel transform. Borrow-split: we need `&ws.distances` while
        // writing `ws.kernel`.
        let distances = std::mem::take(&mut ws.distances);
        distances_to_kernel_into(&distances, &self.model.kernel_params, &mut ws.kernel);
        ws.distances = distances;

        // 3) Decision function. Needs &ws.kernel plus &mut ws (for
        // kernel_weighted_scores scratch); swap kernel + decisions out.
        let kernel = std::mem::take(&mut ws.kernel);
        let mut decisions = std::mem::take(&mut ws.decisions);
        self.decision_function_into(&kernel, ws, &mut decisions);
        ws.kernel = kernel;

        // 4) Probabilities. Same borrow-split trick for decisions.
        let probabilities = self.decision_to_probabilities_with(&decisions, ws);
        ws.decisions = decisions;

        let result = process_probabilities(
            &probabilities,
            &self.model.label_mapper,
            self.model.thresholds.as_deref(),
        );

        (probabilities, result)
    }

    /// Return a reference to the underlying model. Useful for sizing
    /// [`SvmWorkspace`] via [`SvmWorkspace::for_model`] when the caller holds
    /// a predictor but not the model directly.
    pub fn model(&self) -> &DtwSvmModel {
        self.model
    }
}

/// Classify a fingerprint using a trained SVM model.
///
/// Convenience function that creates a predictor and runs prediction.
///
/// # Arguments
///
/// * `model` - Trained SVM model
/// * `fingerprint` - Query fingerprint
///
/// # Returns
///
/// Tuple of (probabilities, prediction result)
pub fn classify_with_svm(model: &SvmModel, fingerprint: &[f64]) -> (Vec<f64>, ProbabilityResult) {
    let predictor = SvmPredictor::new(model);
    predictor.predict(fingerprint)
}

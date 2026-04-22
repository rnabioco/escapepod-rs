//! SVM prediction for barcode classification.
//!
//! This module implements SVM prediction with precomputed kernel,
//! compatible with WarpDemuX's DTW-SVM models.
//!
//! The workflow is:
//! 1. Compute DTW distances from query to training fingerprints
//! 2. Convert distances to RBF kernel: K = exp(-gamma * dist^power)
//! 3. Apply SVM decision function using dual coefficients
//! 4. Convert decision values to probabilities via Platt scaling

use escapepod_signal::dtw::dtw_distance;
use rayon::prelude::*;

use crate::model::{DtwSvmModel, KernelParams};
use crate::probability::{ProbabilityResult, process_probabilities};

// Re-export SvmModel as an alias for DtwSvmModel for backwards compatibility
pub type SvmModel = DtwSvmModel;

/// Compute RBF kernel from distances.
///
/// K = exp(-gamma * distance^power)
///
/// # Arguments
///
/// * `distances` - DTW distances to training samples
/// * `params` - Kernel parameters
///
/// # Returns
///
/// Kernel values (similarity scores)
pub fn distances_to_kernel(distances: &[f64], params: &KernelParams) -> Vec<f64> {
    distances
        .iter()
        .map(|&d| (-params.gamma * d.powf(params.power)).exp())
        .collect()
}

/// In-place variant of [`distances_to_kernel`] that writes into a caller-owned
/// buffer. Avoids a `Vec<f64>` allocation per read in the SVM pipeline.
fn distances_to_kernel_into(distances: &[f64], params: &KernelParams, out: &mut Vec<f64>) {
    out.clear();
    out.extend(
        distances
            .iter()
            .map(|&d| (-params.gamma * d.powf(params.power)).exp()),
    );
}

/// Compute DTW distances from a query fingerprint to all training fingerprints.
///
/// # Arguments
///
/// * `query` - Query fingerprint
/// * `training` - Training fingerprints
/// * `window` - Optional Sakoe-Chiba band constraint
///
/// # Returns
///
/// Vector of DTW distances
pub fn compute_distances(query: &[f64], training: &[Vec<f64>], window: Option<usize>) -> Vec<f64> {
    let mut ws = SvmWorkspace::new();
    compute_distances_into(query, training, window, &mut ws);
    std::mem::take(&mut ws.distances)
}

/// Workspace-backed variant of [`compute_distances`]. Reuses f32 conversion
/// buffers and writes results into `ws.distances` (cleared first).
fn compute_distances_into(
    query: &[f64],
    training: &[Vec<f64>],
    window: Option<usize>,
    ws: &mut SvmWorkspace,
) {
    ws.query_f32.clear();
    ws.query_f32.extend(query.iter().map(|&x| x as f32));

    ws.distances.clear();
    ws.distances.reserve(training.len());
    // Split the borrow so the inner loop can write to `ws.distances` while
    // rewriting `ws.train_scratch` on each iteration.
    let query_f32 = ws.query_f32.as_slice();
    let train_scratch = &mut ws.train_scratch;
    let distances = &mut ws.distances;
    for train_fp in training {
        train_scratch.clear();
        train_scratch.extend(train_fp.iter().map(|&x| x as f32));
        distances.push(dtw_distance(query_f32, train_scratch, window) as f64);
    }
}

/// Reusable scratch buffers for the SVM prediction pipeline.
///
/// `SvmPredictor::predict` in a tight loop (e.g. `par_iter` over 100k reads)
/// otherwise allocates 6–8 `Vec<f64>`s per read — and `couple_probabilities`
/// alone allocates two `k × k` matrices and two `k`-vectors. For k = 32 classes
/// across 100k reads that's ~10M heap allocations just for coupling scratch.
///
/// Hand one workspace per rayon worker (via `par_iter().map_init`) and pass
/// `&mut` to [`SvmPredictor::predict_with_workspace`]; buffers resize up to
/// the max ever needed and stay there.
#[derive(Default, Debug, Clone)]
pub struct SvmWorkspace {
    /// f32 cast of the query, reused across training fingerprints.
    query_f32: Vec<f32>,
    /// f32 cast of the current training fingerprint (rewritten per-SV).
    train_scratch: Vec<f32>,
    /// DTW distances from query to every training fingerprint.
    distances: Vec<f64>,
    /// RBF kernel values (same length as `distances`).
    kernel: Vec<f64>,
    /// Per-pair decision values from the OvO SVM.
    decisions: Vec<f64>,
    /// Per-pair Platt-scaled probabilities (coupling input).
    pair_probs: Vec<f64>,
    /// Kernel-weighted score per class (fallback + kernel-weighted path).
    class_scores: Vec<f64>,
    /// Per-class training sample counts (kernel-weighted path).
    class_counts: Vec<usize>,
    /// Flattened `k × k` row-major pairwise probability matrix.
    r: Vec<f64>,
    /// Flattened `k × k` row-major `Q` matrix for multiclass coupling.
    q: Vec<f64>,
    /// Current probability estimate in the coupling iteration.
    p: Vec<f64>,
    /// `Q p` product, recycled across the 100+ coupling iterations.
    qp: Vec<f64>,
}

impl SvmWorkspace {
    /// Empty workspace. Buffers grow lazily as predictions are run.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a workspace sized for the shape of `model`. Avoids the first
    /// few grow-and-copy rounds when you already know the model.
    pub fn for_model(model: &DtwSvmModel) -> Self {
        let mut ws = Self::new();
        ws.reserve_for_model(model);
        ws
    }

    fn reserve_for_model(&mut self, model: &DtwSvmModel) {
        let k = model.n_classes;
        let n_train = model.training_fingerprints.len();
        let n_pairs = k * (k - 1) / 2;
        let flen = model.training_fingerprints.first().map_or(0, Vec::len);
        self.query_f32.reserve(flen);
        self.train_scratch.reserve(flen);
        self.distances.reserve(n_train);
        self.kernel.reserve(n_train);
        self.decisions.reserve(n_pairs);
        self.pair_probs.reserve(n_pairs);
        self.class_scores.reserve(k);
        self.class_counts.reserve(k);
        self.r.reserve(k * k);
        self.q.reserve(k * k);
        self.p.reserve(k);
        self.qp.reserve(k);
    }
}

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

        Self {
            model,
            training_class,
            sv_class,
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

    fn decision_to_probabilities_with(&self, decisions: &[f64], ws: &mut SvmWorkspace) -> Vec<f64> {
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
        // 1) DTW distances → ws.distances
        compute_distances_into(
            query,
            &self.model.training_fingerprints,
            self.model.window,
            ws,
        );

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

/// Classify a batch of fingerprints with an SVM model on the GPU.
///
/// Runs a single batched DTW distance-matrix kernel on the device, then runs
/// the RBF-kernel / SVM decision / Platt-scaling pipeline on CPU per query.
/// Prefer this over calling [`classify_with_svm`] in a loop when you have
/// many queries — kernel launch and NVRTC compile costs amortize across the
/// whole batch.
///
/// Default chunk budget (in matrix cells) for the GPU batch classifier.
/// 4G cells × f32 ≈ 16 GB of device distance matrix per call. Sized for
/// a 24 GB A30 — at 16 GB matrix + 2-4 GB DTW kernel scratch + driver
/// overhead, peak sits around 20-22 GB with a slim margin. Halves the
/// chunk count at the 10k-SV / 30M-query eval workload (142 -> 71
/// chunks), which matters because the per-chunk kernel launch + host-
/// device transfer overhead stops being the dominant cost.
///
/// Smaller cards (T4 at 16 GB) should drop this to 512M-1G via
/// `--gpu-chunk-cells`; bigger cards (H100/A100-80G) can push to 8G+.
pub const DEFAULT_GPU_CHUNK_CELLS: usize = 4 * 1024 * 1024 * 1024;

/// Only available with the `gpu` feature.
#[cfg(feature = "gpu")]
pub fn classify_with_svm_batch_gpu(
    model: &SvmModel,
    fingerprints: &[Vec<f64>],
) -> Result<Vec<(Vec<f64>, ProbabilityResult)>, escapepod_signal::dtw::GpuDtwError> {
    let ctx = escapepod_signal::dtw::GpuDtwContext::new()?;
    classify_with_svm_batch_gpu_with_ctx(&ctx, model, fingerprints, DEFAULT_GPU_CHUNK_CELLS)
}

/// Same as [`classify_with_svm_batch_gpu`] but reuses an existing
/// [`escapepod_signal::dtw::GpuDtwContext`] and lets callers tune the
/// per-call GPU distance-matrix budget.
///
/// `chunk_matrix_cells` caps `queries.len() × refs.len()` for each GPU
/// call. On an A30 (24 GB VRAM), `DEFAULT_GPU_CHUNK_CELLS` (4 G cells ≈
/// 16 GB f32 distance matrix) is a good starting point. Going too high
/// triggers `CUDA_ERROR_OUT_OF_MEMORY`; too low leaves throughput on the
/// table.
///
/// **Implementation:** runs the full SVM pipeline on-device — DTW kernel,
/// RBF transform, OvO decision — via [`gpu_pipeline::GpuSvmContext`].
/// Only the per-query `(n_pairs × f32)` decision vector is dtoh'd back
/// to host (~3000× less PCIe than dtoh-ing the full distance matrix).
/// The host then runs Platt sigmoid + coupling + argmax in parallel
/// across rayon workers.
#[cfg(feature = "gpu")]
pub fn classify_with_svm_batch_gpu_with_ctx(
    ctx: &escapepod_signal::dtw::GpuDtwContext,
    model: &SvmModel,
    fingerprints: &[Vec<f64>],
    chunk_matrix_cells: usize,
) -> Result<Vec<(Vec<f64>, ProbabilityResult)>, escapepod_signal::dtw::GpuDtwError> {
    if fingerprints.is_empty() {
        return Ok(Vec::new());
    }

    let n_refs = model.training_fingerprints.len();
    // Cap chunk_size at the actual input length: the persistent
    // `dist_dev` / `decisions_dev` buffers in `GpuSvmContext` are sized
    // for `max_chunk_q × n_sv`, and a 4G-cell budget on a small test
    // input would otherwise pre-alloc tens of GB of VRAM we'll never
    // touch. The min() also lets the producer's chunk loop stop early
    // (chunks() yields only one chunk).
    let chunk_size = (chunk_matrix_cells / n_refs.max(1))
        .max(1)
        .min(fingerprints.len());

    let mut svm_ctx = gpu_pipeline::GpuSvmContext::new(ctx, model, chunk_size)?;

    // Producer/consumer split. Producer runs the full GPU pipeline
    // (DTW + RBF + OvO decision) per chunk and ships the small
    // per-chunk decisions table over the channel. Consumer fans
    // per-row Platt + argmax across rayon. Channel item is now a tiny
    // `(n_q × n_pairs)` matrix instead of the giant raw distance
    // matrix, so depth 2 is essentially free (~10 MB host-buffered).
    use std::sync::mpsc::sync_channel;
    let (tx, rx) =
        sync_channel::<Result<ndarray::Array2<f32>, escapepod_signal::dtw::GpuDtwError>>(2);
    let chunks: Vec<&[Vec<f64>]> = fingerprints.chunks(chunk_size).collect();
    let n_chunks = chunks.len();

    let predictor = SvmPredictor::new(model);
    let mut out = Vec::with_capacity(fingerprints.len());
    let predictor_ref = &predictor;
    let model_ref = model;
    let n_pairs = model.n_classes * (model.n_classes - 1) / 2;

    // indicatif progress bar over chunks. Drawn to stderr so it
    // composes with snakemake's per-rule log capture (`2> {log}`),
    // and auto-suppresses to a no-op if stderr isn't a TTY (e.g.
    // piped to a file) — a per-chunk println goes out instead so
    // batch logs still record progression.
    let pb = indicatif::ProgressBar::new(n_chunks as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "[gpu classify] {bar:40.cyan/blue} {pos}/{len} chunks ({percent}%) elapsed {elapsed_precise} eta {eta_precise}",
        )
        .expect("static template")
        .progress_chars("##-"),
    );
    let pb_for_producer = pb.clone();

    std::thread::scope(|scope| -> Result<(), escapepod_signal::dtw::GpuDtwError> {
        let producer = scope.spawn(move || {
            for chunk in chunks.into_iter() {
                let result = svm_ctx.classify_chunk(chunk);
                if result.is_ok() {
                    pb_for_producer.inc(1);
                }
                if tx.send(result).is_err() {
                    break;
                }
            }
            pb_for_producer.finish_and_clear();
        });

        for _chunk_idx in 0..n_chunks {
            let decisions = match rx.recv() {
                Ok(Ok(d)) => d,
                Ok(Err(e)) => {
                    drop(rx);
                    let _ = producer.join();
                    return Err(e);
                }
                Err(_) => break,
            };

            // Per-row host work: Platt sigmoid → coupling → argmax. The
            // RBF + OvO decision used to be here too; now they live on
            // the GPU side and `decisions` already carries the per-pair
            // f32 values straight from the device.
            let chunk_results: Vec<(Vec<f64>, ProbabilityResult)> = (0..decisions.nrows())
                .into_par_iter()
                .map_init(
                    || SvmWorkspace::for_model(model_ref),
                    |ws, i| {
                        let row = decisions.row(i);
                        ws.decisions.clear();
                        ws.decisions.extend(row.iter().map(|&d| d as f64));

                        let mut decisions_buf = std::mem::take(&mut ws.decisions);
                        let probabilities =
                            predictor_ref.decision_to_probabilities_with(&decisions_buf, ws);
                        decisions_buf.clear();
                        ws.decisions = decisions_buf;

                        let result = process_probabilities(
                            &probabilities,
                            &model_ref.label_mapper,
                            model_ref.thresholds.as_deref(),
                        );
                        (probabilities, result)
                    },
                )
                .collect();
            out.extend(chunk_results);

            debug_assert_eq!(decisions.ncols(), n_pairs);
        }

        let _ = producer.join();
        Ok(())
    })?;

    Ok(out)
}

/// On-device SVM classification pipeline.
///
/// Builds the persistent device-side SVM state once (refs uploaded,
/// flat OvO coefficient table, intercepts, output buffers sized for the
/// max chunk we'll see) and then runs `classify_chunk` per query batch,
/// keeping the full distance + kernel + decision pipeline on the GPU.
/// Only `(n_q × n_pairs) × f32` per chunk crosses PCIe back to host
/// — vs the prior `(n_q × n_sv) × f32` distance matrix dtoh, which was
/// the dominant per-chunk cost (~640 ms PCIe at 16 GB/chunk on
/// A30 / Gen4 ×16).
#[cfg(feature = "gpu")]
pub mod gpu_pipeline {
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use ndarray::Array2;

    use escapepod_signal::dtw::{
        DTW_KERNEL_NAME, DTW_MODULE_NAME, GpuDtwContext, GpuDtwError, OVO_DECISION_KERNEL_NAME,
        RBF_KERNEL_NAME, SVM_MODULE_NAME,
    };

    use super::DtwSvmModel;

    /// Persistent device-side state for the on-GPU SVM pipeline.
    pub struct GpuSvmContext<'a> {
        ctx: &'a GpuDtwContext,
        device: Arc<CudaDevice>,
        // Pre-loaded kernels.
        dtw_func: cudarc::driver::CudaFunction,
        rbf_func: cudarc::driver::CudaFunction,
        decision_func: cudarc::driver::CudaFunction,
        // Persistent device buffers — uploaded / sized at construction.
        refs_dev: CudaSlice<f32>,
        r_off_dev: CudaSlice<i32>,
        coef_dev: CudaSlice<f32>,        // (n_pairs × n_sv) row-major
        intercept_dev: CudaSlice<f32>,   // (n_pairs)
        dist_dev: CudaSlice<f32>,        // (max_chunk_q × n_sv) reused per call
        decisions_dev: CudaSlice<f32>,   // (max_chunk_q × n_pairs) reused per call
        // Sizing.
        n_sv: usize,
        n_pairs: usize,
        max_m: usize,         // max ref length (in samples)
        max_chunk_q: usize,   // capacity of dist/decisions buffers
        window: Option<usize>,
        gamma: f32,
        power: f32,
    }

    impl<'a> GpuSvmContext<'a> {
        /// Build the device-side state for a given model.
        ///
        /// `max_chunk_q` is the upper bound on `queries.len()` for any
        /// future `classify_chunk` call — used to pre-allocate the
        /// persistent `dist` and `decisions` buffers (the biggest device
        /// allocations in the pipeline). Pass the chunk size you'd
        /// previously have computed from `chunk_matrix_cells / n_sv`.
        pub fn new(
            ctx: &'a GpuDtwContext,
            model: &DtwSvmModel,
            max_chunk_q: usize,
        ) -> Result<Self, GpuDtwError> {
            let device = ctx.device().clone();

            let dtw_func = ctx.function(DTW_MODULE_NAME, DTW_KERNEL_NAME)?;
            let rbf_func = ctx.function(SVM_MODULE_NAME, RBF_KERNEL_NAME)?;
            let decision_func = ctx.function(SVM_MODULE_NAME, OVO_DECISION_KERNEL_NAME)?;

            // Refs (training_fingerprints) — flatten + upload once.
            let n_sv = model.training_fingerprints.len();
            let mut flat_r: Vec<f32> = Vec::with_capacity(n_sv * 32);
            let mut r_off: Vec<i32> = Vec::with_capacity(n_sv + 1);
            r_off.push(0);
            let mut max_m = 0usize;
            for fp in &model.training_fingerprints {
                flat_r.extend(fp.iter().map(|&x| x as f32));
                let len = fp.len();
                if len > max_m {
                    max_m = len;
                }
                r_off.push(flat_r.len() as i32);
            }
            let refs_dev = device.htod_sync_copy(&flat_r)?;
            let r_off_dev = device.htod_sync_copy(&r_off)?;

            // Build the flat OvO coef table on host. Size: n_pairs × n_sv.
            //
            // Two source modes (mirrors the host predictor at svm.rs:296-352):
            //   - `use_kernel_weighted`: reproduce the kernel-weighted
            //     scoring path — `decision[pair(i,j)] = (Σ_{sv∈class i}
            //     K) / count[i] - (Σ_{sv∈class j} K) / count[j]`. Encoded
            //     as `coef[pair][sv] = +1/count[i]` for class-i SVs,
            //     `-1/count[j]` for class-j SVs, 0 otherwise.
            //   - real OvO dual coefficients: `coef[pair(i,j)][sv] =
            //     dual_coef[j-1][sv]` for class-i SVs, `dual_coef[i][sv]`
            //     for class-j SVs (libsvm layout).
            let n_classes = model.n_classes;
            let n_pairs = n_classes * (n_classes - 1) / 2;

            // Resolve sv_class once — same logic as `SvmPredictor::new`.
            let label_to_class: std::collections::HashMap<i32, usize> = model
                .classes
                .iter()
                .enumerate()
                .map(|(idx, &label)| (label, idx))
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

            let mut coef_flat = vec![0.0f32; n_pairs * n_sv];
            let mut intercept_f32 = vec![0.0f32; n_pairs];

            if model.use_kernel_weighted {
                // Per-class counts for the normalization.
                let mut counts = vec![0usize; n_classes];
                for c in &sv_class {
                    if let Some(idx) = c {
                        counts[*idx] += 1;
                    }
                }
                let mut pair_idx = 0;
                for i in 0..n_classes {
                    for j in (i + 1)..n_classes {
                        let coef_i = if counts[i] > 0 {
                            1.0 / counts[i] as f32
                        } else {
                            0.0
                        };
                        let coef_j = if counts[j] > 0 {
                            1.0 / counts[j] as f32
                        } else {
                            0.0
                        };
                        let row = &mut coef_flat[pair_idx * n_sv..(pair_idx + 1) * n_sv];
                        for (sv_local, c) in sv_class.iter().enumerate() {
                            row[sv_local] = match *c {
                                Some(cl) if cl == i => coef_i,
                                Some(cl) if cl == j => -coef_j,
                                _ => 0.0,
                            };
                        }
                        pair_idx += 1;
                    }
                }
                // Intercept stays 0 in this mode.
            } else {
                // libsvm OvO dual layout.
                let dual = model.dual_coef.as_slice();
                let mut pair_idx = 0;
                for i in 0..n_classes {
                    for j in (i + 1)..n_classes {
                        let coef_i_row = &dual[j - 1];
                        let coef_j_row = &dual[i];
                        let row = &mut coef_flat[pair_idx * n_sv..(pair_idx + 1) * n_sv];
                        for (sv_local, c) in sv_class.iter().enumerate() {
                            row[sv_local] = match *c {
                                Some(cl) if cl == i => coef_i_row[sv_local] as f32,
                                Some(cl) if cl == j => coef_j_row[sv_local] as f32,
                                _ => 0.0,
                            };
                        }
                        intercept_f32[pair_idx] = model.intercept[pair_idx] as f32;
                        pair_idx += 1;
                    }
                }
            }

            let coef_dev = device.htod_sync_copy(&coef_flat)?;
            let intercept_dev = device.htod_sync_copy(&intercept_f32)?;

            // Persistent output buffers sized for the maximum chunk.
            let dist_dev = device.alloc_zeros::<f32>(max_chunk_q * n_sv)?;
            let decisions_dev = device.alloc_zeros::<f32>(max_chunk_q * n_pairs)?;

            Ok(Self {
                ctx,
                device,
                dtw_func,
                rbf_func,
                decision_func,
                refs_dev,
                r_off_dev,
                coef_dev,
                intercept_dev,
                dist_dev,
                decisions_dev,
                n_sv,
                n_pairs,
                max_m,
                max_chunk_q,
                window: model.window,
                gamma: model.kernel_params.gamma as f32,
                power: model.kernel_params.power as f32,
            })
        }

        /// Run the full GPU pipeline for one query chunk and return the
        /// `(n_q × n_pairs)` decisions matrix.
        ///
        /// `queries` must have `<= self.max_chunk_q` rows; otherwise the
        /// pre-allocated `dist_dev` / `decisions_dev` buffers wouldn't
        /// hold the result.
        pub fn classify_chunk(
            &mut self,
            queries: &[Vec<f64>],
        ) -> Result<Array2<f32>, GpuDtwError> {
            let n_q = queries.len();
            if n_q == 0 {
                return Ok(Array2::zeros((0, self.n_pairs)));
            }
            assert!(
                n_q <= self.max_chunk_q,
                "classify_chunk called with n_q={n_q} > max_chunk_q={}",
                self.max_chunk_q
            );

            // Flatten + upload queries.
            let mut flat_q: Vec<f32> = Vec::with_capacity(n_q * 32);
            let mut q_off: Vec<i32> = Vec::with_capacity(n_q + 1);
            q_off.push(0);
            let mut max_n = 0usize;
            for fp in queries {
                flat_q.extend(fp.iter().map(|&x| x as f32));
                if fp.len() > max_n {
                    max_n = fp.len();
                }
                q_off.push(flat_q.len() as i32);
            }
            let queries_dev = self.device.htod_sync_copy(&flat_q)?;
            let q_off_dev = self.device.htod_sync_copy(&q_off)?;

            if max_n > i32::MAX as usize || self.max_m > i32::MAX as usize {
                return Err(GpuDtwError::InputTooLarge {
                    what: "fingerprint length exceeds i32::MAX",
                });
            }

            // ---- DTW kernel ---------------------------------------------------
            // Same launch params as `GpuDtwContext::distance_matrix` — we
            // mirror the shared-memory math here because we're driving the
            // kernel directly through the shared `CudaFunction` handle.
            let max_n_u = max_n as u32;
            let max_m_u = self.max_m as u32;
            let floats: u32 = max_n_u
                .checked_add(max_m_u)
                .and_then(|v| v.checked_add(3u32.checked_mul(max_n_u.saturating_add(1))?))
                .ok_or(GpuDtwError::InputTooLarge {
                    what: "shared memory size overflowed u32",
                })?;
            let shared_mem_bytes: u32 = floats
                .checked_mul(std::mem::size_of::<f32>() as u32)
                .ok_or(GpuDtwError::InputTooLarge {
                    what: "shared memory size overflowed u32",
                })?;

            let window_i32: i32 = match self.window {
                None => -1,
                Some(w) if w > i32::MAX as usize => {
                    return Err(GpuDtwError::InputTooLarge {
                        what: "window exceeds i32::MAX",
                    });
                }
                Some(w) => w as i32,
            };

            let dtw_cfg = LaunchConfig {
                grid_dim: (n_q as u32, self.n_sv as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes,
            };

            unsafe {
                self.dtw_func.clone().launch(
                    dtw_cfg,
                    (
                        &queries_dev,
                        &q_off_dev,
                        &self.refs_dev,
                        &self.r_off_dev,
                        &mut self.dist_dev,
                        n_q as i32,
                        self.n_sv as i32,
                        max_n as i32,
                        self.max_m as i32,
                        window_i32,
                    ),
                )?;
            }

            // ---- RBF in-place ---------------------------------------------------
            let n_cells_i64 = (n_q as i64) * (self.n_sv as i64);
            let rbf_block: u32 = 256;
            let rbf_grid_max: u32 = 65535; // safe portable cap
            let needed_blocks = ((n_cells_i64 + rbf_block as i64 - 1) / rbf_block as i64) as u32;
            let rbf_grid: u32 = needed_blocks.min(rbf_grid_max);
            let rbf_cfg = LaunchConfig {
                grid_dim: (rbf_grid, 1, 1),
                block_dim: (rbf_block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.rbf_func.clone().launch(
                    rbf_cfg,
                    (&mut self.dist_dev, n_cells_i64, self.gamma, self.power),
                )?;
            }

            // ---- OvO decision ---------------------------------------------------
            let dec_cfg = LaunchConfig {
                grid_dim: (n_q as u32, self.n_pairs as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.decision_func.clone().launch(
                    dec_cfg,
                    (
                        &self.dist_dev,
                        n_q as i32,
                        self.n_sv as i32,
                        self.n_pairs as i32,
                        &self.coef_dev,
                        &self.intercept_dev,
                        &mut self.decisions_dev,
                    ),
                )?;
            }

            // ---- dtoh just the decisions table -----------------------------
            // Note: `dtoh_sync_copy` copies the *whole* slice. We sized
            // `decisions_dev` for `max_chunk_q × n_pairs` so a partial
            // chunk would copy stale tail. Slice via a temporary view
            // around the relevant prefix.
            let total_out = n_q * self.n_pairs;
            let mut host_out = vec![0.0f32; total_out];
            // Synchronous copy of just the first `total_out` elements.
            // cudarc's CudaSlice supports slicing via `.try_clone()` +
            // index, but the simplest path is a sized helper:
            let decisions_view = self.decisions_dev.slice(0..total_out);
            self.device.dtoh_sync_copy_into(&decisions_view, &mut host_out)?;

            Ok(Array2::from_shape_vec((n_q, self.n_pairs), host_out)
                .expect("shape matches by construction"))
        }
    }

    // Suppress unused-field warnings — the fields are part of the device
    // ownership story even though Rust doesn't see them read.
    #[allow(dead_code)]
    fn _keep_alive_check<'a>(c: &GpuSvmContext<'a>) {
        let _ = (
            &c.ctx,
            &c.device,
            &c.refs_dev,
            &c.r_off_dev,
            &c.coef_dev,
            &c.intercept_dev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_model() -> DtwSvmModel {
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 4);
        label_mapper.insert(1, 5);
        label_mapper.insert(2, 6);

        DtwSvmModel {
            version: "1.0".to_string(),
            training_fingerprints: vec![
                vec![0.0, 0.0, 0.0], // Class 0
                vec![1.0, 1.0, 1.0], // Class 1
                vec![2.0, 2.0, 2.0], // Class 2
            ],
            training_labels: vec![4, 5, 6],
            support_indices: vec![0, 1, 2],
            dual_coef: vec![vec![1.0, -1.0, 0.5], vec![-0.5, 0.5, 1.0]],
            intercept: vec![0.0, 0.0, 0.0],
            classes: vec![4, 5, 6],
            kernel_params: KernelParams::default(),
            window: None,
            label_mapper,
            thresholds: None,
            prob_a: None,
            prob_b: None,
            n_classes: 3,
            noise_class: false,
            use_kernel_weighted: false,
        }
    }

    #[test]
    fn test_distances_to_kernel() {
        let distances = vec![0.0, 1.0, 2.0];
        let params = KernelParams::default();
        let kernel = distances_to_kernel(&distances, &params);

        assert!((kernel[0] - 1.0).abs() < 1e-10); // exp(0) = 1
        assert!((kernel[1] - (-1.0f64).exp()).abs() < 1e-10);
        assert!((kernel[2] - (-2.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_distances() {
        let query = vec![0.0, 0.0, 0.0];
        let training = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];

        let distances = compute_distances(&query, &training, None);

        assert!(distances[0] < 0.1); // Same vector, distance ~0
        assert!(distances[1] > distances[0]); // Different vector, larger distance
    }

    #[test]
    fn test_svm_predictor_decision() {
        let model = create_test_model();
        let predictor = SvmPredictor::new(&model);

        // Kernel values that strongly favor class 0
        let kernel_values = vec![1.0, 0.1, 0.05];
        let decisions = predictor.decision_function(&kernel_values);

        // Should have n_classes * (n_classes - 1) / 2 = 3 decision values
        assert_eq!(decisions.len(), 3);
    }

    #[test]
    fn test_svm_predictor_probabilities() {
        let model = create_test_model();
        let predictor = SvmPredictor::new(&model);

        let kernel_values = vec![1.0, 0.1, 0.05];
        let decisions = predictor.decision_function(&kernel_values);
        let probs = predictor.decision_to_probabilities(&decisions);

        // Should have n_classes probabilities
        assert_eq!(probs.len(), 3);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_with_svm() {
        let model = create_test_model();

        // Query close to class 0 training point
        let query = vec![0.1, 0.1, 0.1];
        let (probs, result) = classify_with_svm(&model, &query);

        assert_eq!(probs.len(), 3);
        assert!(result.is_confident);
    }

    #[test]
    fn test_kernel_weighted_scores() {
        let model = create_test_model();
        let predictor = SvmPredictor::new(&model);

        // Kernel values: high for class 0 (index 0), low for others
        // Training: sample 0 = class 4, sample 1 = class 5, sample 2 = class 6
        let kernel_values = vec![0.9, 0.1, 0.05];
        let scores = predictor.kernel_weighted_scores(&kernel_values);

        println!("Kernel values: {:?}", kernel_values);
        println!("Scores: {:?}", scores);
        println!("Training labels: {:?}", model.training_labels);
        println!("Classes: {:?}", model.classes);

        // Class 0 (barcode 4) should have highest score
        assert!(scores[0] > scores[1], "Class 0 should beat class 1");
        assert!(scores[0] > scores[2], "Class 0 should beat class 2");
    }

    #[test]
    fn test_classify_different_classes() {
        // Create a model with clear class separation
        let mut label_mapper = HashMap::new();
        label_mapper.insert(0, 0);
        label_mapper.insert(1, 1);
        label_mapper.insert(2, 2);

        let model = DtwSvmModel {
            version: "1.0".to_string(),
            training_fingerprints: vec![
                vec![0.1, 0.1, 0.1], // Class 0
                vec![0.9, 0.9, 0.9], // Class 1
                vec![0.5, 0.5, 0.5], // Class 2
            ],
            training_labels: vec![0, 1, 2],
            support_indices: vec![0, 1, 2],
            dual_coef: vec![vec![0.333, 0.333, 0.333], vec![0.333, 0.333, 0.333]],
            intercept: vec![0.0, 0.0, 0.0],
            classes: vec![0, 1, 2],
            kernel_params: KernelParams::default(),
            window: None,
            label_mapper,
            thresholds: None,
            prob_a: None,
            prob_b: None,
            n_classes: 3,
            noise_class: false,
            use_kernel_weighted: true, // Use kernel-weighted voting
        };

        // Query close to class 0
        let query0 = vec![0.12, 0.12, 0.12];
        let (probs0, result0) = classify_with_svm(&model, &query0);
        println!(
            "Query near class 0: probs={:?}, predicted={}",
            probs0, result0.predicted_barcode
        );
        assert_eq!(result0.predicted_barcode, 0, "Should predict class 0");

        // Query close to class 1
        let query1 = vec![0.88, 0.88, 0.88];
        let (probs1, result1) = classify_with_svm(&model, &query1);
        println!(
            "Query near class 1: probs={:?}, predicted={}",
            probs1, result1.predicted_barcode
        );
        assert_eq!(result1.predicted_barcode, 1, "Should predict class 1");

        // Query close to class 2
        let query2 = vec![0.52, 0.52, 0.52];
        let (probs2, result2) = classify_with_svm(&model, &query2);
        println!(
            "Query near class 2: probs={:?}, predicted={}",
            probs2, result2.predicted_barcode
        );
        assert_eq!(result2.predicted_barcode, 2, "Should predict class 2");
    }
}

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
/// 512M cells × f32 ≈ 2 GB of device distance matrix per call. Fits
/// comfortably on a 24 GB A30 with ~22 GB of headroom for DTW kernel
/// scratch. Smaller cards (T4 at 16 GB, older K80) may need to drop
/// this; bigger cards (H100/A100 80G) can go 2–4× larger. Surfaced as
/// a tunable parameter on the per-ctx entry point so callers can size
/// it from a CLI flag or runtime device-memory query.
pub const DEFAULT_GPU_CHUNK_CELLS: usize = 512 * 1024 * 1024;

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
/// call — the kernel allocates that many f32 cells for the result plus
/// scratch for the DTW kernel. On an A30 (24 GB VRAM) `512 * 1024 * 1024`
/// is a good starting point (`DEFAULT_GPU_CHUNK_CELLS`). Going too high
/// triggers `CUDA_ERROR_OUT_OF_MEMORY`; too low leaves throughput on the
/// table (kernel launch + host↔device copy overhead amortizes poorly).
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

    // Convert refs once (shared across all chunks).
    let refs_f32: Vec<Vec<f32>> = model
        .training_fingerprints
        .iter()
        .map(|fp| fp.iter().map(|&x| x as f32).collect())
        .collect();

    // Chunk queries to fit the per-call `Array2<f32>` distance matrix in
    // GPU VRAM. Previously this was one shot (`distance_matrix(all_queries,
    // all_refs)`) which OOM'd on realistic eval sets — 17M queries × 10k
    // refs × 4 bytes is ~680 GB, and the A30 has 24 GB VRAM.
    let chunk_size = (chunk_matrix_cells / refs_f32.len().max(1)).max(1);

    // Producer/consumer split:
    //   producer thread: GPU distance_matrix(chunk_queries, refs)
    //   main thread:     CPU kernel + decision_function + probabilities
    //
    // Without this overlap the GPU sat idle during per-row post-processing
    // and the CPU sat idle during GPU compute. A bounded channel of
    // depth 1 (sync_channel capacity = 1) means the producer can be one
    // chunk ahead — enough to fully amortize the pipeline bubble without
    // letting GPU results pile up in host memory (each result is up to
    // chunk_size × n_refs × f32 = chunk_matrix_cells × 4 bytes).
    use std::sync::mpsc::sync_channel;
    let (tx, rx) = sync_channel::<Result<ndarray::Array2<f32>, escapepod_signal::dtw::GpuDtwError>>(
        1,
    );
    let chunks: Vec<&[Vec<f64>]> = fingerprints.chunks(chunk_size).collect();
    let n_chunks = chunks.len();
    let refs_for_producer = &refs_f32;
    let window = model.window;

    let predictor = SvmPredictor::new(model);
    let mut ws = SvmWorkspace::for_model(model);
    let mut out = Vec::with_capacity(fingerprints.len());

    std::thread::scope(|scope| -> Result<(), escapepod_signal::dtw::GpuDtwError> {
        // Producer thread: walks chunks in order, runs GPU DTW, ships
        // each Array2<f32> over the channel. If a chunk fails, the error
        // is forwarded so the consumer can short-circuit.
        let producer = scope.spawn(move || {
            for chunk in chunks {
                let queries_f32: Vec<Vec<f32>> = chunk
                    .iter()
                    .map(|fp| fp.iter().map(|&x| x as f32).collect())
                    .collect();
                let result = ctx.distance_matrix(&queries_f32, refs_for_producer, window);
                // Receiver hung up means consumer is bailing on an
                // earlier error; quietly stop.
                if tx.send(result).is_err() {
                    break;
                }
            }
            // Implicit drop(tx) closes the channel on producer exit so
            // the consumer's recv() loop terminates.
        });

        // Consumer (this thread): pulls finished distance matrices in
        // order and runs the per-row CPU pipeline against them. While
        // we're crunching chunk N, the producer is uploading chunk N+1.
        for _chunk_idx in 0..n_chunks {
            let dist = match rx.recv() {
                Ok(Ok(d)) => d,
                Ok(Err(e)) => {
                    // Drain remaining sends so the producer can exit
                    // cleanly, then propagate.
                    drop(rx);
                    let _ = producer.join();
                    return Err(e);
                }
                Err(_) => break, // producer exited / channel closed
            };

            for i in 0..dist.nrows() {
                let row = dist.row(i);
                ws.kernel.clear();
                ws.kernel.extend(row.iter().map(|&d| {
                    let d = d as f64;
                    (-model.kernel_params.gamma * d.powf(model.kernel_params.power)).exp()
                }));

                let kernel = std::mem::take(&mut ws.kernel);
                let mut decisions = std::mem::take(&mut ws.decisions);
                predictor.decision_function_into(&kernel, &mut ws, &mut decisions);
                ws.kernel = kernel;

                let probabilities =
                    predictor.decision_to_probabilities_with(&decisions, &mut ws);
                ws.decisions = decisions;

                let result = process_probabilities(
                    &probabilities,
                    &model.label_mapper,
                    model.thresholds.as_deref(),
                );
                out.push((probabilities, result));
            }
        }

        let _ = producer.join();
        Ok(())
    })?;

    Ok(out)
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

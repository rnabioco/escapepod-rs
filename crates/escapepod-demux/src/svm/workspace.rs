//! Reusable scratch buffers for the SVM prediction pipeline.

use crate::model::DtwSvmModel;

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
///
/// [`SvmPredictor::predict_with_workspace`]: super::SvmPredictor::predict_with_workspace
#[derive(Default, Debug, Clone)]
pub struct SvmWorkspace {
    /// f32 cast of the query, reused across training fingerprints.
    pub(super) query_f32: Vec<f32>,
    /// f32 cast of the current training fingerprint (rewritten per-SV).
    pub(super) train_scratch: Vec<f32>,
    /// DTW distances from query to every training fingerprint.
    pub(super) distances: Vec<f64>,
    /// RBF kernel values (same length as `distances`).
    pub(super) kernel: Vec<f64>,
    /// Per-pair decision values from the OvO SVM.
    pub(super) decisions: Vec<f64>,
    /// Per-pair Platt-scaled probabilities (coupling input).
    pub(super) pair_probs: Vec<f64>,
    /// Kernel-weighted score per class (fallback + kernel-weighted path).
    pub(super) class_scores: Vec<f64>,
    /// Per-class training sample counts (kernel-weighted path).
    pub(super) class_counts: Vec<usize>,
    /// Flattened `k × k` row-major pairwise probability matrix.
    pub(super) r: Vec<f64>,
    /// Flattened `k × k` row-major `Q` matrix for multiclass coupling.
    pub(super) q: Vec<f64>,
    /// Current probability estimate in the coupling iteration.
    pub(super) p: Vec<f64>,
    /// `Q p` product, recycled across the 100+ coupling iterations.
    pub(super) qp: Vec<f64>,
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

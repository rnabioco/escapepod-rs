//! RBF kernel computation and DTW distance helpers for the SVM pipeline.

use escapepod_signal::dtw::dtw_distance_bounded_penalty_into;

use crate::model::KernelParams;

use super::workspace::SvmWorkspace;

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
    distances_to_kernel_iter(distances, params).collect()
}

/// In-place variant of [`distances_to_kernel`] that writes into a caller-owned
/// buffer. Avoids a `Vec<f64>` allocation per read in the SVM pipeline.
pub(super) fn distances_to_kernel_into(
    distances: &[f64],
    params: &KernelParams,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.extend(distances_to_kernel_iter(distances, params));
}

/// Shared RBF-kernel mapping backing both [`distances_to_kernel`] and
/// [`distances_to_kernel_into`], so the `exp(-gamma * d^power)` map lives once.
fn distances_to_kernel_iter<'a>(
    distances: &'a [f64],
    params: &'a KernelParams,
) -> impl Iterator<Item = f64> + 'a {
    distances.iter().map(move |&d| kernel_value(d, params))
}

/// Compute one RBF kernel value: `exp(-gamma * distance^power)`.
///
/// Specializes the two common cases (`power == 1.0`, the default; and
/// `power == 2.0`, classic RBF) to skip the transcendental `f64::powf`.
/// `powf(1.0)` is the WarpDemuX default and shows up in every per-(read
/// × support-vector) kernel evaluation.
#[inline]
fn kernel_value(distance: f64, params: &KernelParams) -> f64 {
    let scaled = if params.power == 1.0 {
        distance
    } else if params.power == 2.0 {
        distance * distance
    } else {
        distance.powf(params.power)
    };
    (-params.gamma * scaled).exp()
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
pub fn compute_distances(
    query: &[f64],
    training: &[Vec<f64>],
    window: Option<usize>,
    penalty: f32,
) -> Vec<f64> {
    let mut ws = SvmWorkspace::new();
    compute_distances_into(query, training, window, penalty, &mut ws);
    std::mem::take(&mut ws.distances)
}

/// Workspace-backed variant of [`compute_distances`]. Reuses f32 conversion
/// buffers and writes results into `ws.distances` (cleared first).
pub(super) fn compute_distances_into(
    query: &[f64],
    training: &[Vec<f64>],
    window: Option<usize>,
    penalty: f32,
    ws: &mut SvmWorkspace,
) {
    ws.query_f32.clear();
    ws.query_f32.extend(query.iter().map(|&x| x as f32));

    ws.distances.clear();
    ws.distances.reserve(training.len());
    // Split the borrow so the inner loop can write to `ws.distances` while
    // rewriting `ws.train_scratch` and reusing the shared DTW row buffers.
    let query_f32 = ws.query_f32.as_slice();
    let train_scratch = &mut ws.train_scratch;
    let distances = &mut ws.distances;
    let dtw = &mut ws.dtw;
    for train_fp in training {
        train_scratch.clear();
        train_scratch.extend(train_fp.iter().map(|&x| x as f32));
        distances.push(dtw_distance_bounded_penalty_into(
            query_f32,
            train_scratch,
            window,
            f32::INFINITY,
            penalty,
            dtw,
        ) as f64);
    }
}

/// Compute DTW distances from an already-`f32` query to a bank of already-`f32`
/// training fingerprints, reusing the workspace's row buffers. This skips the
/// per-call `f64 -> f32` reconversion of the training set — the training
/// fingerprints are constant across reads, so the caller converts them once.
pub(super) fn compute_distances_f32_into(
    query_f32: &[f32],
    training_f32: &[Vec<f32>],
    window: Option<usize>,
    penalty: f32,
    distances: &mut Vec<f64>,
    dtw: &mut escapepod_signal::dtw::DtwScratch,
) {
    distances.clear();
    distances.reserve(training_f32.len());
    for train_fp in training_f32 {
        distances.push(dtw_distance_bounded_penalty_into(
            query_f32,
            train_fp,
            window,
            f32::INFINITY,
            penalty,
            dtw,
        ) as f64);
    }
}

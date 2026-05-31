//! Batched GPU SVM classification (opt-in `gpu` feature).
//!
//! Runs a single batched DTW distance-matrix kernel on the device, then runs
//! the RBF-kernel / SVM decision / Platt-scaling pipeline on the GPU per
//! chunk and finishes coupling + argmax on the host.

#[cfg(feature = "gpu")]
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use super::{SvmModel, SvmPredictor, SvmWorkspace};
#[cfg(feature = "gpu")]
use crate::probability::{ProbabilityResult, process_probabilities};

/// Default chunk budget (in matrix cells) for the GPU batch classifier.
///
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

/// Classify a batch of fingerprints with an SVM model on the GPU.
///
/// Runs a single batched DTW distance-matrix kernel on the device, then runs
/// the RBF-kernel / SVM decision / Platt-scaling pipeline on CPU per query.
/// Prefer this over calling [`classify_with_svm`] in a loop when you have
/// many queries — kernel launch and NVRTC compile costs amortize across the
/// whole batch.
///
/// Only available with the `gpu` feature.
///
/// [`classify_with_svm`]: super::classify_with_svm
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

    use crate::model::DtwSvmModel;

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
        coef_dev: CudaSlice<f32>,      // (n_pairs × n_sv) row-major
        intercept_dev: CudaSlice<f32>, // (n_pairs)
        dist_dev: CudaSlice<f32>,      // (max_chunk_q × n_sv) reused per call
        decisions_dev: CudaSlice<f32>, // (max_chunk_q × n_pairs) reused per call
        // Sizing.
        n_sv: usize,
        n_pairs: usize,
        max_m: usize,       // max ref length (in samples)
        max_chunk_q: usize, // capacity of dist/decisions buffers
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
            // Two source modes (mirrors the host predictor):
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
                for idx in sv_class.iter().flatten() {
                    counts[*idx] += 1;
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
        pub fn classify_chunk(&mut self, queries: &[Vec<f64>]) -> Result<Array2<f32>, GpuDtwError> {
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
            self.device
                .dtoh_sync_copy_into(&decisions_view, &mut host_out)?;

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

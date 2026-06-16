//! GPU boundary-CNN inference via onnxruntime (CUDA execution provider).
//!
//! Opt-in via the `cnn-gpu` feature. Mirrors [`AdapterCnn`](crate::AdapterCnn)
//! but runs the ONNX graph through onnxruntime with the CUDA execution
//! provider, in batches. Preprocessing (`prep_adapter_signal`) and decoding
//! (`decode_adapter_end`) are the *same shared helpers* the CPU tract path
//! uses, so results match bit-for-bit modulo float reassociation across runtimes
//! (well below the argmax granularity in practice; a parity test guards it).
//!
//! `load-dynamic`: onnxruntime is dlopened at runtime rather than linked at
//! build time. Point `ORT_DYLIB_PATH` at a CUDA-enabled `libonnxruntime.so`
//! and ensure a CUDA device + cuDNN are visible. If the CUDA EP cannot be
//! initialized, onnxruntime falls back to CPU — which would be slow but
//! correct.

use std::path::Path;
use std::sync::Mutex;

use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;

use crate::adapter_cnn::{decode_adapter_end, group_by_len, prep_adapter_signal};
use crate::{AdapterCnnConfig, AdapterCnnError};

/// Batched boundary-CNN adapter-end detector backed by onnxruntime + CUDA.
///
/// `ort::Session::run` takes `&mut self`, so the session sits behind a `Mutex`:
/// callers share `&AdapterCnnGpu` across rayon workers (parallel decode/prep),
/// and the actual GPU `run` calls serialize on the lock — which is what we want
/// anyway, since there's one device.
pub struct AdapterCnnGpu {
    session: Mutex<Session>,
    config: AdapterCnnConfig,
}

impl AdapterCnnGpu {
    /// Load an ONNX model with the default (ADAPTed/rna004) preprocessing config.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AdapterCnnError> {
        Self::load_with_config(path, AdapterCnnConfig::default())
    }

    /// Load with an explicit preprocessing config, registering the CUDA EP.
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: AdapterCnnConfig,
    ) -> Result<Self, AdapterCnnError> {
        let session = Session::builder()
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| AdapterCnnError::Load(e.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    /// Preprocessing config in effect.
    pub fn config(&self) -> AdapterCnnConfig {
        self.config
    }

    /// Batched adapter-end detection. Same contract and the *same* bit-exact
    /// length-grouping as
    /// [`AdapterCnn::detect_adapter_end_batch`](crate::AdapterCnn::detect_adapter_end_batch):
    /// one result per input signal in order, too-short reads excluded, and each
    /// exact-length group run as an unpadded `[group, 1, len]` onnxruntime
    /// batch (no cross-read padding). Length-bucket upstream so groups are big.
    pub fn detect_adapter_end_batch(
        &self,
        signals: &[&[f32]],
    ) -> Vec<Result<usize, AdapterCnnError>> {
        let cfg = self.config;
        let prepped: Vec<Option<Vec<f32>>> = signals
            .iter()
            .map(|&s| prep_adapter_signal(s, &cfg))
            .collect();
        let valid_idx: Vec<usize> = (0..signals.len())
            .filter(|&i| prepped[i].is_some())
            .collect();

        let short_err = |i: usize| AdapterCnnError::SignalTooShort {
            len: signals[i].len(),
            required: cfg.min_obs_adapter + cfg.downscale_factor,
        };
        let mut out: Vec<Result<usize, AdapterCnnError>> =
            (0..signals.len()).map(|i| Err(short_err(i))).collect();

        for (len, group) in group_by_len(&prepped, &valid_idx) {
            let run = (|| -> Result<Vec<usize>, AdapterCnnError> {
                let g = group.len();
                let mut data = vec![0f32; g * len];
                for (row, &i) in group.iter().enumerate() {
                    data[row * len..(row + 1) * len].copy_from_slice(prepped[i].as_ref().unwrap());
                }
                let input = Tensor::from_array(([g, 1, len], data))
                    .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
                let mut session = self.session.lock().expect("ort session mutex poisoned");
                let outputs = session
                    .run(ort::inputs![input])
                    .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
                let (shape, scores) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| AdapterCnnError::Run(e.to_string()))?;
                // Expect row-major `[group, 2, length_out]`.
                if shape.len() != 3 || shape[0] as usize != g || shape[1] != 2 {
                    return Err(AdapterCnnError::BadShape {
                        got: shape.iter().map(|&d| d as usize).collect(),
                    });
                }
                let length_out = shape[2] as usize;
                Ok((0..g)
                    .map(|row| {
                        // Channel-0 (adapter_end) of row `row`.
                        let base = row * 2 * length_out;
                        decode_adapter_end(&cfg, length_out, len, |k| scores[base + k])
                    })
                    .collect())
            })();
            match run {
                Ok(ends) => {
                    for (&i, end) in group.iter().zip(ends) {
                        out[i] = Ok(end);
                    }
                }
                Err(e) => {
                    for &i in &group {
                        out[i] = Err(e.clone());
                    }
                }
            }
        }
        out
    }
}

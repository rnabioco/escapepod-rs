// SPDX-License-Identifier: MIT

//! The per-base-feature network: the `feature_model` variant of a charging
//! bundle.
//!
//! A charging bundle names one of three models under one format tag. Two of
//! them read the **same per-base features** this crate already computes — the
//! GBM ([`crate::bundle::ChargingScorer::Gbm`]) and this one — and differ only
//! in what consumes the flat column vector [`crate::ChargingBundle::select_columns`]
//! produces. The third, a CNN over the raw signal, is a different model with a
//! different input and is not implemented here.
//!
//! It is worth the extra path: measured on a held-out flowcell over three
//! paired seeds, the network scores AUROC 0.9621 against the GBM's 0.9475 and
//! MCC 0.8399 against 0.7928 — and, in the terms that decide individual
//! molecules, calls **0.727 of reads at 99% precision against the GBM's
//! 0.449**.
//!
//! Three rules stand between the flat vector and the graph, and none of them
//! may be guessed — a consumer that gets any one wrong produces a **confident
//! wrong answer, not an error**, so all three are declared in the bundle and
//! reproduced here from what it declares:
//!
//! 1. **Fold.** `features.order` is offsets-outer / channels-inner, so the
//!    k-th selected column is offset `k / n_val`, value channel `k % n_val`;
//!    the tensor is `[channel, offset]`. Folding the other way transposes
//!    every input and still scores.
//! 2. **Standardise.** Per-channel `(x - mu[c]) / sd[c]` with constants fitted
//!    on the training split and *shipped*. Recomputing them per batch would
//!    make one read's answer depend on which reads it was run beside.
//! 3. **Missingness.** `NaN` is never handed to the graph. Unlike the GBM,
//!    which routes missing values natively, this model is told about them:
//!    the value channel is zeroed and a paired observed channel carries the
//!    indicator. Passing `NaN` through — which is exactly what the GBM path
//!    correctly does — yields `NaN` logits.
//!
//! The reference implementation is `escapepod_models.charging`'s
//! `feature_nn_fold` / `feature_nn_input`; [`FeatureNet::input_tensor`] is a
//! transcription of those two, and `tests/charging_fnn_parity.rs` pins it
//! against golden vectors that side produced.

use anyhow::{Result, anyhow, bail};
use escapepod_demux::onnx_rewrite::hoist_conv_padding;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::framework::Framework;
use tract_onnx::tract_core::model::TypedRunnableModel;


/// A loaded per-base-feature network, with the input contract it was
/// declared with.
///
/// Batch is pinned to 1 at load: classification fans out across reads with
/// rayon, and tract has no efficient batched convolution, so a batch axis
/// would buy nothing and cost a re-optimisation per batch size. The plan is
/// immutable, so the handle is `Sync` and one instance serves every worker.
pub struct FeatureNet {
    plan: Arc<TypedRunnableModel>,
    /// Value channels — half the tensor's channels; the other half are their
    /// observed-mask partners.
    n_val: usize,
    /// Offsets, i.e. the tensor's length axis.
    n_off: usize,
    /// Per-value-channel standardisation, already narrowed to `f32` and with
    /// the reference implementation's `sd <= 0 -> 1.0` guard applied.
    mu: Vec<f32>,
    sd: Vec<f32>,
}

impl std::fmt::Debug for FeatureNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The plan is a whole optimised graph; printing it in a bundle dump
        // is noise, and `ChargingBundle` derives Debug.
        f.debug_struct("FeatureNet")
            .field("n_val", &self.n_val)
            .field("n_off", &self.n_off)
            .finish_non_exhaustive()
    }
}

impl FeatureNet {
    /// Load an ONNX feature model and check it against the contract the
    /// bundle declared for it.
    ///
    /// `mu`/`sd` are per **value** channel; the observed-mask channels are
    /// indicators and are never standardised.
    pub fn load(path: &Path, n_val: usize, n_off: usize, mu: &[f64], sd: &[f64]) -> Result<Self> {
        if mu.len() != n_val || sd.len() != n_val {
            bail!(
                "feature_model.standardisation has {} mu / {} sd for {} value channels",
                mu.len(),
                sd.len(),
                n_val
            );
        }
        let n_ch = 2 * n_val;
        let onnx = tract_onnx::onnx();
        // The graph is taken as a proto first so the padding can be hoisted out
        // of any convolution before tract lowers it — see
        // [`hoist_conv_padding`]. Batch is 1 here and pinned as 1 below; the
        // zero blocks are concrete, so the two have to agree.
        let mut proto = onnx
            .proto_model_for_path(path)
            .map_err(|e| anyhow!("cannot read feature model {}: {e}", path.display()))?;
        let hoisted = hoist_conv_padding(&mut proto, 1);
        if hoisted > 0 {
            tracing::debug!(
                "feature model {}: hoisted the padding out of {hoisted} convolution(s)",
                path.display()
            );
        }
        let plan = onnx
            .model_for_proto_model(&proto)
            .map_err(|e| anyhow!("cannot parse feature model {}: {e}", path.display()))?
            .with_input_fact(0, f32::fact([1, n_ch, n_off]).into())
            .map_err(|e| {
                anyhow!(
                    "feature model {} does not accept the declared input [1, {n_ch}, {n_off}]: {e}",
                    path.display()
                )
            })?
            .into_optimized()
            .map_err(|e| anyhow!("cannot optimize feature model {}: {e}", path.display()))?
            .into_runnable()
            .map_err(|e| anyhow!("cannot plan feature model {}: {e}", path.display()))?;

        let net = Self {
            plan,
            n_val,
            n_off,
            mu: mu.iter().map(|&v| v as f32).collect(),
            // The reference implementation's guard, applied on the f64 before
            // narrowing so the two agree on the boundary.
            sd: sd
                .iter()
                .map(|&v| if v > 0.0 { v as f32 } else { 1.0 })
                .collect(),
        };
        net.probe_output_contract()?;
        Ok(net)
    }

    /// Number of feature columns the model consumes, i.e. what
    /// `features.order` must name.
    pub fn n_features(&self) -> usize {
        self.n_val * self.n_off
    }

    pub fn n_value_channels(&self) -> usize {
        self.n_val
    }

    pub fn n_offsets(&self) -> usize {
        self.n_off
    }

    /// Run one zeroed input at load and insist the output is `[1, 2]`.
    ///
    /// The same discipline `escapepod-demux`'s `adapter_cnn` uses, for the
    /// same reason: a graph with the wrong head — a regressor, a 16-class
    /// barcode net, the raw-signal CNN with its own input rank — must fail
    /// *here*, with the file named, rather than downstream where a wrong
    /// shape becomes a wrong probability on every read.
    fn probe_output_contract(&self) -> Result<()> {
        let t = Tensor::zero::<f32>(&[1, 2 * self.n_val, self.n_off])
            .map_err(|e| anyhow!("cannot build the probe tensor: {e}"))?;
        let out = self
            .plan
            .run(tvec!(t.into()))
            .map_err(|e| anyhow!("feature model failed on a zeroed probe input: {e}"))?;
        if out.len() != 1 {
            bail!(
                "feature model has {} outputs; the contract is exactly one, [1, 2] logits",
                out.len()
            );
        }
        let shape = out[0].shape();
        if shape != [1, 2] {
            bail!(
                "feature model output is {:?} but the contract is [1, 2] (one logit per \
                 class); a raw-signal CNN bundle or a differently-headed graph would look \
                 exactly like this",
                shape
            );
        }
        Ok(())
    }

    /// Flat selected columns → the network's `[2 * n_val, n_off]` input,
    /// row-major (channel-outer).
    ///
    /// A transcription of `escapepod_models.charging.feature_nn_fold` followed
    /// by `feature_nn_input`, kept in the same arithmetic order so the two are
    /// bit-comparable: the reference substitutes `0.0` for the missing value,
    /// standardises unconditionally, *then* multiplies by the mask — so an
    /// unresolved base leaves an exact zero rather than `-mu/sd`.
    pub fn input_tensor(&self, columns: &[f64]) -> Result<Vec<f32>> {
        fold_standardise(columns, self.n_val, self.n_off, &self.mu, &self.sd)
    }

    /// Score one read: fold, standardise, run, softmax.
    ///
    /// Returns `[P(classes[0]), P(classes[1])]`.
    pub fn predict(&self, columns: &[f64]) -> Result<[f64; 2]> {
        let flat = self.input_tensor(columns)?;
        let t = Tensor::from_shape(&[1, 2 * self.n_val, self.n_off], &flat)
            .map_err(|e| anyhow!("cannot build the feature tensor: {e}"))?;
        let out = self
            .plan
            .run(tvec!(t.into()))
            .map_err(|e| anyhow!("feature model inference failed: {e}"))?;
        let view = out[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| anyhow!("feature model output is not f32: {e}"))?;
        let logits: Vec<f32> = view.iter().copied().collect();
        // The probe pinned this at load, so a short slice here would be a
        // graph that changed shape with its input — worth an error, not an
        // index panic.
        if logits.len() != 2 {
            bail!("feature model emitted {} logits, expected 2", logits.len());
        }
        Ok(softmax2(logits[0] as f64, logits[1] as f64))
    }
}

/// The fold + standardise + mask arithmetic, free of the graph.
///
/// Separate from [`FeatureNet`] because it is the part with no dependency on
/// tract and the part that has to match Python bit-for-bit, so it is testable
/// without an ONNX file — and because the golden-parity test compares this
/// tensor directly, not just the probability it leads to. A wrong fold and a
/// wrong standardisation both move the probability; only the tensor says
/// which.
pub fn fold_standardise(
    columns: &[f64],
    n_val: usize,
    n_off: usize,
    mu: &[f32],
    sd: &[f32],
) -> Result<Vec<f32>> {
    if columns.len() != n_val * n_off {
        bail!(
            "feature model wants {} columns ({n_val} value channels x {n_off} offsets), got {}",
            n_val * n_off,
            columns.len()
        );
    }
    let mut t = vec![0.0f32; 2 * n_val * n_off];
    for (k, &x) in columns.iter().enumerate() {
        // `features.order` is offsets-outer, channels-inner.
        let off = k / n_val;
        let c = k % n_val;
        // `np.isfinite`: NaN *and* the infinities are "not observed".
        let observed = x.is_finite();
        let m = if observed { 1.0f32 } else { 0.0f32 };
        let xv = if observed { x as f32 } else { 0.0f32 };
        // Written in the reference implementation's order — substitute, then
        // standardise unconditionally, then mask — so an unresolved base
        // leaves an exact (signed) zero rather than `-mu/sd`, and the two
        // sides agree to the bit.
        t[c * n_off + off] = ((xv - mu[c]) / sd[c]) * m;
        t[(n_val + c) * n_off + off] = m;
    }
    Ok(t)
}

/// Two-class softmax, shifted by the max so a large logit cannot overflow.
fn softmax2(a: f64, b: f64) -> [f64; 2] {
    let m = a.max(b);
    let (ea, eb) = ((a - m).exp(), (b - m).exp());
    let z = ea + eb;
    [ea / z, eb / z]
}

#[cfg(test)]
mod tests {
    use super::*;


    /// The fold, standardisation and missingness rules, without a graph.
    ///
    /// Mirrors escapepod-models' `test_feature_nn_input_encodes_missingness_explicitly`
    /// and `test_feature_nn_fold_is_offsets_outer` on the same numbers, which
    /// is the point: these are the two sides of one contract.
    #[test]
    fn fold_standardises_and_masks() {
        // 2 value channels x 3 offsets, offsets-outer:
        //   o0: [1.0, 4.0]  o1: [NaN, 5.0]  o2: [3.0, NaN]
        let cols = [1.0, 4.0, f64::NAN, 5.0, 3.0, f64::NAN];
        let t = fold_standardise(&cols, 2, 3, &[2.0, 4.0], &[2.0, 1.0]).unwrap();
        assert_eq!(t.len(), 12);
        // value channel 0: (1-2)/2, masked, (3-2)/2
        assert_eq!(&t[0..3], &[-0.5, 0.0, 0.5]);
        // value channel 1: (4-4)/1, (5-4)/1, masked
        assert_eq!(&t[3..6], &[0.0, 1.0, 0.0]);
        // observed masks, same order
        assert_eq!(&t[6..9], &[1.0, 0.0, 1.0]);
        assert_eq!(&t[9..12], &[1.0, 1.0, 0.0]);
    }

    /// An unresolved base must land on zero, not on `-mu/sd`. This is the one
    /// place the mask multiply is load-bearing rather than cosmetic: with a
    /// non-zero `mu` the two differ by the whole gauge.
    #[test]
    fn unresolved_bases_are_zero_not_the_negative_mean() {
        let t = fold_standardise(&[f64::NAN], 1, 1, &[7.0], &[0.5]).unwrap();
        assert_eq!(t[0], 0.0);
        assert_eq!(t[1], 0.0);
    }

    /// Infinities are not observed either — `np.isfinite`, not `!isnan`.
    #[test]
    fn infinities_are_not_observed() {
        let t = fold_standardise(&[f64::INFINITY], 1, 1, &[0.0], &[1.0]).unwrap();
        assert_eq!(&t[..], &[0.0, 0.0]);
    }

    #[test]
    fn wrong_column_count_is_an_error() {
        let err = fold_standardise(&[1.0, 2.0], 2, 3, &[0.0, 0.0], &[1.0, 1.0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("wants 6 columns"), "{err}");
    }

    #[test]
    fn softmax_is_stable_and_normalised() {
        let [a, b] = softmax2(0.0, 0.0);
        assert_eq!((a, b), (0.5, 0.5));
        let [a, b] = softmax2(-1000.0, 1000.0);
        assert!(a.is_finite() && b.is_finite() && (a + b - 1.0).abs() < 1e-12);
        assert!(b > 0.999);
    }
}

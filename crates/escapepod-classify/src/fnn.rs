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
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::pb;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::framework::Framework;
use tract_onnx::tract_core::model::TypedRunnableModel;

/// ONNX `TensorProto.data_type` for `float32`.
const ONNX_FLOAT: i32 = 1;

/// Rewrite each zero-padded `Conv` into an explicit `Concat` of zero blocks
/// feeding an *unpadded* `Conv`. Returns how many were hoisted.
///
/// Identical arithmetic — zero padding **is** a concatenation of zeros — and
/// worth 4.9× on the shipped charging CNN, because of where tract lowers the
/// padding. A `Conv` becomes im2col + matmul, and tract's im2col has a fast
/// block-copy path that it abandons the moment `pads != 0`, falling back to a
/// per-element bounds-checked loop. Measured on rna, one core, the second
/// convolution of `charging_fnn_ldx16x_rna004` (96→96, k=3, over 33 offsets):
///
/// ```text
///                             im2col     whole graph
///   Conv(pads=[1,1])          257.7 us     310.6 us
///   Concat(zeros) + Conv      9.4 us        63.0 us
/// ```
///
/// The im2col buffer is 9,504 floats, so the padded path spends ~26 ns per
/// element — about 100× a memcpy — and 82% of the entire model's runtime goes
/// into rearranging 38 KB. The matmul it feeds is fine (63 GFLOP/s); only the
/// packing is broken.
///
/// Two spellings do *not* work, both tried:
///
/// * an ONNX `Pad` node before an unpadded `Conv` — tract's optimizer fuses it
///   straight back into the convolution and restores the slow path (272 µs);
/// * a bigger batch — the cost is per row, not per call, so batching amortises
///   nothing (252 µs/read at batch 64).
///
/// `Concat` survives optimization, which is the whole reason it is the spelling
/// used here. That makes this a workaround pinned to a tract behaviour: if a
/// later version fuses `Concat` too, or fixes its padded im2col, this becomes a
/// no-op that costs a graph node, never a wrong answer. Correctness does not
/// rest on the workaround holding — `tests/charging_fnn_parity.rs` scores the
/// fixture bundle (the same two-padded-conv architecture) against golden
/// vectors bit-exactly, and it runs through this rewrite.
///
/// Deliberately conservative: anything it is not sure of, it leaves alone. Only
/// a single spatial axis, only the default ONNX domain, only `group = 1`, only
/// explicit non-negative `pads`, and never when `auto_pad` is doing the work.
/// The charging models are all 1-D `group = 1`, and a convolution shape this
/// has never seen is not one to rewrite blind.
///
/// `batch` must be the batch the model is about to be pinned to: the zero
/// blocks are concrete tensors and have to match on the non-concatenated axes.
fn hoist_conv_padding(proto: &mut pb::ModelProto, batch: usize) -> usize {
    let Some(graph) = proto.graph.as_mut() else {
        return 0;
    };
    // Input channel counts come from the weight initializers, which must be
    // read before the node list is rebuilt.
    let weight_dims: HashMap<String, Vec<i64>> = graph
        .initializer
        .iter()
        .map(|t| (t.name.clone(), t.dims.clone()))
        .collect();

    let mut nodes = Vec::with_capacity(graph.node.len() + 4);
    let mut zeros = Vec::new();
    let mut hoisted = 0usize;

    for (idx, mut node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        let Some((lo, hi)) = hoistable_pads(&node, &weight_dims) else {
            nodes.push(node);
            continue;
        };
        // [out_c, in_c / group, k] — `group` is pinned at 1 above, so dim 1 is
        // the convolution's input channel count.
        let in_c = weight_dims[&node.input[1]][1];

        // One zero block per non-empty side, concatenated along the spatial
        // axis of an `[N, C, L]` input.
        let mut inputs = Vec::with_capacity(3);
        let mut zero_block = |pad: i64, side: &str| -> String {
            let name = format!("escpod_hoisted_pad_{idx}_{side}");
            zeros.push(pb::TensorProto {
                dims: vec![batch as i64, in_c, pad],
                data_type: ONNX_FLOAT,
                name: name.clone(),
                raw_data: vec![0u8; batch * in_c as usize * pad as usize * 4],
                ..Default::default()
            });
            name
        };
        if lo > 0 {
            inputs.push(zero_block(lo, "lo"));
        }
        inputs.push(node.input[0].clone());
        if hi > 0 {
            inputs.push(zero_block(hi, "hi"));
        }

        let padded = format!("escpod_hoisted_pad_{idx}_out");
        nodes.push(pb::NodeProto {
            input: inputs,
            output: vec![padded.clone()],
            name: format!("escpod_hoisted_pad_{idx}_concat"),
            op_type: "Concat".to_string(),
            attribute: vec![pb::AttributeProto {
                name: "axis".to_string(),
                r#type: pb::attribute_proto::AttributeType::Int as i32,
                i: 2,
                ..Default::default()
            }],
            ..Default::default()
        });

        node.input[0] = padded;
        for attr in &mut node.attribute {
            if attr.name == "pads" {
                attr.ints = vec![0, 0];
            }
        }
        nodes.push(node);
        hoisted += 1;
    }

    graph.node = nodes;
    graph.initializer.extend(zeros);
    hoisted
}

/// `(pad_before, pad_after)` if this node is a convolution whose padding this
/// rewrite is sure it can hoist. See [`hoist_conv_padding`] on why each guard
/// is a refusal rather than a best effort.
fn hoistable_pads(
    node: &pb::NodeProto,
    weight_dims: &HashMap<String, Vec<i64>>,
) -> Option<(i64, i64)> {
    if node.op_type != "Conv" || !node.domain.is_empty() {
        return None;
    }
    let mut pads: Option<(i64, i64)> = None;
    let mut group = 1i64;
    for attr in &node.attribute {
        match attr.name.as_str() {
            // Padding computed from the output shape rather than stated: the
            // amount is not in the graph, so there is nothing to hoist.
            "auto_pad" if attr.s.as_slice() != b"NOTSET" => return None,
            "pads" => match attr.ints.as_slice() {
                &[lo, hi] if lo >= 0 && hi >= 0 => pads = Some((lo, hi)),
                // Two or more spatial axes need a concat per axis, and no
                // charging model has ever had one to test against.
                _ => return None,
            },
            "group" => group = attr.i,
            _ => {}
        }
    }
    if group != 1 {
        return None;
    }
    let (lo, hi) = pads?;
    if lo == 0 && hi == 0 {
        return None;
    }
    // The weight has to be a graph initializer for its channel count to be
    // knowable here; a computed kernel is not something to guess at.
    let dims = weight_dims.get(node.input.get(1)?)?;
    (dims.len() == 3).then_some((lo, hi))
}

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

    /// The padding hoist ([`hoist_conv_padding`]) as a graph transform.
    ///
    /// Numeric equivalence is pinned where it counts —
    /// `tests/charging_fnn_parity.rs` scores the fixture bundle, whose two
    /// padded convolutions are the same architecture the shipped model uses,
    /// against golden vectors bit-exactly, and it loads through this rewrite.
    /// What is left to check here is the *shape* of the transform and, more
    /// importantly, that every guard refuses rather than guesses.
    mod hoist {
        use super::*;

        fn attr_ints(name: &str, ints: &[i64]) -> pb::AttributeProto {
            pb::AttributeProto {
                name: name.to_string(),
                r#type: pb::attribute_proto::AttributeType::Ints as i32,
                ints: ints.to_vec(),
                ..Default::default()
            }
        }

        fn attr_int(name: &str, i: i64) -> pb::AttributeProto {
            pb::AttributeProto {
                name: name.to_string(),
                r#type: pb::attribute_proto::AttributeType::Int as i32,
                i,
                ..Default::default()
            }
        }

        fn attr_str(name: &str, s: &str) -> pb::AttributeProto {
            pb::AttributeProto {
                name: name.to_string(),
                r#type: pb::attribute_proto::AttributeType::String as i32,
                s: s.as_bytes().to_vec(),
                ..Default::default()
            }
        }

        /// One `Conv` over `[1, 8, 33]` with the given attributes, its weight
        /// declared as an initializer the way an export writes it.
        fn model(attrs: Vec<pb::AttributeProto>) -> pb::ModelProto {
            pb::ModelProto {
                graph: Some(pb::GraphProto {
                    node: vec![pb::NodeProto {
                        input: vec!["x".into(), "w".into(), "b".into()],
                        output: vec!["y".into()],
                        name: "conv".into(),
                        op_type: "Conv".into(),
                        attribute: attrs,
                        ..Default::default()
                    }],
                    initializer: vec![pb::TensorProto {
                        // [out_c, in_c, k]
                        dims: vec![8, 8, 3],
                        data_type: ONNX_FLOAT,
                        name: "w".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            }
        }

        fn symmetric() -> Vec<pb::AttributeProto> {
            vec![attr_ints("kernel_shape", &[3]), attr_ints("pads", &[1, 1])]
        }

        #[test]
        fn a_padded_conv_becomes_zero_blocks_and_an_unpadded_conv() {
            let mut m = model(symmetric());
            assert_eq!(hoist_conv_padding(&mut m, 1), 1);
            let g = m.graph.unwrap();
            assert_eq!(g.node.len(), 2, "a Concat is inserted before the Conv");

            let (cat, conv) = (&g.node[0], &g.node[1]);
            assert_eq!(cat.op_type, "Concat");
            // Zeros, the original input, zeros — in that order, or the window
            // shifts by one base and every read still scores.
            assert_eq!(cat.input.len(), 3);
            assert_eq!(cat.input[1], "x");
            assert_eq!(cat.attribute[0].name, "axis");
            assert_eq!(cat.attribute[0].i, 2, "the spatial axis of [N, C, L]");

            assert_eq!(conv.op_type, "Conv");
            assert_eq!(
                conv.input[0], cat.output[0],
                "the Conv reads the padded tensor"
            );
            let pads = conv.attribute.iter().find(|a| a.name == "pads").unwrap();
            assert_eq!(pads.ints, vec![0, 0], "tract's fast im2col needs pads == 0");

            // One zero block per side, each [batch, in_c, pad], f32.
            let zeros: Vec<_> = g.initializer.iter().filter(|t| t.name != "w").collect();
            assert_eq!(zeros.len(), 2);
            for z in zeros {
                assert_eq!(z.dims, vec![1, 8, 1]);
                assert_eq!(z.data_type, ONNX_FLOAT);
                assert_eq!(z.raw_data.len(), 8 * 4, "1 x 8 x 1 f32");
                assert!(z.raw_data.iter().all(|&b| b == 0));
            }
        }

        /// The zero blocks are concrete tensors, so they have to match the
        /// batch the model is pinned to.
        #[test]
        fn the_zero_blocks_take_the_batch_they_are_given() {
            let mut m = model(symmetric());
            assert_eq!(hoist_conv_padding(&mut m, 16), 1);
            let g = m.graph.unwrap();
            let z = g.initializer.iter().find(|t| t.name != "w").unwrap();
            assert_eq!(z.dims, vec![16, 8, 1]);
            assert_eq!(z.raw_data.len(), 16 * 8 * 4);
        }

        #[test]
        fn asymmetric_padding_only_adds_the_side_it_needs() {
            let mut m = model(vec![attr_ints("pads", &[2, 0])]);
            assert_eq!(hoist_conv_padding(&mut m, 1), 1);
            let g = m.graph.unwrap();
            let cat = &g.node[0];
            assert_eq!(cat.input.len(), 2, "no trailing zero block");
            assert_eq!(cat.input[1], "x", "the pad goes before the input");
            let zeros: Vec<_> = g.initializer.iter().filter(|t| t.name != "w").collect();
            assert_eq!(zeros.len(), 1);
            assert_eq!(zeros[0].dims, vec![1, 8, 2]);
        }

        /// Every guard, each of which is a case the rewrite cannot be sure it
        /// would reproduce — so it leaves the graph exactly as it found it.
        #[test]
        fn anything_unfamiliar_is_left_alone() {
            let cases: Vec<(&str, Vec<pb::AttributeProto>)> = vec![
                ("nothing to hoist", vec![attr_ints("pads", &[0, 0])]),
                ("no pads attribute", vec![attr_ints("kernel_shape", &[3])]),
                ("two spatial axes", vec![attr_ints("pads", &[1, 1, 1, 1])]),
                (
                    "auto_pad computes the padding",
                    vec![
                        attr_ints("pads", &[1, 1]),
                        attr_str("auto_pad", "SAME_UPPER"),
                    ],
                ),
                (
                    "grouped convolution",
                    vec![attr_ints("pads", &[1, 1]), attr_int("group", 2)],
                ),
                ("negative pad", vec![attr_ints("pads", &[-1, 1])]),
            ];
            for (why, attrs) in cases {
                let mut m = model(attrs);
                let before = m.clone();
                assert_eq!(hoist_conv_padding(&mut m, 1), 0, "{why}: should not hoist");
                assert_eq!(m, before, "{why}: the graph must be untouched");
            }
        }

        /// `auto_pad` set to its default is not `auto_pad` doing the work.
        #[test]
        fn an_explicit_notset_auto_pad_still_hoists() {
            let mut m = model(vec![
                attr_ints("pads", &[1, 1]),
                attr_str("auto_pad", "NOTSET"),
            ]);
            assert_eq!(hoist_conv_padding(&mut m, 1), 1);
        }

        /// A kernel that is not a graph initializer has no knowable channel
        /// count here, and the zero blocks need one.
        #[test]
        fn a_computed_kernel_is_left_alone() {
            let mut m = model(symmetric());
            m.graph.as_mut().unwrap().initializer.clear();
            assert_eq!(hoist_conv_padding(&mut m, 1), 0);
        }

        #[test]
        fn a_graph_without_convolutions_is_untouched() {
            let mut m = model(symmetric());
            m.graph.as_mut().unwrap().node[0].op_type = "Gemm".into();
            let before = m.clone();
            assert_eq!(hoist_conv_padding(&mut m, 1), 0);
            assert_eq!(m, before);
        }

        /// Custom-domain ops share ONNX's names but not its semantics.
        #[test]
        fn a_custom_domain_conv_is_left_alone() {
            let mut m = model(symmetric());
            m.graph.as_mut().unwrap().node[0].domain = "com.example".into();
            assert_eq!(hoist_conv_padding(&mut m, 1), 0);
        }

        /// Two convolutions, as every charging CNN has: both hoisted, and the
        /// generated names stay distinct so the second cannot clobber the
        /// first's zero blocks.
        #[test]
        fn each_convolution_gets_its_own_zero_blocks() {
            let mut m = model(symmetric());
            let g = m.graph.as_mut().unwrap();
            let mut second = g.node[0].clone();
            second.name = "conv2".into();
            second.input[0] = "y".into();
            second.output[0] = "z".into();
            g.node.push(second);

            assert_eq!(hoist_conv_padding(&mut m, 1), 2);
            let g = m.graph.unwrap();
            let names: HashMap<&str, usize> =
                g.initializer.iter().fold(HashMap::new(), |mut acc, t| {
                    *acc.entry(t.name.as_str()).or_default() += 1;
                    acc
                });
            assert!(
                names.values().all(|&n| n == 1),
                "duplicate initializer name"
            );
            assert_eq!(g.node.len(), 4);
            // The second Conv still reads the first Conv's output, via its own
            // Concat — a rewrite that dropped the rewiring would still run.
            let cat2 = &g.node[2];
            assert_eq!(cat2.op_type, "Concat");
            assert!(cat2.input.contains(&"y".to_string()));
        }
    }

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

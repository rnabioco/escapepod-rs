//! Graph rewrites applied to an ONNX proto before tract lowers it.
//!
//! Two rewrites, and both exist for the same reason: tract lowers a standard
//! ONNX operator in a way that is correct for what this workspace has always
//! asked of it and wrong, or merely slow, the moment it is asked for something
//! else. They live here rather than in a loader because every tract consumer in
//! the workspace can hit them.
//!
//! [`hoist_conv_padding`] is about *where* tract lowers convolution padding.
//! The boundary CNN, the CTC-CRF encoder and the charging networks all ship
//! zero-padded 1-D convolutions. Measured on the boundary CNN (`adapter_rna004`,
//! nine padded convs, dilations 1-8): 600 -> 380 CPU-seconds over 119k reads
//! with the input fact pinned, classifications identical.
//!
//! [`expand_instance_norm`] is about *which axes* tract reduces over in an
//! `InstanceNormalization` — every axis but the channel, batch included. At
//! batch 1, which is what every tract graph in `escpod` is pinned to, that is
//! the spec. Above batch 1 it is a different function, and it is also what
//! stops a graph from running entirely on a GPU.

use std::collections::HashMap;
use tract_onnx::pb;

/// ONNX `TensorProto.data_type` for `float32`.
const ONNX_FLOAT: i32 = 1;
/// ONNX `TensorProto.data_type` for `int64`.
const ONNX_INT64: i32 = 7;

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
pub fn hoist_conv_padding(proto: &mut pb::ModelProto, batch: usize) -> usize {
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

/// Rewrite each `InstanceNormalization` into the explicit per-instance graph
/// the ONNX spec defines. Returns how many were rewritten.
///
/// # Why this exists
///
/// tract lowers `InstanceNormalization` by reducing over **every axis but the
/// channel** (`tract-onnx/src/ops/nn/instance_norm.rs`):
///
/// ```text
/// let axes: Vec<_> = (0..rank as i64).filter(|&axis| axis != 1).collect();
/// ```
///
/// For a rank-3 `[N, C, W]` input that is `axes = [0, 2]` — the batch axis
/// included. The ONNX operator is defined per *instance*: the mean and
/// variance of sample `n`, channel `c` are taken over that sample's spatial
/// positions alone. At `N = 1` the two agree exactly, which is why nothing in
/// this workspace has ever noticed: every tract graph `escpod` runs is pinned
/// to batch 1.
///
/// Two consequences, and the second is the reason this is not merely a
/// tidy-up:
///
/// * **A batched score is not the per-read score.** Reduce over the batch and
///   each read's normalisation depends on the reads it was batched with, so
///   the same read gets a different logit in a different batch. That is not a
///   tolerance question; it is a different function.
/// * **It is why the CUDA transform gives up 29 times.** `GpuReduce::new`
///   accepts a single axis only, and `split_multi_axis_reduce` — the rewrite
///   that would split a multi-axis one — covers `Sum | Prod | Min | Max | Any
///   | All` and deliberately not `MeanOfSquares`, which does not chain
///   (`mean(mean(x²)²) != mean(x²)`). So each variance reduction lands back on
///   the host with a device sync on either side of it.
///
/// One axis instead of two fixes both at once: the reduction becomes
/// per-instance, and single-axis reductions are exactly what the device
/// accepts.
///
/// # What it emits
///
/// The spec's own formula, spelled out, with the spatial axis named
/// explicitly:
///
/// ```text
///   mean = ReduceMean(X, axes=[2], keepdims=1)
///   diff = X - mean
///   var  = ReduceMean(diff * diff, axes=[2], keepdims=1)
///   Y    = diff / Sqrt(var + epsilon) * Unsqueeze(scale) + Unsqueeze(B)
/// ```
///
/// `Unsqueeze(_, axes=[0, 2])` lifts the `[C]` scale and bias to `[1, C, 1]`
/// without needing to know `C`, so nothing here reads a weight.
///
/// This is not bit-identical to tract's lowering even at batch 1: tract
/// multiplies by `rsqrt(var + eps)` where this divides by `sqrt(var + eps)`,
/// and ONNX has no `Rsqrt` to spell the former with. The difference is a last
/// ulp per normalisation; the contract, as everywhere else in this family, is
/// agreement inside the bundle's own tolerance rather than bit-exactness.
///
/// # What it refuses
///
/// Conservative in the same way [`hoist_conv_padding`] is — anything it is not
/// sure of, it leaves alone, because a wrong rewrite here is a silently wrong
/// answer rather than a load error:
///
/// * a non-default domain, or a rank other than 3 (the 1-D case, which is
///   every model in this workspace — a 2-D norm would need `axes = [2, 3]`
///   and a `split_multi_axis_reduce` that reaches `MeanOfSquares`);
/// * an opset below 13, where `Unsqueeze` still carries its axes as an
///   attribute. Below 18 `ReduceMean` does too, and that spelling *is*
///   handled, because the older charging exports use it.
pub fn expand_instance_norm(proto: &mut pb::ModelProto, rank: usize) -> usize {
    if rank != 3 {
        return 0;
    }
    let opset = proto
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .map(|o| o.version)
        .unwrap_or(0);
    if opset < 13 {
        return 0;
    }
    // `axes` became an input in opset 18 for `ReduceMean` and in 13 for
    // `Unsqueeze`. Emitting the wrong spelling is a load error rather than a
    // wrong answer, but there is no reason to emit one.
    let reduce_axes_is_input = opset >= 18;

    let Some(graph) = proto.graph.as_mut() else {
        return 0;
    };

    let mut nodes = Vec::with_capacity(graph.node.len() + 16);
    let mut consts = Vec::new();
    let mut rewritten = 0usize;

    for (idx, node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        if node.op_type != "InstanceNormalization"
            || !node.domain.is_empty()
            || node.input.len() != 3
            || node.output.len() != 1
        {
            nodes.push(node);
            continue;
        }
        let epsilon = node
            .attribute
            .iter()
            .find(|a| a.name == "epsilon")
            .map(|a| a.f)
            .unwrap_or(1e-5);

        let tag = format!("escpod_instnorm_{idx}");
        let n = |suffix: &str| format!("{tag}_{suffix}");

        // Constants are little-endian raw bytes, like every other initializer
        // this module writes.
        let (spatial, lift) = (n("spatial_axis"), n("lift_axes"));
        consts.push(int64_initializer(spatial.clone(), &[2]));
        consts.push(int64_initializer(lift.clone(), &[0, 2]));
        consts.push(pb::TensorProto {
            dims: vec![],
            data_type: ONNX_FLOAT,
            name: n("eps"),
            raw_data: epsilon.to_le_bytes().to_vec(),
            ..Default::default()
        });

        let mut wire = |op: &str, ins: Vec<String>, out: String, attrs: Vec<pb::AttributeProto>| {
            nodes.push(pb::NodeProto {
                input: ins,
                output: vec![out.clone()],
                name: out.clone(),
                op_type: op.to_string(),
                attribute: attrs,
                ..Default::default()
            });
            out
        };
        let keepdims = vec![pb::AttributeProto {
            name: "keepdims".to_string(),
            r#type: pb::attribute_proto::AttributeType::Int as i32,
            i: 1,
            ..Default::default()
        }];
        // Pre-18 spells the same reduction with the axes as an attribute.
        let (reduce_in, reduce_attrs) = if reduce_axes_is_input {
            (Some(spatial.clone()), keepdims.clone())
        } else {
            let mut attrs = keepdims.clone();
            attrs.push(pb::AttributeProto {
                name: "axes".to_string(),
                r#type: pb::attribute_proto::AttributeType::Ints as i32,
                ints: vec![2],
                ..Default::default()
            });
            (None, attrs)
        };
        let reduce_inputs = |data: String| match &reduce_in {
            Some(axes) => vec![data, axes.clone()],
            None => vec![data],
        };

        let x = node.input[0].clone();
        let mean = wire(
            "ReduceMean",
            reduce_inputs(x.clone()),
            n("mean"),
            reduce_attrs.clone(),
        );
        let diff = wire("Sub", vec![x, mean], n("diff"), vec![]);
        let sq = wire("Mul", vec![diff.clone(), diff.clone()], n("sq"), vec![]);
        let var = wire("ReduceMean", reduce_inputs(sq), n("var"), reduce_attrs);
        let vare = wire("Add", vec![var, n("eps")], n("var_eps"), vec![]);
        let sd = wire("Sqrt", vec![vare], n("sd"), vec![]);
        let norm = wire("Div", vec![diff, sd], n("norm"), vec![]);
        let scale3 = wire(
            "Unsqueeze",
            vec![node.input[1].clone(), lift.clone()],
            n("scale3"),
            vec![],
        );
        let bias3 = wire(
            "Unsqueeze",
            vec![node.input[2].clone(), lift],
            n("bias3"),
            vec![],
        );
        let scaled = wire("Mul", vec![norm, scale3], n("scaled"), vec![]);
        // The last node keeps the original output name, so every consumer is
        // rewired by construction.
        wire("Add", vec![scaled, bias3], node.output[0].clone(), vec![]);
        rewritten += 1;
    }

    graph.node = nodes;
    graph.initializer.extend(consts);
    rewritten
}

/// A rank-1 `int64` initializer holding `values`, for the `axes` inputs the
/// post-13 spellings of `ReduceMean` and `Unsqueeze` take.
fn int64_initializer(name: String, values: &[i64]) -> pb::TensorProto {
    pb::TensorProto {
        dims: vec![values.len() as i64],
        data_type: ONNX_INT64,
        name,
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The padding hoist ([`hoist_conv_padding`]) as a graph transform.
    //
    // Numeric equivalence is pinned where it counts —
    // `tests/charging_fnn_parity.rs` scores the fixture bundle, whose two
    // padded convolutions are the same architecture the shipped model uses,
    // against golden vectors bit-exactly, and it loads through this rewrite.
    // What is left to check here is the *shape* of the transform and, more
    // importantly, that every guard refuses rather than guesses.

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

    // ---------------------------------------------------------------------
    // `expand_instance_norm`
    //
    // The transform's whole point is which axes the reductions name, so that
    // is what these assert. Numeric equivalence at batch 1 is pinned where it
    // counts, by the waveform bundle's own verification against the training
    // corpus; what a unit test can add is that the emitted graph says `[2]`
    // and not `[0, 2]`, that every consumer is rewired, and that each guard
    // refuses rather than guesses.

    /// One `InstanceNormalization` feeding a `Relu`, at the given opset.
    fn norm_model(opset: i64) -> pb::ModelProto {
        pb::ModelProto {
            opset_import: vec![pb::OperatorSetIdProto {
                domain: String::new(),
                version: opset,
            }],
            graph: Some(pb::GraphProto {
                node: vec![
                    pb::NodeProto {
                        input: vec!["x".into(), "scale".into(), "bias".into()],
                        output: vec!["normed".into()],
                        name: "norm".into(),
                        op_type: "InstanceNormalization".into(),
                        attribute: vec![pb::AttributeProto {
                            name: "epsilon".into(),
                            r#type: pb::attribute_proto::AttributeType::Float as i32,
                            f: 1e-5,
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    pb::NodeProto {
                        input: vec!["normed".into()],
                        output: vec!["y".into()],
                        name: "relu".into(),
                        op_type: "Relu".into(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// The `int64` axes an initializer holds, as `expand_instance_norm` writes
    /// them: little-endian raw bytes.
    fn axes_const(initializers: &[pb::TensorProto], name: &str) -> Vec<i64> {
        let t = initializers
            .iter()
            .find(|t| t.name == name)
            .expect("axes const");
        t.raw_data
            .as_chunks::<8>()
            .0
            .iter()
            .copied()
            .map(i64::from_le_bytes)
            .collect()
    }

    fn reduce_axes(g: &pb::GraphProto, initializers: &[pb::TensorProto]) -> Vec<Vec<i64>> {
        g.node
            .iter()
            .filter(|n| n.op_type == "ReduceMean")
            .map(|n| match n.input.get(1) {
                // opset >= 18: axes are an int64 initializer
                Some(name) => axes_const(initializers, name),
                // opset < 18: axes are an attribute
                None => n
                    .attribute
                    .iter()
                    .find(|a| a.name == "axes")
                    .expect("axes attribute")
                    .ints
                    .clone(),
            })
            .collect()
    }

    #[test]
    fn instance_norm_reduces_over_the_spatial_axis_alone() {
        let mut m = norm_model(18);
        assert_eq!(expand_instance_norm(&mut m, 3), 1);
        let g = m.graph.as_ref().unwrap();

        assert!(
            g.node.iter().all(|n| n.op_type != "InstanceNormalization"),
            "the op it replaces must be gone"
        );
        // Mean and variance, each over axis 2 only. `[0, 2]` — what tract's
        // own lowering asks for — would reduce across the batch and make one
        // read's score depend on the reads beside it.
        assert_eq!(reduce_axes(g, &g.initializer), vec![vec![2], vec![2]]);
        // The scale and bias lift to `[1, C, 1]` without reading `C`.
        let unsqueezes: Vec<&pb::NodeProto> =
            g.node.iter().filter(|n| n.op_type == "Unsqueeze").collect();
        assert_eq!(unsqueezes.len(), 2);
        for u in unsqueezes {
            assert_eq!(axes_const(&g.initializer, &u.input[1]), vec![0, 2]);
        }
    }

    #[test]
    fn instance_norm_keeps_the_output_name_so_consumers_are_rewired() {
        let mut m = norm_model(18);
        expand_instance_norm(&mut m, 3);
        let g = m.graph.as_ref().unwrap();

        // The `Relu` downstream still reads `normed`, and exactly one node
        // still produces it.
        let relu = g
            .node
            .iter()
            .find(|n| n.op_type == "Relu")
            .expect("relu survives");
        assert_eq!(relu.input, vec!["normed".to_string()]);
        assert_eq!(
            g.node
                .iter()
                .filter(|n| n.output.contains(&"normed".to_string()))
                .count(),
            1
        );
        // Node names have to stay unique; a norm emits two of several op types.
        let mut names: Vec<&String> = g.node.iter().map(|n| &n.name).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate node name");
    }

    #[test]
    fn instance_norm_spells_pre_18_axes_as_an_attribute() {
        let mut m = norm_model(17);
        assert_eq!(expand_instance_norm(&mut m, 3), 1);
        let g = m.graph.as_ref().unwrap();
        // Same axes, the older spelling: `ReduceMean` did not take `axes` as
        // an input until opset 18, and emitting the wrong one does not load.
        assert!(
            g.node
                .iter()
                .filter(|n| n.op_type == "ReduceMean")
                .all(|n| n.input.len() == 1)
        );
        assert_eq!(reduce_axes(g, &g.initializer), vec![vec![2], vec![2]]);
    }

    #[test]
    fn instance_norm_refuses_what_it_cannot_be_sure_of() {
        // Rank 4 would need `axes = [2, 3]`, and a two-axis `MeanOfSquares` is
        // exactly what has no device kernel — so rewriting it would fix
        // nothing while changing the numbers.
        let mut m = norm_model(18);
        assert_eq!(expand_instance_norm(&mut m, 4), 0);
        assert_eq!(m.graph.as_ref().unwrap().node.len(), 2, "left alone");

        // Below 13 `Unsqueeze` still carries its axes as an attribute.
        let mut m = norm_model(12);
        assert_eq!(expand_instance_norm(&mut m, 3), 0);

        // A custom-domain op of the same name is somebody else's operator.
        let mut m = norm_model(18);
        m.graph.as_mut().unwrap().node[0].domain = "com.example".into();
        assert_eq!(expand_instance_norm(&mut m, 3), 0);
    }
}

//! Graph rewrites applied to an ONNX proto before tract lowers it.
//!
//! One rewrite today: [`hoist_conv_padding`]. It exists because of where
//! tract lowers convolution padding, and it lives here rather than in
//! `escapepod-classify` because every tract loader in the workspace needs
//! it: the boundary CNN, the CTC-CRF encoder and the charging networks all
//! ship zero-padded 1-D convolutions. Measured on the boundary CNN
//! (`adapter_rna004`, nine padded convs, dilations 1-8): 600 -> 380
//! CPU-seconds over 119k reads with the input fact pinned, classifications
//! identical.

use std::collections::HashMap;
use tract_onnx::pb;

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

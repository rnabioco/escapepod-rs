//! Shared ONNX-proto helpers for this crate's and `escapepod-classify`'s
//! native-kernel recognizers.
//!
//! `crf::encoder_native` (this crate, the CRF barcode encoder's five
//! unidirectional LSTM layers) and `escapepod_classify::fnn_lstm` (the
//! charging classifier's single bidirectional layer) each recognise an
//! exported ONNX graph and lift its weights into
//! [`escapepod_signal::lstm`]'s kernel rather than run the graph through
//! tract. Both walk the same `ModelProto`/`GraphProto`/`NodeProto` structures
//! with the same primitives — producer/consumer lookups, attribute readers,
//! tensor decoders — and validate the same `LSTM`-node shape (forbidden
//! attributes, `input_forget`, `layout`, `W`/`R`/`B` dimensions, per-sequence
//! lengths, a zero `ConstantOfShape` initial state, peephole weights),
//! differing only in how many directions the node carries and, since one
//! recogniser matches several stacked layers and the other matches one, how
//! its error messages are worded. This module is the one copy of both, plus
//! the synthetic-graph builders both files' tests use to construct one.

use std::collections::HashMap;
use tract_onnx::pb;

/// ONNX `TensorProto.data_type` for `float32`.
pub const ONNX_FLOAT: i32 = 1;
/// ONNX `TensorProto.data_type` for `int64`.
pub const ONNX_INT64: i32 = 7;

// ---- proto-walking primitives -------------------------------------------

/// The node that produces `name` as one of its outputs, if any.
pub fn producer<'a>(graph: &'a pb::GraphProto, name: &str) -> Option<&'a pb::NodeProto> {
    graph
        .node
        .iter()
        .find(|n| n.output.iter().any(|o| o == name))
}

/// Every node that consumes `name` as one of its inputs.
pub fn consumers<'a>(graph: &'a pb::GraphProto, name: &str) -> Vec<&'a pb::NodeProto> {
    graph
        .node
        .iter()
        .filter(|n| n.input.iter().any(|i| i == name))
        .collect()
}

/// `name`'s one consumer, and only if it is a `op`.
pub fn sole_consumer<'a>(
    graph: &'a pb::GraphProto,
    name: &str,
    op: &str,
) -> Result<&'a pb::NodeProto, String> {
    match consumers(graph, name).as_slice() {
        [only] if only.op_type == op => Ok(only),
        [only] => Err(format!("`{name}` feeds {} rather than {op}", only.op_type)),
        cs => Err(format!(
            "`{name}` has {} consumers, expected one {op}",
            cs.len()
        )),
    }
}

/// A node's attribute, by name.
pub fn attr<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a pb::AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

/// A node's `int` attribute, by name.
pub fn attr_int(node: &pb::NodeProto, name: &str) -> Option<i64> {
    attr(node, name).map(|a| a.i)
}

/// A node's `string` attribute, by name.
pub fn attr_str<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a str> {
    attr(node, name).and_then(|a| std::str::from_utf8(&a.s).ok())
}

/// A node's `ints` (list) attribute, by name.
pub fn attr_ints<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a [i64]> {
    attr(node, name).map(|a| a.ints.as_slice())
}

/// A node's `tensor` attribute, by name.
pub fn attr_tensor<'a>(node: &'a pb::NodeProto, name: &str) -> Option<&'a pb::TensorProto> {
    attr(node, name).and_then(|a| a.t.as_ref())
}

/// A float initializer's values, from whichever field the export used.
pub fn tensor_f32(t: &pb::TensorProto) -> Result<Vec<f32>, String> {
    if t.data_type != ONNX_FLOAT {
        return Err(format!("`{}` is not float32", t.name));
    }
    let n: usize = t.dims.iter().map(|&d| d.max(0) as usize).product();
    let v: Vec<f32> = if !t.float_data.is_empty() {
        t.float_data.clone()
    } else {
        t.raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    if v.len() != n {
        return Err(format!(
            "`{}` holds {} values for dims {:?}",
            t.name,
            v.len(),
            t.dims
        ));
    }
    Ok(v)
}

/// An int64 initializer's values, from whichever field the export used.
pub fn tensor_i64(t: &pb::TensorProto) -> Result<Vec<i64>, String> {
    if t.data_type != ONNX_INT64 {
        return Err(format!("`{}` is not int64", t.name));
    }
    Ok(if !t.int64_data.is_empty() {
        t.int64_data.clone()
    } else {
        t.raw_data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect()
    })
}

// ---- the LSTM-node validator ---------------------------------------------

/// A validated ONNX `LSTM` node's hidden size and raw weights, direction-major
/// exactly as the initializers laid them out.
#[derive(Debug)]
pub struct LstmNode {
    /// Hidden units per direction (`hidden_size`).
    pub hidden: usize,
    /// Input channels the recurrence itself takes per timestep (not
    /// `n_dirs · 4 · hidden` — the feature width `W`'s last axis carries).
    pub n_in: usize,
    /// `W`, flattened `[n_dirs, 4·hidden, n_in]`.
    pub w: Vec<f32>,
    /// `R`, flattened `[n_dirs, 4·hidden, hidden]`.
    pub r: Vec<f32>,
    /// `B`, flattened `[n_dirs, 8·hidden]` (zero-filled if the graph omits it).
    pub b: Vec<f32>,
}

impl LstmNode {
    /// Validate `node` is an ONNX `LSTM` shaped the way
    /// [`escapepod_signal::lstm`]'s native kernel can run, and lift its raw
    /// weights out.
    ///
    /// Refuses every variation ONNX's LSTM semantics permit that the kernel
    /// does not implement: custom activations or clipping, coupled
    /// input/forget gates (`input_forget`), batch-major layout, per-sequence
    /// lengths, a non-zero initial state, and peephole weights.
    ///
    /// `n_dirs` is 1 for a unidirectional layer (this crate's CRF encoder
    /// stack) or 2 for a bidirectional one (the charging network's single
    /// layer) — it decides both the expected `direction` attribute
    /// (`"forward"`/`"bidirectional"`) and the expected leading dimension of
    /// `W`/`R`/`B`. `index` labels which stacked layer this is, for a
    /// multi-layer caller's error messages; pass `None` when there is only
    /// one `LSTM` node in the graph to blame.
    pub fn parse(
        node: &pb::NodeProto,
        graph: &pb::GraphProto,
        init: &HashMap<&str, &pb::TensorProto>,
        n_dirs: usize,
        index: Option<usize>,
    ) -> Result<Self, String> {
        // Two prefixing conventions, both matching what each of today's two
        // callers already produced before this validator was one function:
        // whole-node complaints name "LSTM" (bare, or "LSTM {i}" when there
        // is more than one node to blame); field-specific complaints (which
        // input, which tensor) carry no "LSTM" at all for a lone node, and
        // "LSTM {i} " only to disambiguate a stacked layer.
        let node_tag = match index {
            Some(i) => format!("LSTM {i}"),
            None => "LSTM".to_string(),
        };
        let field_tag = match index {
            Some(i) => format!("LSTM {i} "),
            None => String::new(),
        };

        let expect_dir = if n_dirs == 2 {
            "bidirectional"
        } else {
            "forward"
        };
        if attr_str(node, "direction").unwrap_or("forward") != expect_dir {
            return Err(if n_dirs == 2 {
                format!("{node_tag} is not bidirectional")
            } else {
                format!("{node_tag} is not the default forward direction")
            });
        }
        let h = attr_int(node, "hidden_size").ok_or("LSTM has no hidden_size")? as usize;
        if h == 0 {
            return Err(format!("{field_tag}hidden_size is 0"));
        }
        for forbidden in ["activations", "activation_alpha", "activation_beta", "clip"] {
            if node.attribute.iter().any(|a| a.name == forbidden) {
                return Err(format!("{node_tag} sets `{forbidden}`"));
            }
        }
        if attr_int(node, "input_forget").unwrap_or(0) != 0 {
            return Err(format!("{node_tag} couples input and forget gates"));
        }
        if attr_int(node, "layout").unwrap_or(0) != 0 {
            return Err(format!("{node_tag} uses batch-major layout"));
        }
        let input_at = |k: usize| node.input.get(k).map(String::as_str).unwrap_or("");
        if node.input.len() < 3 {
            return Err(format!("{node_tag} has fewer than 3 inputs"));
        }
        let g = 4 * h;
        let w_t = init
            .get(input_at(1))
            .ok_or_else(|| format!("{field_tag}W is not an initializer"))?;
        let r_t = init
            .get(input_at(2))
            .ok_or_else(|| format!("{field_tag}R is not an initializer"))?;
        let n_in = match w_t.dims.as_slice() {
            [d0, gg, n_in] if *d0 as usize == n_dirs && *gg as usize == g => *n_in as usize,
            d => {
                return Err(format!(
                    "{field_tag}W dims {d:?}, expected [{n_dirs}, {g}, n_in]"
                ));
            }
        };
        if r_t.dims != [n_dirs as i64, g as i64, h as i64] {
            return Err(format!(
                "{field_tag}R dims {:?}, expected [{n_dirs}, {g}, {h}]",
                r_t.dims
            ));
        }
        let w = tensor_f32(w_t)?;
        let r = tensor_f32(r_t)?;
        let b = match input_at(3) {
            "" => vec![0.0f32; n_dirs * 8 * h],
            name => {
                let t = init
                    .get(name)
                    .ok_or_else(|| format!("{field_tag}B is not an initializer"))?;
                if t.dims != [n_dirs as i64, 8 * h as i64] {
                    return Err(format!(
                        "{field_tag}B dims {:?}, expected [{n_dirs}, {}]",
                        t.dims,
                        8 * h
                    ));
                }
                tensor_f32(t)?
            }
        };
        if !input_at(4).is_empty() {
            return Err(format!("{node_tag} has per-sequence lengths"));
        }
        for (i, what) in [(5, "initial_h"), (6, "initial_c")] {
            let name = input_at(i);
            if name.is_empty() {
                continue;
            }
            let p = producer(graph, name)
                .ok_or_else(|| format!("{field_tag}{what} has no producer"))?;
            if p.op_type != "ConstantOfShape" {
                return Err(format!(
                    "{field_tag}{what} comes from {}, not zeros",
                    p.op_type
                ));
            }
            if let Some(t) = attr_tensor(p, "value")
                && tensor_f32(t)?.iter().any(|&v| v != 0.0)
            {
                return Err(format!("{field_tag}{what} is a non-zero constant"));
            }
        }
        if !input_at(7).is_empty() {
            return Err(format!("{node_tag} has peephole weights"));
        }

        Ok(LstmNode {
            hidden: h,
            n_in,
            w,
            r,
            b,
        })
    }
}

// ---- synthetic-graph builders for tests ----------------------------------

/// Builders for small synthetic ONNX graphs, shared by this crate's and
/// `escapepod-classify`'s LSTM-recognizer tests so there is one copy instead
/// of two. Public under `test` (this crate's own tests) or the
/// `test-support` feature (an external crate's dev-dependency on this one —
/// `cfg(test)` does not cross a crate boundary).
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use tract_onnx::pb;

    /// A deterministic xorshift stream in `[-scale, scale)`.
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self, scale: f32) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0) * scale
        }
    }

    pub fn f32_init(name: &str, dims: &[i64], rng: &mut Rng, scale: f32) -> pb::TensorProto {
        let n: usize = dims.iter().map(|&d| d as usize).product();
        pb::TensorProto {
            name: name.into(),
            dims: dims.to_vec(),
            data_type: super::ONNX_FLOAT,
            float_data: (0..n).map(|_| rng.next(scale)).collect(),
            ..Default::default()
        }
    }

    pub fn node(
        op: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: Vec<pb::AttributeProto>,
    ) -> pb::NodeProto {
        pb::NodeProto {
            op_type: op.into(),
            name: format!("{op}_{}", outputs[0]),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            attribute: attrs,
            ..Default::default()
        }
    }

    pub fn a_int(name: &str, i: i64) -> pb::AttributeProto {
        pb::AttributeProto {
            name: name.into(),
            r#type: pb::attribute_proto::AttributeType::Int as i32,
            i,
            ..Default::default()
        }
    }

    pub fn a_ints(name: &str, ints: &[i64]) -> pb::AttributeProto {
        pb::AttributeProto {
            name: name.into(),
            r#type: pb::attribute_proto::AttributeType::Ints as i32,
            ints: ints.to_vec(),
            ..Default::default()
        }
    }

    pub fn a_tensor(name: &str, t: pb::TensorProto) -> pb::AttributeProto {
        pb::AttributeProto {
            name: name.into(),
            r#type: pb::attribute_proto::AttributeType::Tensor as i32,
            t: Some(t),
            ..Default::default()
        }
    }

    pub fn value_info(name: &str, dims: &[i64]) -> pb::ValueInfoProto {
        pb::ValueInfoProto {
            name: name.into(),
            r#type: Some(pb::TypeProto {
                value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                    elem_type: super::ONNX_FLOAT,
                    shape: Some(pb::TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&d| pb::tensor_shape_proto::Dimension {
                                value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(d)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

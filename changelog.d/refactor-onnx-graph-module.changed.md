### Changed

- **The ONNX-proto helper toolkit duplicated between the CRF barcode encoder
  and the charging classifier's LSTM recognizer is now one module.**
  Proto-walking primitives (`producer`/`consumers`/`sole_consumer`,
  `attr`/`attr_int`/`attr_str`/`attr_ints`/`attr_tensor`,
  `tensor_f32`/`tensor_i64`), the `LSTM`-node shape validator (forbidden
  attributes, `input_forget`, `layout`, `W`/`R`/`B` dimensions, per-sequence
  lengths, a zero `ConstantOfShape` initial state, peephole weights), and the
  synthetic-graph builders both files' own tests used to construct one, were
  ~90 near-identical lines apiece in `escapepod-demux/src/crf/encoder_native.rs`
  and `escapepod-classify/src/fnn_lstm.rs`. They now live once, in
  `escapepod_demux::onnx_graph` beside `onnx_rewrite`, as
  `onnx_graph::LstmNode::parse` (taking a direction count and an optional
  stacked-layer index so one function serves the CRF encoder's five
  unidirectional layers and the charging network's single bidirectional one)
  plus `onnx_graph::test_support` (gated on `test` or the new `test-support`
  feature, since `cfg(test)` does not cross the crate boundary
  `escapepod-classify`'s own tests need it to cross). The ort CUDA session
  builder duplicated in `adapter_cnn_gpu.rs` and `crf/encoder_gpu.rs` is
  similarly folded into `ort_ep::cuda_session`. Pure code motion — no
  behavior change — verified by identical `cargo nextest`/`cargo test --doc`
  pass counts before and after (demux 172/172, classify 137/137, doctests
  4/4); net effect across the 9 touched files is 516 insertions(+), 584
  deletions(-).

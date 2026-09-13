# escapepod-python

Cluster/build policy: root `CLAUDE.md`. Published to PyPI as `escapepod`.

## Purpose

- pyo3 (`abi3-py39`) bindings over `escapepod-signal` and `escapepod-classify` (default features — no ONNX).
- Exposes `Reader`, `DatasetReader`, `Writer`, `ReadData`, `RunInfo`, `create_run_info`, `Pod5Error`; `mad_normalize`, `normalize_signal`, `refine_signal_map`, `span_statistics{,_batch}`, `KmerTable`; `AnchoredReads` + `ANCHOR_SOURCES`/`MASK_SOURCES` for the charging corpus builder.
- Marshal only: rules about windows, features, anchors live in `escapepod-classify`.

## Build and test

```bash
pixi run -e python-test build-python   # maturin develop
pixi run -e python-test test-python    # pytest tests/python/
pixi run -e python-test test-compat    # tests/compat/ against reference pod5
```

- `test-compat` is **not** in `cargo nextest` and is the only check that catches an `escpod view` regression (#242).
- `.cargo/config.toml`'s `target-cpu=x86-64-v3` applies to wheels built here.
- Heavy work off the GIL via `py.detach`; `rayon` inside.

## Invariants and traps

- Keep lazy `for rd in reader:` fast — columns resolved once per batch (`ReadsBatchView`), never per row; eager `reads()` is not the benchmark.
- `to_dict/to_pandas/to_polars` must agree between `Reader` and `DatasetReader`; `Reader` uses `build_columns` (`reader.rs:565`), `DatasetReader` does not yet.
- `AnchoredReads.__new__` takes `motif_offset` explicitly — column bundles are +3, waveform +2; never give it a default (a wrong offset anchors one base off and still returns plausible windows).
- `storage_order` / `extract` (`anchored.rs`) must order through `Pod5Index::storage_key`, never a local re-spelling — the missing-read sentinel differs and the order silently diverges from the Rust path.
- `ANCHOR_SOURCES`/`MASK_SOURCES` (`anchored.rs:549-556`) hand-copy enum discriminant order; derive from the enums.
- `adc_to_pa` (`reader.rs:14`) duplicates `classify::pipeline::signal_pa`; one `calibrate()` belongs in `escapepod-signal`.
- `Writer`: exception path aborts, GC path finalises with `ResourceWarning` — both deliberate.

## Known debt

- `read_data.rs:305-351` `PyRunInfo::new` ≡ `writer.rs:425-471` `create_run_info`.
- `read_data.rs:55-112` `PyReadData::new` vs `writer.rs:163-227` `add_read`: 23-kwarg assembly ×2.
- `reader.rs` / `dataset.rs`: `get_signal{,_pa}`, `get_signals{,_pa}`, `byte_count`, `to_*` duplicated; share free fns over `&escapepod_signal::Reader`.
- `anchored.rs:41-48` `mask_source_name` → `MaskSource::as_str`.

## Do not

- Do not materialise every `ReadData` to answer a column question — `read_columns()` exists.
- Do not add a window/mask/anchor rule here — put it in `escapepod-classify` so Rust inference reproduces it.
- Do not enable classify's ONNX features here as a drive-by — the wheel deliberately links no tract today (size, licence surface); that is a decision, not a default.
- Do not skip `test-compat` because nextest is green.

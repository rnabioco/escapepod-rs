# escapepod-classify

Cluster/build/profiling policy: root `CLAUDE.md` ("Build Commands", "Build baseline and SLURM builds"). History: `CHANGELOG.md`.

## Purpose and layering

- tRNA charging classification over POD5 + aligned BAM, against an escapepod-models bundle.
- Depends on `escapepod-signal` (k-mer, `chunk`, `lstm`, `mapping`, POD5) and `escapepod-demux` for exactly `GbmModel, load_gbm_model, GbmPredictor`, `onnx_rewrite::{hoist_conv_padding, expand_instance_norm}`. The shared ONNX-proto toolkit consolidates as `escapepod_demux::onnx_graph`; import it from there — the demux edge stays.
- Consumed by `escapepod-cli` (`escpod classify`) and `escapepod-python` (`AnchoredReads`).

## Module map

- `bundle` — closed `metadata.json` schema, checksums, variant → `ChargingBundle`/`ChargingScorer`. `recipe` — `FeatureRecipe`, the feature space as a borrowed view.
- `geometry` (FASTA motif/arm → `RefGeometry`) → `anchor` (ref→query→signal, orientation vote, `finalize`) → `features` (dwell/mean/std/resid grid) / `window` (raw pA window).
- `waveform` — windowed variant: own scan, `MD` reference, chunk assembly, CPU/GPU loops; `waveform_net{,_gpu}` — tract / tract-cuda loaders.
- `fnn` — `feature_model` loader; `fnn_lstm` — BiLSTM export recogniser (kernel in `signal::lstm`).
- `pipeline` — `scan_bam`, `Pod5Index`, `classify_reads`, `ClassifyStats`. `bam_tags` — `mv`/`ns`/`ts`/`MD`. `lib` — re-exports, `cl_from_probability`.

## Three bundle variants, one feature space

| | `gbm` | `feature_model` | `waveform_model` |
|---|---|---|---|
| input | columns per `features.order` | same, folded `[channel, offset]` + mask channels | 3 tensors via `signal::chunk` |
| scorer | trees, `NaN` native | native `signal::lstm`; `ESCAPEPOD_FNN_TRACT` = way back | tract batch 1 (`expand_instance_norm` off) / tract-cuda batched (on) |
| anchor / reference | motif +3 / basecalled query | same | motif **+2** / read's **`MD` tag**, never FASTA |
| frame | voted per run | voted | declared; `--orientation` ignored |

Shared: `geometry`, `Pod5Index` + storage-order sort, `ClassifyStats`; column variants share everything above `ChargingScorer`.

## Build, test this crate alone

```bash
srun -p rna -c 32 --mem=32G pixi run cargo nextest run -p escapepod-classify --features fnn-onnx,waveform-onnx  # → `onnx` once collapsed
srun -p rna -c 32 --mem=32G pixi run cargo check -p escapepod-classify --features cuda   # build here, run on gpu
ESCAPEPOD_CHARGING_BUNDLE=<dir> srun -p rna -c 8 pixi run cargo bench -p escapepod-classify --bench charging --features fnn-onnx
cargo run -p escapepod-classify --example verify_feature_model --features fnn-onnx   # pairs with scripts/dump_feature_model_reference.py
```

- All `tests/*.rs` run from committed `tests/fixtures/`; none need `ext/`. Goldens: `tests/fixtures/gen_*_golden.py`.
- `fnn_lstm::tests` must pass under every `ESCAPEPOD_LSTM_BACKEND` cap — the dispatch never reaches the AVX2 batched kernels on an AVX-512 node.

## Invariants and traps

- Unknown metadata key → refused, not dropped → `bundle::tests::schema`.
- Refused: non-empty `refinement.opts`, transforming `feature_set`. Carried, not applied: `abstain`, Platt, `basecaller`, `adapter_window` → `bundle::tests::feature_set`, `charging_abstain`.
- Fold checked against `features.order`; a wrong fold transposes and still scores → `bundle::tests::fold`.
- Missingness is a mask channel, never `NaN` into the graph → `charging_fnn_parity`.
- `positive_class` ∈ `classes`; shipped bundle has `charged = 0` → `bundle::tests::waveform`.
- `preprocessing.motif/motif_offset` must equal `anchor` → same.
- `Pod5Index::build` via the read index **and** sort on `storage_key` — both mandatory, ~60× otherwise; rotated locator must fail → `tests/charging_index_order.rs`.
- Orientation vote: ≥50 reads, ≥95% consensus, else error → `anchor::tests`.
- BGZF pools sized from rayon; `MultithreadedReader::new` is one worker.
- `classify_reads` feeds `par_chunks` of 8 scorer groups so LSTM lockstep keeps weights in L2.
- `features::median_of_keys` is the measured 11.5% win — upstream to `signal::stats`, don't fork.
- Every definition of the model's input lives here; cli/python only marshal.

## Env levers

| var | effect | default | test / doc |
|---|---|---|---|
| `ESCAPEPOD_FNN_TRACT` | `feature_model` through tract | native | – / root |
| `ESCAPEPOD_WAVEFORM_HOIST` | hoist conv padding, CPU (slower, bit-identical) | off | bench / – |
| `ESCAPEPOD_WAVEFORM_GPU_HOIST` | same on GPU — **changes logits** | off | – / – |
| `ESCAPEPOD_WAVEFORM_GPU_GROUPS_IN_FLIGHT` | channel depth | 64 | – / – |
| `ESCAPEPOD_WAVEFORM_GPU_PREP_CHUNK` | reads per prep burst | max(8·batch, 4096) | – / – |
| `ESCAPEPOD_LSTM_BACKEND` | cap kernel dispatch (via `signal::lstm`) | best | tests / root |
| `ESCAPEPOD_CHARGING_BUNDLE` | real bundle for bench/examples | unset | bench / root |

## Known debt

- `pipeline.rs:56-141` / `waveform.rs:288-358`: two `scan_bam`s; `move_table` pasted (`anchor.rs:183`, `waveform.rs:274`).
- `pipeline.rs:476-530`: `Scorer` mirrors `ChargingScorer`.
- `bundle.rs:207-213, 1102-1107, 1922-2004`: parallel `Option`s + `.expect("checked above")` ×5 → `enum ModelBlock`.
- `waveform_net.rs:235-298` / `waveform_net_gpu.rs:275-369`: duplicated open/pin/probe.
- `window::BaseJustify` duplicates `signal::chunk::BaseJustify`.
- `anchor::ref_to_query`: second CIGAR walk beside `signal::mapping::ref_to_signal`.
- `fnn_lstm.rs` `(backend, n)` match at `400-514` → `signal::lstm::run_batch`.

## Do not

- Do not read a rule from a flag or default the bundle can declare — a wrong guess scores, it does not error.
- Do not slice the waveform reference from the FASTA — one `N` blanks nine k-mers (#306).
- Do not vote the waveform frame or apply Platt/`abstain` silently — bundle decisions.
- Do not batch the waveform graph without `expand_instance_norm` — tract reduces over the batch axis.
- Do not inline the LSTM kernels or test one backend only — a miscompile hid that way.
- Do not scan the reads table or read in BAM order — right calls, 60× slower, invisible without a stopwatch.

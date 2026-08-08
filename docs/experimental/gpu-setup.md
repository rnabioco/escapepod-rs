# GPU setup

Three opt-in Cargo features accelerate demux stages on NVIDIA GPUs:

| Feature | Commands | What runs on the GPU |
|---------|----------|----------------------|
| `cnn-gpu` | `demux detect --method cnn --gpu`, fused `demux --method cnn --gpu` | CNN/TCN adapter detection through onnxruntime's CUDA execution provider. This is the GPU path that pays off most — detection is inference-bound, and batched GPU inference measures ~7× faster end-to-end than the CPU path on an A30. |
| `crf-gpu` | `demux basecall --gpu`, fused `demux --gpu` with a CRF model | The CTC-CRF encoder for barcode basecalling, ~4× end-to-end. Implies `crf-decode`. |
| `gpu` | `demux classify --gpu`, `demux train-svm --gpu`, fused `demux --gpu` | Batched DTW distance for SVM classification. **Experimental and usually slower** than giving the CPU a full node — see the note in [demux](demux.md). |

The features are independent; enable any combination.

## Building

Nothing CUDA-related is needed at **build** time — all three features load
their GPU runtimes with `dlopen` at **run** time, so you can build on any
machine:

```bash
cargo build --release --features cnn-gpu,crf-gpu -p escapepod-cli
# add `gpu` too if you want the experimental DTW path:
cargo build --release --features gpu,cnn-gpu,crf-gpu -p escapepod-cli
```

## Runtime libraries: the pixi environment (easiest)

At run time the features need two things, and the repository's
[pixi](https://pixi.sh) `gpu` environment supplies both:

1. **The CUDA 12 runtime libraries** — `libcublas`/`libcublasLt`,
   `libcudart`, `libcufft`, cuDNN 9 (all for the onnxruntime CUDA execution
   provider) and `libnvrtc` (for the `gpu` DTW kernels). These come from
   conda-forge via the environment's dependencies.
2. **A CUDA-enabled `libonnxruntime`** for `cnn-gpu`/`crf-gpu` — fetched once
   by the `install-ort` task into `.pixi/ort/` (it is extracted from the
   `onnxruntime-gpu` wheel; nothing is pip-installed). The version is pinned
   to match the `ort` crate the binary was built with.

Setup, in full:

```bash
# once, on a machine with network access (on clusters: the login node —
# this also creates the environment on first use)
pixi run -e gpu install-ort

# then, on a node with a visible NVIDIA GPU — no env vars needed
pixi run -e gpu ./target/release/escpod demux detect --method cnn --gpu \
    --cnn-model adapter_rna004.onnx reads.pod5 -o boundaries.csv
```

Activating the environment (`pixi run -e gpu …` or `pixi shell -e gpu`) sets
`LD_LIBRARY_PATH` and `ORT_DYLIB_PATH` automatically. The only system
requirement on the node itself is an NVIDIA driver new enough for CUDA 12.

If you only use the DTW `gpu` feature (no onnxruntime), you can skip
`install-ort` — the environment alone provides `libnvrtc`.

## Verifying the GPU is actually in use

At the default log level, escpod announces which device each stage runs on:

```
INFO Detecting adapter boundaries using boundary CNN (GPU)
INFO Encoder runs on: GPU (onnxruntime CUDA)
```

Those lines say what was *requested*; onnxruntime failures that demote the
work to CPU surface as **warnings** (visible by default), so a warning-free
run on a GPU node is a healthy one. For positive confirmation that the CUDA
execution provider loaded, raise the dependency log level — escpod pins
third-party logs at `warn` unless `RUST_LOG` overrides it:

```bash
RUST_LOG=ort=info escpod demux basecall --gpu … 2>&1 | grep CUDAExecutionProvider
# INFO [ort::ep] Successfully registered `CUDAExecutionProvider`
```

| Symptom | Cause |
|---------|-------|
| Warning that the execution provider *"may fall back to CPU"*, run is slow | A CUDA runtime library is missing (typically `libcublasLt.so.12` or `libcudnn.so.9`). Run inside the pixi `gpu` environment so `LD_LIBRARY_PATH` includes them. |
| Clear startup error: could not load onnxruntime | `ORT_DYLIB_PATH` is unset (e.g. `install-ort` was never run) or points at a CPU-only build of onnxruntime. |
| Process hangs at startup and prints **nothing**, not even a status line | `ORT_DYLIB_PATH` is set but points at a file that does not exist. The pixi activation only sets it when the library is present, so this normally means a stale manual override. |

!!! tip "On a cluster, redirect output to a file"
    When running under a scheduler, don't pipe the job's output through
    `tail` or similar — those buffer until EOF, so a healthy job looks
    identical to a hung one. Redirect to a log file and follow that instead.

## Manual setup (without pixi)

Reproduce what the environment provides:

- Put the CUDA 12 runtime on `LD_LIBRARY_PATH`: `libcublas.so.12`,
  `libcublasLt.so.12`, `libcudart.so.12`, `libcufft.so.11`,
  `libcudnn.so.9`, and `libnvrtc.so.12` for the DTW `gpu` feature.
- For `cnn-gpu`/`crf-gpu`, set `ORT_DYLIB_PATH` to a **CUDA-enabled**
  `libonnxruntime`, e.g. extracted from the `onnxruntime-gpu` wheel
  (`onnxruntime/capi/libonnxruntime.so.<version>`). The onnxruntime version
  must be compatible with the `ort` crate the binary was built with —
  current pins: onnxruntime 1.28.0 with `ort` 2.0.0-rc.13.

## HPC notes

- Anything that downloads (`pixi run -e gpu install-ort`, and the pixi
  environment creation it triggers) must run on a **networked** node —
  compute nodes typically cannot reach the internet. Neither step needs a
  GPU, so the login node is fine.
- Both the environment (`.pixi/envs/gpu`) and the onnxruntime download
  (`.pixi/ort`) live inside the project directory. On a shared filesystem
  the GPU nodes see them with no further staging.
- Model files are also fetched explicitly, never at run time — see
  `escpod demux models fetch` in [demux](demux.md).

## See Also

- [demux](demux.md) — the demultiplexing workflow these features accelerate
- [Installation](../getting-started/installation.md) — feature flags overview

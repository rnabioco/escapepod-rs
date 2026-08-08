# Sourced by pixi when activating any environment that includes the `gpu`
# feature (see [feature.gpu.activation] in pixi.toml).
#
# `ort` — the onnxruntime bindings behind the cargo features `cnn-gpu` and
# `crf-gpu` — is built with `load-dynamic`: it dlopens the library named by
# ORT_DYLIB_PATH at run time. `pixi run -e gpu install-ort` places a
# CUDA-enabled libonnxruntime under `.pixi/ort/`; point at it here, but only
# if it actually exists — a dangling ORT_DYLIB_PATH makes the ort features
# hang silently at startup, whereas leaving it unset fails fast with a clear
# "could not load onnxruntime" error that names the fix.
#
# An ORT_DYLIB_PATH already exported by the caller wins.
if [ -z "${ORT_DYLIB_PATH:-}" ]; then
    for _ort_so in "$PIXI_PROJECT_ROOT"/.pixi/ort/onnxruntime/capi/libonnxruntime.so.*; do
        if [ -e "$_ort_so" ]; then
            export ORT_DYLIB_PATH="$_ort_so"
            break
        fi
    done
    unset _ort_so
fi

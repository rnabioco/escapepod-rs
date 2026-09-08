#!/usr/bin/env bash
# Supply the unversioned `libcudart.so` that cudarc's dynamic loader tries
# FIRST — `get_lib_name_candidates` in cudarc's `src/lib.rs` checks the bare
# `libcudart.so` before any versioned name (`libcudart.so.12`, ...). conda-forge's
# `cuda-cudart`/`cuda-cudart-dev` packages do not ship that unversioned symlink,
# only `libcudart.so.<major>` and `libcudart.so.<major>.<minor>.<patch>`.
#
# Without it, `dlopen("libcudart.so")` skips straight past this env's pinned
# CUDA 12 runtime to whatever unversioned `libcudart.so` is on the node's
# system-wide library search path — a bare-metal CUDA toolkit install of a
# possibly different major version. rnabioco/escapepod-rs#347: a node with a
# system CUDA 13 install resolves there instead, and CUDA 13 dropped the
# `cudaGetDeviceProperties_v2` symbol this env's pinned CUDA 12.x runtime
# still carries — so tract-cuda's context creation panics deep inside
# cudarc's dlsym wrapper (`cudarc-0.19.9/src/runtime/sys/mod.rs`), which does
# not return a `Result`, on a symbol the correct library has.
#
# Idempotent: a no-op once the symlink exists, so a re-activation or a second
# `pixi run -e gpu` does nothing further.
if [ -n "${CONDA_PREFIX:-}" ] \
    && [ ! -e "$CONDA_PREFIX/lib/libcudart.so" ] \
    && [ -e "$CONDA_PREFIX/lib/libcudart.so.12" ]; then
    ln -s libcudart.so.12 "$CONDA_PREFIX/lib/libcudart.so"
fi

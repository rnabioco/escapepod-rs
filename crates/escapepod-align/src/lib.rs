// SPDX-License-Identifier: MIT

//! All-vs-all read-to-reference alignment for small reference panels.
//!
//! Every read is scored against every reference with an affine-gap dynamic
//! program, the best-scoring reference(s) are reported, ties are reported as
//! ties, and a traceback is computed only for the winners. There is no seed
//! index, so memory is one read's DP and runtime is a function of
//! `reads × references × lengths` alone — the right trade for a tRNA panel
//! (~160 references of ~150 nt), and the wrong one for a genome. The intended
//! range is **≤ ~10 k references of ≤ ~1 kb**; nothing here seeds, so a larger
//! panel is simply proportionally slower.
//!
//! Written clean-room from rnabioco/escapepod-rs#395. The functional model is
//! gpu-tRNA-mapper (every read against every reference, ties explicit,
//! tracebacks for winners only); none of its code or its semi-global
//! definition is reproduced.
//!
//! # Layers
//!
//! * [`alphabet`] — 2-bit A/C/G/T with U folded to T and case ignored; every
//!   other letter is a *wildcard* that pairs with anything as a match.
//! * [`Scoring`] / [`Mode`] — the scoring convention (a gap of length *k*
//!   costs `gap_open + (k − 1)·gap_extend`) and the two alignment modes.
//! * [`scalar`] — the Gotoh reference implementation, score-only and with
//!   traceback. The oracle every other path is tested against by equality.
//! * [`simd`] — score-only inter-sequence kernels (one lane per reference,
//!   i16 saturating), AVX2 and AVX-512BW, runtime-dispatched.
//! * [`sam`] — CIGAR strings and the `MD`/`NM` pair, reproducing
//!   `samtools calmd` including at reference ambiguity codes.
//! * [`Aligner`] — the panel plus the dispatch plus winner selection: what a
//!   caller actually uses.
//! * `cuda` (feature `gpu`) — the same score, one CUDA thread per (read,
//!   reference) pair, for a batch of reads at once; its [`ScoreMatrix`] goes
//!   back into [`Aligner::map_reads_scored`], which does everything after
//!   scoring exactly as the CPU path does.
//!
//! This crate does no I/O and depends on nothing but `std` (plus cudarc under
//! `gpu`); reading BAM, FASTA and FASTQ is `escapepod-cli`'s job
//! (`escpod align`).

pub mod alphabet;
#[cfg(feature = "gpu")]
pub mod cuda;
mod error;
mod mapper;
mod panel;
pub mod sam;
pub mod scalar;
mod scoring;
pub mod simd;

pub use error::AlignError;
pub use mapper::{Aligner, Hit, MapOptions, ReadMapping, ScoreMatrix};
pub use panel::{Panel, Reference};
pub use scalar::{Alignment, CigarOp};
pub use scoring::{Mode, Scoring};
pub use simd::Backend;

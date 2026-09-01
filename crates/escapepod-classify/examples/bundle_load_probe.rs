// SPDX-License-Identifier: MIT

//! Does `ChargingBundle::load` accept this bundle, and if not, why?
//!
//! Takes bundle directories (or `metadata.json` paths) and prints one verdict
//! each — the variant it holds, or the full error chain. It calls exactly the
//! loader `escpod classify` calls (`commands/classify.rs`), so a verdict here
//! is the CLI's verdict, and getting one costs no POD5, no BAM and no
//! reference.
//!
//! That last part is the point. `verify_waveform_model` answers a deeper
//! question — do the assembled tensors and the logit match the training corpus
//! — but it needs a corpus dump plus all three inputs to answer anything at
//! all. When the question is the shallow one, "will this bundle load", that
//! setup is the reason the question goes unanswered.
//!
//! # The companion probe
//!
//! `escapepod-demux/examples/tract_dynamo_probe.rs` is the *graph*-level
//! counterpart: it takes a bare `.onnx` and reports where tract gives up. Use
//! it when this one fails inside tract. Use this one otherwise, because a
//! bundle has several ways to be unloadable that have nothing to do with the
//! graph — a missing k-mer table, a checksum mismatch, or a `metadata.json` key
//! this runtime does not implement. The bundle schema is closed
//! (`deny_unknown_fields`), so a *newer* bundle is refused by an *older*
//! escpod even when its graph is fine, and that failure reads nothing like a
//! shape-inference failure.
//!
//! # The measurement (escapepod-rs 0.19.0, tract 0.23.5)
//!
//! ```text
//! charging_tcn_rna004@v0.1.0   FAIL  Failed analyse for node #192 node_conv1d ConvHir
//! charging_tcn_rna004@v0.1.1   OK    variant=waveform (onnx)
//! charging_tcn_rna004@v0.1.2   OK    variant=waveform (onnx)
//! ```
//!
//! `@v0.1.0`'s refusal is correct and permanent: it is the dynamo export tract
//! cannot shape-analyse, fixed upstream by re-exporting rather than by anything
//! this crate could do at load time (rnabioco/escapepod-models#96, leech
//! 0.10.0). `@v0.1.2` carries the *same* ONNX as `@v0.1.1` byte for byte and
//! differs only in metadata, which is why a bundle-level probe and a
//! graph-level one can disagree about it — and did, across escpod versions:
//! releases up to and including 0.18.1 have no waveform variant at all and
//! refuse both.
//!
//! # Running it
//!
//! ```bash
//! cargo run --release --example bundle_load_probe \
//!     --features waveform-onnx,fnn-onnx -- <bundle dir>...
//! ```
//!
//! Build with the features the *variant under test* needs, or the refusal you
//! measure is your own feature flags rather than the bundle: without
//! `waveform-onnx` a waveform bundle is refused at load by design, with a
//! rebuild hint.
//!
//! Git tracks only `metadata.json` and `provenance.json` in an
//! escapepod-models bundle directory — the `.onnx` and `9mer_levels_v1.txt`
//! are release artifacts. Probing a bundle straight out of that checkout
//! therefore fails on the missing k-mer table and tells you nothing about the
//! model; fetch the release, or symlink the two artifacts alongside the
//! metadata first.
use escapepod_classify::bundle::ChargingBundle;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bundle_load_probe <bundle-dir>...");
        std::process::exit(2);
    }

    // Exit non-zero if any bundle failed, so this is usable as a check and not
    // only as something to read.
    let mut failed = false;
    for a in &args {
        let name = Path::new(a)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| a.clone());
        match ChargingBundle::load(Path::new(a)) {
            Ok(b) => println!(
                "OK    {name}\n        variant={} id={} ver={:?} classes={:?}",
                b.scorer.kind(),
                b.model_id,
                b.model_version,
                b.classes
            ),
            Err(e) => {
                failed = true;
                println!("FAIL  {name}");
                // The whole chain, not just the outermost message: the cause
                // is routinely the innermost link (a missing file, a tract
                // analysis failure) while the outer one only names the bundle.
                for (i, c) in e.chain().enumerate() {
                    println!("        [{i}] {c}");
                }
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}

//! `escpod demux --model <MODEL> --info`: what is this model, and what will it do?
//!
//! Answers the questions you would otherwise answer by reading a sidecar and a
//! provenance file side by side: what the model is, what it needs from you,
//! what ships inside it, and what it scored. Prints without touching any POD5,
//! so it is safe to run against a model you are about to trust.
//!
//! Supports the CTC-CRF bundle (rich: geometry, references, pinned detector,
//! metrics) and the DTW-SVM/GBM classifier JSONs (barcode set and shape). The
//! CRF is the one that benefits, because it is the one carrying a bundle.

use std::path::Path;

use crate::style;
use escapepod_demux::{AnyModel, load_any_model};

/// Print everything the model can tell us about itself.
pub fn run(model_path: &Path) -> anyhow::Result<()> {
    #[cfg(feature = "crf-decode")]
    if let Some(dir) = super::run::crf_bundle_dir(model_path) {
        return crf_info(&dir);
    }
    classifier_info(model_path)
}

/// Human-readable byte size; bundles are small enough that MB is the ceiling.
fn human(n: u64) -> String {
    const K: u64 = 1024;
    match n {
        0..K => format!("{n} B"),
        K..0x100000 => format!("{:.1} KB", n as f64 / K as f64),
        _ => format!("{:.1} MB", n as f64 / (K * K) as f64),
    }
}

fn heading(s: &str) {
    println!("\n{}", style::action(s));
}

fn field(k: &str, v: impl std::fmt::Display) {
    println!("  {:<22} {}", style::label(k), v);
}

/// Flatten nested metric JSON into `a.b.c = value` lines.
///
/// Metrics are untyped by design (see `CrfMetadata::metrics`), so rather than
/// guessing a shape, walk whatever is there. Arrays of scalars collapse onto
/// one line; anything deeper keeps its path, which is what makes a
/// per-recovery-level table readable.
fn walk_metrics(prefix: &str, v: &serde_json::Value, out: &mut Vec<(String, String)>) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                walk_metrics(&p, val, out);
            }
        }
        serde_json::Value::Array(items) => {
            if items.iter().all(|i| !i.is_object() && !i.is_array()) {
                let joined: Vec<String> = items.iter().map(render_scalar).collect();
                out.push((prefix.to_string(), joined.join(", ")));
            } else {
                for (i, item) in items.iter().enumerate() {
                    walk_metrics(&format!("{prefix}[{i}]"), item, out);
                }
            }
        }
        other => out.push((prefix.to_string(), render_scalar(other))),
    }
}

fn render_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        // Metrics arrive as f64, so a recall prints as 0.9877713334625322.
        // Four decimals is the precision these are quoted at everywhere else;
        // integral floats (counts) lose the trailing `.0`.
        serde_json::Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() && f.fract() == 0.0 => format!("{f:.0}"),
            Some(f) if n.is_f64() => format!("{f:.4}"),
            _ => n.to_string(),
        },
        other => other.to_string(),
    }
}

#[cfg(feature = "crf-decode")]
fn crf_info(dir: &Path) -> anyhow::Result<()> {
    use escapepod_demux::crf::{BarcodeRefs, CrfMetadata};

    let meta = CrfMetadata::load(dir.join("metadata.json"))?;

    heading("Model");
    field("kind", "CTC-CRF barcode basecaller");
    field("bundle", dir.display());
    if let Some(m) = &meta.model {
        field("id", &m.id);
        if let Some(v) = &m.version {
            field("version", v);
        }
        if let Some(c) = &m.chemistry {
            field("chemistry", c);
        }
        if let Some(n) = &m.notes {
            field("notes", n);
        }
    }

    heading("Signal geometry");
    let anchor = meta.signal.anchor();
    let end = match anchor {
        escapepod_demux::crf::Anchor::AdapterEnd => "adapter_end",
        escapepod_demux::crf::Anchor::ReadEnd => "read_end",
    };
    field(
        "window",
        format!(
            "[{end} - {}, {end}]  ({} samples)",
            meta.signal.chunk, meta.signal.chunk
        ),
    );
    field("stride", meta.signal.stride);
    field("timesteps", meta.signal.chunk / meta.signal.stride);
    // The margin and the clamp both decide whether a read is decoded at all, so
    // a bundle that sets them is materially different to run and `--info` is
    // where you look before trusting a model. Neither exists without a
    // detector, and printing "min adapter_end" for a model that consumes no
    // adapter_end is how `--info` would repeat the very confusion this anchor
    // was added to end.
    if meta.needs_boundary() {
        field(
            "min adapter_end",
            format!(
                "{}  (chunk + margin {})",
                meta.min_adapter_end(),
                meta.boundary_margin()
            ),
        );
        field(
            "window clamp",
            match meta.clamp_max_shift() {
                0 => "off — a read whose adapter ends before chunk is refused".to_string(),
                n => format!(
                    "adapter_end down to {} decodes from [0, {}] (max shift {})",
                    meta.signal.chunk.saturating_sub(n),
                    meta.signal.chunk,
                    n
                ),
            },
        );
    } else {
        field(
            "boundary detector",
            "none — the window is anchored on the read end, so a read is refused \
             only if it is shorter than chunk",
        );
    }
    field(
        "standardisation",
        format!(
            "mean {:.3}, stdev {:.3}",
            meta.standardisation.mean, meta.standardisation.stdev
        ),
    );

    heading("Decoder");
    field("state_len", meta.crf.state_len);
    field("n_base", meta.crf.n_base);
    field("alphabet", meta.crf.alphabet.join(""));
    field(
        "states",
        meta.crf.n_base.pow(meta.crf.state_len as u32).to_string(),
    );
    // The single most misread property of this model: it cannot emit the first
    // `state_len` bases of its target (escapepod-models#36).
    field(
        "emits",
        format!(
            "target[{}:]  — the first {} target bases fix the initial state and are never emitted",
            meta.crf.state_len, meta.crf.state_len
        ),
    );

    heading("Barcode references");
    match &meta.barcodes {
        Some(entries) => {
            let refs = BarcodeRefs::from_pairs(
                entries.iter().map(|e| (e.name.clone(), e.sequence.clone())),
            )?;
            field("source", "bundled (no --barcodes needed)");
            field("count", refs.len());
            let lens: Vec<usize> = entries.iter().map(|e| e.sequence.len()).collect();
            let (lo, hi) = (
                lens.iter().min().copied().unwrap_or(0),
                lens.iter().max().copied().unwrap_or(0),
            );
            field(
                "length",
                if lo == hi {
                    format!("{lo} nt")
                } else {
                    format!("{lo}-{hi} nt")
                },
            );
            field(
                "min pairwise distance",
                refs.min_pairwise_distance()
                    .map_or_else(|| "n/a".into(), |d| d.to_string()),
            );
            println!();
            for e in entries {
                println!("    {:<10} {}", style::label(&e.name), e.sequence);
            }
        }
        None => {
            field("source", "NOT bundled — --barcodes <FILE> is required");
        }
    }

    heading("Boundary detector");
    match &meta.boundary {
        Some(b) => {
            field("method", &b.method);
            if let Some(id) = &b.model_id {
                field("model", id);
            }
            match &b.onnx {
                Some(o) => field("weights", format!("{o} (bundled)")),
                None => field("weights", "not bundled — supply --cnn-model"),
            }
            if let Some(sha) = &b.sha256 {
                field("sha256", format!("{sha} (checked at run)"));
            }
            match &b.input {
                Some(i) => field(
                    "input",
                    format!(
                        "raw [{}..{}] /{} -> {} positions, pad {}",
                        i.min_obs_adapter,
                        i.max_obs_trace,
                        i.downscale_factor,
                        i.input_len,
                        i.pad_value
                    ),
                ),
                None => field("input", "not declared — assuming legacy rna004 geometry"),
            }
            field("pinned", "yes — this model was calibrated against it");
        }
        // Two different "no boundary block": one means the operator has to
        // choose a detector, the other means there is nothing to choose.
        None if !meta.needs_boundary() => field(
            "pinned",
            "n/a — this model consumes no boundary detector (--method is refused)",
        ),
        None => field("pinned", "no — you must pass --method"),
    }

    if let Some(metrics) = &meta.metrics {
        heading("Published metrics");
        let mut rows = Vec::new();
        walk_metrics("", metrics, &mut rows);
        let w = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(0).min(48);
        for (k, v) in rows {
            println!("  {:<w$}  {}", style::label(&k), v, w = w);
        }
    }

    if let Some(m) = &meta.model
        && !m.caveats.is_empty()
    {
        heading("Caveats");
        for c in &m.caveats {
            println!("  {} {}", style::label("!"), c);
        }
    }

    heading("Bundled files");
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .collect();
    files.sort_by_key(std::fs::DirEntry::file_name);
    for f in files {
        let size = f.metadata().map(|m| m.len()).unwrap_or(0);
        println!(
            "  {:<28} {:>10}",
            f.file_name().to_string_lossy(),
            human(size)
        );
    }

    heading("Run it");
    let needs_barcodes = meta.barcodes.is_none();
    // A read-end model needs no detector and *refuses* `--method`, so printing
    // it here would hand back a command line that errors.
    let needs_method = meta.needs_boundary() && meta.boundary.is_none();
    let mut cmd = format!("  escpod demux <in.pod5> --model {} -d out/", dir.display());
    if needs_barcodes {
        cmd.push_str(" \\\n      --barcodes <refs.csv>");
    }
    if needs_method {
        cmd.push_str(" \\\n      --method cnn --cnn-model <adapter.onnx>");
    }
    println!("{cmd}");
    println!();
    Ok(())
}

fn classifier_info(path: &Path) -> anyhow::Result<()> {
    let model = load_any_model(path)?;
    heading("Model");
    field("bundle", path.display());
    let mapper = match &model {
        AnyModel::Svm(m) => {
            field("kind", "DTW-SVM fingerprint classifier");
            Some(&m.label_mapper)
        }
        AnyModel::Gbm(m) => {
            field("kind", "GBM fingerprint classifier");
            Some(&m.label_mapper)
        }
        AnyModel::WarpDemux(_) => {
            field(
                "kind",
                "WarpDemuX reference bank (demux classify --reference)",
            );
            None
        }
    };
    if let Some(mapper) = mapper {
        let mut ids: Vec<i32> = mapper.values().copied().filter(|&i| i >= 0).collect();
        ids.sort_unstable();
        ids.dedup();
        field("barcodes", ids.len());
        field(
            "labels",
            ids.iter()
                .map(|i| format!("BC{i:02}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    heading("Boundary detector");
    field("pinned", "no — you must pass --method");
    heading("Bundled files");
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {:<28} {:>10}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        human(size)
    );
    println!();
    Ok(())
}

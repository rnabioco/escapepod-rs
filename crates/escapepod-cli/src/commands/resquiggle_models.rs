//! Named k-mer level models: pinned manifest + verified local cache.
//!
//! Resquiggle needs a k-mer level table, but Oxford Nanopore's canonical tables
//! (<https://github.com/nanoporetech/kmer_models>, MPL-2.0) are too large — and
//! wrong-licensed — to vendor into this MIT tree. Instead we refer to models by
//! name and resolve them against a local cache populated by an **explicit**
//! prefetch (`escpod resquiggle models fetch`).
//!
//! The runtime resolution path (`resolve`) never touches the network: on this
//! project's HPC target the compute nodes generally can't reach GitHub, so a
//! lazy fetch would hang the job. Downloading is confined to the opt-in
//! `models-download` feature and is meant to be run from a networked login node
//! before submitting compute jobs. Integrity is enforced at fetch time against
//! a pinned commit + sha256; cached reads are trusted.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Subcommand;

/// Upstream commit the manifest is pinned to (nanoporetech/kmer_models).
const PINNED_COMMIT: &str = "4e56daed7fbb79b538f58e41262d5c54b07356ea";

/// Base URL for raw file access at the pinned commit. Only used by the
/// download path.
#[cfg_attr(not(feature = "models-download"), allow(dead_code))]
const RAW_BASE: &str = "https://raw.githubusercontent.com/nanoporetech/kmer_models";

/// One entry in the pinned k-mer model manifest.
pub struct ModelEntry {
    /// Model name used with `--kmer-model` and as the cached basename.
    pub name: &'static str,
    /// Path of the level table within the upstream repo at [`PINNED_COMMIT`].
    #[cfg_attr(not(any(feature = "models-download", test)), allow(dead_code))]
    pub upstream_path: &'static str,
    /// Expected sha256 of the file contents (lowercase hex).
    #[cfg_attr(not(any(feature = "models-download", test)), allow(dead_code))]
    pub sha256: &'static str,
    /// K-mer length (for display).
    pub k: usize,
}

/// The pinned manifest. sha256 values were computed by downloading each file at
/// [`PINNED_COMMIT`] and hashing (see `docs/experimental/resquiggle.md`).
pub const MODELS: &[ModelEntry] = &[
    ModelEntry {
        name: "dna_r10.4.1_e8.2_400bps",
        upstream_path: "dna_r10.4.1_e8.2_400bps/9mer_levels_v1.txt",
        sha256: "e5e154511d50c7e288c020a933fa66cc15d87ec06f903baa739e0a5c390cf23c",
        k: 9,
    },
    ModelEntry {
        name: "dna_r10.4.1_e8.2_260bps",
        upstream_path: "dna_r10.4.1_e8.2_260bps/9mer_levels_v1.txt",
        sha256: "cf9adf3650bd5ae53dae3d2a6f7c5a3f5d56dc5710d5db872e2fd29a2cc25f47",
        k: 9,
    },
    ModelEntry {
        name: "rna004",
        upstream_path: "rna004/9mer_levels_v1.txt",
        sha256: "1d366c9e060f31be1d6489b88b415666fba42f1c4006e1e549666f4a50f13e63",
        k: 9,
    },
    ModelEntry {
        name: "rna_r9.4_180mv_70bps",
        upstream_path: "rna_r9.4_180mv_70bps/5mer_levels_v1.txt",
        sha256: "a7d879305798c87cb35a6aac65a92c08f55a07103400db9bb59b2455f9ba428f",
        k: 5,
    },
];

/// Look up a model entry by name.
fn find(name: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().find(|m| m.name == name)
}

/// Comma-separated list of known model names, for error messages.
fn known_names() -> String {
    MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ")
}

/// Resolve the on-disk cache directory for k-mer models.
///
/// Precedence: `$ESCAPEPOD_KMER_CACHE` → `$XDG_CACHE_HOME/escapepod/kmer_models`
/// → `$HOME/.cache/escapepod/kmer_models`. Per the XDG spec, an empty
/// `XDG_CACHE_HOME` is treated as unset.
pub fn cache_dir() -> Result<PathBuf> {
    if let Some(d) = std::env::var_os("ESCAPEPOD_KMER_CACHE").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d).join("escapepod").join("kmer_models"));
    }
    if let Some(h) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        return Ok(PathBuf::from(h)
            .join(".cache")
            .join("escapepod")
            .join("kmer_models"));
    }
    bail!("cannot determine cache directory: set ESCAPEPOD_KMER_CACHE, XDG_CACHE_HOME, or HOME");
}

/// Cache path for a given model entry (`<cache_dir>/<name>.txt`).
fn cache_path(entry: &ModelEntry) -> Result<PathBuf> {
    Ok(cache_dir()?.join(format!("{}.txt", entry.name)))
}

/// Resolve a `--kmer-model <name>` to a cached level-table path.
///
/// Never downloads: if the model isn't cached, errors with a hint to prefetch
/// from a networked node. Loading is left to `KmerTable::from_file`.
pub fn resolve(name: &str) -> Result<PathBuf> {
    let Some(entry) = find(name) else {
        bail!(
            "unknown k-mer model '{name}'; known models: {}",
            known_names()
        );
    };
    let path = cache_path(entry)?;
    if !path.exists() {
        bail!(
            "k-mer model '{name}' is not cached (expected at {}).\n\
             Run 'escpod resquiggle models fetch {name}' from a networked node \
             (e.g. an HPC login node) before submitting compute jobs.",
            path.display()
        );
    }
    Ok(path)
}

/// K-mer model management subcommands.
#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// List known k-mer models and whether each is cached locally
    List,
    /// Print the k-mer model cache directory
    Path,
    /// Download k-mer model(s) into the cache (run from a networked node).
    ///
    /// Requires building with `--features models-download`.
    #[command(after_help = "\
Examples:
  escpod resquiggle models fetch dna_r10.4.1_e8.2_400bps
  escpod resquiggle models fetch --all
")]
    Fetch {
        /// Model name to download (omit and pass --all for every model)
        name: Option<String>,
        /// Download every model in the manifest
        #[arg(long)]
        all: bool,
    },
}

/// Dispatch a `resquiggle models` subcommand.
pub fn run(command: ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::List => list(),
        ModelsCommand::Path => {
            println!("{}", cache_dir()?.display());
            Ok(())
        }
        ModelsCommand::Fetch { name, all } => fetch(name, all),
    }
}

/// Print each manifest model with its cached/not-cached status.
fn list() -> Result<()> {
    let dir = cache_dir()?;
    println!("cache:  {}", dir.display());
    println!("pinned: nanoporetech/kmer_models@{}", &PINNED_COMMIT[..12]);
    println!();
    for m in MODELS {
        let status = if dir.join(format!("{}.txt", m.name)).exists() {
            "cached"
        } else {
            "not cached"
        };
        println!("  {:<26} {}mer  {}", m.name, m.k, status);
    }
    Ok(())
}

#[cfg(feature = "models-download")]
fn fetch(name: Option<String>, all: bool) -> Result<()> {
    match (name, all) {
        (Some(_), true) => bail!("pass either a model name or --all, not both"),
        (None, false) => {
            bail!("specify a model name or --all; see 'escpod resquiggle models list'")
        }
        (Some(name), false) => {
            let Some(entry) = find(&name) else {
                bail!(
                    "unknown k-mer model '{name}'; known models: {}",
                    known_names()
                );
            };
            fetch_entry(entry)
        }
        (None, true) => {
            for entry in MODELS {
                fetch_entry(entry)?;
            }
            Ok(())
        }
    }
}

#[cfg(not(feature = "models-download"))]
fn fetch(_name: Option<String>, _all: bool) -> Result<()> {
    bail!(
        "downloading k-mer models requires building with '--features models-download'.\n\
         Rebuild the binary with that feature (on a networked machine), or obtain the \
         table another way and pass it with 'escpod resquiggle --kmer-table <path>'."
    );
}

/// Download one model, verify its sha256 against the pinned manifest, and write
/// it atomically into the cache. Skips the download if a valid copy is present.
#[cfg(feature = "models-download")]
fn fetch_entry(entry: &ModelEntry) -> Result<()> {
    use anyhow::Context;
    use tracing::{info, warn};

    let dir = cache_dir()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating cache directory {}", dir.display()))?;
    let dest = dir.join(format!("{}.txt", entry.name));

    if dest.exists() {
        let existing = std::fs::read(&dest)?;
        if sha256_hex(&existing) == entry.sha256 {
            info!("{} already cached ({})", entry.name, dest.display());
            return Ok(());
        }
        warn!("{} cached copy failed checksum; re-downloading", entry.name);
    }

    let url = format!("{RAW_BASE}/{PINNED_COMMIT}/{}", entry.upstream_path);
    info!("downloading {} from {}", entry.name, url);
    let body = ureq::get(&url)
        .call()
        .with_context(|| format!("fetching {url}"))?
        .into_body()
        .into_with_config()
        .limit(64 * 1024 * 1024)
        .read_to_vec()
        .with_context(|| format!("reading response body from {url}"))?;

    let got = sha256_hex(&body);
    if got != entry.sha256 {
        bail!(
            "checksum mismatch for {}: expected {}, got {} (upstream changed or download corrupted)",
            entry.name,
            entry.sha256,
            got
        );
    }

    // Atomic write: temp file in the same dir, fsync-free rename.
    let tmp = dir.join(format!(".{}.tmp", entry.name));
    std::fs::write(&tmp, &body).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &dest)
        .with_context(|| format!("moving {} into place at {}", tmp.display(), dest.display()))?;

    info!(
        "cached {} ({} bytes) -> {}",
        entry.name,
        body.len(),
        dest.display()
    );
    Ok(())
}

/// Lowercase-hex sha256 of a byte slice.
#[cfg(feature = "models-download")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_well_formed() {
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{} sha256 must be 64 hex chars", m.name);
            assert!(
                m.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{} sha256 must be hex",
                m.name
            );
            assert!(m.k % 2 == 1, "{} k must be odd", m.name);
            assert!(
                m.upstream_path.contains(&format!("{}mer", m.k)),
                "{} path should reference its k",
                m.name
            );
        }
    }

    #[test]
    fn cache_dir_precedence() {
        // ESCAPEPOD_KMER_CACHE wins outright.
        temp_env(
            &[
                ("ESCAPEPOD_KMER_CACHE", Some("/x/cache")),
                ("XDG_CACHE_HOME", Some("/x/xdg")),
                ("HOME", Some("/x/home")),
            ],
            || assert_eq!(cache_dir().unwrap(), PathBuf::from("/x/cache")),
        );
        // Falls back to XDG when the explicit var is unset/empty.
        temp_env(
            &[
                ("ESCAPEPOD_KMER_CACHE", None),
                ("XDG_CACHE_HOME", Some("/x/xdg")),
                ("HOME", Some("/x/home")),
            ],
            || {
                assert_eq!(
                    cache_dir().unwrap(),
                    PathBuf::from("/x/xdg/escapepod/kmer_models")
                )
            },
        );
        // Empty XDG is treated as unset → HOME.
        temp_env(
            &[
                ("ESCAPEPOD_KMER_CACHE", None),
                ("XDG_CACHE_HOME", Some("")),
                ("HOME", Some("/x/home")),
            ],
            || {
                assert_eq!(
                    cache_dir().unwrap(),
                    PathBuf::from("/x/home/.cache/escapepod/kmer_models")
                )
            },
        );
    }

    #[test]
    fn resolve_unknown_model_errors() {
        let err = resolve("not_a_real_model").unwrap_err().to_string();
        assert!(err.contains("unknown k-mer model"), "{err}");
    }

    #[test]
    fn resolve_uncached_model_hints_prefetch() {
        temp_env(
            &[("ESCAPEPOD_KMER_CACHE", Some("/nonexistent/escpod-cache"))],
            || {
                let err = resolve("rna004").unwrap_err().to_string();
                assert!(err.contains("not cached"), "{err}");
                assert!(err.contains("models fetch rna004"), "{err}");
            },
        );
    }

    /// Set env vars for the duration of `f`, restoring prior values after.
    /// Serialized via a mutex since env is process-global.
    fn temp_env(vars: &[(&str, Option<&str>)], f: impl FnOnce()) {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(String, Option<std::ffi::OsString>)> = vars
            .iter()
            .map(|(k, _)| ((*k).to_string(), std::env::var_os(k)))
            .collect();
        for (k, v) in vars {
            match v {
                Some(val) => unsafe { std::env::set_var(k, val) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
        f();
        for (k, v) in saved {
            match v {
                Some(val) => unsafe { std::env::set_var(&k, val) },
                None => unsafe { std::env::remove_var(&k) },
            }
        }
    }
}

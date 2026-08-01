//! Named demux models: pinned manifest + verified local cache.
//!
//! Boundary and barcode model binaries are gitignored in escapepod-models and
//! distributed through GitHub Releases, so there was previously no supported
//! way to obtain them from escpod at all. This is the same shape as
//! [`crate::commands::resquiggle_models`] does for k-mer tables: refer to models
//! by name, resolve against a local cache populated by an **explicit** prefetch
//! (`escpod demux models fetch`).
//!
//! # Resolution never touches the network
//!
//! [`resolve`] only ever reads the cache. On this project's HPC target the
//! compute nodes generally cannot reach GitHub, so a lazy fetch would hang the
//! job rather than fail it. Downloading is confined to [`fetch`], meant to run
//! on a networked login node before submitting work. Integrity is enforced at
//! fetch time; cached reads are trusted.
//!
//! # The fetch unit is a bundle, not a model
//!
//! The barcode GBM is trained against a specific boundary model's output, and
//! mixing them is a silent accuracy loss — using LLR boundaries instead of the
//! matched CNN costs 17.2 points of balanced recall, and even swapping between
//! two *good* boundary models costs 0.0059 (McNemar p=3.8e-08) unless the GBM
//! is retrained. escapepod-models therefore publishes the pair as one release,
//! and fetching that release as a unit is what makes the coupling impossible to
//! break by accident. Individual members are still addressable by id for
//! [`resolve`]; they just cannot be *fetched* apart.
//!
//! # Authentication
//!
//! The models are CC-BY-4.0, but `escapepod-models` is currently a **private**
//! repository, so release assets 404 for anonymous requests — the
//! `browser_download_url` GitHub advertises is not usable without credentials.
//! Fetching therefore goes through the REST asset endpoint and sends a bearer
//! token from `$GITHUB_TOKEN` or `$GH_TOKEN` when one is set. The same endpoint
//! serves public repositories anonymously, so if the repository is opened up
//! later this keeps working with no token and no code change.
//!
//! # What is verified
//!
//! Each member's sha256 is checked after extraction, against the value pinned
//! below — which is the same value escapepod-models publishes in the release's
//! `BUNDLE.json` and per-model `provenance.json`. The archive's own checksum is
//! deliberately *not* pinned: re-packing the zip (different compression, a
//! rebuilt archive) would change it without changing a single model byte, so
//! pinning it would produce false failures while adding nothing — the member
//! hashes already cover every byte that gets used.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Subcommand;

/// Upstream repository the manifest is pinned against.
const REPO: &str = "rnabioco/escapepod-models";

/// GitHub REST API base. Used instead of `browser_download_url` because that
/// URL is unusable while the repository is private; this endpoint works for
/// both private (with a token) and public (anonymous) repositories.
#[cfg_attr(not(feature = "model-fetch"), allow(dead_code))]
const API_BASE: &str = "https://api.github.com";

/// What a model is used for. Determines which command consumes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Adapter-boundary detector for `demux detect --method cnn`.
    Boundary,
    /// Barcode classifier for `demux classify`.
    Barcode,
}

impl ModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Boundary => "boundary",
            Self::Barcode => "barcode",
        }
    }
}

/// One model inside a release bundle.
pub struct Member {
    /// Stable id, used with [`resolve`] and as the cache subdirectory prefix.
    pub id: &'static str,
    /// Pinned version, without a leading `v`.
    pub version: &'static str,
    pub kind: ModelKind,
    /// Chemistry the model is valid for. A model must never be applied blindly
    /// across chemistries, so this is carried rather than assumed.
    pub chemistry: &'static str,
    /// Path of the file within the release archive.
    #[cfg_attr(not(any(feature = "model-fetch", test)), allow(dead_code))]
    pub archive_path: &'static str,
    /// Basename the file is cached under.
    pub file: &'static str,
    /// Expected sha256 of the file contents (lowercase hex).
    #[cfg_attr(not(any(feature = "model-fetch", test)), allow(dead_code))]
    pub sha256: &'static str,
}

impl Member {
    /// `id@vversion` — how escapepod-models names the artifact.
    pub fn full_id(&self) -> String {
        format!("{}@v{}", self.id, self.version)
    }
}

/// A release that ships one or more coupled models.
pub struct Bundle {
    /// Name used with `escpod demux models fetch <name>`.
    pub name: &'static str,
    /// Release tag in [`REPO`].
    #[cfg_attr(not(any(feature = "model-fetch", test)), allow(dead_code))]
    pub tag: &'static str,
    /// Release asset filename.
    #[cfg_attr(not(any(feature = "model-fetch", test)), allow(dead_code))]
    pub asset: &'static str,
    /// One-line description for `models list`.
    pub description: &'static str,
    pub members: &'static [Member],
}

/// The pinned manifest.
///
/// sha256 values are taken from the release's `BUNDLE.json` and independently
/// re-hashed from the downloaded artifacts before being recorded here.
pub const BUNDLES: &[Bundle] = &[Bundle {
    name: "wdx4_rna004",
    tag: "wdx4_rna004_bundle@v1.1.0",
    asset: "wdx4_rna004_bundle@v1.1.0.zip",
    description: "RNA004 boundary CNN + WDX4 barcode GBM (matched pair)",
    members: &[
        Member {
            id: "adapter_rna004",
            version: "1.1.0",
            kind: ModelKind::Boundary,
            chemistry: "rna004",
            archive_path: "adapter_rna004@v1.1.0/adapter_rna004.onnx",
            file: "adapter_rna004.onnx",
            sha256: "b59f8667187ef9fa7e940cd37b108f8d5f3c6d6213ca841cda6eced0e33d26b5",
        },
        Member {
            id: "barcode_wdx4_rna004",
            version: "1.1.0",
            kind: ModelKind::Barcode,
            chemistry: "rna004",
            archive_path: "barcode_wdx4_rna004@v1.1.0/barcode_wdx4_rna004.gbm.json",
            file: "barcode_wdx4_rna004.gbm.json",
            sha256: "75b74a92a9996bba250b5936c536221f3e156a3dcd1947cb8a046daded4fec1d",
        },
    ],
}];

/// Every member across every bundle.
pub fn members() -> impl Iterator<Item = (&'static Bundle, &'static Member)> {
    BUNDLES
        .iter()
        .flat_map(|b| b.members.iter().map(move |m| (b, m)))
}

/// Look up a bundle by name.
fn find_bundle(name: &str) -> Option<&'static Bundle> {
    BUNDLES.iter().find(|b| b.name == name)
}

/// Look up a model by id, or by the fully-qualified `id@vversion`.
fn find_member(id: &str) -> Option<(&'static Bundle, &'static Member)> {
    members().find(|(_, m)| m.id == id || m.full_id() == id)
}

fn known_bundles() -> String {
    BUNDLES
        .iter()
        .map(|b| b.name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn known_models() -> String {
    members().map(|(_, m)| m.id).collect::<Vec<_>>().join(", ")
}

/// Resolve the on-disk cache directory for demux models.
///
/// Precedence: `$ESCAPEPOD_DEMUX_MODEL_CACHE` → `$XDG_CACHE_HOME/escapepod/demux_models`
/// → `$HOME/.cache/escapepod/demux_models`. Per the XDG spec, an empty
/// `XDG_CACHE_HOME` is treated as unset.
pub fn cache_dir() -> Result<PathBuf> {
    if let Some(d) = std::env::var_os("ESCAPEPOD_DEMUX_MODEL_CACHE").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d).join("escapepod").join("demux_models"));
    }
    if let Some(h) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        return Ok(PathBuf::from(h)
            .join(".cache")
            .join("escapepod")
            .join("demux_models"));
    }
    bail!(
        "cannot determine cache directory: set ESCAPEPOD_DEMUX_MODEL_CACHE, XDG_CACHE_HOME, or HOME"
    );
}

/// Cache path for one member: `<cache>/<id>@v<version>/<file>`.
///
/// Versioned so two versions of a model can coexist and a run stays traceable
/// to the exact artifact it used.
fn member_path(member: &Member) -> Result<PathBuf> {
    Ok(cache_dir()?.join(member.full_id()).join(member.file))
}

/// Resolve a model id to a cached path. Never downloads.
///
/// Accepts a bare id (`adapter_rna004`, giving the pinned version) or a
/// fully-qualified `adapter_rna004@v1.1.0`.
pub fn resolve(id: &str) -> Result<PathBuf> {
    let Some((bundle, member)) = find_member(id) else {
        bail!(
            "unknown demux model '{id}'; known models: {}",
            known_models()
        );
    };
    let path = member_path(member)?;
    if !path.exists() {
        bail!(
            "demux model '{id}' is not cached (expected at {}).\n\
             Run 'escpod demux models fetch {}' from a networked node \
             (e.g. an HPC login node) before submitting compute jobs.",
            path.display(),
            bundle.name
        );
    }
    Ok(path)
}

/// Demux model management subcommands.
#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// List known demux models and whether each is cached locally
    List,
    /// Print the demux model cache directory
    Path,
    /// Download model bundle(s) into the cache (run from a networked node).
    #[command(after_help = "\
Examples:
  escpod demux models fetch wdx4_rna004
  escpod demux models fetch --all

Fetches a matched bundle, not individual models: the barcode classifier is
trained against a specific boundary model's output, and mixing them silently
loses accuracy.
")]
    Fetch {
        /// Bundle name to download (omit and pass --all for every bundle)
        name: Option<String>,
        /// Download every bundle in the manifest
        #[arg(long)]
        all: bool,
    },
}

/// Dispatch a `demux models` subcommand.
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

/// Print each manifest entry with its cached/not-cached status.
fn list() -> Result<()> {
    let dir = cache_dir()?;
    println!("cache: {}", dir.display());
    println!("repo:  {REPO}");
    for b in BUNDLES {
        println!();
        println!("{}  ({})", b.name, b.description);
        for m in b.members {
            let status = if dir.join(m.full_id()).join(m.file).exists() {
                "cached"
            } else {
                "not cached"
            };
            println!(
                "  {:<28} {:<9} {:<8} {}",
                m.full_id(),
                m.kind.as_str(),
                m.chemistry,
                status
            );
        }
    }
    Ok(())
}

#[cfg(feature = "model-fetch")]
fn fetch(name: Option<String>, all: bool) -> Result<()> {
    match (name, all) {
        (Some(_), true) => bail!("pass either a bundle name or --all, not both"),
        (None, false) => bail!("specify a bundle name or --all; see 'escpod demux models list'"),
        (Some(name), false) => {
            let Some(bundle) = find_bundle(&name) else {
                // A model id is the likely mistake, so name the bundle it lives in.
                if let Some((b, _)) = find_member(&name) {
                    bail!(
                        "'{name}' is a model, not a bundle; fetch its bundle instead: \
                         'escpod demux models fetch {}'",
                        b.name
                    );
                }
                bail!(
                    "unknown bundle '{name}'; known bundles: {}",
                    known_bundles()
                );
            };
            fetch_bundle(bundle)
        }
        (None, true) => {
            for bundle in BUNDLES {
                fetch_bundle(bundle)?;
            }
            Ok(())
        }
    }
}

#[cfg(not(feature = "model-fetch"))]
fn fetch(_name: Option<String>, _all: bool) -> Result<()> {
    bail!(
        "downloading demux models requires building with '--features model-fetch'.\n\
         Rebuild the binary with that feature (on a networked machine), or obtain the \
         files another way and pass them explicitly with '--cnn-model' / '--model'."
    );
}

/// Download one bundle, verify every member's sha256, and write them into the
/// cache. Members already present with a matching hash are left alone, so a
/// re-run is cheap and a partially-populated cache self-heals.
#[cfg(feature = "model-fetch")]
fn fetch_bundle(bundle: &Bundle) -> Result<()> {
    use anyhow::Context;
    use std::io::Read;
    use tracing::{info, warn};

    let dir = cache_dir()?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating cache directory {}", dir.display()))?;

    // Skip the download entirely when every member is already good.
    let mut missing = Vec::new();
    for m in bundle.members {
        let dest = dir.join(m.full_id()).join(m.file);
        match std::fs::read(&dest) {
            Ok(bytes) if sha256_hex(&bytes) == m.sha256 => {
                info!("{} already cached ({})", m.full_id(), dest.display());
            }
            Ok(_) => {
                warn!(
                    "{} cached copy failed checksum; re-downloading",
                    m.full_id()
                );
                missing.push(m);
            }
            Err(_) => missing.push(m),
        }
    }
    if missing.is_empty() {
        return Ok(());
    }

    let body = download_asset(bundle)?;

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&body))
        .with_context(|| format!("opening {} as a zip archive", bundle.asset))?;

    for m in missing {
        let mut entry = archive.by_name(m.archive_path).with_context(|| {
            format!(
                "{} does not contain {} (release layout changed?)",
                bundle.asset, m.archive_path
            )
        })?;
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut bytes)
            .with_context(|| format!("extracting {}", m.archive_path))?;

        let got = sha256_hex(&bytes);
        if got != m.sha256 {
            bail!(
                "checksum mismatch for {}: expected {}, got {} \
                 (release re-published or download corrupted)",
                m.full_id(),
                m.sha256,
                got
            );
        }

        let subdir = dir.join(m.full_id());
        std::fs::create_dir_all(&subdir)
            .with_context(|| format!("creating {}", subdir.display()))?;
        let dest = subdir.join(m.file);
        // Atomic publish: write beside the target, then rename, so a killed
        // fetch never leaves a half-written model that `resolve` would accept.
        let tmp = subdir.join(format!(".{}.tmp", m.file));
        std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &dest)
            .with_context(|| format!("moving {} into place", dest.display()))?;
        info!(
            "cached {} ({} bytes) -> {}",
            m.full_id(),
            bytes.len(),
            dest.display()
        );
    }
    Ok(())
}

/// Percent-encode the characters that appear in our release tags but are not
/// safe in a URL path segment.
///
/// Tags are `id@vX.Y.Z`. Deliberately minimal rather than a general encoder:
/// the inputs are compile-time constants from the manifest, not user data.
#[cfg(feature = "model-fetch")]
fn encode_path_segment(s: &str) -> String {
    s.replace('@', "%40")
}

/// Bearer token for the GitHub API, if the environment provides one.
///
/// `GITHUB_TOKEN` then `GH_TOKEN` — the two names CI and the `gh` CLI already
/// use, so a machine that can `gh release download` can usually fetch too.
#[cfg(feature = "model-fetch")]
fn github_token() -> Option<String> {
    ["GITHUB_TOKEN", "GH_TOKEN"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()))
}

/// Download a bundle's release asset through the REST API.
///
/// Two requests: resolve the tag to its asset list, then fetch the asset by id.
/// The asset id is deliberately *not* pinned in the manifest — it changes if a
/// release is re-uploaded, which would turn a routine re-publish into a
/// confusing 404. The tag plus asset name is the pin, and the member checksums
/// are what actually guarantee the bytes.
#[cfg(feature = "model-fetch")]
fn download_asset(bundle: &Bundle) -> Result<Vec<u8>> {
    use anyhow::Context;
    use tracing::info;

    let token = github_token();
    let auth = |req: ureq::RequestBuilder<_>| match &token {
        Some(t) => req.header("Authorization", &format!("Bearer {t}")),
        None => req,
    };
    // GitHub rejects API requests without a User-Agent.
    let ua = concat!("escpod/", env!("CARGO_PKG_VERSION"));

    let tag_url = format!(
        "{API_BASE}/repos/{REPO}/releases/tags/{}",
        encode_path_segment(bundle.tag)
    );
    // Parsed with serde_json rather than ureq's `json` feature, which is not
    // enabled — the workspace pins ureq to `default-features = false` plus
    // rustls so the static-musl release stays OpenSSL-free.
    let meta_body = auth(ureq::get(&tag_url))
        .header("User-Agent", ua)
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| annotate_auth(e, &token))
        .with_context(|| format!("looking up release {}", bundle.tag))?
        .into_body()
        .read_to_string()
        .with_context(|| format!("reading release metadata for {}", bundle.tag))?;
    let meta: serde_json::Value = serde_json::from_str(&meta_body)
        .with_context(|| format!("parsing release metadata for {}", bundle.tag))?;

    let asset_id = meta["assets"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|a| a["name"].as_str() == Some(bundle.asset))
        .and_then(|a| a["id"].as_u64())
        .with_context(|| {
            format!(
                "release {} has no asset named {} (release layout changed?)",
                bundle.tag, bundle.asset
            )
        })?;

    let asset_url = format!("{API_BASE}/repos/{REPO}/releases/assets/{asset_id}");
    info!("downloading {} from {}", bundle.name, asset_url);
    auth(ureq::get(&asset_url))
        .header("User-Agent", ua)
        // Asks for the bytes rather than the JSON description of the asset.
        .header("Accept", "application/octet-stream")
        .call()
        .map_err(|e| annotate_auth(e, &token))
        .with_context(|| format!("fetching {asset_url}"))?
        .into_body()
        .into_with_config()
        .limit(256 * 1024 * 1024)
        .read_to_vec()
        .with_context(|| format!("reading response body from {asset_url}"))
}

/// Turn an anonymous 404 into the explanation it almost always is.
///
/// A private repository returns 404 rather than 401 for unauthenticated
/// requests — deliberately, so it cannot be probed for existence — which makes
/// the bare error deeply misleading here.
#[cfg(feature = "model-fetch")]
fn annotate_auth(err: ureq::Error, token: &Option<String>) -> anyhow::Error {
    let is_404 = matches!(&err, ureq::Error::StatusCode(404));
    let err = anyhow::Error::new(err);
    if is_404 && token.is_none() {
        return err.context(
            "not found, and no GitHub token was provided. escapepod-models is currently a \
             private repository, which returns 404 rather than 401 for anonymous requests. \
             Set GITHUB_TOKEN or GH_TOKEN to a token with read access and retry.",
        );
    }
    err
}

/// Lowercase-hex sha256 of a byte slice.
#[cfg(feature = "model-fetch")]
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
        for b in BUNDLES {
            assert!(!b.members.is_empty(), "{} has no members", b.name);
            assert!(
                b.asset.ends_with(".zip"),
                "{} asset should be a zip",
                b.name
            );
            assert!(b.tag.contains('@'), "{} tag should be id@vX.Y.Z", b.name);
            for m in b.members {
                assert_eq!(m.sha256.len(), 64, "{} sha256 must be 64 hex chars", m.id);
                assert!(
                    m.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                    "{} sha256 must be hex",
                    m.id
                );
                assert!(
                    m.archive_path.ends_with(m.file),
                    "{} archive_path should end with its cached basename",
                    m.id
                );
                assert!(
                    m.archive_path.starts_with(&m.full_id()),
                    "{} archive_path should be namespaced by full_id",
                    m.id
                );
            }
        }
    }

    /// Ids must be unique across bundles or `resolve` becomes ambiguous.
    #[test]
    fn model_ids_are_unique() {
        let mut seen = Vec::new();
        for (_, m) in members() {
            assert!(!seen.contains(&m.id), "duplicate model id {}", m.id);
            seen.push(m.id);
        }
    }

    /// The coupling this whole design exists to protect: a bundle that ships a
    /// barcode model must also ship the boundary model it was trained against.
    #[test]
    fn barcode_models_travel_with_a_boundary_model() {
        for b in BUNDLES {
            if b.members.iter().any(|m| m.kind == ModelKind::Barcode) {
                assert!(
                    b.members.iter().any(|m| m.kind == ModelKind::Boundary),
                    "{} ships a barcode model without its boundary model",
                    b.name
                );
            }
        }
    }

    #[test]
    fn lookup_accepts_bare_and_qualified_ids() {
        let (_, m) = find_member("adapter_rna004").expect("bare id");
        assert_eq!(m.version, "1.1.0");
        let (_, q) = find_member("adapter_rna004@v1.1.0").expect("qualified id");
        assert_eq!(q.id, m.id);
        assert!(find_member("adapter_rna004@v9.9.9").is_none());
        assert!(find_member("nope").is_none());
    }

    #[test]
    fn resolve_is_offline_and_explains_how_to_prefetch() {
        let tmp = tempfile::tempdir().unwrap();
        crate::test_env::temp_env(
            &[(
                "ESCAPEPOD_DEMUX_MODEL_CACHE",
                Some(tmp.path().to_str().unwrap()),
            )],
            || {
                let err = resolve("adapter_rna004").unwrap_err().to_string();
                assert!(err.contains("not cached"), "{err}");
                assert!(err.contains("demux models fetch wdx4_rna004"), "{err}");
            },
        );
    }

    #[test]
    fn cache_dir_precedence() {
        crate::test_env::temp_env(
            &[
                ("ESCAPEPOD_DEMUX_MODEL_CACHE", Some("/x/cache")),
                ("XDG_CACHE_HOME", Some("/x/xdg")),
                ("HOME", Some("/x/home")),
            ],
            || assert_eq!(cache_dir().unwrap(), PathBuf::from("/x/cache")),
        );
        crate::test_env::temp_env(
            &[
                ("ESCAPEPOD_DEMUX_MODEL_CACHE", None),
                ("XDG_CACHE_HOME", Some("/x/xdg")),
                ("HOME", Some("/x/home")),
            ],
            || {
                assert_eq!(
                    cache_dir().unwrap(),
                    PathBuf::from("/x/xdg/escapepod/demux_models")
                )
            },
        );
        // Empty XDG is treated as unset, per the spec.
        crate::test_env::temp_env(
            &[
                ("ESCAPEPOD_DEMUX_MODEL_CACHE", None),
                ("XDG_CACHE_HOME", Some("")),
                ("HOME", Some("/x/home")),
            ],
            || {
                assert_eq!(
                    cache_dir().unwrap(),
                    PathBuf::from("/x/home/.cache/escapepod/demux_models")
                )
            },
        );
    }

    #[cfg(feature = "model-fetch")]
    #[test]
    fn tags_are_url_encoded() {
        assert_eq!(
            encode_path_segment("wdx4_rna004_bundle@v1.1.0"),
            "wdx4_rna004_bundle%40v1.1.0"
        );
    }

    #[cfg(feature = "model-fetch")]
    #[test]
    fn sha256_matches_a_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

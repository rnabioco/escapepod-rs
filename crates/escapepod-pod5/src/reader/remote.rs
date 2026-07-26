//! Reading POD5 objects out of object storage over range requests.
//!
//! A POD5 file is self-describing from its tail: the last few dozen bytes point
//! at a FlatBuffer footer, which points at the embedded Arrow tables. So
//! inspecting a multi-GB object costs a handful of range GETs — the tail, the
//! footer, and whichever table the command actually reads — instead of a full
//! download. That is the whole point of this module; [`RemoteSource`] is just a
//! [`ByteSource`] that turns a byte range into a GET.
//!
//! **Read-only, and metadata-shaped.** Commands that consume the reads table
//! (`inspect`, `view`, `summary`) transfer a few MB. Commands that consume
//! *signal* still pull the entire signal table, because signal is currently
//! fetched a whole table at a time — see the `signal_table_bytes` docs. Writing
//! remotely is not supported at all.
//!
//! ## Async bridge
//!
//! `object_store` is async; [`Reader`](super::Reader) is not, and making it
//! async would infect every iterator, CLI command, and rayon pipeline
//! downstream. Instead one process-wide runtime is created on first use and
//! `block_on` bridges each request. It must be process-wide rather than
//! per-call: rayon workers call into signal extraction concurrently, and
//! building a runtime per request would be both ruinous and prone to nesting
//! panics.

use crate::error::{Error, Result};
use bytes::Bytes;
use object_store::{ObjectStore, path::Path as ObjectPath};
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Runtime;
use url::Url;

use super::byte_source::{ByteSource, check_range};

/// The single runtime that drives every remote request in this process.
static REMOTE_RT: OnceLock<Runtime> = OnceLock::new();

/// Render an error together with its source chain.
///
/// `object_store` wraps `reqwest`, which wraps `hyper`, and each layer's
/// `Display` shows only its own message — the top-level text is routinely a
/// content-free "builder error" or "error sending request" while the actual
/// cause (DNS, TLS, a 403) sits two links down. Users get the chain.
fn chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut src = e.source();
    while let Some(cause) = src {
        out.push_str(&format!(": {cause}"));
        src = cause.source();
    }
    out
}

fn runtime() -> Result<&'static Runtime> {
    if let Some(rt) = REMOTE_RT.get() {
        return Ok(rt);
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        // Remote reads are latency-bound, not CPU-bound; a small pool is
        // plenty and leaves the cores for rayon's decompression workers.
        .worker_threads(4)
        .thread_name("escapepod-remote")
        .build()
        .map_err(|e| Error::Remote(format!("could not start async runtime: {e}")))?;
    Ok(REMOTE_RT.get_or_init(|| rt))
}

/// A [`ByteSource`] backed by an object-storage or HTTP object.
pub struct RemoteSource {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    len: u64,
    url: String,
}

impl RemoteSource {
    /// Connect to `url` and resolve the object's size.
    ///
    /// Credentials come from `object_store`'s standard chain — for S3 that is
    /// `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`,
    /// `AWS_REGION`, `AWS_ENDPOINT` (set that for MinIO or any other
    /// S3-compatible endpoint), and the instance/profile providers.
    pub fn open(url: &str) -> Result<Self> {
        Self::open_with_options(url, Vec::<(String, String)>::new())
    }

    /// Like [`Self::open`] but with explicit `object_store` configuration keys,
    /// which take precedence over the environment.
    pub fn open_with_options<I, K, V>(url: &str, options: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let parsed =
            Url::parse(url).map_err(|e| Error::Remote(format!("invalid URL {url}: {e}")))?;

        // object_store refuses cleartext HTTP unless told otherwise. When the
        // user typed `http://` themselves that refusal is just noise, so allow
        // it for that scheme only — `https://` and the cloud schemes keep the
        // strict default. (An S3-compatible endpoint reached over cleartext is
        // a different setting, and stays opt-in via `AWS_ALLOW_HTTP=true`.)
        let mut opts: Vec<(String, String)> = Vec::new();
        if parsed.scheme() == "http" {
            opts.push(("allow_http".to_string(), "true".to_string()));
        }
        // Caller-supplied options come last so they override the defaults above.
        opts.extend(
            options
                .into_iter()
                .map(|(k, v)| (k.as_ref().to_string(), v.into())),
        );

        let (store, path) = object_store::parse_url_opts(&parsed, opts)
            .map_err(|e| Error::Remote(format!("cannot address {url}: {}", chain(&e))))?;
        let store: Arc<dyn ObjectStore> = Arc::from(store);

        let rt = runtime()?;
        let meta = rt
            .block_on(store.head(&path))
            .map_err(|e| Error::Remote(format!("HEAD {url} failed: {}", chain(&e))))?;

        Ok(Self {
            store,
            path,
            len: meta.size,
            url: url.to_string(),
        })
    }
}

impl ByteSource for RemoteSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_range(&self, offset: u64, len: u64) -> Result<Bytes> {
        let r = check_range(offset, len, self.len, "object")?;
        if r.is_empty() {
            return Ok(Bytes::new());
        }
        let rt = runtime()?;
        rt.block_on(self.store.get_range(&self.path, r.clone()))
            .map_err(|e| {
                Error::Remote(format!(
                    "GET {} bytes {}..{} failed: {}",
                    self.url,
                    r.start,
                    r.end,
                    chain(&e)
                ))
            })
    }

    fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        for r in ranges {
            check_range(r.start, r.end.saturating_sub(r.start), self.len, "object")?;
        }
        let rt = runtime()?;
        // `get_ranges` coalesces adjacent/nearby ranges into fewer, larger GETs
        // and issues the rest concurrently — the reason this override exists.
        rt.block_on(self.store.get_ranges(&self.path, ranges))
            .map_err(|e| Error::Remote(format!("GET {} ranges failed: {}", self.url, chain(&e))))
    }

    fn describe(&self) -> String {
        self.url.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every scheme `is_remote_url` claims must be one `parse_url_opts` can
    /// actually route — otherwise the CLI sends an input down the remote path
    /// and it dies at the URL layer instead of falling back to a local path.
    ///
    /// Each entry carries the minimum configuration that backend needs to
    /// *construct* a store. Azure is the reason this is a table rather than a
    /// bare list: an `az://container/path` URL carries no account name, so the
    /// builder needs one from the environment (`AZURE_STORAGE_ACCOUNT_NAME`)
    /// exactly as S3 needs credentials.
    #[test]
    fn every_advertised_scheme_resolves_to_a_store() {
        let azure = [("azure_storage_account_name", "testaccount")];
        let cases: [(&str, &[(&str, &str)]); 7] = [
            ("s3://bucket/reads.pod5", &[]),
            ("s3a://bucket/reads.pod5", &[]),
            ("gs://bucket/reads.pod5", &[]),
            ("az://container/reads.pod5", &azure),
            ("abfs://container/reads.pod5", &azure),
            ("http://example.org/reads.pod5", &[("allow_http", "true")]),
            ("https://example.org/reads.pod5", &[]),
        ];

        for (url, opts) in cases {
            assert!(
                crate::is_remote_url(url),
                "{url} should be recognised as remote"
            );
            let parsed = Url::parse(url).unwrap();
            // Store construction is offline — it does not touch the network.
            if let Err(e) = object_store::parse_url_opts(&parsed, opts.iter().copied()) {
                panic!("object_store cannot address {url}: {}", chain(&e));
            }
        }
    }
}

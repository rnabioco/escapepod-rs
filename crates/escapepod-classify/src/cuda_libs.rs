// SPDX-License-Identifier: MIT

//! Make sure the cuBLAS that `tract-cuda` is about to use is paired with the
//! cuBLASLt it was built against (rnabioco/escapepod-rs#416).
//!
//! # What went wrong
//!
//! cudarc's `dynamic-loading` opens cuBLAS by trying a list of names, and the
//! first is the **unversioned** `libcublas.so` (`cudarc::get_lib_name_candidates`).
//! A conda/pixi CUDA *runtime* environment ships only the versioned
//! `libcublas.so.12` — the unversioned symlink is a `-dev` file — so
//! `LD_LIBRARY_PATH` cannot answer that name and the dynamic loader falls
//! through to the host's `ldconfig` cache. On a node with a system CUDA
//! toolkit that is registered there, that is the *host's* `libcublas.so`
//! (compgpu03: `/usr/local/cuda-12.8`). Its own `NEEDED libcublasLt.so.12`,
//! though, **is** answered by `LD_LIBRARY_PATH` — the environment's 12.9. The
//! process ends up with cuBLAS 12.8 driving cuBLASLt 12.9.
//!
//! Nothing errors. The pair loads, creates handles, and runs GEMMs whose
//! results are garbage: the windowed charging TCN's GPU output correlated
//! 0.009 with the CPU scorer on 20k real reads, and which wrong answer came
//! back depended on the batch size and on the build (allocation alignment
//! steers cuBLASLt's heuristic). Pointing either side at its own sibling —
//! 12.8/12.8 or 12.9/12.9 — restored exact parity, which is the proof that the
//! pairing, not either library, is the fault. A node without a host toolkit in
//! the cache (compgpu01) never sees the mix, which is why the bug looked like
//! a property of the node.
//!
//! # What this does
//!
//! Before the graph is moved onto CUDA, [`ensure_cublas_pairing`] opens cuBLAS
//! by the same names in the same order cudarc will, reads which release each
//! loaded file belongs to (its resolved file name — see [`release_of`] for
//! why not the libraries' own version calls), and if they disagree, closes cuBLAS again (which unloads the stray cuBLASLt with it)
//! and pre-loads the cuBLASLt that sits **beside the cuBLAS that won**, by
//! absolute path. The loader matches a `NEEDED` entry against already-loaded
//! objects by soname, so the reopened cuBLAS binds to that sibling. The result
//! is re-checked, and a pairing that still disagrees is an error — never a
//! GPU run.
//!
//! The handles opened here are deliberately leaked: they pin the libraries
//! cudarc will find, so nothing between this check and the first GEMM can
//! change which files those are.

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};

/// The two libraries' versions and where they were loaded from, as found.
#[derive(Debug, Clone)]
pub struct CublasPairing {
    pub cublas_path: PathBuf,
    pub cublas_version: String,
    pub lt_path: PathBuf,
    pub lt_version: String,
    /// True when the first pairing disagreed and was repaired by pre-loading
    /// the sibling cuBLASLt.
    pub repaired: bool,
}

impl CublasPairing {
    fn agrees(&self) -> bool {
        self.cublas_version == self.lt_version
    }
}

impl std::fmt::Display for CublasPairing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cuBLAS {} ({}) with cuBLASLt {} ({})",
            self.cublas_version,
            self.cublas_path.display(),
            self.lt_version,
            self.lt_path.display()
        )
    }
}

/// Every loaded object whose file name starts with `prefix`, as the loader
/// recorded its path.
fn loaded_objects(prefix: &str) -> Vec<PathBuf> {
    struct Acc<'a> {
        prefix: &'a str,
        out: Vec<PathBuf>,
    }
    unsafe extern "C" fn cb(
        info: *mut libc::dl_phdr_info,
        _size: libc::size_t,
        data: *mut c_void,
    ) -> c_int {
        // Safety: `data` is the `Acc` passed below, and `dlpi_name` is a
        // NUL-terminated string owned by the loader for the callback's span.
        unsafe {
            let acc = &mut *(data as *mut Acc);
            let name = (*info).dlpi_name;
            if !name.is_null() {
                let path = PathBuf::from(CStr::from_ptr(name).to_string_lossy().into_owned());
                if path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .is_some_and(|f| f.starts_with(acc.prefix))
                {
                    acc.out.push(path);
                }
            }
        }
        0
    }
    let mut acc = Acc {
        prefix,
        out: Vec::new(),
    };
    // Safety: `cb` only reads loader-owned data and writes into `acc`.
    unsafe {
        libc::dl_iterate_phdr(Some(cb), &mut acc as *mut Acc as *mut c_void);
    }
    acc.out
}

fn dl_error() -> String {
    // Safety: dlerror returns a thread-local string or null.
    unsafe {
        let e = libc::dlerror();
        if e.is_null() {
            "unknown dlopen error".to_string()
        } else {
            CStr::from_ptr(e).to_string_lossy().into_owned()
        }
    }
}

fn dlopen(name: &str, flags: c_int) -> Option<*mut c_void> {
    let c = CString::new(name).ok()?;
    // Safety: dlopen with a valid C string.
    let h = unsafe { libc::dlopen(c.as_ptr(), flags) };
    (!h.is_null()).then_some(h)
}

/// The release a loaded library file belongs to: the version suffix of the
/// file its path resolves to (`libcublas.so.12.8.3.14` -> `12.8.3.14`), which
/// is how every CUDA toolkit and conda package names them. Falls back to the
/// resolved directory when the file carries no full version, so two files
/// from one install still compare equal.
///
/// Not `cublasGetProperty`: in CUDA 12 that entry point answers with the
/// *cuBLASLt* it is linked to — on compgpu03 a cuBLAS 12.8.3 reported 12.9.2,
/// the version of the stray cuBLASLt, which is exactly the lie this check is
/// here to see through.
fn release_of(path: &Path, stem: &str) -> Result<String, String> {
    let real = std::fs::canonicalize(path)
        .map_err(|e| format!("cannot resolve {}: {e}", path.display()))?;
    let name = real
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default();
    match name.strip_prefix(stem) {
        Some(v) if v.split('.').count() >= 3 && v.split('.').all(|p| p.parse::<u32>().is_ok()) => {
            Ok(v.to_string())
        }
        _ => Ok(format!(
            "(unversioned, in {})",
            real.parent().unwrap_or(Path::new("/")).display()
        )),
    }
}

/// Open cuBLAS by cudarc's candidate names, in cudarc's order, and read what
/// got paired with it.
fn open_and_inspect() -> Result<(*mut c_void, CublasPairing), String> {
    let candidates = cudarc::get_lib_name_candidates("cublas");
    let (name, handle) = candidates
        .iter()
        .find_map(|c| dlopen(c, libc::RTLD_LAZY | libc::RTLD_LOCAL).map(|h| (c.clone(), h)))
        .ok_or_else(|| {
            format!(
                "no cuBLAS library could be opened (tried {})",
                candidates.join(", ")
            )
        })?;
    let first = |prefix: &str, what: &str| {
        loaded_objects(prefix)
            .into_iter()
            .next()
            .ok_or_else(|| format!("{what} is not loaded after opening {name}"))
    };
    let cublas_path = first("libcublas.so", "libcublas")?;
    let lt_path = first("libcublasLt.so", "libcublasLt")?;
    let pairing = CublasPairing {
        cublas_version: release_of(&cublas_path, "libcublas.so.")?,
        lt_version: release_of(&lt_path, "libcublasLt.so.")?,
        cublas_path,
        lt_path,
        repaired: false,
    };
    Ok((handle, pairing))
}

/// Check — and where possible repair — the cuBLAS/cuBLASLt pairing before
/// `tract-cuda` creates its cuBLAS handle. See the module doc.
///
/// `Ok` means the pair `tract-cuda` will use agrees on its version; `Err`
/// means it does not and could not be made to, and the GPU path must not run.
pub fn ensure_cublas_pairing() -> Result<CublasPairing, String> {
    if !loaded_objects("libcublas.so").is_empty() {
        // Someone opened cuBLAS already; cudarc will get that one. Inspect it
        // but do not try to swap libraries out from under a live user.
        let (_h, pairing) = open_and_inspect()?;
        return if pairing.agrees() {
            Ok(pairing)
        } else {
            Err(format!(
                "{pairing}: mismatched, and cuBLAS was already in use"
            ))
        };
    }

    let (handle, first) = open_and_inspect()?;
    if first.agrees() {
        return Ok(first); // `handle` leaked on purpose — see the module doc.
    }

    // The sibling: the cuBLASLt the winning cuBLAS was shipped with, under the
    // soname its NEEDED entry asks for (the file name the loader recorded).
    let cublas_real = std::fs::canonicalize(&first.cublas_path).map_err(|e| {
        format!(
            "{first}: cannot resolve {}: {e}",
            first.cublas_path.display()
        )
    })?;
    let soname = first
        .lt_path
        .file_name()
        .ok_or_else(|| format!("{first}: cuBLASLt path has no file name"))?;
    let dir = cublas_real
        .parent()
        .ok_or_else(|| format!("{first}: cuBLAS path has no directory"))?;
    let sibling = dir.join(soname);
    if !sibling.exists() {
        return Err(format!(
            "{first}: mismatched, and there is no {} beside cuBLAS to pair it with",
            sibling.display()
        ));
    }

    // Safety: nothing has called into cuBLAS and no handle was created; closing it drops the only reference to it and to
    // the cuBLASLt it pulled in.
    unsafe { libc::dlclose(handle) };
    if !loaded_objects("libcublasLt.so").is_empty() {
        return Err(format!(
            "{first}: mismatched, and the stray cuBLASLt stayed loaded after cuBLAS was closed"
        ));
    }
    let sibling_str = sibling
        .to_str()
        .ok_or_else(|| format!("{first}: non-UTF-8 path {}", sibling.display()))?;
    // Leaked on purpose: this is the cuBLASLt every later open must bind to.
    dlopen(sibling_str, libc::RTLD_NOW | libc::RTLD_LOCAL)
        .ok_or_else(|| format!("{first}: cannot open {}: {}", sibling.display(), dl_error()))?;

    let (_handle, mut second) = open_and_inspect()?;
    if !second.agrees() {
        return Err(format!(
            "{second}: still mismatched after pre-loading {}",
            sibling.display()
        ));
    }
    second.repaired = true;
    tracing::warn!(
        "cuBLAS pairing repaired: the dynamic loader had paired cuBLAS {} ({}) with cuBLASLt \
         {} ({}), a mix that returns wrong GEMM results without any error \
         (rnabioco/escapepod-rs#416); now using {second}. The usual cause is a CUDA runtime \
         environment without the unversioned `libcublas.so`, so the host toolkit's is found \
         through the ldconfig cache instead",
        first.cublas_version,
        first.cublas_path.display(),
        first.lt_version,
        first.lt_path.display(),
    );
    Ok(second)
}

#[cfg(test)]
mod tests {
    use super::release_of;
    use std::os::unix::fs::symlink;

    /// The toolkit layout: soname symlink -> fully versioned file.
    #[test]
    fn release_follows_the_soname_link_to_the_versioned_file() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("libcublas.so.12.8.3.14"), b"").unwrap();
        symlink("libcublas.so.12.8.3.14", d.path().join("libcublas.so.12")).unwrap();
        symlink("libcublas.so.12", d.path().join("libcublas.so")).unwrap();
        let v = release_of(&d.path().join("libcublas.so"), "libcublas.so.").unwrap();
        assert_eq!(v, "12.8.3.14");
    }

    /// #416's pair: same soname, different releases, must not compare equal.
    #[test]
    fn two_releases_differ_and_one_release_agrees() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        std::fs::write(a.path().join("libcublas.so.12.8.3.14"), b"").unwrap();
        std::fs::write(a.path().join("libcublasLt.so.12.8.3.14"), b"").unwrap();
        std::fs::write(b.path().join("libcublasLt.so.12.9.2.10"), b"").unwrap();
        let cublas = release_of(&a.path().join("libcublas.so.12.8.3.14"), "libcublas.so.").unwrap();
        let lt_a = release_of(
            &a.path().join("libcublasLt.so.12.8.3.14"),
            "libcublasLt.so.",
        )
        .unwrap();
        let lt_b = release_of(
            &b.path().join("libcublasLt.so.12.9.2.10"),
            "libcublasLt.so.",
        )
        .unwrap();
        assert_eq!(cublas, lt_a);
        assert_ne!(cublas, lt_b);
    }

    /// No full version in the name: fall back to the directory, so a pair from
    /// one install still agrees and a pair from two does not.
    #[test]
    fn unversioned_files_compare_by_directory() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        for d in [&a, &b] {
            std::fs::write(d.path().join("libcublas.so"), b"").unwrap();
            std::fs::write(d.path().join("libcublasLt.so"), b"").unwrap();
        }
        let c = release_of(&a.path().join("libcublas.so"), "libcublas.so.").unwrap();
        let la = release_of(&a.path().join("libcublasLt.so"), "libcublasLt.so.").unwrap();
        let lb = release_of(&b.path().join("libcublasLt.so"), "libcublasLt.so.").unwrap();
        assert_eq!(c, la);
        assert_ne!(c, lb);
    }
}

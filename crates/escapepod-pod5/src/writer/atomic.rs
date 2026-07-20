//! Atomic file staging: write to a temp file, rename into place on success.
//!
//! POD5 archives are only valid once the footer and trailing signature land,
//! which happens at the very end of a write. Writing straight to the user's
//! destination therefore leaves a window — the whole write — during which a
//! crash, error, or interrupt leaves an unreadable stump at a path that is
//! supposed to hold a finished archive. Worse, `File::create` truncates an
//! existing file up front, so an interrupted overwrite destroys the previous
//! archive before producing a replacement.
//!
//! [`AtomicFile`] closes that window: bytes go to a temp file in the same
//! directory, and the destination is only touched by the final `rename(2)`.
//! Drop without [`AtomicFile::commit`] unlinks the temp and leaves the
//! destination exactly as it was.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use tempfile::{Builder, NamedTempFile};

use crate::error::{Error, Result};

/// Filename prefix for in-flight staging files.
///
/// Leading dot keeps strays out of `ls` and out of the `*.pod5` globs used to
/// collect inputs. The fixed infix makes them greppable when a hard kill
/// leaves some behind: `find <dir> -name '.escpod-tmp-*'`.
pub const TEMP_PREFIX: &str = ".escpod-tmp-";

/// How hard to push bytes toward stable storage before renaming into place.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Durability {
    /// Rename only. The destination is always either the old file or a
    /// complete new one, but a machine crash shortly after a write may still
    /// lose the data — on some filesystems the rename can reach the journal
    /// before the data blocks do, leaving a zero-length destination.
    ///
    /// This is the default: it costs nothing, and on the HPC filesystems this
    /// tool targets the outputs are regenerable from inputs that are
    /// themselves on the same storage.
    #[default]
    None,
    /// `fsync` the staging file before renaming it into place. Survives
    /// machine crashes at the cost of one sync per output file.
    File,
    /// Also `fsync` the parent directory after the rename, so the rename
    /// record itself is durable.
    FileAndDir,
}

/// Set once a shutdown is in progress so no new staging files are created
/// behind the cleanup sweep.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);
static NEXT_ID: AtomicU64 = AtomicU64::new(0);
static IN_FLIGHT: LazyLock<Mutex<HashMap<u64, PathBuf>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Unlink every staging file currently in flight and refuse to create more.
///
/// Intended for a `ctrlc`-style handler just before exiting: `NamedTempFile`'s
/// own cleanup runs on drop, which never happens when the process is killed by
/// a signal.
///
/// The bounded `try_lock` retry keeps a wedged writer from turning Ctrl-C into
/// a hang; giving up and leaking a temp is strictly better than not exiting.
pub fn abort_all_in_flight_writes() {
    SHUTDOWN.store(true, Ordering::SeqCst);

    for _ in 0..100 {
        if let Ok(mut map) = IN_FLIGHT.try_lock() {
            for (_, path) in map.drain() {
                let _ = fs::remove_file(&path);
            }
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Directory to stage in: the destination's parent, treating both "no parent"
/// and the empty parent of a bare filename as the current directory.
fn staging_dir(dest: &Path) -> &Path {
    match dest.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    }
}

/// A file that appears at its destination only when fully written.
///
/// The staging file lives in the same directory as the destination so the
/// final rename stays within one filesystem and is therefore atomic.
///
/// # Flushing
///
/// [`commit`](Self::commit) syncs and renames the underlying file, but it
/// cannot flush buffers it does not own. Callers that wrap
/// [`reopen`](Self::reopen) in a `BufWriter` **must** flush or drop that
/// wrapper first, or the buffered tail is lost.
///
/// # Permissions
///
/// `tempfile` creates staging files 0600. On commit the mode becomes the
/// destination's existing mode when overwriting, else 0644 — matching what
/// `File::create` yields under a conventional umask. The process umask is
/// deliberately not consulted: `umask(2)` is get-and-set only, so reading it
/// races against other threads in the host process.
#[derive(Debug)]
pub struct AtomicFile {
    /// `None` once committed, which is what makes `Drop` a no-op after success.
    temp: Option<NamedTempFile>,
    dest: PathBuf,
    durability: Durability,
    id: u64,
}

impl AtomicFile {
    /// Stage a write for `dest` with the default durability.
    pub fn new(dest: impl AsRef<Path>) -> Result<Self> {
        Self::with_durability(dest, Durability::default())
    }

    /// Stage a write for `dest`.
    ///
    /// The parent directory must already exist; this deliberately does not
    /// create it, since callers that want that do it explicitly and a typo
    /// should not silently produce a new directory tree.
    pub fn with_durability(dest: impl AsRef<Path>, durability: Durability) -> Result<Self> {
        let dest = dest.as_ref();

        if SHUTDOWN.load(Ordering::SeqCst) {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::Interrupted,
                "shutting down; refusing to start a new output file",
            )));
        }

        let dir = staging_dir(dest);
        let temp = Builder::new()
            .prefix(TEMP_PREFIX)
            .suffix(".partial")
            .tempfile_in(dir)
            // Name the directory rather than the temp path, which the caller
            // never chose and cannot act on.
            .map_err(|e| {
                Error::Io(io::Error::new(
                    e.kind(),
                    format!("cannot create a temporary file in {}: {e}", dir.display()),
                ))
            })?;

        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut map) = IN_FLIGHT.lock() {
            map.insert(id, temp.path().to_path_buf());
        }

        Ok(Self {
            temp: Some(temp),
            dest: dest.to_path_buf(),
            durability,
            id,
        })
    }

    fn temp(&self) -> Result<&NamedTempFile> {
        self.temp.as_ref().ok_or(Error::WriterFinalized)
    }

    /// Path of the in-flight staging file.
    pub fn temp_path(&self) -> Result<&Path> {
        Ok(self.temp()?.path())
    }

    /// The destination this will be renamed to.
    pub fn dest(&self) -> &Path {
        &self.dest
    }

    /// An independent owned handle on the staging file.
    ///
    /// Use this to hand a `File` to a `BufWriter` or move one into a thread
    /// while the `AtomicFile` stays behind as the cleanup guard. The handle
    /// shares the inode, so [`commit`](Self::commit) can sync bytes written
    /// through it without needing the handle back.
    pub fn reopen(&self) -> Result<File> {
        Ok(self.temp()?.reopen()?)
    }

    /// Mode to apply on commit: an existing destination keeps its permissions,
    /// a new file gets 0644.
    #[cfg(unix)]
    fn target_mode(&self) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(&self.dest)
            .map(|m| m.permissions().mode())
            .unwrap_or(0o644)
    }

    /// Sync as configured, fix up permissions, and rename into place.
    ///
    /// Any buffered writer wrapping [`reopen`](Self::reopen) must already be
    /// flushed — see the type-level note.
    pub fn commit(mut self) -> Result<()> {
        let temp = self.temp.take().ok_or(Error::WriterFinalized)?;

        // Sync before the rename, never after: renaming first can journal the
        // directory entry ahead of the data blocks, so a crash leaves the
        // destination pointing at garbage.
        if self.durability != Durability::None {
            temp.as_file().sync_all()?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = self.target_mode();
            fs::set_permissions(temp.path(), fs::Permissions::from_mode(mode))?;
        }

        temp.persist(&self.dest).map_err(|e| {
            Error::Io(io::Error::other(format!(
                "cannot move the completed file into place at {}: {}",
                self.dest.display(),
                e.error
            )))
        })?;

        if self.durability == Durability::FileAndDir {
            // Best effort: opening a directory for sync is not portable, and
            // failing here would discard an output that is already in place.
            if let Ok(dir) = File::open(staging_dir(&self.dest)) {
                let _ = dir.sync_all();
            }
        }

        self.unregister();
        Ok(())
    }

    /// Discard the staged file explicitly, surfacing any unlink error.
    ///
    /// The destination is left untouched, or absent if it never existed.
    pub fn abort(mut self) -> Result<()> {
        let temp = self.temp.take().ok_or(Error::WriterFinalized)?;
        self.unregister();
        Ok(temp.close()?)
    }

    fn unregister(&self) {
        if let Ok(mut map) = IN_FLIGHT.lock() {
            map.remove(&self.id);
        }
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        // Still Some means commit() never ran — the caller bailed or unwound.
        // Dropping the NamedTempFile unlinks it; the destination is untouched.
        if self.temp.is_some() {
            self.unregister();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn strays(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| {
                let name = e.ok()?.file_name().to_string_lossy().into_owned();
                name.starts_with(TEMP_PREFIX).then_some(name)
            })
            .collect()
    }

    #[test]
    fn commit_moves_file_into_place() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");

        let atomic = AtomicFile::new(&dest).unwrap();
        let mut f = atomic.reopen().unwrap();
        f.write_all(b"finished").unwrap();
        f.flush().unwrap();
        atomic.commit().unwrap();

        assert_eq!(fs::read(&dest).unwrap(), b"finished");
        assert!(strays(dir.path()).is_empty(), "staging file left behind");
    }

    #[test]
    fn drop_without_commit_leaves_no_destination() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");

        {
            let atomic = AtomicFile::new(&dest).unwrap();
            let mut f = atomic.reopen().unwrap();
            f.write_all(b"partial").unwrap();
            f.flush().unwrap();
        }

        assert!(!dest.exists(), "destination created despite no commit");
        assert!(strays(dir.path()).is_empty(), "staging file left behind");
    }

    /// The data-loss regression: an interrupted overwrite must not damage the
    /// file that was already there.
    #[test]
    fn failed_overwrite_preserves_the_original() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");
        fs::write(&dest, b"the original contents").unwrap();

        {
            let atomic = AtomicFile::new(&dest).unwrap();
            let mut f = atomic.reopen().unwrap();
            f.write_all(b"replacement that never lands").unwrap();
            f.flush().unwrap();
        }

        assert_eq!(fs::read(&dest).unwrap(), b"the original contents");
        assert!(strays(dir.path()).is_empty());
    }

    #[test]
    fn abort_discards_the_staged_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");

        let atomic = AtomicFile::new(&dest).unwrap();
        atomic.reopen().unwrap().write_all(b"discarded").unwrap();
        atomic.abort().unwrap();

        assert!(!dest.exists());
        assert!(strays(dir.path()).is_empty());
    }

    #[test]
    fn commit_overwrites_an_existing_destination() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");
        fs::write(&dest, b"old").unwrap();

        let atomic = AtomicFile::new(&dest).unwrap();
        let mut f = atomic.reopen().unwrap();
        f.write_all(b"new").unwrap();
        f.flush().unwrap();
        atomic.commit().unwrap();

        assert_eq!(fs::read(&dest).unwrap(), b"new");
    }

    /// A bare filename has an empty parent, which is not a usable directory.
    #[test]
    fn bare_filename_stages_in_current_directory() {
        assert_eq!(staging_dir(Path::new("out.pod5")), Path::new("."));
        assert_eq!(staging_dir(Path::new("sub/out.pod5")), Path::new("sub"));
    }

    #[test]
    fn missing_parent_directory_names_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("does-not-exist").join("out.pod5");

        let err = AtomicFile::new(&dest).unwrap_err().to_string();
        assert!(
            err.contains("does-not-exist"),
            "error should name the missing directory, got: {err}"
        );
    }

    #[test]
    fn durability_defaults_to_none() {
        assert_eq!(Durability::default(), Durability::None);
    }

    #[test]
    fn syncing_durability_still_commits() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");

        let atomic = AtomicFile::with_durability(&dest, Durability::FileAndDir).unwrap();
        let mut f = atomic.reopen().unwrap();
        f.write_all(b"durable").unwrap();
        f.flush().unwrap();
        atomic.commit().unwrap();

        assert_eq!(fs::read(&dest).unwrap(), b"durable");
    }

    #[cfg(unix)]
    #[test]
    fn new_file_is_readable_not_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");

        let atomic = AtomicFile::new(&dest).unwrap();
        atomic.reopen().unwrap().write_all(b"x").unwrap();
        atomic.commit().unwrap();

        let mode = fs::metadata(&dest).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o644, "expected 0644, got {mode:o}");
    }

    #[cfg(unix)]
    #[test]
    fn overwrite_preserves_the_existing_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out.pod5");
        fs::write(&dest, b"old").unwrap();
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o640)).unwrap();

        let atomic = AtomicFile::new(&dest).unwrap();
        atomic.reopen().unwrap().write_all(b"new").unwrap();
        atomic.commit().unwrap();

        let mode = fs::metadata(&dest).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640, "expected 0640 to be preserved, got {mode:o}");
    }
}

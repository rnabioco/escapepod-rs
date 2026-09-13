use std::process::Command;

fn main() {
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let profile = std::env::var("PROFILE").unwrap_or_default();

    // Check if HEAD is an exact release tag — if so, use the plain version.
    let is_tagged = Command::new("git")
        .args(["describe", "--tags", "--exact-match", "HEAD"])
        .output()
        .ok()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let escpod_version = if is_tagged {
        version
    } else {
        let git_sha = Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                let sha = String::from_utf8(o.stdout).ok()?;
                let sha = sha.trim().to_string();
                if sha.is_empty() { None } else { Some(sha) }
            });

        match git_sha {
            Some(sha) => {
                let dirty = Command::new("git")
                    .args(["diff", "--quiet", "HEAD"])
                    .status()
                    .map(|s| !s.success())
                    .unwrap_or(false);

                let dev = if profile != "release" { "-dev" } else { "" };
                if dirty {
                    format!("{version}{dev}+{sha}.dirty")
                } else {
                    format!("{version}{dev}+{sha}")
                }
            }
            None => {
                if profile != "release" {
                    format!("{version}-dev")
                } else {
                    version
                }
            }
        }
    };

    println!("cargo:rustc-env=ESCPOD_VERSION={escpod_version}");

    // Rerun when git state changes (branch switch, commit, new tags).
    // `../.git/HEAD` assumes the workspace root's `.git` is one level up, which
    // is false here (this crate is two levels down) and always false in a
    // linked worktree (`.git` there is a file, not a directory) — either way
    // cargo can't stat a path that doesn't exist, so it treats the build
    // script as perpetually stale and reruns it (and the three `git`
    // subprocesses above) on every build. `git rev-parse --git-path` resolves
    // the real, absolute location for both cases; when it fails (no git
    // checkout at all, e.g. a release tarball) the path is simply omitted
    // rather than pointing at nothing.
    //
    // Deliberately not watching the index: `git diff --quiet HEAD` above
    // refreshes and rewrites it (git's own stat-cache behavior) on every run,
    // so a rerun-if-changed on it would just make the script trigger its own
    // next rerun forever. The `.dirty` suffix is still recomputed correctly
    // whenever the script runs for any other reason; it just won't force a
    // rebuild by itself the moment a file is edited without a commit.
    if let Some(head_path) = git_path("HEAD") {
        // A symbolic HEAD's target ref (the branch tip) can live in the
        // shared repo dir even when HEAD itself is per-worktree, so it needs
        // its own `--git-path` resolution rather than a path built by hand.
        if let Ok(contents) = std::fs::read_to_string(&head_path)
            && let Some(target_ref) = contents.trim().strip_prefix("ref: ")
            && let Some(ref_path) = git_path(target_ref)
        {
            println!("cargo:rerun-if-changed={ref_path}");
        }
        println!("cargo:rerun-if-changed={head_path}");
    }
    if let Some(tags_path) = git_path("refs/tags") {
        println!("cargo:rerun-if-changed={tags_path}");
    }
}

/// Resolve `git rev-parse --git-path <subpath>` to an absolute path, or
/// `None` when there is no git checkout to resolve it against.
fn git_path(subpath: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--git-path", subpath])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    if path.is_empty() {
        None
    } else {
        Some(path.to_string())
    }
}

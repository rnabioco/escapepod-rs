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

    // Rerun when git state changes (branch switch, commit, staging, new tags).
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
    println!("cargo:rerun-if-changed=../.git/refs/tags");
}

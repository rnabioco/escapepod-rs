// SPDX-License-Identifier: MIT

//! `bam-filter --region` needs a BAI. It must use one that already exists
//! (`<name>.bam.bai` or `<name>.bai`) instead of rebuilding it, and it must
//! still work when the BAM's directory is read-only, where the index can only
//! live in memory (#449).

#![cfg(feature = "cli")]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const REGION: &str = "tRNA-Ala-GGC-1-1";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("escapepod-classify/tests/fixtures")
}

/// A fresh scratch dir under cargo's per-target tmp (removed on drop).
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("bamidx-{name}"));
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_only(dir: &Path) {
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o555)).unwrap();
}

/// Whether the process can still create files in a 0o555 directory (root).
fn dir_is_writable_anyway(dir: &Path) -> bool {
    let probe = dir.join(".probe");
    let ok = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

fn bam_filter(bam: &Path, out: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_escpod"))
        .args(["bam-filter"])
        .arg(fixtures().join("trna_reads.pod5"))
        .arg("-b")
        .arg(bam)
        .arg("-o")
        .arg(out)
        .args(["--region", REGION])
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to run escpod")
}

fn assert_ok(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "bam-filter failed:\n{stderr}");
    stderr
}

#[test]
fn read_only_bam_dir_without_bai_uses_in_memory_index() {
    let bams = Scratch::new("ro-bam");
    let outs = Scratch::new("ro-out");
    let bam = bams.0.join("reads.bam");
    std::fs::copy(fixtures().join("trna_mappings_padded.bam"), &bam).unwrap();
    read_only(&bams.0);
    if dir_is_writable_anyway(&bams.0) {
        eprintln!("skipping: directory permissions are not enforced for this user");
        return;
    }

    let out = outs.0.join("out.pod5");
    let stderr = assert_ok(&bam_filter(&bam, &out));
    assert!(out.exists(), "no output written");
    assert!(
        stderr.contains("in memory"),
        "expected the in-memory fallback to be reported:\n{stderr}"
    );
    assert!(!bams.0.join("reads.bam.bai").exists());
}

#[test]
fn existing_bai_is_used_not_rebuilt() {
    for (label, bai_name) in [("bam-bai", "reads.bam.bai"), ("alt-bai", "reads.bai")] {
        let bams = Scratch::new(label);
        let outs = Scratch::new(&format!("{label}-out"));
        let bam = bams.0.join("reads.bam");
        std::fs::copy(fixtures().join("trna_mappings_padded.bam"), &bam).unwrap();
        std::fs::copy(
            fixtures().join("trna_mappings_padded.bam.bai"),
            bams.0.join(bai_name),
        )
        .unwrap();
        read_only(&bams.0);

        let stderr = assert_ok(&bam_filter(&bam, &outs.0.join("out.pod5")));
        assert!(
            !stderr.contains("BAI index not found"),
            "{bai_name}: existing index was ignored:\n{stderr}"
        );
        // Nothing was added beside the BAM.
        let mut names: Vec<_> = std::fs::read_dir(&bams.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        let mut want = vec!["reads.bam".to_string(), bai_name.to_string()];
        want.sort();
        assert_eq!(names, want);
    }
}

// SPDX-License-Identifier: MIT

//! `MD`/`NM` against what `samtools calmd` writes for the same CIGAR,
//! including reference `N` (and other IUPAC, lower-case, `U`) positions —
//! `fixtures/calmd_golden.json`, from `fixtures/gen_calmd_golden.py`.

use escapepod_align::CigarOp;
use escapepod_align::sam::{cigar_string, md_nm};
use serde_json::Value;

/// Split a SAM CIGAR into its soft clips and aligned ops.
fn parse_cigar(s: &str) -> (usize, Vec<(CigarOp, u32)>, usize) {
    let (mut lead, mut tail) = (0usize, 0usize);
    let mut ops = Vec::new();
    let mut num = 0u32;
    for c in s.chars() {
        if let Some(d) = c.to_digit(10) {
            num = num * 10 + d;
            continue;
        }
        match c {
            'M' => ops.push((CigarOp::Match, num)),
            'I' => ops.push((CigarOp::Ins, num)),
            'D' => ops.push((CigarOp::Del, num)),
            'S' if ops.is_empty() => lead = num as usize,
            'S' => tail = num as usize,
            other => panic!("unexpected CIGAR op {other}"),
        }
        num = 0;
    }
    (lead, ops, tail)
}

#[test]
fn md_nm_match_samtools_calmd() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/calmd_golden.json");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let reference = g["reference"]["seq"].as_str().unwrap().as_bytes();
    let records = g["records"].as_array().unwrap();
    // The cases the golden exists for must be in it.
    for name in ["n_under_read_base", "n_under_read_n", "n_inside_deletion"] {
        assert!(
            records.iter().any(|r| r["name"] == name),
            "golden lacks {name}"
        );
    }
    for r in records {
        let name = r["name"].as_str().unwrap();
        let seq = r["seq"].as_str().unwrap().as_bytes();
        let cigar = r["cigar"].as_str().unwrap();
        let pos = r["pos"].as_u64().unwrap() as usize - 1;
        let (lead, ops, tail) = parse_cigar(cigar);
        let (md, nm) = md_nm(seq, reference, lead, pos, &ops);
        assert_eq!(md, r["md"].as_str().unwrap(), "{name}: MD");
        assert_eq!(nm as u64, r["nm"].as_u64().unwrap(), "{name}: NM");
        // And the CIGAR we would write for it round-trips.
        assert_eq!(
            cigar_string(&ops, seq.len(), lead, seq.len() - tail),
            cigar,
            "{name}"
        );
    }
}

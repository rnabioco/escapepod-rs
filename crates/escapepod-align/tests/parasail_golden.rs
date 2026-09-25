// SPDX-License-Identifier: MIT

//! The scalar Gotoh against parasail (`sw_trace` / `sg_trace`), from
//! `fixtures/align_golden.json` (`fixtures/gen_align_golden.py`): random
//! nanopore-like pairs and real tRNA reads, both modes, two scoring schemes.

use escapepod_align::alphabet::encode;
use escapepod_align::{Mode, Scoring, scalar};
use serde_json::Value;

fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/align_golden.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// `(label, query codes, reference codes, scoring, mode, expected result)`.
type Case = (String, Vec<u8>, Vec<u8>, Scoring, Mode, Value);

/// Every (pair, result) in the golden, with the scoring and mode it names.
fn cases(g: &Value) -> Vec<Case> {
    let schemes = g["schemes"].as_object().unwrap();
    let mut out = Vec::new();
    for p in g["pairs"].as_array().unwrap() {
        let q = encode(p["query"].as_str().unwrap().as_bytes());
        let r = encode(p["reference"].as_str().unwrap().as_bytes());
        for res in p["results"].as_array().unwrap() {
            let v: Vec<i32> = schemes[res["scheme"].as_str().unwrap()]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            let scoring = Scoring::new(v[0], v[1], v[2], v[3]).unwrap();
            let mode: Mode = res["mode"].as_str().unwrap().parse().unwrap();
            let label = format!(
                "{} {} {}",
                p["name"].as_str().unwrap(),
                res["mode"].as_str().unwrap(),
                res["scheme"].as_str().unwrap()
            );
            out.push((label, q.clone(), r.clone(), scoring, mode, res.clone()));
        }
    }
    out
}

#[test]
fn scalar_matches_parasail_golden() {
    let g = golden();
    let cases = cases(&g);
    assert!(cases.len() >= 4 * 40, "golden is smaller than generated");
    let at = |v: &Value, k: &str| v[k].as_u64().unwrap() as usize;
    for (label, q, r, scoring, mode, want) in &cases {
        let a = scalar::align(q, r, scoring, *mode);
        assert_eq!(
            a.score as i64,
            want["score"].as_i64().unwrap(),
            "{label}: score"
        );
        assert_eq!(
            scalar::score(q, r, scoring, *mode),
            a.score,
            "{label}: score-only disagrees with traceback DP"
        );
        assert_eq!(
            (a.query_start, a.query_end, a.ref_start, a.ref_end),
            (
                at(want, "query_start"),
                at(want, "query_end"),
                at(want, "ref_start"),
                at(want, "ref_end")
            ),
            "{label}: coordinates (query_start, query_end, ref_start, ref_end)"
        );
    }
}

/// Optimal tracebacks are not unique, so the CIGAR is checked by re-scoring it
/// under the same scoring rather than by comparing strings.
#[test]
fn cigar_rescores_to_reported_score() {
    let g = golden();
    for (label, q, r, scoring, mode, _) in cases(&g) {
        let a = scalar::align(&q, &r, &scoring, mode);
        assert_eq!(a.rescore(&q, &r, &scoring), a.score, "{label}");
        // The ops must also exactly span the reported coordinates.
        let (mut qi, mut rj) = (0usize, 0usize);
        for (op, len) in &a.ops {
            match op {
                escapepod_align::CigarOp::Match => {
                    qi += *len as usize;
                    rj += *len as usize;
                }
                escapepod_align::CigarOp::Ins => qi += *len as usize,
                escapepod_align::CigarOp::Del => rj += *len as usize,
            }
        }
        assert_eq!(qi, a.query_end - a.query_start, "{label}: query span");
        assert_eq!(rj, a.ref_end - a.ref_start, "{label}: reference span");
        assert!(
            matches!(a.ops.first(), Some((escapepod_align::CigarOp::Match, _))),
            "{label}: alignment starts with a gap"
        );
        assert!(
            matches!(a.ops.last(), Some((escapepod_align::CigarOp::Match, _))),
            "{label}: alignment ends with a gap"
        );
    }
}

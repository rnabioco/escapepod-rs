// SPDX-License-Identifier: MIT

//! The CUDA score kernel against the scalar oracle, by equality.
//!
//! Compiled only with `--features gpu`. Without a usable device (no driver,
//! no NVRTC, no GPU) the test prints why and passes, so a GPU-less host can
//! still run `cargo nextest run --features gpu` (the
//! `escapepod-signal/tests/gpu_dtw.rs` pattern).

#![cfg(feature = "gpu")]

use escapepod_align::alphabet::{encode, reverse_complement_codes};
use escapepod_align::cuda::{GpuScorer, MAX_READ_LEN, MAX_REF_LEN};
use escapepod_align::{Mode, Panel, Scoring, scalar};

/// Build a scorer, or say why there is no GPU here and return `None`.
fn scorer(panel: &Panel, scoring: Scoring, mode: Mode) -> Option<GpuScorer> {
    match std::panic::catch_unwind(|| GpuScorer::new(panel, scoring, mode)) {
        Ok(Ok(s)) => Some(s),
        Ok(Err(e)) => {
            eprintln!("[gpu_parity] skipping: no usable CUDA device ({e})");
            None
        }
        Err(_) => {
            eprintln!("[gpu_parity] skipping: CUDA initialisation panicked");
            None
        }
    }
}

/// xorshift64*, the generator the SIMD kernels' tests use, so the reads here
/// are the same kind of reads.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
}

fn random_seq(rng: &mut Rng, len: usize, n_rate: f64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            if rng.chance(n_rate) {
                b'N'
            } else {
                b"ACGT"[rng.below(4)]
            }
        })
        .collect()
}

/// Nanopore-ish: substitutions, insertions and deletions at a few % each.
fn mutate(rng: &mut Rng, s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() + 8);
    for &b in s {
        if rng.chance(0.04) {
            continue;
        }
        if rng.chance(0.05) {
            out.push(b"ACGT"[rng.below(4)]);
        } else {
            out.push(b);
        }
        if rng.chance(0.03) {
            out.push(b"ACGT"[rng.below(4)]);
        }
    }
    out
}

fn random_panel(rng: &mut Rng, n_refs: usize, min_len: usize, spread: usize) -> Panel {
    Panel::new((0..n_refs).map(|k| {
        let len = min_len + rng.below(spread);
        (format!("r{k}"), random_seq(rng, len, 0.01))
    }))
    .unwrap()
}

fn schemes() -> [Scoring; 3] {
    [
        Scoring::default(),
        Scoring::new(1, -1, -2, -1).unwrap(),
        Scoring::new(3, -2, -5, -2).unwrap(),
    ]
}

/// Reads drawn from the panel (so real maxima exist), unrelated random reads
/// (so the zero floor and negative semi-global cells do), and every length
/// around the kernel's 16-row stripe.
fn reads_for(rng: &mut Rng, panel: &Panel, n: usize) -> Vec<Vec<u8>> {
    let mut reads: Vec<Vec<u8>> = (1..=40).map(|len| random_seq(rng, len, 0.02)).collect();
    for t in 0..n {
        let read = if t % 4 == 3 {
            let len = 1 + rng.below(300);
            random_seq(rng, len, 0.02)
        } else {
            let r = &panel.get(rng.below(panel.len())).seq;
            let a = rng.below(r.len() / 3 + 1);
            let b = r.len() - rng.below(r.len() / 3 + 1);
            let (lead, tail) = (rng.below(30), rng.below(30));
            let mut read = random_seq(rng, lead, 0.0);
            read.extend(mutate(rng, &r[a..b.max(a)]));
            read.extend(random_seq(rng, tail, 0.0));
            read
        };
        reads.push(read);
    }
    reads.iter().map(|r| encode(r)).collect()
}

/// Every query against every reference, GPU against `scalar::score`, in one
/// batch (reverse complements included, as `--strand both` sends them).
/// Returns the number of pairs compared, or `None` when there is no device.
fn check(
    panel: &Panel,
    reads: &[Vec<u8>],
    scoring: Scoring,
    mode: Mode,
    what: &str,
) -> Option<usize> {
    let mut gpu = scorer(panel, scoring, mode)?;
    let mut queries: Vec<Vec<u8>> = reads.to_vec();
    queries.extend(reads.iter().map(|q| reverse_complement_codes(q)));
    let refs: Vec<&[u8]> = queries.iter().map(|q| &q[..]).collect();
    let m = gpu.score_batch(&refs).expect("score_batch");
    assert_eq!(m.len(), queries.len());
    assert_eq!(m.n_refs(), panel.len());
    // The oracle is the slow half (this crate is unoptimised in the test
    // profile), so it runs on every core; each thread checks every k-th query.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (m, queries) = (&m, &queries);
            scope.spawn(move || {
                for (k, q) in queries.iter().enumerate().skip(t).step_by(threads) {
                    let row = m.row(k).unwrap_or_else(|| {
                        panic!("{what}: query {k} ({} nt) left unscored", q.len())
                    });
                    for (r, &got) in row.iter().enumerate() {
                        let want = scalar::score(q, &panel.get(r).codes, &scoring, mode);
                        assert_eq!(
                            got as i32,
                            want,
                            "{what} {} {scoring}: query {k} ({} nt) vs reference {r} ({} nt)",
                            mode.name(),
                            q.len(),
                            panel.get(r).codes.len()
                        );
                    }
                }
            });
        }
    });
    let pairs = queries.len() * panel.len();
    Some(pairs)
}

fn fixture_panel(fasta: &str) -> Panel {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(
        root.join("../escapepod-classify/tests/fixtures")
            .join(fasta),
    )
    .unwrap();
    let mut refs: Vec<(String, Vec<u8>)> = Vec::new();
    for line in text.lines() {
        if let Some(h) = line.strip_prefix('>') {
            refs.push((h.split_whitespace().next().unwrap().to_string(), Vec::new()));
        } else {
            refs.last_mut().unwrap().1.extend(line.trim().bytes());
        }
    }
    Panel::new(refs).unwrap()
}

fn fixture_reads() -> Vec<Vec<u8>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("tests/fixtures/align_golden.json")).unwrap(),
    )
    .unwrap();
    golden["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["source"].as_str() != Some("random"))
        .map(|p| encode(p["query"].as_str().unwrap().as_bytes()))
        .collect()
}

#[test]
fn gpu_scores_equal_scalar() {
    let mut total = 0usize;
    let mut rng = Rng(0x5eed_0101);

    // 37 references: one block with a ragged warp of empty lanes.
    let small = random_panel(&mut rng, 37, 20, 280);
    let reads = reads_for(&mut rng, &small, 60);
    for scoring in schemes() {
        for mode in [Mode::Local, Mode::SemiGlobal] {
            let Some(n) = check(&small, &reads, scoring, mode, "random 37-ref panel") else {
                return;
            };
            total += n;
        }
    }

    // 300 references up to 620 nt: the panel no longer fits one block's
    // shared memory, so it is split into reference groups.
    let wide = random_panel(&mut rng, 300, 20, 600);
    let reads = reads_for(&mut rng, &wide, 6);
    for mode in [Mode::Local, Mode::SemiGlobal] {
        total += check(
            &wide,
            &reads,
            Scoring::default(),
            mode,
            "random 300-ref panel",
        )
        .unwrap();
    }

    // The real tRNA reads against both 47-reference fixture panels.
    let reads = fixture_reads();
    assert!(reads.len() >= 5, "golden carries too few fixture reads");
    for fasta in ["trna_reference.fa", "trna_reference_ambiguous.fa"] {
        let panel = fixture_panel(fasta);
        assert_eq!(panel.len(), 47);
        for scoring in schemes() {
            for mode in [Mode::Local, Mode::SemiGlobal] {
                total += check(&panel, &reads, scoring, mode, fasta).unwrap();
            }
        }
    }
    eprintln!("[gpu_parity] {total} (query, reference) pairs equal to scalar::score");
}

/// What the kernel does not take comes back unscored (for the CPU), and does
/// not disturb the rows around it.
#[test]
fn gpu_leaves_what_it_cannot_score_to_the_cpu() {
    let mut rng = Rng(0x5eed_0102);
    let panel = random_panel(&mut rng, 20, 50, 100);
    let scoring = Scoring::default();
    let Some(mut gpu) = scorer(&panel, scoring, Mode::Local) else {
        return;
    };
    let long = encode(&random_seq(&mut rng, MAX_READ_LEN + 1, 0.0));
    let ok = encode(&random_seq(&mut rng, 120, 0.0));
    let edge = encode(&random_seq(&mut rng, MAX_READ_LEN, 0.0));
    let queries: Vec<&[u8]> = vec![&ok, &[], &long, &edge, &ok];
    let m = gpu.score_batch(&queries).unwrap();
    assert!(m.row(1).is_none(), "empty read");
    assert!(m.row(2).is_none(), "read over MAX_READ_LEN");
    for k in [0, 3, 4] {
        let row = m.row(k).expect("scored");
        for (r, &s) in row.iter().enumerate() {
            let want = scalar::score(queries[k], &panel.get(r).codes, &scoring, Mode::Local);
            assert_eq!(s as i32, want, "query {k} reference {r}");
        }
    }
    assert!(gpu.score_batch(&[]).unwrap().is_empty());
}

#[test]
fn gpu_refuses_a_reference_it_cannot_hold() {
    let panel = Panel::new([("big", vec![b'A'; MAX_REF_LEN + 1])]).unwrap();
    // Refused on every host: for its length where there is a device, for the
    // missing device (or libraries) where there is not.
    match GpuScorer::new(&panel, Scoring::default(), Mode::Local) {
        Err(e) => eprintln!("[gpu_parity] refused as expected: {e}"),
        Ok(_) => panic!("a {} nt reference was accepted", MAX_REF_LEN + 1),
    }
}

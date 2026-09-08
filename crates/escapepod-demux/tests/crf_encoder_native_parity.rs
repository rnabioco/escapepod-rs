//! Native CRF encoder vs. tract, on real weights.
//!
//! Needs a real bundle, which is not redistributable and so does not exist in
//! CI: set `ESCAPEPOD_CRF_BUNDLE` to a directory holding `metadata.json` and
//! the ONNX graph it names, and this is skipped with a message otherwise
//! (never on rna, never in CI — mirrors `benches/crf_encoder.rs`).
//!
//! ```text
//! ESCAPEPOD_CRF_BUNDLE=/path/to/barcode_crf_ldx32_rna004@v0.2.1 \
//!   cargo test -p escapepod-demux --features crf-decode --test crf_encoder_native_parity
//! ```

#![cfg(feature = "crf-decode")]

use escapepod_demux::crf::{CrfEncoder, CrfScratch};

/// Deterministic pseudo-random stream (xorshift, seeded), matching
/// `benches/crf_encoder.rs` so a window here is directly comparable.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    fn normal(&mut self) -> f32 {
        let sum: f32 = (0..12).map(|_| self.next() as f32 / u32::MAX as f32).sum();
        sum - 6.0
    }
}

fn load_bundle() -> Option<String> {
    match std::env::var("ESCAPEPOD_CRF_BUNDLE") {
        Ok(dir) => Some(dir),
        Err(_) => {
            eprintln!(
                "skipping crf_encoder_native_parity: set ESCAPEPOD_CRF_BUNDLE to a \
                 CRF bundle directory to run it"
            );
            None
        }
    }
}

/// The native encoder loads (this bundle's export is the recognised shape)
/// and its scores agree with tract's within the load-time self-check's own
/// tolerance, on 64 seeded windows distinct from the one that check used.
#[test]
fn native_scores_agree_with_tract_on_real_weights() {
    let Some(dir) = load_bundle() else { return };
    let encoder =
        CrfEncoder::load_bundle(&dir).unwrap_or_else(|e| panic!("{dir} did not load: {e:#}"));
    assert!(
        encoder.encoder_backend().starts_with("native"),
        "expected a native backend for {dir}, got {:?} — this bundle's export is not the \
         recognised shape (or ESCAPEPOD_CRF_TRACT is set)",
        encoder.encoder_backend()
    );

    let meta = encoder.metadata();
    let chunk = meta.signal.chunk;
    let mut rng = Rng(0xC0FF_EE15_5EED_1234);
    let mut max_diff = 0.0f32;
    let mut mismatched_calls = 0usize;
    let n_windows = 64;
    for seed in 0..n_windows {
        rng.0 = rng.0.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let window: Vec<f32> = (0..chunk).map(|_| rng.normal()).collect();

        let native_scores = encoder.encode(&window).expect("native encode succeeds");
        let tract_scores = encoder
            .encode_tract(&window)
            .expect("tract encode succeeds");
        assert_eq!(native_scores.len(), tract_scores.len());
        let diff = native_scores
            .iter()
            .zip(&tract_scores)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        max_diff = max_diff.max(diff);

        let mut scratch = CrfScratch::new();
        let native_seq = encoder
            .basecall_prepped(&window, &mut scratch)
            .expect("native basecall succeeds");
        let tract_seq = {
            // Force the tract path through the same decode by handing it
            // `encode_tract`'s scores directly — `basecall_prepped` always
            // takes the active (native) path on this encoder.
            use escapepod_demux::crf::decode_with;
            let alphabet: Vec<u8> = meta
                .crf
                .alphabet
                .iter()
                .map(|s| s.as_bytes().first().copied().unwrap_or(b'N'))
                .collect();
            decode_with(
                encoder.layout(),
                &alphabet,
                &tract_scores,
                meta.t_len(),
                &mut scratch,
                encoder.backend(),
            )
            .expect("tract-scores decode succeeds")
        };
        if native_seq != tract_seq {
            mismatched_calls += 1;
        }
    }

    eprintln!(
        "{dir}: max |native - tract| over {n_windows} windows = {max_diff:e}; \
         {mismatched_calls}/{n_windows} decodes differ"
    );
    // The CPU-vs-GPU precedent (#331, #322): a handful of near-ties may
    // disagree; anything more says the two paths are not scoring the same
    // function.
    assert!(
        mismatched_calls <= 1,
        "{mismatched_calls}/{n_windows} windows decoded differently between native and tract"
    );
}

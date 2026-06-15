//! GPU/CPU parity tests for the batched SVB16 decoder.
//!
//! Compiled only with `--features gpu`. Skips gracefully (printing a message)
//! when no CUDA device / NVRTC is available, mirroring `gpu_dtw.rs`.

#![cfg(feature = "gpu")]

use escapepod_signal::compression::svb16;
use escapepod_signal::dtw::{GpuDtwContext, GpuDtwError};
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Build a GPU context, skipping gracefully if no device/NVRTC is present.
///
/// cudarc panics on the first call when `libcuda`/`libnvrtc` are missing, so
/// catch unwinds in addition to matching the `Result` (same as `gpu_dtw.rs`).
fn try_context() -> Option<GpuDtwContext> {
    match std::panic::catch_unwind(GpuDtwContext::new) {
        Ok(Ok(ctx)) => Some(ctx),
        Ok(Err(GpuDtwError::Driver(e))) => {
            eprintln!("[gpu_svb16] skipping — CUDA driver not usable: {e}");
            None
        }
        Ok(Err(e)) => {
            eprintln!("[gpu_svb16] skipping — GPU context init failed: {e}");
            None
        }
        Err(_) => {
            eprintln!("[gpu_svb16] skipping — GPU context init panicked (no CUDA libs?)");
            None
        }
    }
}

/// A signal whose deltas span the full 1-byte / 2-byte SVB16 key space:
/// small steps (1-byte values) interleaved with large jumps (2-byte values),
/// plus wraparound near `i16::MIN`/`i16::MAX` to exercise the wrapping math.
fn varied_signal(seed: u64, len: usize) -> Vec<i16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v = Vec::with_capacity(len);
    let mut cur: i32 = 0;
    for i in 0..len {
        // Mix of tiny deltas and occasional big jumps.
        let step: i32 = if i % 7 == 0 {
            rng.random::<i16>() as i32 // big jump → 2-byte value
        } else {
            (rng.random::<i8>() % 4) as i32 // small step → usually 1-byte
        };
        cur = (cur + step).clamp(i16::MIN as i32, i16::MAX as i32);
        v.push(cur as i16);
    }
    v
}

#[test]
fn gpu_svb16_matches_cpu_scalar() {
    let Some(ctx) = try_context() else { return };

    // Reads of assorted lengths, including edge cases (empty, 1, key-byte
    // boundaries at multiples of 8).
    let lens = [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 100, 1000, 4096, 50_000];
    let signals: Vec<Vec<i16>> = lens
        .iter()
        .enumerate()
        .map(|(i, &len)| varied_signal(0xC0FFEE + i as u64, len))
        .collect();

    let encoded: Vec<Vec<u8>> = signals
        .iter()
        .map(|s| svb16::encode(s).expect("encode"))
        .collect();

    let reads: Vec<(&[u8], usize)> = encoded
        .iter()
        .zip(&signals)
        .map(|(bytes, sig)| (bytes.as_slice(), sig.len()))
        .collect();

    let gpu_out = ctx.decode_svb16_batch(&reads).expect("gpu decode");

    assert_eq!(gpu_out.len(), signals.len());
    for (i, (gpu, expected)) in gpu_out.iter().zip(&signals).enumerate() {
        // Cross-check the CPU scalar decoder agrees with the original too.
        let cpu = svb16::decode_scalar(&encoded[i], expected.len()).expect("cpu decode");
        assert_eq!(&cpu, expected, "cpu scalar decode mismatch at read {i}");
        assert_eq!(
            gpu,
            expected,
            "gpu decode mismatch at read {i} (len {})",
            expected.len()
        );
    }
}

#[test]
fn gpu_svb16_large_batch() {
    let Some(ctx) = try_context() else { return };

    // Many reads at once → exercises the grid-stride launch and offset arrays.
    let n = 20_000usize;
    let signals: Vec<Vec<i16>> = (0..n)
        .map(|i| varied_signal(i as u64, 64 + (i % 256)))
        .collect();
    let encoded: Vec<Vec<u8>> = signals.iter().map(|s| svb16::encode(s).unwrap()).collect();
    let reads: Vec<(&[u8], usize)> = encoded
        .iter()
        .zip(&signals)
        .map(|(b, s)| (b.as_slice(), s.len()))
        .collect();

    let gpu_out = ctx.decode_svb16_batch(&reads).expect("gpu decode");
    assert_eq!(gpu_out, signals);
}

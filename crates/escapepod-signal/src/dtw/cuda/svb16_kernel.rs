//! GPU SVB16 decode kernel (one thread per read).
//!
//! Port of the scalar reference decoder in
//! `escapepod_pod5::compression::svb16::decode_scalar`. SVB16 decode is
//! inherently serial *within* a read (the control-byte stream sets each
//! value's byte offset, and the delta reconstruction is a prefix sum), but
//! embarrassingly parallel *across* reads — so one thread decodes one read
//! linearly, and a launch covers tens of thousands of reads at once.
//!
//! This is the first stage of the fused GPU signal chain: the host
//! zstd-decompresses each VBZ chunk on the CPU (no nvCOMP dependency) and
//! ships the SVB16 byte stream here; the decoded `i16` signal then stays
//! resident in VRAM for detect → fingerprint → classify, never crossing
//! PCIe back to the host.

/// Module name registered with the GPU device for the SVB16 decode kernel.
pub const MODULE_NAME: &str = "escapepod_gpu_svb16";
/// Kernel name registered with the GPU device. See [`MODULE_NAME`].
pub const KERNEL_NAME: &str = "svb16_decode_kernel";

/// CUDA-C source compiled at runtime via NVRTC.
///
/// Layout mirrors `decode_scalar`:
/// - `keys` section: `ceil(n/8)` bytes, 1 bit per sample (0 = 1-byte value,
///   1 = 2-byte little-endian value).
/// - `values` section: the variable-length data stream that follows.
///
/// Per sample: read the control bit, pull 1 or 2 bytes, zigzag-decode the
/// `u16` (`(v >> 1) ^ -(v & 1)`), add to the running `u16` accumulator
/// (wrapping), and store the result reinterpreted as `i16`.
pub const KERNEL_SRC: &str = r#"
extern "C" __global__
void svb16_decode_kernel(
    const unsigned char* __restrict__ data,    // all reads' SVB16 bytes, concatenated
    const long long*     __restrict__ data_off, // [n_reads + 1] byte offsets into `data`
    const int*           __restrict__ counts,   // [n_reads] sample count per read
    const long long*     __restrict__ out_off,  // [n_reads + 1] sample offsets into `out`
    short*               __restrict__ out,       // all reads' i16 samples, concatenated
    int n_reads)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_reads) return;

    int n = counts[r];
    if (n <= 0) return;

    const unsigned char* d = data + data_off[r];
    short* o = out + out_off[r];

    // keys_len = ceil(n / 8); values follow the key bytes.
    long long keys_len = ((long long)n + 7) >> 3;
    const unsigned char* keys   = d;
    const unsigned char* values = d + keys_len;

    long long data_offset = 0;
    unsigned short prev = 0;

    for (int i = 0; i < n; ++i) {
        int kb  = i >> 3;
        int bit = i & 7;
        int two = (keys[kb] >> bit) & 1;

        unsigned short value;
        if (two) {
            value = (unsigned short)values[data_offset]
                  | ((unsigned short)values[data_offset + 1] << 8);
            data_offset += 2;
        } else {
            value = (unsigned short)values[data_offset];
            data_offset += 1;
        }

        // Zigzag decode: (v >> 1) ^ -(v & 1)   (mask is 0x0000 or 0xFFFF).
        unsigned short mask  = (value & 1) ? (unsigned short)0xFFFF : (unsigned short)0;
        unsigned short delta = (unsigned short)((value >> 1) ^ mask);

        prev = (unsigned short)(prev + delta);
        o[i] = (short)prev;
    }
}
"#;

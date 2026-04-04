use super::config::UsqConfig;
use super::quantizer::UsqEncoded;

/// Number of vectors per fastscan block (32 = 2 cache lines of nibbles).
pub const BLOCK_SIZE: usize = 32;

/// SoA (Structure of Arrays) store for encoded vectors.
///
/// Also holds transposed 1-bit fastscan codes for fast SIMD candidate filtering.
/// The fastscan layout matches the nibble format expected by the AVX-512 kernel:
///   - Groups: padded_dim / 4
///   - Per group, per block: 16 bytes covering 32 vectors (4 bits/vector packed 2-per-byte)
pub struct UsqLayout {
    // IDs
    ids: Vec<i64>,
    // Metadata arrays (SoA)
    norms: Vec<f32>,
    norms_sq: Vec<f32>,
    vmaxs: Vec<f32>,
    quant_qualities: Vec<f32>,
    // B-bit codes (flat, code_bytes per vector)
    packed_bits: Vec<u8>,
    code_bytes: usize,
    // 1-bit fastscan transposed codes
    fastscan_codes: Vec<u8>,
    fastscan_block_size: usize,
    n_blocks: usize,
    // Dimensions
    padded_dim: usize,
}

impl UsqLayout {
    /// Build a UsqLayout from encoded vectors and their IDs.
    pub fn build(config: &UsqConfig, encoded: &[UsqEncoded], ids: &[i64]) -> Self {
        assert_eq!(
            encoded.len(),
            ids.len(),
            "encoded and ids must have the same length"
        );

        let code_bytes = config.code_bytes();
        let padded_dim = config.padded_dim();
        let n = encoded.len();

        let mut norms = Vec::with_capacity(n);
        let mut norms_sq = Vec::with_capacity(n);
        let mut vmaxs = Vec::with_capacity(n);
        let mut quant_qualities = Vec::with_capacity(n);
        let mut packed_bits = Vec::with_capacity(n * code_bytes);

        for item in encoded {
            assert_eq!(
                item.packed_bits.len(),
                code_bytes,
                "encoded packed_bits length mismatch: expected {code_bytes}, got {}",
                item.packed_bits.len()
            );
            norms.push(item.norm);
            norms_sq.push(item.norm_sq);
            vmaxs.push(item.vmax);
            quant_qualities.push(item.quant_quality);
            packed_bits.extend_from_slice(&item.packed_bits);
        }

        // Build fastscan transposed codes from sign_bits.
        let (fastscan_codes, n_blocks, fastscan_block_size) =
            transpose_sign_bits(encoded, n, padded_dim);

        Self {
            ids: ids.to_vec(),
            norms,
            norms_sq,
            vmaxs,
            quant_qualities,
            packed_bits,
            code_bytes,
            fastscan_codes,
            fastscan_block_size,
            n_blocks,
            padded_dim,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn id_at(&self, idx: usize) -> i64 {
        self.ids[idx]
    }

    pub fn norm_at(&self, idx: usize) -> f32 {
        self.norms[idx]
    }

    pub fn norm_sq_at(&self, idx: usize) -> f32 {
        self.norms_sq[idx]
    }

    pub fn vmax_at(&self, idx: usize) -> f32 {
        self.vmaxs[idx]
    }

    pub fn quant_quality_at(&self, idx: usize) -> f32 {
        self.quant_qualities[idx]
    }

    /// Returns the B-bit packed codes for vector at `idx`.
    /// Slice length is exactly `config.code_bytes()`.
    pub fn packed_bits_at(&self, idx: usize) -> &[u8] {
        let start = idx * self.code_bytes;
        &self.packed_bits[start..start + self.code_bytes]
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    pub fn fastscan_block_size(&self) -> usize {
        self.fastscan_block_size
    }

    /// Returns the fastscan transposed block at `block_idx`.
    /// Block format: `(padded_dim / 4)` groups of 16 bytes each.
    pub fn fastscan_block(&self, block_idx: usize) -> &[u8] {
        let start = block_idx * self.fastscan_block_size;
        &self.fastscan_codes[start..start + self.fastscan_block_size]
    }

    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }
}

/// Transpose 1-bit sign codes into the fastscan nibble format.
///
/// Layout per block:
///   - `n_groups = padded_dim / 4` groups of 4 consecutive dimensions
///   - Per group: 16 bytes covering all 32 vectors in the block
///     - Each vector contributes 1 nibble (4 bits) = sign bits for 4 dimensions
///     - 2 vectors per byte: low nibble = even slot, high nibble = odd slot
///
/// This matches the format consumed by the AVX-512 fastscan kernel.
fn transpose_sign_bits(
    encoded: &[UsqEncoded],
    count: usize,
    padded_dim: usize,
) -> (Vec<u8>, usize, usize) {
    let n_blocks = count.div_ceil(BLOCK_SIZE);
    // Each group covers 4 dims → 32 vectors × 4 bits / 8 = 16 bytes per group
    let n_groups = padded_dim.div_ceil(4);
    let fastscan_block_size = n_groups * 16;
    let mut fastscan_codes = vec![0u8; n_blocks * fastscan_block_size];

    for block_idx in 0..n_blocks {
        let block_base = block_idx * fastscan_block_size;
        for group_idx in 0..n_groups {
            let group_base = block_base + group_idx * 16;
            for slot in 0..BLOCK_SIZE {
                let vid = block_idx * BLOCK_SIZE + slot;
                let mut nibble = 0u8;
                if vid < count {
                    // sign_bits: 1 bit per dimension, packed LSB-first
                    let sign_code = &encoded[vid].sign_bits;
                    for bit_pos in 0..4usize {
                        let dim_idx = group_idx * 4 + bit_pos;
                        if dim_idx >= padded_dim {
                            break;
                        }
                        // Bounds check — sign_bits may be shorter than padded_dim/8
                        // if it was built before padding; tolerate gracefully.
                        let byte_idx = dim_idx / 8;
                        if byte_idx >= sign_code.len() {
                            break;
                        }
                        let bit = (sign_code[byte_idx] >> (dim_idx % 8)) & 1;
                        nibble |= bit << bit_pos;
                    }
                }

                // 2 vectors per byte: even slot → low nibble, odd slot → high nibble
                let dst = group_base + slot / 2;
                if slot % 2 == 0 {
                    fastscan_codes[dst] |= nibble;
                } else {
                    fastscan_codes[dst] |= nibble << 4;
                }
            }
        }
    }

    (fastscan_codes, n_blocks, fastscan_block_size)
}

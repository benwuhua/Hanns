#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsqConfig {
    pub dim: usize,
    pub nbits: u8,
    pub seed: u64,
}

impl UsqConfig {
    pub fn new(dim: usize, nbits: u8) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be > 0".to_string());
        }
        if nbits == 0 || nbits > 8 {
            return Err(format!("nbits must be 1..=8, got {nbits}"));
        }
        Ok(Self { dim, nbits, seed: 42 })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Padded to multiple of 64 for SIMD alignment.
    pub fn padded_dim(&self) -> usize {
        self.dim.div_ceil(64) * 64
    }

    /// Number of quantization levels (2^nbits).
    pub fn levels(&self) -> u32 {
        1u32 << self.nbits
    }

    /// Packed code size in bytes (B-bit codes for padded_dim dimensions).
    pub fn code_bytes(&self) -> usize {
        (self.padded_dim() * self.nbits as usize).div_ceil(8)
    }

    /// Per-vector metadata size: norm + norm_sq + vmax + quant_quality (4 x f32 = 16 bytes).
    pub fn meta_bytes(&self) -> usize {
        4 * std::mem::size_of::<f32>()
    }

    /// Size of the 1-bit sign code in bytes (1 bit per padded dimension).
    /// Note: when `nbits == 1`, sign_bytes() == code_bytes() because sign bits ARE the codes.
    /// For `nbits > 1`, sign_bytes() is the size of the fastscan approximation code (separate from packed_bits).
    pub fn sign_bytes(&self) -> usize {
        self.padded_dim() / 8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExRaBitQConfig {
    pub dim: usize,
    pub bits_per_dim: usize,
    pub rotation_seed: u64,
}

impl ExRaBitQConfig {
    pub fn new(dim: usize, bits_per_dim: usize) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be > 0".to_string());
        }
        if !matches!(bits_per_dim, 3 | 4 | 5 | 7 | 8 | 9) {
            return Err(format!(
                "unsupported bits_per_dim {}; expected one of 3, 4, 5, 7, 8, 9",
                bits_per_dim
            ));
        }

        Ok(Self {
            dim,
            bits_per_dim,
            rotation_seed: 42,
        })
    }

    pub fn with_rotation_seed(mut self, rotation_seed: u64) -> Self {
        self.rotation_seed = rotation_seed;
        self
    }

    pub fn padded_dim(&self) -> usize {
        self.dim.div_ceil(64) * 64
    }

    pub fn short_bits(&self) -> usize {
        1
    }

    pub fn ex_bits(&self) -> usize {
        self.bits_per_dim - self.short_bits()
    }

    pub fn short_code_bytes(&self) -> usize {
        self.padded_dim() / 8
    }

    pub fn long_code_bytes(&self) -> usize {
        match self.ex_bits() {
            2 => self.padded_dim() / 4,
            3 => (self.padded_dim() / 64) * (16 + 8),
            4 => self.padded_dim() / 2,
            6 => (self.padded_dim() / 64) * 48,
            7 => (self.padded_dim() / 64) * (48 + 8),
            8 => self.padded_dim(),
            bits => unreachable!("unsupported ex_bits {bits}"),
        }
    }
}

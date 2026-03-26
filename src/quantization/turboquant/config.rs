#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurboQuantMode {
    Mse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurboRotationBackend {
    DenseOrthogonal,
    Hadamard,
}

impl Default for TurboRotationBackend {
    fn default() -> Self {
        Self::Hadamard
    }
}

#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    pub dim: usize,
    pub bits_per_dim: u8,
    pub mode: TurboQuantMode,
    pub rotation_seed: u64,
    pub normalize_for_cosine: bool,
    pub rotation_backend: TurboRotationBackend,
}

impl TurboQuantConfig {
    pub fn new(dim: usize, bits_per_dim: u8) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(
            (1..=8).contains(&bits_per_dim),
            "bits_per_dim must be in 1..=8"
        );
        Self {
            dim,
            bits_per_dim,
            mode: TurboQuantMode::Mse,
            rotation_seed: 42,
            normalize_for_cosine: false,
            rotation_backend: TurboRotationBackend::default(),
        }
    }

    pub fn with_rotation_seed(mut self, rotation_seed: u64) -> Self {
        self.rotation_seed = rotation_seed;
        self
    }

    pub fn with_normalize_for_cosine(mut self, normalize_for_cosine: bool) -> Self {
        self.normalize_for_cosine = normalize_for_cosine;
        self
    }

    pub fn with_hadamard(mut self) -> Self {
        self.rotation_backend = TurboRotationBackend::Hadamard;
        self
    }

    pub fn with_dense_rotation(mut self) -> Self {
        self.rotation_backend = TurboRotationBackend::DenseOrthogonal;
        self
    }

    pub fn padded_dim(&self) -> usize {
        match self.rotation_backend {
            TurboRotationBackend::Hadamard => self.dim.next_power_of_two(),
            TurboRotationBackend::DenseOrthogonal => self.dim,
        }
    }

    pub fn code_bytes(&self) -> usize {
        (self.dim * self.bits_per_dim as usize).div_ceil(8)
    }
}

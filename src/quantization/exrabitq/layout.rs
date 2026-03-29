use super::{EncodedVector, ExFactor, ExRaBitQConfig};

pub const FAST_SIZE: usize = 32;

#[derive(Clone)]
pub struct ExRaBitQLayout {
    ids: Vec<i64>,
    short_codes: Vec<u8>,
    long_codes: Vec<u8>,
    factors: Vec<ExFactor>,
    short_ip_factors: Vec<f32>,
    short_sum_xb: Vec<f32>,
    short_err_factors: Vec<f32>,
    x_norms: Vec<f32>,
    x2: Vec<f32>,
    fastscan_codes: Vec<u8>,
    n_blocks: usize,
    fastscan_block_size: usize,
    short_code_bytes: usize,
    long_code_bytes: usize,
    padded_dim: usize,
}

impl ExRaBitQLayout {
    pub fn build(config: &ExRaBitQConfig, encoded: &[EncodedVector], ids: &[i64]) -> Self {
        assert_eq!(encoded.len(), ids.len());

        let short_code_bytes = config.short_code_bytes();
        let long_code_bytes = config.long_code_bytes();
        let padded_dim = config.padded_dim();
        let mut short_codes = Vec::with_capacity(encoded.len() * short_code_bytes);
        let mut long_codes = Vec::with_capacity(encoded.len() * long_code_bytes);
        let mut factors = Vec::with_capacity(encoded.len());
        let mut short_ip_factors = Vec::with_capacity(encoded.len());
        let mut short_sum_xb = Vec::with_capacity(encoded.len());
        let mut short_err_factors = Vec::with_capacity(encoded.len());
        let mut x_norms = Vec::with_capacity(encoded.len());
        let mut x2 = Vec::with_capacity(encoded.len());
        let mut raw_short = Vec::with_capacity(encoded.len());

        for item in encoded {
            assert_eq!(item.short_code.len(), short_code_bytes);
            assert_eq!(item.long_code.len(), long_code_bytes);
            short_codes.extend_from_slice(&item.short_code);
            long_codes.extend_from_slice(&item.long_code);
            factors.push(item.factor);
            short_ip_factors.push(item.short_factors.ip);
            short_sum_xb.push(item.short_factors.sum_xb);
            short_err_factors.push(item.short_factors.err);
            x_norms.push(item.x_norm);
            x2.push(item.x2);
            raw_short.push(item.short_code.clone());
        }

        let (fastscan_codes, n_blocks, fastscan_block_size) =
            transpose_short_codes(&raw_short, padded_dim);

        Self {
            ids: ids.to_vec(),
            short_codes,
            long_codes,
            factors,
            short_ip_factors,
            short_sum_xb,
            short_err_factors,
            x_norms,
            x2,
            fastscan_codes,
            n_blocks,
            fastscan_block_size,
            short_code_bytes,
            long_code_bytes,
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

    pub fn short_code_at(&self, idx: usize) -> &[u8] {
        let start = idx * self.short_code_bytes;
        &self.short_codes[start..start + self.short_code_bytes]
    }

    pub fn long_code_at(&self, idx: usize) -> &[u8] {
        let start = idx * self.long_code_bytes;
        &self.long_codes[start..start + self.long_code_bytes]
    }

    pub fn factor_at(&self, idx: usize) -> ExFactor {
        self.factors[idx]
    }

    pub fn x_norm_at(&self, idx: usize) -> f32 {
        self.x_norms[idx]
    }

    pub fn short_ip_at(&self, idx: usize) -> f32 {
        self.short_ip_factors[idx]
    }

    pub fn short_sum_xb_at(&self, idx: usize) -> f32 {
        self.short_sum_xb[idx]
    }

    pub fn short_err_at(&self, idx: usize) -> f32 {
        self.short_err_factors[idx]
    }

    pub fn x2_at(&self, idx: usize) -> f32 {
        self.x2[idx]
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    pub fn fastscan_block_size(&self) -> usize {
        self.fastscan_block_size
    }

    pub fn fastscan_block(&self, block_idx: usize) -> &[u8] {
        let start = block_idx * self.fastscan_block_size;
        &self.fastscan_codes[start..start + self.fastscan_block_size]
    }

    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }
}

fn transpose_short_codes(raw_codes: &[Vec<u8>], dim: usize) -> (Vec<u8>, usize, usize) {
    let n_blocks = raw_codes.len().div_ceil(FAST_SIZE);
    let fastscan_block_size = dim.div_ceil(4) * 16;
    let mut fastscan_codes = vec![0u8; n_blocks * fastscan_block_size];

    for block_idx in 0..n_blocks {
        let block_base = block_idx * fastscan_block_size;
        for group_idx in 0..dim.div_ceil(4) {
            let group_base = block_base + group_idx * 16;
            for slot in 0..FAST_SIZE {
                let vid = block_idx * FAST_SIZE + slot;
                let mut nibble = 0u8;
                if vid < raw_codes.len() {
                    for bit_pos in 0..4usize {
                        let dim_idx = group_idx * 4 + bit_pos;
                        if dim_idx >= dim {
                            break;
                        }
                        let byte = raw_codes[vid][dim_idx / 8];
                        let bit = (byte >> (dim_idx % 8)) & 1;
                        nibble |= bit << bit_pos;
                    }
                }

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

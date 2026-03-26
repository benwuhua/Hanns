pub fn pack_codes(codes: &[u16], bits_per_dim: u8, out: &mut Vec<u8>) {
    assert!((1..=8).contains(&bits_per_dim), "bits_per_dim must be in 1..=8");
    let total_bits = codes.len() * bits_per_dim as usize;
    out.clear();
    out.resize(total_bits.div_ceil(8), 0);

    let mask_limit = 1u16 << bits_per_dim;
    let mut bit_offset = 0usize;
    for &code in codes {
        assert!(code < mask_limit, "code out of range for bits_per_dim");
        let mut value = code as u32;
        let mut remaining = bits_per_dim as usize;

        while remaining > 0 {
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            let take = remaining.min(8 - bit_idx);
            let chunk_mask = (1u32 << take) - 1;
            out[byte_idx] |= ((value & chunk_mask) as u8) << bit_idx;
            value >>= take;
            bit_offset += take;
            remaining -= take;
        }
    }
}

pub fn unpack_codes(bytes: &[u8], n: usize, bits_per_dim: u8) -> Vec<u16> {
    assert!((1..=8).contains(&bits_per_dim), "bits_per_dim must be in 1..=8");

    let mut codes = Vec::with_capacity(n);
    let mut bit_offset = 0usize;
    for _ in 0..n {
        let mut value = 0u16;
        let mut written = 0usize;
        let mut remaining = bits_per_dim as usize;

        while remaining > 0 {
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            let take = remaining.min(8 - bit_idx);
            let chunk_mask = ((1u16 << take) - 1) as u8;
            let chunk = (bytes.get(byte_idx).copied().unwrap_or(0) >> bit_idx) & chunk_mask;
            value |= (chunk as u16) << written;
            written += take;
            bit_offset += take;
            remaining -= take;
        }

        codes.push(value);
    }

    codes
}

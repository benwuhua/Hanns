use knowhere_rs::quantization::usq::{UsqConfig, UsqQuantizer, UsqRotator};

#[test]
fn test_config_padded_dim() {
    // dim=128 → padded=128 (already multiple of 64)
    assert_eq!(UsqConfig::new(128, 4).unwrap().padded_dim(), 128);
    // dim=100 → padded=128
    assert_eq!(UsqConfig::new(100, 4).unwrap().padded_dim(), 128);
    // dim=65 → padded=128
    assert_eq!(UsqConfig::new(65, 4).unwrap().padded_dim(), 128);
}

#[test]
fn test_config_code_bytes() {
    let c = UsqConfig::new(128, 4).unwrap();
    // 128 dims * 4 bits / 8 = 64 bytes
    assert_eq!(c.code_bytes(), 64);

    let c = UsqConfig::new(128, 1).unwrap();
    // 128 dims * 1 bit / 8 = 16 bytes
    assert_eq!(c.code_bytes(), 16);

    let c = UsqConfig::new(128, 8).unwrap();
    // 128 dims * 8 bits / 8 = 128 bytes
    assert_eq!(c.code_bytes(), 128);
}

#[test]
fn test_rotator_preserves_norm() {
    let config = UsqConfig::new(128, 4).unwrap();
    let rotator = UsqRotator::new(&config);
    let padded = config.padded_dim();

    let v: Vec<f32> = (0..padded).map(|i| (i as f32 * 0.1).sin()).collect();
    let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    let rotated = rotator.rotate(&v);
    let norm_after: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        (norm_before - norm_after).abs() < 1e-3,
        "rotation should preserve norm: before={norm_before}, after={norm_after}"
    );
}

#[test]
fn test_rotator_inverse() {
    let config = UsqConfig::new(128, 4).unwrap();
    let rotator = UsqRotator::new(&config);
    let padded = config.padded_dim();

    let v: Vec<f32> = (0..padded).map(|i| (i as f32 * 0.37).sin()).collect();
    let rotated = rotator.rotate(&v);
    let recovered = rotator.inverse_rotate(&rotated);

    for i in 0..padded {
        assert!(
            (v[i] - recovered[i]).abs() < 1e-3,
            "inverse rotation failed at dim {i}: expected {}, got {}",
            v[i], recovered[i]
        );
    }
}

#[test]
fn test_rotator_deterministic() {
    let config = UsqConfig::new(128, 4).unwrap();
    let r1 = UsqRotator::new(&config);
    let r2 = UsqRotator::new(&config);
    assert_eq!(r1.matrix(), r2.matrix(), "same seed should produce same matrix");
}

// ---- Task 2: UsqQuantizer tests ----

#[test]
fn test_encode_4bit_basic() {
    let config = UsqConfig::new(128, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; 128]);

    let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
    let encoded = quantizer.encode(&v);

    assert!(encoded.norm > 0.0, "norm should be positive");
    assert!(encoded.vmax > 0.0, "vmax should be positive");
    assert!(encoded.quant_quality > 0.0, "quant_quality should be positive");
    assert_eq!(
        encoded.packed_bits.len(),
        config.code_bytes(),
        "packed_bits wrong size"
    );
    assert_eq!(
        encoded.sign_bits.len(),
        config.sign_bytes(),
        "sign_bits wrong size"
    );
}

#[test]
fn test_encode_1bit() {
    let config = UsqConfig::new(128, 1).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; 128]);

    let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
    let encoded = quantizer.encode(&v);

    // 1-bit: code size = padded_dim/8 = 128/8 = 16 bytes
    assert_eq!(encoded.packed_bits.len(), 16);
    // sign_bits should equal packed_bits for 1-bit
    assert_eq!(encoded.sign_bits, encoded.packed_bits);
}

#[test]
fn test_score_is_reasonable() {
    let dim = 128;
    let config = UsqConfig::new(dim, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; dim]);

    // Use the same vector as query (self-similarity should be high)
    let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
    let encoded = quantizer.encode(&v);

    // score(v, v) ≈ ‖v‖² (since centroid=0, rotation preserves inner products)
    let self_score = quantizer.score(&encoded, &v);
    let true_norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let relative_err =
        ((self_score - true_norm_sq) / true_norm_sq.max(1e-6)).abs();
    assert!(
        relative_err < 0.3,
        "self-score should approximate ‖v‖²: self_score={self_score:.4}, true_norm_sq={true_norm_sq:.4}, rel_err={relative_err:.4}"
    );
}

#[test]
fn test_encode_8bit() {
    let config = UsqConfig::new(64, 8).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; 64]);

    let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.5).cos()).collect();
    let encoded = quantizer.encode(&v);

    assert!(encoded.norm > 0.0);
    assert_eq!(encoded.packed_bits.len(), config.code_bytes());
    assert_eq!(encoded.sign_bits.len(), config.sign_bytes());
    // For 8-bit: code_bytes = padded_dim*8/8 = padded_dim; sign_bytes = padded_dim/8 — different sizes
    assert_ne!(encoded.sign_bits.len(), encoded.packed_bits.len());
}

#[test]
fn test_encode_2bit() {
    let config = UsqConfig::new(128, 2).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; 128]);

    let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
    let encoded = quantizer.encode(&v);

    // 2-bit: code_bytes = padded_dim * 2 / 8 = 128*2/8 = 32 bytes
    assert_eq!(encoded.packed_bits.len(), config.code_bytes());
    // sign_bytes = padded_dim / 8 = 16 bytes
    assert_eq!(encoded.sign_bits.len(), config.sign_bytes());
    assert!(encoded.norm > 0.0);
    assert!(encoded.quant_quality > 0.0);
}

#[test]
fn test_nonzero_centroid() {
    let dim = 64;
    let config = UsqConfig::new(dim, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    let centroid: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
    quantizer.set_centroid(&centroid);

    let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();
    let encoded = quantizer.encode(&v);

    assert!(encoded.norm > 0.0);
    assert_eq!(encoded.packed_bits.len(), config.code_bytes());
}

#[test]
fn test_encode_rotated_matches_encode() {
    let dim = 128;
    let config = UsqConfig::new(dim, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; dim]);

    let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();

    // encode() and encode_rotated() (with manual padding + rotation) must agree.
    let encoded_direct = quantizer.encode(&v);

    let padded_dim = config.padded_dim();
    let mut padded = vec![0.0f32; padded_dim];
    padded[..dim].copy_from_slice(&v);
    let rotated = quantizer.rotator().rotate(&padded);
    let encoded_via_rotated = quantizer.encode_rotated(&rotated);

    assert_eq!(encoded_direct.packed_bits, encoded_via_rotated.packed_bits);
    assert!((encoded_direct.norm - encoded_via_rotated.norm).abs() < 1e-5);
    assert!((encoded_direct.vmax - encoded_via_rotated.vmax).abs() < 1e-5);
}

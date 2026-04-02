use knowhere_rs::quantization::usq::{UsqConfig, UsqRotator};

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

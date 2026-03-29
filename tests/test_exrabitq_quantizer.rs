use knowhere_rs::quantization::exrabitq::{ExRaBitQConfig, ExRaBitQQuantizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut values: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for value in &mut values {
        *value /= norm;
    }
    values
}

#[test]
fn test_exrabitq_config_pads_dim_to_multiple_of_64() {
    let cfg = ExRaBitQConfig::new(768, 4).unwrap();
    assert_eq!(cfg.padded_dim(), 768);

    let cfg = ExRaBitQConfig::new(770, 4).unwrap();
    assert_eq!(cfg.padded_dim(), 832);
}

#[test]
fn test_exrabitq_config_rejects_unsupported_bits() {
    assert!(ExRaBitQConfig::new(768, 6).is_err());
}

#[test]
fn test_exrabitq_rotation_is_seed_deterministic() {
    let cfg = ExRaBitQConfig::new(96, 4).unwrap().with_rotation_seed(7);
    let q1 = ExRaBitQQuantizer::new(cfg.clone()).unwrap();
    let q2 = ExRaBitQQuantizer::new(cfg).unwrap();
    assert_eq!(q1.rotation_matrix(), q2.rotation_matrix());
}

#[test]
fn test_fast_quantize_is_not_worse_than_reference_greedy() {
    let cfg = ExRaBitQConfig::new(64, 4).unwrap().with_rotation_seed(11);
    let q = ExRaBitQQuantizer::new(cfg).unwrap();
    let v = random_unit_vector(64, 123);
    let fast = q.fast_quantize_for_test(&v);
    let greedy = q.greedy_quantize_for_test(&v, 3);
    assert!(fast.objective + 1e-6 >= greedy.objective);
}

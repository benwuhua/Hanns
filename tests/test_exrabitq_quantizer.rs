use knowhere_rs::quantization::exrabitq::{ExRaBitQConfig, ExRaBitQQuantizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut values: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for value in &mut values {
        *value /= norm;
    }
    values
}

#[derive(Clone, Copy, Debug)]
struct ReferenceThresholdEvent {
    threshold: f64,
    dim: usize,
}

impl PartialEq for ReferenceThresholdEvent {
    fn eq(&self, other: &Self) -> bool {
        self.threshold.to_bits() == other.threshold.to_bits() && self.dim == other.dim
    }
}

impl Eq for ReferenceThresholdEvent {}

impl PartialOrd for ReferenceThresholdEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReferenceThresholdEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .threshold
            .total_cmp(&self.threshold)
            .then_with(|| other.dim.cmp(&self.dim))
    }
}

fn normalize_abs_padded(vector: &[f32], padded_dim: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; padded_dim];
    padded[..vector.len()].copy_from_slice(vector);
    let norm = padded.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for value in &mut padded {
        *value = (*value / norm).abs();
    }
    padded
}

fn reference_fast_quantize(values: &[f32], ex_bits: usize) -> (Vec<u8>, f32, f32, f32) {
    let max_level = ((1usize << ex_bits) - 1) as u8;
    let max_o = values.iter().copied().fold(0.0f32, f32::max).max(1e-12) as f64;
    let t_start = (((1usize << ex_bits) - 1) / 3) as f64 / max_o;
    let t_end = (((1usize << ex_bits) - 1) + 10) as f64 / max_o;

    let mut cur_o_bar = vec![0u8; values.len()];
    let mut sqr_denominator = values.len() as f64 * 0.25;
    let mut numerator = 0.0f64;
    for (idx, &value) in values.iter().enumerate() {
        let level = (t_start * value as f64 + 1e-5f64)
            .floor()
            .clamp(0.0, max_level as f64) as u8;
        cur_o_bar[idx] = level;
        let level = f64::from(level);
        sqr_denominator += level * level + level;
        numerator += (level + 0.5) * value as f64;
    }

    let mut next_t = BinaryHeap::new();
    for (dim, &value) in values.iter().enumerate() {
        let threshold = if value > 0.0 {
            (f64::from(cur_o_bar[dim]) + 1.0) / value as f64
        } else {
            f64::INFINITY
        };
        next_t.push(ReferenceThresholdEvent { threshold, dim });
    }

    let mut max_ip = 0.0f64;
    let mut best_t = 0.0f64;
    while let Some(event) = next_t.pop() {
        let dim = event.dim;
        if cur_o_bar[dim] >= max_level {
            continue;
        }

        cur_o_bar[dim] += 1;
        let updated = f64::from(cur_o_bar[dim]);
        sqr_denominator += 2.0 * updated;
        numerator += values[dim] as f64;

        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = event.threshold;
        }

        if cur_o_bar[dim] < max_level {
            let value = values[dim];
            let t_next = if value > 0.0 {
                (f64::from(cur_o_bar[dim]) + 1.0) / value as f64
            } else {
                f64::INFINITY
            };
            if t_next < t_end {
                next_t.push(ReferenceThresholdEvent {
                    threshold: t_next,
                    dim,
                });
            }
        }
    }

    let mut codes = vec![0u8; values.len()];
    sqr_denominator = values.len() as f64 * 0.25;
    numerator = 0.0;
    for (idx, &value) in values.iter().enumerate() {
        let level = (best_t * value as f64 + 1e-5f64)
            .floor()
            .clamp(0.0, max_level as f64) as u8;
        codes[idx] = level;
        let level = f64::from(level);
        sqr_denominator += level * level + level;
        numerator += (level + 0.5) * value as f64;
    }

    (
        codes,
        numerator as f32,
        sqr_denominator as f32,
        if sqr_denominator > 0.0 {
            ((numerator * numerator) / sqr_denominator) as f32
        } else {
            0.0
        },
    )
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

#[test]
fn test_fast_quantize_matches_reference_scan() {
    for &bits_per_dim in &[3usize, 4, 5, 7, 8, 9] {
        let cfg = ExRaBitQConfig::new(64, bits_per_dim)
            .unwrap()
            .with_rotation_seed(19 + bits_per_dim as u64);
        let q = ExRaBitQQuantizer::new(cfg.clone()).unwrap();
        let v = random_unit_vector(cfg.dim, 1000 + bits_per_dim as u64);
        let abs_padded = normalize_abs_padded(&v, cfg.padded_dim());
        let expected = reference_fast_quantize(&abs_padded, cfg.ex_bits());
        let actual = q.fast_quantize_for_test(&v);

        assert_eq!(actual.codes, expected.0, "bits_per_dim={bits_per_dim}");
        assert!(
            (actual.numerator - expected.1).abs() < 1e-5,
            "bits_per_dim={bits_per_dim}: numerator actual={} expected={}",
            actual.numerator,
            expected.1
        );
        assert!(
            (actual.denominator - expected.2).abs() < 1e-5,
            "bits_per_dim={bits_per_dim}: denominator actual={} expected={}",
            actual.denominator,
            expected.2
        );
        assert!(
            (actual.objective - expected.3).abs() < 1e-5,
            "bits_per_dim={bits_per_dim}: objective actual={} expected={}",
            actual.objective,
            expected.3
        );
    }
}

#[test]
fn test_compacted_long_code_roundtrip_and_ip() {
    let dim = 64usize;
    let query = random_unit_vector(dim, 20260329);

    for &bits_per_dim in &[3usize, 4, 5, 7, 8, 9] {
        let cfg = ExRaBitQConfig::new(dim, bits_per_dim).unwrap();
        let q = ExRaBitQQuantizer::new(cfg.clone()).unwrap();
        let max_level = ((1usize << cfg.ex_bits()) - 1) as u8;
        let mut rng = StdRng::seed_from_u64(5000 + bits_per_dim as u64);
        let raw_levels: Vec<u8> = (0..cfg.padded_dim())
            .map(|_| rng.gen_range(0..=max_level))
            .collect();

        let compact = q.compact_long_code_for_test(&raw_levels);
        assert_eq!(compact.len(), cfg.long_code_bytes(), "bits_per_dim={bits_per_dim}");

        let decoded = q.decode_long_code_levels_for_test(&compact);
        assert_eq!(decoded, raw_levels, "bits_per_dim={bits_per_dim}");

        let expected = query
            .iter()
            .zip(raw_levels.iter())
            .map(|(lhs, rhs)| lhs * *rhs as f32)
            .sum::<f32>();
        let via_quantizer = q.long_code_inner_product(&query, &compact);
        assert!(
            (via_quantizer - expected).abs() < 1e-5,
            "bits_per_dim={bits_per_dim}: via_quantizer={} expected={}",
            via_quantizer,
            expected
        );
    }
}

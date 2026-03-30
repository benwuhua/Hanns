use knowhere_rs::quantization::exrabitq::{
    reference_short_distance, rerank_candidates, scalar_scan_layout, scan_and_rerank,
    EncodedVector, ExRaBitQConfig, ExRaBitQFastScanState, ExRaBitQLayout, ExRaBitQQuantizer,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * dim];
    for value in &mut data {
        *value = rng.gen_range(-1.0f32..1.0f32);
    }
    data
}

fn build_fixture(
    n: usize,
    dim: usize,
    bits: usize,
) -> (
    ExRaBitQQuantizer,
    Vec<f32>,
    Vec<f32>,
    Vec<i64>,
    Vec<EncodedVector>,
    ExRaBitQLayout,
) {
    let config = ExRaBitQConfig::new(dim, bits)
        .unwrap()
        .with_rotation_seed(7);
    let quantizer = ExRaBitQQuantizer::new(config).unwrap();
    let centroid = vec![0.0f32; dim];
    let data = random_vectors(n, dim, 1234);
    let ids: Vec<i64> = (0..n as i64).collect();
    let encoded: Vec<EncodedVector> = data
        .chunks_exact(dim)
        .map(|row| quantizer.encode_with_centroid(row, &centroid))
        .collect();
    let layout = ExRaBitQLayout::build(quantizer.config(), &encoded, &ids);
    (quantizer, data, centroid, ids, encoded, layout)
}

#[test]
fn test_short_code_layout_roundtrips_blockwise() {
    let (_quantizer, _data, _centroid, ids, encoded, layout) = build_fixture(40, 32, 4);
    assert_eq!(layout.len(), ids.len());
    for i in 0..ids.len() {
        assert_eq!(layout.id_at(i), ids[i]);
        assert_eq!(layout.short_code_at(i), encoded[i].short_code.as_slice());
        assert_eq!(layout.long_code_at(i), encoded[i].long_code.as_slice());
    }
}

#[test]
fn test_scalar_fastscan_matches_reference_lower_bounds() {
    let (quantizer, data, centroid, _ids, _encoded, layout) = build_fixture(96, 32, 4);
    let query = &data[0..32];
    let (q_rot, y2) = quantizer.rotate_query_residual(query, &centroid);
    let state = ExRaBitQFastScanState::new(&q_rot, y2);

    let fast = scalar_scan_layout(&layout, &state, layout.len());
    let mut slow = Vec::with_capacity(layout.len());
    for idx in 0..layout.len() {
        slow.push((
            idx,
            reference_short_distance(
                &state,
                layout.short_code_at(idx),
                layout.short_ip_at(idx),
                layout.short_sum_xb_at(idx),
                layout.short_err_at(idx),
                layout.x2_at(idx),
            ),
        ));
    }
    slow.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    assert_eq!(fast.len(), slow.len());
    for (got, expected) in fast.iter().zip(slow.iter()) {
        assert_eq!(got.idx, expected.0);
        assert!((got.distance - expected.1).abs() < 1e-5);
    }
}

#[test]
fn test_high_accuracy_fastscan_matches_reference_lower_bounds() {
    let (quantizer, data, centroid, _ids, _encoded, layout) = build_fixture(96, 32, 4);
    let query = &data[1 * 32..2 * 32];
    let (q_rot, y2) = quantizer.rotate_query_residual(query, &centroid);
    let state = ExRaBitQFastScanState::new_high_accuracy(&q_rot, y2);

    let fast = scalar_scan_layout(&layout, &state, layout.len());
    let mut slow = Vec::with_capacity(layout.len());
    for idx in 0..layout.len() {
        slow.push((
            idx,
            reference_short_distance(
                &state,
                layout.short_code_at(idx),
                layout.short_ip_at(idx),
                layout.short_sum_xb_at(idx),
                layout.short_err_at(idx),
                layout.x2_at(idx),
            ),
        ));
    }
    slow.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    assert_eq!(fast.len(), slow.len());
    for (got, expected) in fast.iter().zip(slow.iter()) {
        assert_eq!(got.idx, expected.0);
        assert!((got.distance - expected.1).abs() < 1e-4);
    }
}

#[test]
fn test_rerank_returns_self_for_identical_query() {
    let (quantizer, data, centroid, ids, _encoded, layout) = build_fixture(96, 32, 4);
    let target = 0usize;
    let query = &data[target * 32..(target + 1) * 32];
    let (q_rot, y2) = quantizer.rotate_query_residual(query, &centroid);
    let state = ExRaBitQFastScanState::new(&q_rot, y2);
    let candidates = scalar_scan_layout(&layout, &state, 16);
    let reranked = rerank_candidates(&quantizer, &layout, &state, &candidates, 5);

    assert!(!reranked.is_empty());
    assert_eq!(reranked[0].0, ids[target]);
    for pair in reranked.windows(2) {
        assert!(pair[0].1 <= pair[1].1);
    }
}

#[test]
fn test_fastscan_avx512_matches_scalar_for_4bit() {
    #[cfg(target_arch = "x86_64")]
    {
        if !(std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512f"))
        {
            return;
        }

        let (quantizer, data, centroid, _ids, _encoded, layout) = build_fixture(160, 128, 4);
        let query = &data[2 * 128..3 * 128];
        let (q_rot, y2) = quantizer.rotate_query_residual(query, &centroid);
        let state = ExRaBitQFastScanState::new(&q_rot, y2);

        let scalar = scalar_scan_layout(&layout, &state, layout.len());
        let simd = knowhere_rs::quantization::exrabitq::simd_scan_layout(&layout, &state, layout.len())
            .expect("avx512 fast scan");

        assert_eq!(scalar.len(), simd.len());
        for (lhs, rhs) in scalar.iter().zip(simd.iter()) {
            assert_eq!(lhs.idx, rhs.idx);
            assert!((lhs.distance - rhs.distance).abs() < 1e-5);
            assert!((lhs.rabitq_ip - rhs.rabitq_ip).abs() < 1e-5);
        }
    }
}

#[test]
fn test_fastscan_avx512_matches_scalar_for_8bit() {
    #[cfg(target_arch = "x86_64")]
    {
        if !(std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512f"))
        {
            return;
        }

        let (quantizer, data, centroid, _ids, _encoded, layout) = build_fixture(160, 128, 8);
        let query = &data[3 * 128..4 * 128];
        let (q_rot, y2) = quantizer.rotate_query_residual(query, &centroid);
        let state = ExRaBitQFastScanState::new(&q_rot, y2);

        let scalar = scalar_scan_layout(&layout, &state, layout.len());
        let simd = knowhere_rs::quantization::exrabitq::simd_scan_layout(&layout, &state, layout.len())
            .expect("avx512 fast scan");

        assert_eq!(scalar.len(), simd.len());
        for (lhs, rhs) in scalar.iter().zip(simd.iter()) {
            assert_eq!(lhs.idx, rhs.idx);
            assert!((lhs.distance - rhs.distance).abs() < 1e-5);
            assert!((lhs.rabitq_ip - rhs.rabitq_ip).abs() < 1e-5);
        }
    }
}

#[test]
fn test_high_accuracy_bitmask_scan_matches_scalar_rerank() {
    let (quantizer, data, centroid, _ids, _encoded, layout) = build_fixture(192, 64, 4);
    let top_k = 10usize;

    for query_idx in 0..4usize {
        let query = &data[query_idx * 64..(query_idx + 1) * 64];
        let (q_rot, y2) = quantizer.rotate_query_residual(query, &centroid);
        let state = ExRaBitQFastScanState::new_high_accuracy(&q_rot, y2);

        let scalar_candidates = scalar_scan_layout(&layout, &state, layout.len());
        let exact_scalar = rerank_candidates(&quantizer, &layout, &state, &scalar_candidates, top_k);
        let bitmask = scan_and_rerank(&quantizer, &layout, &state, 32, top_k);

        let hits = exact_scalar
            .iter()
            .filter(|(id, _)| bitmask.iter().any(|(other_id, _)| other_id == id))
            .count();
        let recall = hits as f32 / top_k as f32;
        assert!(
            recall >= 1.0,
            "query_idx={query_idx}: recall={} exact={:?} bitmask={:?}",
            recall,
            exact_scalar,
            bitmask
        );

        assert_eq!(exact_scalar.len(), bitmask.len());
        for (lhs, rhs) in exact_scalar.iter().zip(bitmask.iter()) {
            assert_eq!(lhs.0, rhs.0, "query_idx={query_idx}");
            assert!((lhs.1 - rhs.1).abs() < 1e-4, "query_idx={query_idx}: lhs={lhs:?} rhs={rhs:?}");
        }
    }
}

use knowhere_rs::quantization::exrabitq::{
    reference_short_distance, rerank_candidates, scalar_scan_layout, EncodedVector, ExRaBitQConfig,
    ExRaBitQFastScanState, ExRaBitQLayout, ExRaBitQQuantizer,
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

use knowhere_rs::quantization::usq::*;

fn make_data(n: usize, dim: usize) -> (Vec<f32>, Vec<i64>) {
    let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.13).sin()).collect();
    let ids: Vec<i64> = (0..n as i64).collect();
    (data, ids)
}

fn build_test_layout(n: usize, dim: usize, nbits: u8) -> (UsqQuantizer, UsqLayout, Vec<i64>) {
    let config = UsqConfig::new(dim, nbits).unwrap();
    let mut q = UsqQuantizer::new(config.clone());
    q.set_centroid(&vec![0.0f32; dim]);
    let (data, ids) = make_data(n, dim);
    let encoded: Vec<_> = (0..n).map(|i| q.encode(&data[i * dim..(i + 1) * dim])).collect();
    let layout = UsqLayout::build(&config, &encoded, &ids);
    (q, layout, ids)
}

#[test]
fn test_fastscan_state_zero_query() {
    let config = UsqConfig::new(128, 4).unwrap();
    let q_rot = vec![0.0f32; config.padded_dim()];
    let state = UsqFastScanState::new(&q_rot, &config);
    // Zero query → zero LUT entries.
    assert!(state.lut.iter().all(|&v| v == 0), "zero query should give zero LUT");
    assert_eq!(state.lut_scale, 1e-6); // clamped minimum
}

#[test]
fn test_fastscan_topk_returns_n() {
    let (q, layout, _) = build_test_layout(100, 128, 4);
    let config = q.config().clone();
    let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.71).cos()).collect();
    let mut q_padded = vec![0.0f32; config.padded_dim()];
    q_padded[..128].copy_from_slice(&query);
    let q_rot = q.rotator().rotate(&q_padded);
    let state = UsqFastScanState::new(&q_rot, &config);

    let candidates = fastscan_topk(&layout, &state, 20);
    assert!(candidates.len() <= 20);
    assert!(!candidates.is_empty());
}

#[test]
fn test_scan_and_rerank_recall() {
    let dim = 128;
    let n = 200; // n > n_candidates(150) for nbits=4 → exercises fastscan code path
    let k = 10;
    let (quantizer, layout, _ids) = build_test_layout(n, dim, 4);
    let config = quantizer.config().clone();

    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.71).cos()).collect();
    let mut q_padded = vec![0.0f32; config.padded_dim()];
    q_padded[..dim].copy_from_slice(&query);
    let q_rot = quantizer.rotator().rotate(&q_padded);
    let q_norm_sq: f32 = q_rot.iter().map(|x| x * x).sum();

    // Use UsqQueryState for scoring
    let query_state = quantizer.precompute_query_state(&query);

    // Brute force reference (identical scoring path).
    let mut bf: Vec<(i64, f32)> = (0..n)
        .map(|i| {
            let score = quantizer.score_with_meta(
                &query_state,
                layout.norm_at(i),
                layout.vmax_at(i),
                layout.quant_quality_at(i),
                layout.packed_bits_at(i),
            );
            let dist = (q_norm_sq + layout.norm_sq_at(i) - 2.0 * score).max(0.0);
            (i as i64, dist)
        })
        .collect();
    bf.sort_by(|a, b| a.1.total_cmp(&b.1));
    let bf_top: Vec<i64> = bf.iter().take(k).map(|r| r.0).collect();

    // scan_and_rerank (n=200 > n_candidates=150 for nbits=4, so fastscan path is taken).
    let fs_state = UsqFastScanState::new(&q_rot, &config);
    let results =
        scan_and_rerank(&quantizer, &layout, &fs_state, &query_state, q_norm_sq, k);
    let result_ids: Vec<i64> = results.iter().map(|r| r.0).collect();

    let overlap = bf_top.iter().filter(|id| result_ids.contains(id)).count();
    assert!(
        overlap >= 4,
        "scan_and_rerank top-{k} fastscan path should have >=40% overlap with brute force, got {overlap}/{k}\nbf_top={bf_top:?}\nresult={result_ids:?}"
    );
}

#[test]
fn test_scan_and_rerank_exercises_fastscan() {
    // With n=200, nbits=4: n_candidates = 10*15 = 150, so 200 > 150 → fastscan path
    let dim = 128;
    let n = 200;
    let k = 10;
    let config = UsqConfig::new(dim, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; dim]);

    let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.13).sin()).collect();
    let ids: Vec<i64> = (0..n as i64).collect();
    let encoded: Vec<_> = (0..n).map(|i| quantizer.encode(&data[i*dim..(i+1)*dim])).collect();
    let layout = UsqLayout::build(&config, &encoded, &ids);

    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.71).cos()).collect();
    let mut q_padded = vec![0.0f32; config.padded_dim()];
    q_padded[..dim].copy_from_slice(&query);
    let q_rot = quantizer.rotator().rotate(&q_padded);
    let q_norm_sq: f32 = q_rot.iter().map(|x| x * x).sum();

    // Use UsqQueryState for scoring
    let query_state = quantizer.precompute_query_state(&query);

    // Brute force reference
    let mut bf: Vec<(i64, f32)> = (0..n).map(|i| {
        let score = quantizer.score_with_meta(
            &query_state,
            layout.norm_at(i), layout.vmax_at(i),
            layout.quant_quality_at(i), layout.packed_bits_at(i),
        );
        (i as i64, q_norm_sq + layout.norm_sq_at(i) - 2.0 * score)
    }).collect();
    bf.sort_by(|a, b| a.1.total_cmp(&b.1));
    let bf_top: Vec<i64> = bf.iter().take(k).map(|r| r.0).collect();

    // Two-stage with fastscan
    let fs_state = UsqFastScanState::new(&q_rot, &config);
    let results = scan_and_rerank(&quantizer, &layout, &fs_state, &query_state, q_norm_sq, k);
    let result_ids: Vec<i64> = results.iter().map(|r| r.0).collect();

    let overlap = bf_top.iter().filter(|id| result_ids.contains(id)).count();
    assert!(
        overlap >= 4,
        "fastscan+rerank top-{k} should have >=40% overlap with brute force\nbf_top={bf_top:?}\nresult={result_ids:?}\noverlap={overlap}"
    );
}

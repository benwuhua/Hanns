use std::collections::HashSet;

use knowhere_rs::{
    pick_refine_index, IvfRaBitqConfig, IvfRaBitqIndex, MetricType, RefineType, SearchRequest,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn exact_topk(data: &[f32], queries: &[f32], dim: usize, top_k: usize) -> Vec<Vec<i64>> {
    queries
        .chunks(dim)
        .map(|query| {
            let mut distances: Vec<(i64, f32)> = data
                .chunks(dim)
                .enumerate()
                .map(|(id, vector)| {
                    let dist = query
                        .iter()
                        .zip(vector.iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum::<f32>();
                    (id as i64, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.total_cmp(&b.1));
            distances
                .into_iter()
                .take(top_k)
                .map(|(id, _)| id)
                .collect()
        })
        .collect()
}

fn recall_at_k(results: &[Vec<i64>], ground_truth: &[Vec<i64>], top_k: usize) -> f32 {
    let mut hits = 0usize;
    for (result, gt) in results.iter().zip(ground_truth.iter()) {
        let gt_set: HashSet<i64> = gt.iter().copied().collect();
        hits += result
            .iter()
            .take(top_k)
            .filter(|id| gt_set.contains(id))
            .count();
    }
    hits as f32 / (results.len() * top_k) as f32
}

fn build_clustered_dataset(
    clusters: usize,
    points_per_cluster: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(7);
    let mut data = Vec::with_capacity(clusters * points_per_cluster * dim);
    let mut queries = Vec::with_capacity(clusters * dim);

    for cluster in 0..clusters {
        let mut center = vec![0.0f32; dim];
        for value in &mut center {
            *value = rng.gen_range(-1.0..1.0) + cluster as f32 * 0.35;
        }

        for point in 0..points_per_cluster {
            for (d, &base) in center.iter().enumerate() {
                let jitter = ((point * (d + 3) + cluster) % 17) as f32 * 0.008 - 0.06;
                let hard_negative = if point % 9 == 0 { 0.03 } else { -0.02 };
                data.push(base + jitter + hard_negative);
            }
        }

        for (d, &base) in center.iter().enumerate() {
            let jitter = ((cluster * (d + 5)) % 13) as f32 * 0.005 - 0.02;
            queries.push(base + jitter);
        }
    }

    (data, queries)
}

fn search_ids(
    index: &IvfRaBitqIndex,
    queries: &[f32],
    dim: usize,
    req: &SearchRequest,
) -> Vec<Vec<i64>> {
    queries
        .chunks(dim)
        .map(|query| index.search(query, req).unwrap().ids)
        .collect()
}

#[test]
fn test_refine_type_creation() {
    let dim = 4;
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.08, 0.08, 0.08, 0.08, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0,
    ];
    let ids = vec![10, 11, 12, 13];

    for refine_type in [
        RefineType::DataView,
        RefineType::Uint8Quant,
        RefineType::Float16Quant,
        RefineType::Bfloat16Quant,
    ] {
        let refine = pick_refine_index(&data, dim, &ids, MetricType::L2, Some(refine_type))
            .unwrap()
            .unwrap();
        assert_eq!(refine.refine_type(), refine_type);
        assert_eq!(refine.len(), ids.len());
    }
}

#[test]
fn test_refine_distance() {
    let dim = 4;
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.08, 0.08, 0.08, 0.08, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0,
    ];
    let ids = vec![10, 11, 12, 13];
    let query = vec![0.02, 0.02, 0.02, 0.02];

    for refine_type in [
        RefineType::DataView,
        RefineType::Uint8Quant,
        RefineType::Float16Quant,
        RefineType::Bfloat16Quant,
    ] {
        let refine = pick_refine_index(&data, dim, &ids, MetricType::L2, Some(refine_type))
            .unwrap()
            .unwrap();
        let d10 = refine.refine_distance(&query, 10).unwrap();
        let d11 = refine.refine_distance(&query, 11).unwrap();
        let d12 = refine.refine_distance(&query, 12).unwrap();

        assert!(
            d10 <= d11,
            "expected id=10 closer than id=11 for {:?}",
            refine_type
        );
        assert!(
            d11 < d12,
            "expected id=11 closer than id=12 for {:?}",
            refine_type
        );
    }
}

#[test]
fn test_pick_refine_index() {
    let dim = 4;
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.08, 0.08, 0.08, 0.08, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0,
    ];
    let ids = vec![10, 11, 12, 13];
    let query = vec![0.02, 0.02, 0.02, 0.02];
    let coarse_candidates = vec![(12, 0.01), (11, 0.02), (10, 0.03)];

    assert!(pick_refine_index(&data, dim, &ids, MetricType::L2, None)
        .unwrap()
        .is_none());

    let refine = pick_refine_index(
        &data,
        dim,
        &ids,
        MetricType::L2,
        Some(RefineType::Float16Quant),
    )
    .unwrap()
    .unwrap();

    let reranked = refine.rerank(&query, &coarse_candidates, 3);
    assert_eq!(reranked[0].0, 10);
    assert_eq!(reranked[1].0, 11);
}

#[test]
fn test_refine_recall_improves_dataview_and_fp16() {
    let dim = 32;
    let top_k = 10;
    let (data, queries) = build_clustered_dataset(10, 120, dim);
    let ground_truth = exact_topk(&data, &queries, dim, top_k);

    let mut base = IvfRaBitqIndex::new(IvfRaBitqConfig::new(dim, 24).with_nprobe(4));
    base.train(&data).unwrap();
    base.add(&data, None).unwrap();

    let mut dataview = IvfRaBitqIndex::new(
        IvfRaBitqConfig::new(dim, 24)
            .with_nprobe(4)
            .with_refine(RefineType::DataView, 100),
    );
    dataview.train(&data).unwrap();
    dataview.add(&data, None).unwrap();

    let mut fp16 = IvfRaBitqIndex::new(
        IvfRaBitqConfig::new(dim, 24)
            .with_nprobe(4)
            .with_refine(RefineType::Float16Quant, 100),
    );
    fp16.train(&data).unwrap();
    fp16.add(&data, None).unwrap();

    let req = SearchRequest {
        top_k,
        nprobe: 4,
        filter: None,
        params: None,
        radius: None,
    };

    let base_recall = recall_at_k(
        &search_ids(&base, &queries, dim, &req),
        &ground_truth,
        top_k,
    );
    let dataview_recall = recall_at_k(
        &search_ids(&dataview, &queries, dim, &req),
        &ground_truth,
        top_k,
    );
    let fp16_recall = recall_at_k(
        &search_ids(&fp16, &queries, dim, &req),
        &ground_truth,
        top_k,
    );

    println!(
        "recall@{} baseline={:.3}, dataview={:.3}, fp16={:.3}",
        top_k, base_recall, dataview_recall, fp16_recall
    );

    assert!(
        dataview_recall >= base_recall + 0.02,
        "expected dataview refine to improve recall, baseline={:.3}, refined={:.3}",
        base_recall,
        dataview_recall
    );
    assert!(
        fp16_recall >= base_recall + 0.02,
        "expected fp16 refine to improve recall, baseline={:.3}, refined={:.3}",
        base_recall,
        fp16_recall
    );
    assert!(
        dataview_recall >= 0.92,
        "dataview recall too low: {:.3}",
        dataview_recall
    );
    assert!(
        fp16_recall >= 0.92,
        "fp16 recall too low: {:.3}",
        fp16_recall
    );
}

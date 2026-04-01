use knowhere_rs::{pick_refine_index, MetricType, RefineType};

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

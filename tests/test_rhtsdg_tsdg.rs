use knowhere_rs::faiss::rhtsdg::tsdg::{
    stage1_prune_neighbors, stage2_filter_neighbors, DistanceBackend, DistanceMatrix,
};
use knowhere_rs::MetricType;

fn line_with_off_axis_escape_fixture() -> DistanceMatrix<'static> {
    let points = vec![
        0.0, 0.0, // center 0
        1.0, 0.0, // neighbor 1
        2.0, 0.0, // neighbor 2, should be alpha-pruned by 1
        0.0, 3.0, // neighbor 3, should survive
    ];
    DistanceMatrix::from_points(2, Box::leak(points.into_boxed_slice()))
}

fn reverse_edge_gate_fixture() -> DistanceMatrix<'static> {
    let points = vec![
        0.0, 0.0, // center 0
        2.0, 0.0, // alive neighbor 1
        1.0, 0.0, // reverse neighbor 2, occluded by 1
        9.0, 9.0, // unused spacer 3
        0.0, 3.0, // reverse neighbor 4, should survive
    ];
    DistanceMatrix::from_points(2, Box::leak(points.into_boxed_slice()))
}

#[test]
fn alpha_pruning_drops_center_sorted_occluded_neighbor() {
    let fixture = line_with_off_axis_escape_fixture();
    let (alive, occs) = stage1_prune_neighbors(0, &[1, 2, 3], &fixture, 1.2);

    assert_eq!(alive, vec![1, 3]);
    assert_eq!(occs, vec![0, 0]);
}

#[test]
fn l2_distance_matrix_uses_squared_distance_for_internal_ordering() {
    let points = vec![
        0.0, 0.0, //
        2.0, 0.0, //
    ];
    let matrix = DistanceMatrix::from_points(2, Box::leak(points.into_boxed_slice()));

    assert_eq!(matrix.distance(0, 1), 4.0);
}

#[test]
fn l2_distance_matrix_prefers_simd_ptr_kernel_while_non_l2_uses_fallback() {
    let l2_points = vec![
        0.0, 0.0, //
        2.0, 0.0, //
    ];
    let ip_points = vec![
        0.0, 0.0, //
        2.0, 0.0, //
    ];

    let l2_matrix = DistanceMatrix::from_points(2, Box::leak(l2_points.into_boxed_slice()));
    let ip_matrix = DistanceMatrix::from_points_with_metric(
        2,
        Box::leak(ip_points.into_boxed_slice()),
        knowhere_rs::MetricType::Ip,
    );

    assert_eq!(l2_matrix.distance_backend(), DistanceBackend::L2FastPath);
    assert_eq!(ip_matrix.distance_backend(), DistanceBackend::Fallback);
}

#[test]
fn node_id_subset_mapping_still_resolves_logical_indices() {
    let points = vec![
        10.0, 0.0, // physical 0
        20.0, 0.0, // physical 1
        30.0, 0.0, // physical 2
        40.0, 0.0, // physical 3
    ];
    let node_ids = vec![2, 0];
    let matrix = DistanceMatrix::from_points_for_nodes(
        2,
        Box::leak(points.into_boxed_slice()),
        MetricType::L2,
        Some(Box::leak(node_ids.into_boxed_slice())),
    );

    assert_eq!(matrix.distance(0, 1), 400.0);
}

#[test]
fn ip_distance_matrix_keeps_fallback_distance_values() {
    let points = vec![
        1.0, 2.0, //
        3.0, 4.0, //
    ];
    let matrix = DistanceMatrix::from_points_with_metric(
        2,
        Box::leak(points.into_boxed_slice()),
        MetricType::Ip,
    );

    assert_eq!(matrix.distance(0, 1), -11.0);
}

#[test]
fn occurrence_filter_adds_only_reverse_edges_below_threshold() {
    let fixture = reverse_edge_gate_fixture();
    let kept = stage2_filter_neighbors(0, &[1], &[0], &[2, 4], &fixture, 0, 2);

    assert_eq!(kept, vec![1, 4]);
}

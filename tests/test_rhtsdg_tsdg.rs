use knowhere_rs::faiss::rhtsdg::tsdg::{
    stage1_prune_neighbors, stage2_filter_neighbors, DistanceMatrix,
};

fn line_with_off_axis_escape_fixture() -> DistanceMatrix {
    DistanceMatrix::from_points(
        2,
        vec![
            0.0, 0.0, // center 0
            1.0, 0.0, // neighbor 1
            2.0, 0.0, // neighbor 2, should be alpha-pruned by 1
            0.0, 3.0, // neighbor 3, should survive
        ],
    )
}

fn reverse_edge_gate_fixture() -> DistanceMatrix {
    DistanceMatrix::from_points(
        2,
        vec![
            0.0, 0.0, // center 0
            2.0, 0.0, // alive neighbor 1
            1.0, 0.0, // reverse neighbor 2, occluded by 1
            9.0, 9.0, // unused spacer 3
            0.0, 3.0, // reverse neighbor 4, should survive
        ],
    )
}

#[test]
fn alpha_pruning_drops_center_sorted_occluded_neighbor() {
    let fixture = line_with_off_axis_escape_fixture();
    let (alive, occs) = stage1_prune_neighbors(0, &[1, 2, 3], &fixture, 1.2);

    assert_eq!(alive, vec![1, 3]);
    assert_eq!(occs, vec![0, 0]);
}

#[test]
fn occurrence_filter_adds_only_reverse_edges_below_threshold() {
    let fixture = reverse_edge_gate_fixture();
    let kept = stage2_filter_neighbors(0, &[1], &[0], &[2, 4], &fixture, 0, 2);

    assert_eq!(kept, vec![1, 4]);
}

use knowhere_rs::faiss::rhtsdg::neighbor::{Neighbor, NeighborStatus, Neighborhood};
use knowhere_rs::faiss::rhtsdg::xndescent::{XNDescentBuilder, XNDescentConfig};

#[test]
fn insert_neighbor_dedupes_and_keeps_best_k() {
    let neighborhood = Neighborhood::new(2);

    assert!(neighborhood.insert(7, 0.3, NeighborStatus::New));
    assert!(!neighborhood.insert(7, 0.5, NeighborStatus::New));
    assert!(neighborhood.insert(8, 0.2, NeighborStatus::Old));
    assert!(neighborhood.insert(9, 0.1, NeighborStatus::New));

    let snapshot = neighborhood.snapshot();
    let ids: Vec<u32> = snapshot.iter().map(|neighbor| neighbor.id).collect();
    assert_eq!(ids, vec![9, 8]);
}

#[test]
fn local_join_updates_both_endpoints_symmetrically() {
    let mut builder = XNDescentBuilder::new_for_tests(
        2,
        vec![
            0.0, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
        ],
        XNDescentConfig {
            k: 2,
            sample_count: 2,
            iter_count: 1,
            reverse_count: 0,
            use_shortcut: false,
        },
    );

    builder.seed_neighbors_for_test(
        0,
        &[
            Neighbor::new(1, 1.0, NeighborStatus::New),
            Neighbor::new(2, 2.0, NeighborStatus::New),
        ],
    );

    let updates = builder.local_join_once_for_test();
    assert!(updates >= 2);
    assert!(builder.has_edge_for_test(1, 2));
    assert!(builder.has_edge_for_test(2, 1));
}

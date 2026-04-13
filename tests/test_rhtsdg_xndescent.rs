use hanns::faiss::rhtsdg::neighbor::{Neighbor, NeighborStatus, Neighborhood};
use hanns::faiss::rhtsdg::xndescent::{XNDescentBuilder, XNDescentConfig};
use hanns::faiss::rhtsdg::RhtsdgIndex;

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

#[test]
fn local_join_uses_reverse_new_neighbors_to_connect_endpoints() {
    let mut builder = XNDescentBuilder::new_for_tests(
        2,
        vec![
            0.0, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
        ],
        XNDescentConfig {
            k: 2,
            sample_count: 1,
            iter_count: 1,
            reverse_count: 2,
            use_shortcut: false,
        },
    );

    builder.seed_neighbors_for_test(0, &[Neighbor::new(1, 1.0, NeighborStatus::New)]);
    builder.seed_neighbors_for_test(2, &[Neighbor::new(1, 1.0, NeighborStatus::New)]);

    let updates = builder.local_join_once_for_test();
    assert!(
        updates >= 2,
        "reverse-neighbor local join should connect both endpoints"
    );
    assert!(builder.has_edge_for_test(0, 2));
    assert!(builder.has_edge_for_test(2, 0));
}

#[test]
fn reverse_sampling_prefers_nearer_sources_when_truncated() {
    let mut builder = XNDescentBuilder::new_for_tests(
        2,
        vec![
            10.0, 0.0, //
            0.0, 0.0, //
            0.5, 0.0, //
            1.0, 0.0, //
        ],
        XNDescentConfig {
            k: 2,
            sample_count: 1,
            iter_count: 1,
            reverse_count: 1,
            use_shortcut: false,
        },
    );

    builder.seed_neighbors_for_test(0, &[Neighbor::new(1, 100.0, NeighborStatus::New)]);
    builder.seed_neighbors_for_test(1, &[Neighbor::new(3, 1.0, NeighborStatus::New)]);
    builder.seed_neighbors_for_test(2, &[Neighbor::new(1, 0.25, NeighborStatus::New)]);

    let updates = builder.local_join_once_for_test();
    assert!(updates >= 2);
    assert!(
        builder.has_edge_for_test(2, 3),
        "reverse sampling should keep the nearer source when reverse_count truncates"
    );
    assert!(builder.has_edge_for_test(3, 2));
}

#[test]
fn sampled_init_avoids_ring_wrap_on_small_line_fixture() {
    let graph = XNDescentBuilder::build_for_tests(
        2,
        vec![
            0.0, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
        ],
        XNDescentConfig {
            k: 1,
            sample_count: 1,
            iter_count: 1,
            reverse_count: 0,
            use_shortcut: false,
        },
    );

    assert_eq!(
        graph[3],
        vec![2],
        "seed graph should prefer the nearest neighbor"
    );
}

#[test]
fn rebuild_samples_prefers_nearest_new_neighbors_before_old_neighbors() {
    let neighborhood = Neighborhood::new(4);
    neighborhood.insert(40, 4.0, NeighborStatus::Old);
    neighborhood.insert(10, 1.0, NeighborStatus::New);
    neighborhood.insert(30, 3.0, NeighborStatus::New);
    neighborhood.insert(20, 2.0, NeighborStatus::New);

    neighborhood.rebuild_samples(2);
    let (nn_new, nn_old) = neighborhood.sample_lists();

    assert_eq!(nn_new, vec![10, 20]);
    assert_eq!(nn_old, vec![30, 40]);
}

#[test]
fn insert_neighbor_prefers_smaller_id_when_distance_ties_at_capacity() {
    let neighborhood = Neighborhood::new(2);

    assert!(neighborhood.insert(30, 1.0, NeighborStatus::New));
    assert!(neighborhood.insert(20, 1.0, NeighborStatus::New));
    assert!(neighborhood.insert(10, 1.0, NeighborStatus::New));

    let ids: Vec<u32> = neighborhood
        .snapshot()
        .into_iter()
        .map(|neighbor| neighbor.id)
        .collect();
    assert_eq!(ids, vec![10, 20]);
}

#[test]
fn build_trace_reports_nonzero_xndescent_and_tsdg_phases() {
    let dim = 2;
    let width = 16usize;
    let mut vectors = Vec::with_capacity(width * width * dim);
    for y in 0..width {
        for x in 0..width {
            vectors.push(x as f32);
            vectors.push(y as f32);
        }
    }

    let trace = RhtsdgIndex::build_trace_for_tests(dim, vectors);

    assert!(trace.xndescent_iters > 0);
    assert!(trace.stage1_pairs_checked > 0);
    assert!(trace.stage2_reverse_candidates > 0);
}

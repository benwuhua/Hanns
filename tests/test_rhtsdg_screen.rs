use knowhere_rs::faiss::rhtsdg::{run_local_screen_fixture_for_test, RhtsdgIndex};

fn small_grid_fixture() -> (RhtsdgIndex, Vec<f32>, Vec<Vec<u32>>) {
    let vectors = vec![
        0.0, 0.0, //
        1.0, 0.0, //
        0.0, 1.0, //
        1.0, 1.0, //
    ];

    let layer0 = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];

    let index = RhtsdgIndex::new_for_tests(2, vectors, layer0, 0);
    let queries = vec![
        0.05, 0.02, //
        0.92, 0.08, //
        0.10, 0.85, //
        0.95, 0.95, //
    ];
    let truth = vec![vec![0], vec![1], vec![2], vec![3]];
    (index, queries, truth)
}

#[test]
fn search_matches_bruteforce_top1_on_small_grid() {
    let (index, queries, truth) = small_grid_fixture();
    let got = index.search_batch_for_test(&queries, 1, 4);

    assert_eq!(got, truth);
}

#[test]
fn build_from_vectors_recovers_small_grid_top1() {
    let vectors = vec![
        0.0, 0.0, //
        1.0, 0.0, //
        0.0, 1.0, //
        1.0, 1.0, //
    ];

    let queries = vec![
        0.05, 0.02, //
        0.92, 0.08, //
        0.10, 0.85, //
        0.95, 0.95, //
    ];
    let truth = vec![vec![0], vec![1], vec![2], vec![3]];

    let index = RhtsdgIndex::build_for_tests(2, vectors);
    let got = index.search_batch_for_test(&queries, 1, 4);

    assert_eq!(index.build_kind_for_test(), "xndescent_tsdg_fixture");
    assert_eq!(got, truth);
}

#[test]
fn build_from_larger_fixture_produces_multiple_layers() {
    let dim = 2;
    let width = 16usize;
    let mut vectors = Vec::with_capacity(width * width * dim);
    for y in 0..width {
        for x in 0..width {
            vectors.push(x as f32);
            vectors.push(y as f32);
        }
    }

    let index = RhtsdgIndex::build_for_tests(dim, vectors);
    let layer_sizes = index.layer_sizes_for_test();

    assert!(
        layer_sizes.len() > 1,
        "expected multi-layer graph, got {:?}",
        layer_sizes
    );
    assert_eq!(layer_sizes[0], width * width);
    assert!(
        *layer_sizes.last().unwrap() < layer_sizes[0],
        "top layer should be sparser than layer 0, got {:?}",
        layer_sizes
    );
}

#[test]
fn search_trace_reports_layer0_and_upper_layer_work_separately() {
    let dim = 2;
    let width = 16usize;
    let mut vectors = Vec::with_capacity(width * width * dim);
    for y in 0..width {
        for x in 0..width {
            vectors.push(x as f32);
            vectors.push(y as f32);
        }
    }

    let index = RhtsdgIndex::build_for_tests(dim, vectors);
    let trace = index.search_trace_for_test(&[15.0, 15.0], 10, 16);

    assert!(
        trace.upper_layer_visits > 0,
        "expected upper-layer work in layered search trace, got {:?}",
        trace
    );
    assert!(trace.frontier_pops > 0);
}

#[test]
fn search_prunes_dominated_frontier_candidates_once_ef_is_saturated() {
    let vectors = vec![
        5.0, 0.0,  //
        0.0, 0.0,  //
        50.0, 0.0, //
        60.0, 0.0, //
        70.0, 0.0, //
    ];
    let layer0 = vec![
        vec![1, 2, 3, 4],
        vec![0],
        vec![0],
        vec![0],
        vec![0],
    ];

    let index = RhtsdgIndex::new_for_tests(2, vectors, layer0, 0);
    let trace = index.search_trace_for_test(&[0.0, 0.0], 1, 1);

    assert_eq!(trace.results, vec![1]);
    assert_eq!(
        trace.visited,
        5,
        "search should still mark dominated neighbors as seen to avoid repeated distance work"
    );
    assert_eq!(
        trace.frontier_pops,
        2,
        "search should avoid pushing dominated neighbors onto the frontier once ef is saturated"
    );
    assert_eq!(trace.batch4_calls, 1);
}

#[test]
fn search_trace_uses_batch4_l2_distance_when_four_neighbors_are_available() {
    let vectors = vec![
        10.0, 0.0, //
        0.0, 0.0,  //
        1.0, 0.0,  //
        2.0, 0.0,  //
        3.0, 0.0,  //
    ];
    let layer0 = vec![
        vec![1, 2, 3, 4],
        vec![0],
        vec![0],
        vec![0],
        vec![0],
    ];

    let index = RhtsdgIndex::new_for_tests(2, vectors, layer0, 0);
    let trace = index.search_trace_for_test(&[0.0, 0.0], 4, 4);

    assert_eq!(trace.results, vec![1, 2, 3, 4]);
    assert!(
        trace.batch4_calls > 0,
        "layer search should use batch-4 L2 distance when four neighbors are available"
    );
}

#[test]
#[ignore]
fn screen_rhtsdg_recall_gate_on_synthetic_fixture() {
    let summary = run_local_screen_fixture_for_test();
    println!("screen summary: {:?}", summary);

    assert_eq!(summary.build_kind, "xndescent_tsdg_fixture");
    assert!(
        summary.recall_at_10 >= 0.95,
        "recall@10 below gate: {}",
        summary.recall_at_10
    );
}

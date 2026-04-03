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

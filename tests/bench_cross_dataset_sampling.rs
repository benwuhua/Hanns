#![cfg(feature = "long-tests")]

use knowhere_rs::benchmark::{
    generate_cross_dataset_artifact, CROSS_DATASET_OUTPUT_PATH,
};

#[test]
#[ignore = "cross-dataset artifact generation; excluded from default regression"]
fn test_generate_cross_dataset_sampling() {
    let artifact = generate_cross_dataset_artifact(CROSS_DATASET_OUTPUT_PATH)
        .expect("generate cross dataset artifact");

    assert_eq!(artifact.benchmark, "BENCH-P2-003-cross-dataset-sampling");
    assert!(artifact.rows.len() >= 6, "artifact rows must cover >= 3 datasets x 2 indexes");
}

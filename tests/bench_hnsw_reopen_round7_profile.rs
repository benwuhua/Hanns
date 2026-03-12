#[cfg(feature = "long-tests")]
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType};
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::hnsw::HnswCandidateSearchProfileReport;
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::HnswIndex;
#[cfg(feature = "long-tests")]
use knowhere_rs::MetricType;
#[cfg(feature = "long-tests")]
use serde_json::json;
#[cfg(feature = "long-tests")]
use std::fs;
#[cfg(feature = "long-tests")]
use std::path::Path;
#[cfg(feature = "long-tests")]
use std::time::Instant;

#[cfg(feature = "long-tests")]
const HNSW_REOPEN_ROUND7_BASELINE_PATH: &str =
    "benchmark_results/hnsw_reopen_layer0_prefetch_audit_round6.json";
#[cfg(feature = "long-tests")]
const HNSW_REOPEN_ROUND7_FLAT_GRAPH_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_layer0_flat_graph_audit_round7.json";

#[cfg(feature = "long-tests")]
fn profile_dimensions() -> (usize, usize, usize) {
    (12_000, 128, 128)
}

#[cfg(feature = "long-tests")]
fn generate_profile_base_vectors(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(num_vectors * dim);
    for i in 0..num_vectors {
        let cluster = (i % 64) as f32 * 0.01;
        let band = ((i / 64) % 32) as f32 * 0.001;
        for d in 0..dim {
            let raw = ((i * 131 + d * 17 + (i % 29) * (d % 7)) % 1024) as f32 / 1024.0;
            vectors.push(raw + cluster + band + (d % 11) as f32 * 0.0001);
        }
    }
    vectors
}

#[cfg(feature = "long-tests")]
fn generate_profile_query_vectors(base: &[f32], num_queries: usize, dim: usize) -> Vec<f32> {
    let num_base = base.len() / dim;
    let mut queries = Vec::with_capacity(num_queries * dim);
    for i in 0..num_queries {
        let src = (i * 97) % num_base;
        for d in 0..dim {
            let base_value = base[src * dim + d];
            let offset = (((i + d) % 5) as f32 - 2.0) * 0.0002;
            queries.push(base_value + offset);
        }
    }
    queries
}

#[cfg(feature = "long-tests")]
fn build_round7_flat_graph_audit_artifact(
    report: &HnswCandidateSearchProfileReport,
    search_wall_clock_ms: f64,
) -> serde_json::Value {
    let (num_base, num_queries, dim) = profile_dimensions();
    let search_qps = if search_wall_clock_ms > 0.0 {
        num_queries as f64 / (search_wall_clock_ms / 1000.0)
    } else {
        num_queries as f64
    };
    let layout_gap_summary = if report.search_core_shape.rust_layer0_neighbor_layout
        == "flat_u32_adjacency"
    {
        "Round 7 switches layer-0 neighbor reads from per-node Vec<i64> indirection to a flat u32 adjacency slab for the ordered-pool L2 loop. The next step is authority same-schema rerun to measure QPS delta."
    } else {
        "Round 7 still reads layer-0 neighbors from per-node Vec<i64> containers; flat adjacency migration is not active yet."
    };
    let recommended_next_target =
        if report.search_core_shape.rust_layer0_neighbor_layout == "flat_u32_adjacency" {
            "round7_same_schema_authority_rerun"
        } else {
            "layer0_flat_graph_enablement_fix"
        };

    json!({
        "task_id": "HNSW-REOPEN-LAYER0-FLAT-GRAPH-AUDIT-ROUND7",
        "family": "HNSW",
        "benchmark_lane": "hnsw_reopen_layer0_flat_graph_audit_round7",
        "authority_scope": "remote_x86_only",
        "round7_baseline_source": HNSW_REOPEN_ROUND7_BASELINE_PATH,
        "native_reference_files": [
            "thirdparty/faiss/faiss/cppcontrib/knowhere/impl/HnswSearcher.h",
            "thirdparty/faiss/faiss/cppcontrib/knowhere/thirdparty/hnswlib/hnswalg.h"
        ],
        "rust_reference_files": [
            "src/faiss/hnsw.rs"
        ],
        "dataset_source": "synthetic_sift_like_128d",
        "dataset": {
            "base_vectors": num_base,
            "query_vectors": num_queries,
            "dim": dim
        },
        "config": {
            "m": 16,
            "ef_construction": 200,
            "ef_search": 64,
            "top_k": 10
        },
        "search_core_shape": &report.search_core_shape,
        "batch_distance_mode": &report.batch_distance_mode,
        "batch_distance_call_counts": &report.batch_distance_call_counts,
        "prefetch_mode": &report.prefetch_mode,
        "prefetch_call_counts": &report.prefetch_call_counts,
        "layer0_neighbor_access_call_counts": &report.layer0_neighbor_access_call_counts,
        "candidate_search_breakdown": &report.candidate_search_breakdown,
        "call_counts": &report.call_counts,
        "distance_compute_breakdown": &report.distance_compute_breakdown,
        "distance_compute_call_counts": &report.distance_compute_call_counts,
        "hotspot_ranking": &report.hotspot_ranking,
        "recommended_next_target": recommended_next_target,
        "layout_gap_summary": layout_gap_summary,
        "sample_search": {
            "query_count": report.query_count,
            "search_wall_clock_ms": search_wall_clock_ms,
            "qps": search_qps
        },
        "total_profiled_ms": report.total_profiled_ms
    })
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "round 7 layer-0 flat-graph audit artifact generation; excluded from default regression"]
fn test_generate_hnsw_reopen_round7_layer0_flat_graph_audit() {
    let artifact_path = Path::new(HNSW_REOPEN_ROUND7_FLAT_GRAPH_AUDIT_PATH);
    if let Some(parent) = artifact_path.parent() {
        assert!(
            parent.exists(),
            "benchmark_results directory must exist before generating the round 7 flat-graph audit artifact"
        );
    }

    let (num_base, num_queries, dim) = profile_dimensions();
    let base = generate_profile_base_vectors(num_base, dim);
    let queries = generate_profile_query_vectors(&base, num_queries, dim);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create HNSW round 7 flat-graph audit index");
    index
        .train(&base)
        .expect("train HNSW round 7 flat-graph audit index");
    index
        .add(&base, None)
        .expect("add base vectors for HNSW round 7 flat-graph audit index");

    let search_start = Instant::now();
    let report = index
        .candidate_search_profile_report(&queries, 64, 10)
        .expect("profile HNSW round 7 layer-0 flat-graph audit");
    let search_wall_clock_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let artifact = build_round7_flat_graph_audit_artifact(&report, search_wall_clock_ms);

    fs::write(
        artifact_path,
        serde_json::to_string_pretty(&artifact)
            .expect("serialize round 7 layer-0 flat-graph audit artifact"),
    )
    .expect("write round 7 layer-0 flat-graph audit artifact");

    assert_eq!(
        artifact["search_core_shape"]["rust_layer0_neighbor_layout"],
        "flat_u32_adjacency"
    );
    assert_eq!(
        artifact["search_core_shape"]["rust_layer0_neighbor_id_type"],
        "u32"
    );
    assert!(
        artifact["layer0_neighbor_access_call_counts"]["layer0_flat_graph_neighbor_reads"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "generated artifact must show non-zero layer-0 flat-graph neighbor reads"
    );
}

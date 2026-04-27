#![cfg(feature = "long-tests")]
//! Focused HNSW-PQ aligned benchmark lane.
//!
//! Official `benchmark_float.TEST_HNSW_PQ` emits FP32 FLAT-refine rows before
//! failing on the later FP16 combination. This lane uses Hanns HnswPqIndex with
//! retained raw vectors and exact reranking over `top_k * refine_k` candidates.

use hanns::benchmark::average_recall_at_k;
use hanns::dataset::load_sift1m_complete;
use hanns::faiss::{HnswPqConfig, HnswPqIndex};
use hanns::MetricType;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn parse_usize_list(var: &str, default: &[usize]) -> Vec<usize> {
    env::var(var)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn parse_optional_usize(var: &str) -> Option<usize> {
    env::var(var)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn env_optional_string(var: &str) -> Option<String> {
    env::var(var)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn search_queries(
    index: &HnswPqIndex,
    queries: &[f32],
    dim: usize,
    top_k: usize,
    refine_k: usize,
) -> Vec<Vec<i64>> {
    #[cfg(feature = "parallel")]
    {
        queries
            .par_chunks(dim)
            .map(|query| {
                index
                    .search_refined(query, top_k, refine_k, None)
                    .expect("search refined HNSW-PQ")
                    .ids
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        queries
            .chunks(dim)
            .map(|query| {
                index
                    .search_refined(query, top_k, refine_k, None)
                    .expect("search refined HNSW-PQ")
                    .ids
            })
            .collect()
    }
}

#[test]
#[ignore = "SIFT1M remote benchmark lane; run on HannsDB-x86 for authority evidence"]
fn bench_hnsw_pq_aligned_topk100() {
    let dataset_path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());
    let dataset = load_sift1m_complete(&dataset_path).expect("load SIFT1M dataset");
    let nq = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000)
        .min(dataset.num_query());
    let nb = env::var("SIFT_NUM_BASE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(dataset.num_base())
        .min(dataset.num_base());
    let ef_values = parse_usize_list("HNSWPQ_EF_VALUES", &[128]);
    let refine_k_values = parse_usize_list("HNSWPQ_REFINE_K_VALUES", &[1, 2, 4, 8, 16]);
    let m = env::var("HNSWPQ_M")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    let ef_construction = env::var("HNSWPQ_EF_CONSTRUCTION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200);
    let pq_m = env::var("HNSWPQ_PQ_M")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let pq_k = env::var("HNSWPQ_PQ_K")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256);
    let top_k = 100usize;
    let query_threads = query_threads();
    let default_build_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(query_threads);
    let build_threads = parse_optional_usize("HNSWPQ_BUILD_THREADS")
        .unwrap_or(default_build_threads)
        .max(1);

    if let Some(expected_threads) = parse_optional_usize("HNSWPQ_EXPECT_THREADS") {
        assert_eq!(
            query_threads, expected_threads,
            "aligned HNSW-PQ verdict lane must run with expected thread count"
        );
    }
    if let Some(expected_threads) = parse_optional_usize("HNSWPQ_EXPECT_BUILD_THREADS") {
        assert_eq!(
            build_threads, expected_threads,
            "aligned HNSW-PQ verdict lane must run with expected build thread count"
        );
    }

    let base_all = dataset.base.vectors();
    let base = &base_all[..nb * dataset.dim()];
    let queries = dataset.query.vectors();
    let query_batch = &queries[..nq * dataset.dim()];
    let gt = dataset
        .ground_truth
        .iter()
        .take(nq)
        .cloned()
        .collect::<Vec<_>>();

    let run_set_id = format!(
        "hnsw-pq-aligned-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs()
    );
    let mut rows = Vec::new();

    let initial_ef = ef_values.first().copied().unwrap_or(128);
    let config = HnswPqConfig::new(dataset.dim())
        .with_m(m)
        .with_ef_construction(ef_construction)
        .with_ef_search(initial_ef)
        .with_pq_params(pq_m, pq_k)
        .with_metric_type(MetricType::L2)
        .with_num_threads(build_threads);

    let mut index = HnswPqIndex::new(config).expect("create HNSW-PQ");
    let train_start = Instant::now();
    index.train(base).expect("train HNSW-PQ");
    let train_s = train_start.elapsed().as_secs_f64();
    let add_start = Instant::now();
    index.add(base, None).expect("add HNSW-PQ");
    let add_s = add_start.elapsed().as_secs_f64();
    let build_s = train_s + add_s;
    assert!(
        hanns::index::Index::has_raw_data(&index),
        "HNSW-PQ aligned lane requires raw vectors for FLAT refine"
    );

    for ef in ef_values {
        index.set_ef_search(ef);
        for refine_k in &refine_k_values {
            let search_start = Instant::now();
            let results = search_queries(&index, query_batch, dataset.dim(), top_k, *refine_k);
            let search_s = search_start.elapsed().as_secs_f64();
            let qps = nq as f64 / search_s.max(1e-9);

            let recall_at_10 = average_recall_at_k(&results, &gt, 10);
            let recall_at_100 = average_recall_at_k(&results, &gt, 100);

            println!(
                "HNSW-PQ aligned ef={ef} refine_k={refine_k}: build={build_s:.3}s train={train_s:.3}s add={add_s:.3}s qps={qps:.3} R@10={recall_at_10:.4} R@100={recall_at_100:.4}"
            );

            rows.push(json!({
                "family": "HNSW-PQ",
                "implementation": "hanns",
                "top_k": top_k,
                "recall_at_10": recall_at_10,
                "recall_at_100": recall_at_100,
                "m": m,
                "ef_construction": ef_construction,
                "ef": ef,
                "pq_m": pq_m,
                "pq_k": pq_k,
                "refine_k": refine_k,
                "threads": query_threads,
                "query_threads": query_threads,
                "build_threads": build_threads,
                "build_s": build_s,
                "train_s": train_s,
                "add_s": add_s,
                "search_s": search_s,
                "qps": qps,
                "runner": "hanns_hnsw_pq_aligned_raw_refine",
            }));
        }
    }

    let payload = json!({
        "artifact_type": "hnsw_pq_aligned_hanns_rows",
        "authority_surface": env::var("AUTHORITY_SURFACE").unwrap_or_else(|_| "local_non_authority_smoke".to_string()),
        "run_set_id": run_set_id,
        "comparison_scope": "official benchmark_float.TEST_HNSW_PQ FP32 partial-row evidence; official full test currently fails on FP16 bad optional access",
        "dataset": {
            "path": dataset_path,
            "base_count": nb,
            "full_base_count": dataset.num_base(),
            "query_count_used": nq,
            "dim": dataset.dim(),
        },
        "hanns_commit": env::var("HANNS_COMMIT").unwrap_or_else(|_| "local".to_string()),
        "hanns_source_provenance": {
            "local_git_remote": env_optional_string("HANNS_GIT_REMOTE"),
            "local_git_head": env_optional_string("HANNS_GIT_HEAD"),
            "local_git_dirty": env_optional_string("HANNS_GIT_DIRTY"),
            "scoped_source_sha256": env_optional_string("HANNS_SCOPED_SOURCE_SHA256"),
            "remote_source_mode": env_optional_string("HANNS_REMOTE_SOURCE_MODE")
                .unwrap_or_else(|| "unspecified".to_string()),
        },
        "thread_contract": {
            "query_threads": query_threads,
            "build_threads": build_threads,
        },
        "raw_refine_contract": {
            "has_raw_data": true,
            "candidate_pool": "top_k * refine_k",
            "rerank_metric": "exact L2 over retained raw vectors",
        },
        "rows": rows,
    });

    let output_dir = env::var("HNSWPQ_ALIGNED_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/parity"));
    fs::create_dir_all(&output_dir).expect("create output dir");
    let stem = payload["run_set_id"].as_str().expect("run_set_id");
    let json_path = output_dir.join(format!("hanns-knowhere-{stem}.json"));
    fs::write(&json_path, serde_json::to_string_pretty(&payload).unwrap()).unwrap();
    println!("wrote {}", json_path.display());
}

fn query_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

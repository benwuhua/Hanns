#![cfg(feature = "long-tests")]
//! Focused HNSW aligned benchmark lane for Hanns-vs-Zilliz evidence.
//!
//! The lane submits the query set as a batch so the plain HNSW multi-query
//! path exercises the same 8-thread throughput shape used by the verdict
//! contract, instead of timing a single-thread query loop.

use hanns::api::{IndexConfig, IndexParams, SearchRequest};
use hanns::benchmark::average_recall_at_k;
use hanns::dataset::load_sift1m_complete;
use hanns::faiss::HnswIndex;
use hanns::{IndexType, MetricType};
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

#[test]
#[ignore = "SIFT1M remote benchmark lane; run on HannsDB-x86 for authority evidence"]
fn bench_hnsw_aligned_topk100() {
    let dataset_path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());
    let dataset = load_sift1m_complete(&dataset_path).expect("load SIFT1M dataset");
    let nq = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    let ef_values = parse_usize_list("HNSW_EF_VALUES", &[139]);
    let m = env::var("HNSW_M")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32);
    let ef_construction = env::var("HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(400);
    let top_k = 100usize;
    let query_threads = query_threads();
    let build_threads = parse_optional_usize("HNSW_BUILD_THREADS").unwrap_or(query_threads);

    let base = dataset.base.vectors();
    let queries = dataset.query.vectors();
    let query_batch = &queries[..nq * dataset.dim()];
    let gt = dataset
        .ground_truth
        .iter()
        .take(nq)
        .cloned()
        .collect::<Vec<_>>();

    let run_set_id = format!(
        "hnsw-aligned-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs()
    );
    let mut rows = Vec::new();

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: ef_values.first().copied(),
            num_threads: Some(build_threads),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create HNSW");
    let train_start = Instant::now();
    index.train(base).expect("train HNSW");
    let train_s = train_start.elapsed().as_secs_f64();
    let add_start = Instant::now();
    index
        .add_parallel(base, None, Some(true))
        .expect("add HNSW");
    let add_s = add_start.elapsed().as_secs_f64();
    let build_s = train_s + add_s;
    if let Some(expected_threads) = parse_optional_usize("HNSW_EXPECT_THREADS") {
        assert_eq!(
            query_threads, expected_threads,
            "aligned HNSW verdict lane must run with expected thread count"
        );
    }

    for ef in ef_values {
        let request = SearchRequest {
            top_k,
            nprobe: ef,
            params: Some(format!(r#"{{"ef": {ef}}}"#)),
            ..Default::default()
        };

        let search_start = Instant::now();
        let search_result = index.search(query_batch, &request).expect("search HNSW");
        let search_s = search_start.elapsed().as_secs_f64();
        let qps = nq as f64 / search_s.max(1e-9);

        let results = search_result
            .ids
            .chunks(top_k)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let recall_at_10 = average_recall_at_k(&results, &gt, 10);
        let recall_at_100 = average_recall_at_k(&results, &gt, 100);

        println!(
            "HNSW aligned ef={ef}: build={build_s:.3}s train={train_s:.3}s add={add_s:.3}s qps={qps:.3} R@10={recall_at_10:.4} R@100={recall_at_100:.4}"
        );

        rows.push(json!({
            "family": "HNSW",
            "implementation": "hanns",
            "top_k": top_k,
            "recall_at_10": recall_at_10,
            "recall_at_100": recall_at_100,
            "m": m,
            "ef_construction": ef_construction,
            "ef": ef,
            "threads": query_threads,
            "query_threads": query_threads,
            "build_threads": build_threads,
            "build_s": build_s,
            "train_s": train_s,
            "add_s": add_s,
            "qps": qps,
        }));
    }

    let payload = json!({
        "artifact_type": "hnsw_aligned_hanns_rows",
        "authority_surface": env::var("AUTHORITY_SURFACE").unwrap_or_else(|_| "local_non_authority_smoke".to_string()),
        "run_set_id": run_set_id,
        "dataset": {
            "path": dataset_path,
            "base_count": dataset.num_base(),
            "query_count_used": nq,
            "dim": dataset.dim(),
        },
        "hanns_commit": env::var("HANNS_COMMIT").unwrap_or_else(|_| "local".to_string()),
        "thread_contract": {
            "query_threads": query_threads,
            "build_threads": build_threads,
        },
        "rows": rows,
    });

    let output_dir = env::var("HNSW_ALIGNED_OUTPUT_DIR")
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

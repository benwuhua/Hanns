#![cfg(feature = "long-tests")]
//! Focused HNSW-SQ aligned benchmark lane.
//!
//! Official `benchmark_float.TEST_HNSW_SQ` emits FP32 rows before failing on
//! later FP16/BF16 refine combinations. This Hanns lane records SQ8/SQ8Refine
//! rows so verdict logic can compare only explicitly bound, same-top-k evidence.

use hanns::api::{IndexConfig, IndexParams, SearchRequest, SqMode};
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

fn parse_sq_mode() -> SqMode {
    match env::var("HNSWSQ_MODE")
        .unwrap_or_else(|_| "sq8_refine".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "sq8" => SqMode::SQ8,
        "sq8_refine" | "refine" => SqMode::SQ8Refine,
        other => panic!("unsupported HNSWSQ_MODE={other}; use sq8 or sq8_refine"),
    }
}

#[test]
#[ignore = "SIFT1M remote benchmark lane; run on HannsDB-x86 for authority evidence"]
fn bench_hnsw_sq_aligned_topk100() {
    let dataset_path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());
    let dataset = load_sift1m_complete(&dataset_path).expect("load SIFT1M dataset");
    let nq = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000)
        .min(dataset.num_query());
    let ef_values = parse_usize_list("HNSWSQ_EF_VALUES", &[128, 192, 256, 384]);
    let m = env::var("HNSWSQ_M")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    let ef_construction = env::var("HNSWSQ_EF_CONSTRUCTION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200);
    let top_k = 100usize;
    let query_threads = query_threads();
    let build_threads = parse_optional_usize("HNSWSQ_BUILD_THREADS").unwrap_or(query_threads);
    let sq_mode = parse_sq_mode();

    if let Some(expected_threads) = parse_optional_usize("HNSWSQ_EXPECT_THREADS") {
        assert_eq!(
            query_threads, expected_threads,
            "aligned HNSW-SQ verdict lane must run with expected thread count"
        );
    }

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
        "hnsw-sq-aligned-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs()
    );
    let mut rows = Vec::new();

    let config = IndexConfig {
        index_type: IndexType::HnswSq,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: ef_values.first().copied(),
            sq_mode: Some(sq_mode),
            num_threads: Some(build_threads),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create HNSW-SQ");
    let train_start = Instant::now();
    index.train(base).expect("train HNSW-SQ");
    let train_s = train_start.elapsed().as_secs_f64();
    let add_start = Instant::now();
    index
        .add_parallel(base, None, Some(true))
        .expect("add HNSW-SQ");
    let add_s = add_start.elapsed().as_secs_f64();
    let build_s = train_s + add_s;

    for ef in ef_values {
        let request = SearchRequest {
            top_k,
            nprobe: ef,
            params: Some(format!(r#"{{"ef": {ef}}}"#)),
            ..Default::default()
        };

        let search_start = Instant::now();
        let search_result = index.search(query_batch, &request).expect("search HNSW-SQ");
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
            "HNSW-SQ aligned mode={sq_mode:?} ef={ef}: build={build_s:.3}s train={train_s:.3}s add={add_s:.3}s qps={qps:.3} R@10={recall_at_10:.4} R@100={recall_at_100:.4}"
        );

        rows.push(json!({
            "family": "HNSW-SQ",
            "implementation": "hanns",
            "top_k": top_k,
            "recall_at_10": recall_at_10,
            "recall_at_100": recall_at_100,
            "m": m,
            "ef_construction": ef_construction,
            "ef": ef,
            "sq_mode": format!("{sq_mode:?}"),
            "threads": query_threads,
            "query_threads": query_threads,
            "build_threads": build_threads,
            "build_s": build_s,
            "train_s": train_s,
            "add_s": add_s,
            "qps": qps,
            "runner": "hanns_hnsw_sq_aligned",
        }));
    }

    let payload = json!({
        "artifact_type": "hnsw_sq_aligned_hanns_rows",
        "authority_surface": env::var("AUTHORITY_SURFACE").unwrap_or_else(|_| "local_non_authority_smoke".to_string()),
        "run_set_id": run_set_id,
        "comparison_scope": "official benchmark_float.TEST_HNSW_SQ FP32 partial-row evidence; official full test currently fails on FP16/BF16 refine combinations",
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

    let output_dir = env::var("HNSWSQ_ALIGNED_OUTPUT_DIR")
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

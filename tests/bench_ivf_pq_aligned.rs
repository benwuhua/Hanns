#![cfg(feature = "long-tests")]
//! Focused IVF-PQ aligned benchmark lane for the Hanns-vs-Zilliz milestone.
//!
//! This lane is narrower than BENCH-038: it emits machine-readable rows with
//! top_k=100 and Recall@100 so the validator can pair Hanns rows with normalized
//! official `benchmark_float_qps` IVF-PQ evidence.

use hanns::api::{IndexConfig, IndexParams, SearchRequest};
use hanns::benchmark::average_recall_at_k;
use hanns::dataset::load_sift1m_complete;
use hanns::faiss::IvfPqIndex;
use hanns::{IndexType, MetricType};
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

#[test]
#[ignore = "SIFT1M remote benchmark lane; run on HannsDB-x86 for authority evidence"]
fn bench_ivf_pq_aligned_topk100() {
    let dataset_path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());
    let dataset = load_sift1m_complete(&dataset_path).expect("load SIFT1M dataset");
    let nq = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    let nprobe_values = parse_usize_list("IVFPQ_NPROBE_VALUES", &[100]);
    let nlist = env::var("IVFPQ_NLIST")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| ((dataset.num_base() as f64).sqrt() as usize).max(1));
    let m = env::var("IVFPQ_M")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    let nbits = env::var("IVFPQ_NBITS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let top_k = 100usize;

    let base = dataset.base.vectors();
    let queries = dataset.query.vectors();
    let gt = dataset
        .ground_truth
        .iter()
        .take(nq)
        .cloned()
        .collect::<Vec<_>>();

    let run_set_id = format!(
        "ivfpq-aligned-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs()
    );
    let mut rows = Vec::new();

    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: nprobe_values.first().copied(),
            m: Some(m),
            nbits_per_idx: Some(nbits),
            ..Default::default()
        },
    };

    let mut index = IvfPqIndex::new(&config).expect("create IVF-PQ");
    let train_start = Instant::now();
    index.train(base).expect("train IVF-PQ");
    let train_s = train_start.elapsed().as_secs_f64();
    let add_start = Instant::now();
    index.add(base, None).expect("add IVF-PQ");
    let add_s = add_start.elapsed().as_secs_f64();
    let build_s = train_s + add_s;
    let threads = query_threads();
    if let Some(expected_threads) = parse_optional_usize("IVFPQ_EXPECT_THREADS") {
        assert_eq!(
            threads, expected_threads,
            "aligned IVF-PQ verdict lane must run with expected thread count"
        );
    }

    for nprobe in nprobe_values {
        let request = SearchRequest {
            top_k,
            nprobe,
            ..Default::default()
        };

        let search_start = Instant::now();
        #[cfg(feature = "parallel")]
        let results = (0..nq)
            .into_par_iter()
            .map(|query_index| {
                let offset = query_index * dataset.dim();
                let query = &queries[offset..offset + dataset.dim()];
                index.search(query, &request).expect("search IVF-PQ").ids
            })
            .collect::<Vec<_>>();

        #[cfg(not(feature = "parallel"))]
        let mut results = Vec::with_capacity(nq);
        #[cfg(not(feature = "parallel"))]
        for query_index in 0..nq {
            let offset = query_index * dataset.dim();
            let query = &queries[offset..offset + dataset.dim()];
            let result = index.search(query, &request).expect("search IVF-PQ");
            results.push(result.ids);
        }
        let search_s = search_start.elapsed().as_secs_f64();
        let qps = nq as f64 / search_s.max(1e-9);
        let recall_at_10 = average_recall_at_k(&results, &gt, 10);
        let recall_at_100 = average_recall_at_k(&results, &gt, 100);

        println!(
            "IVF-PQ aligned nprobe={nprobe}: build={build_s:.3}s train={train_s:.3}s add={add_s:.3}s qps={qps:.3} R@10={recall_at_10:.4} R@100={recall_at_100:.4}"
        );

        rows.push(json!({
            "family": "IVF-PQ",
            "implementation": "hanns",
            "top_k": top_k,
            "recall_at_10": recall_at_10,
            "recall_at_100": recall_at_100,
            "nlist": nlist,
            "nprobe": nprobe,
            "m": m,
            "nbits": nbits,
            "threads": threads,
            "build_s": build_s,
            "train_s": train_s,
            "add_s": add_s,
            "qps": qps,
        }));
    }

    let payload = json!({
        "artifact_type": "ivfpq_aligned_hanns_rows",
        "authority_surface": env::var("AUTHORITY_SURFACE").unwrap_or_else(|_| "local_non_authority_smoke".to_string()),
        "run_set_id": run_set_id,
        "dataset": {
            "path": dataset_path,
            "base_count": dataset.num_base(),
            "query_count_used": nq,
            "dim": dataset.dim(),
        },
        "hanns_commit": env::var("HANNS_COMMIT").unwrap_or_else(|_| "local".to_string()),
        "rows": rows,
    });

    let output_dir = env::var("IVFPQ_ALIGNED_OUTPUT_DIR")
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

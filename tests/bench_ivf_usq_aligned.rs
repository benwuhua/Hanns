#![cfg(feature = "long-tests")]
//! Focused IVF-USQ / ExRaBitQ aligned benchmark lane.
//!
//! This lane mirrors official Knowhere `benchmark_float.TEST_IVF_RABITQ`
//! evidence more closely than the `benchmark_float_qps` runner because the
//! official qps executable does not expose IVF_RABITQ on the probed commit.

use hanns::api::{MetricType, SearchRequest};
use hanns::benchmark::average_recall_at_k;
use hanns::dataset::load_sift1m_complete;
use hanns::faiss::ivf_usq::{IvfUsqConfig, IvfUsqIndex};
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
fn bench_ivf_usq_aligned_topk100() {
    let dataset_path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());
    let dataset = load_sift1m_complete(&dataset_path).expect("load SIFT1M dataset");
    let nq = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000)
        .min(dataset.num_query());
    let nlist = env::var("IVFUSQ_NLIST")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1024);
    let bits_values = parse_usize_list("IVFUSQ_BITS_VALUES", &[4]);
    let nprobe_values = parse_usize_list("IVFUSQ_NPROBE_VALUES", &[1, 2, 4, 8, 16, 32, 64, 128]);
    let top_k = 100usize;

    if let Some(expected_threads) = parse_optional_usize("IVFUSQ_EXPECT_THREADS") {
        assert_eq!(
            query_threads(),
            expected_threads,
            "aligned IVF-USQ verdict lane must run with expected thread count"
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
        "ivfusq-aligned-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs()
    );
    let mut rows = Vec::new();

    for bits_per_dim in bits_values {
        let config = IvfUsqConfig::new(dataset.dim(), nlist, bits_per_dim)
            .with_metric(MetricType::L2)
            .with_nprobe(nprobe_values.first().copied().unwrap_or(1))
            .with_rerank_k(top_k * 4);
        let mut index = IvfUsqIndex::new(config);

        let train_start = Instant::now();
        index.train(base).expect("train IVF-USQ");
        let train_s = train_start.elapsed().as_secs_f64();
        let add_start = Instant::now();
        index.add(base, None).expect("add IVF-USQ");
        let add_s = add_start.elapsed().as_secs_f64();
        let build_s = train_s + add_s;

        for nprobe in &nprobe_values {
            let request = SearchRequest {
                top_k,
                nprobe: *nprobe,
                ..Default::default()
            };
            let search_start = Instant::now();
            #[cfg(feature = "parallel")]
            let search_result = index
                .search_parallel(query_batch, &request, query_threads())
                .expect("search IVF-USQ parallel");
            #[cfg(not(feature = "parallel"))]
            let search_result = index.search(query_batch, &request).expect("search IVF-USQ");
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
                "IVF-USQ aligned bits={bits_per_dim} nprobe={nprobe}: build={build_s:.3}s train={train_s:.3}s add={add_s:.3}s qps={qps:.3} R@10={recall_at_10:.4} R@100={recall_at_100:.4}"
            );

            rows.push(json!({
                "family": "IVF-USQ/RabitQ",
                "implementation": "hanns",
                "top_k": top_k,
                "recall_at_10": recall_at_10,
                "recall_at_100": recall_at_100,
                "nlist": nlist,
                "nprobe": nprobe,
                "bits_per_dim": bits_per_dim,
                "threads": query_threads(),
                "build_s": build_s,
                "train_s": train_s,
                "add_s": add_s,
                "qps": qps,
                "runner": "hanns_ivf_usq_aligned",
            }));
        }
    }

    let payload = json!({
        "artifact_type": "ivfusq_aligned_hanns_rows",
        "authority_surface": env::var("AUTHORITY_SURFACE").unwrap_or_else(|_| "local_non_authority_smoke".to_string()),
        "run_set_id": run_set_id,
        "comparison_scope": "official benchmark_float.TEST_IVF_RABITQ equivalent evidence; not benchmark_float_qps",
        "dataset": {
            "path": dataset_path,
            "base_count": dataset.num_base(),
            "query_count_used": nq,
            "dim": dataset.dim(),
        },
        "hanns_commit": env::var("HANNS_COMMIT").unwrap_or_else(|_| "local".to_string()),
        "rows": rows,
    });

    let output_dir = env::var("IVFUSQ_ALIGNED_OUTPUT_DIR")
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

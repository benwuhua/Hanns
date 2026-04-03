use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::dataset::load_sift1m_complete;
use knowhere_rs::faiss::{HnswIndex, RhtsdgIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DEFAULT_BASE_LIMIT: usize = 20_000;
const DEFAULT_QUERY_LIMIT: usize = 200;
const DEFAULT_TOP_K: usize = 10;
const DEFAULT_EF_SEARCH: usize = 128;

#[derive(Debug, Clone)]
struct CompareArgs {
    dataset: String,
    data_dir: Option<PathBuf>,
    base_limit: usize,
    query_limit: usize,
    top_k: usize,
    ef_search: usize,
}

#[derive(Debug, Clone)]
struct DatasetBundle {
    name: String,
    source: String,
    dim: usize,
    base: Vec<f32>,
    queries: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AlgoSummary {
    name: &'static str,
    build_secs: f64,
    search_secs: f64,
    qps: f64,
    recall_at_k: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args().skip(1).collect::<Vec<_>>())?;
    let dataset = load_dataset(&args)?;
    let truth = brute_force_topk(&dataset.base, dataset.dim, &dataset.queries, args.top_k);

    println!("=== RHTSDG vs HNSW ===");
    println!("dataset={}", dataset.name);
    println!("dataset_source={}", dataset.source);
    println!(
        "base={} query={} dim={} top_k={} ef_search={}",
        dataset.base.len() / dataset.dim,
        dataset.queries.len() / dataset.dim,
        dataset.dim,
        args.top_k,
        args.ef_search
    );

    let hnsw = run_hnsw(&dataset, &truth, &args)?;
    let rhtsdg = run_rhtsdg(&dataset, &truth, &args)?;

    println!(
        "{:<8} build_s={:>8.3} search_s={:>8.3} qps={:>10.2} recall@{}={:>7.4}",
        hnsw.name, hnsw.build_secs, hnsw.search_secs, hnsw.qps, args.top_k, hnsw.recall_at_k
    );
    println!(
        "{:<8} build_s={:>8.3} search_s={:>8.3} qps={:>10.2} recall@{}={:>7.4}",
        rhtsdg.name,
        rhtsdg.build_secs,
        rhtsdg.search_secs,
        rhtsdg.qps,
        args.top_k,
        rhtsdg.recall_at_k
    );

    Ok(())
}

fn parse_args(argv: Vec<String>) -> Result<CompareArgs, Box<dyn Error>> {
    let mut args = CompareArgs {
        dataset: "synthetic".to_string(),
        data_dir: None,
        base_limit: DEFAULT_BASE_LIMIT,
        query_limit: DEFAULT_QUERY_LIMIT,
        top_k: DEFAULT_TOP_K,
        ef_search: DEFAULT_EF_SEARCH,
    };

    let mut i = 0usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--dataset" => {
                i += 1;
                args.dataset = argv.get(i).ok_or("missing value for --dataset")?.clone();
            }
            "--data-dir" => {
                i += 1;
                args.data_dir = Some(PathBuf::from(
                    argv.get(i).ok_or("missing value for --data-dir")?,
                ));
            }
            "--base-limit" => {
                i += 1;
                args.base_limit = argv
                    .get(i)
                    .ok_or("missing value for --base-limit")?
                    .parse()?;
            }
            "--query-limit" => {
                i += 1;
                args.query_limit = argv
                    .get(i)
                    .ok_or("missing value for --query-limit")?
                    .parse()?;
            }
            "--top-k" => {
                i += 1;
                args.top_k = argv.get(i).ok_or("missing value for --top-k")?.parse()?;
            }
            "--ef-search" => {
                i += 1;
                args.ef_search = argv
                    .get(i)
                    .ok_or("missing value for --ef-search")?
                    .parse()?;
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
        i += 1;
    }

    Ok(args)
}

fn load_dataset(args: &CompareArgs) -> Result<DatasetBundle, Box<dyn Error>> {
    if args.dataset == "sift1m" {
        if let Some(dir) = resolve_sift_dir(args.data_dir.as_deref()) {
            let loaded = load_sift1m_complete(&dir)?;
            return Ok(DatasetBundle {
                name: "sift1m".to_string(),
                source: dir.display().to_string(),
                dim: loaded.base.dim(),
                base: slice_vectors(loaded.base.vectors(), loaded.base.dim(), args.base_limit),
                queries: slice_vectors(
                    loaded.query.vectors(),
                    loaded.query.dim(),
                    args.query_limit,
                ),
            });
        }

        eprintln!(
            "warning: dataset=sift1m requested but no SIFT directory was found; using synthetic fallback"
        );
    }

    Ok(synthetic_bundle(
        &args.dataset,
        128,
        args.base_limit.min(10_000),
        args.query_limit.min(1_000),
    ))
}

fn resolve_sift_dir(explicit: Option<&Path>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(explicit) = explicit {
        candidates.push(explicit.to_path_buf());
    }
    if let Ok(env_path) = env::var("SIFT1M_PATH") {
        candidates.push(PathBuf::from(env_path));
    }
    candidates.push(PathBuf::from("./data/sift"));
    candidates.push(PathBuf::from("./data/sift1m"));
    candidates.push(PathBuf::from("/data/work/knowhere-rs-src/data/sift"));
    candidates.push(PathBuf::from("/data/work/knowhere-rs-src/data/sift1m"));
    candidates.push(PathBuf::from(
        "/data/work/knowhere-rs-integ/knowhere-rs/data/sift",
    ));
    candidates.push(PathBuf::from(
        "/data/work/knowhere-rs-integ/knowhere-rs/data/sift1m",
    ));

    candidates
        .into_iter()
        .find(|path| path.join("base.fvecs").exists() || path.join("sift_base.fvecs").exists())
}

fn synthetic_bundle(name: &str, dim: usize, base: usize, query: usize) -> DatasetBundle {
    let mut rng = StdRng::seed_from_u64(42);
    let mut base_vectors = vec![0.0; base * dim];
    let mut query_vectors = vec![0.0; query * dim];

    for (idx, vector) in base_vectors.chunks_mut(dim).enumerate() {
        let cluster = (idx % 64) as f32;
        for value in vector {
            *value = cluster * 0.125 + rng.gen_range(-0.03f32..0.03f32);
        }
    }

    for (idx, vector) in query_vectors.chunks_mut(dim).enumerate() {
        let cluster = (idx % 64) as f32;
        for value in vector {
            *value = cluster * 0.125 + rng.gen_range(-0.03f32..0.03f32);
        }
    }

    DatasetBundle {
        name: name.to_string(),
        source: "synthetic_fallback".to_string(),
        dim,
        base: base_vectors,
        queries: query_vectors,
    }
}

fn slice_vectors(vectors: &[f32], dim: usize, limit: usize) -> Vec<f32> {
    vectors
        .chunks(dim)
        .take(limit)
        .flat_map(|vector| vector.iter().copied())
        .collect()
}

fn run_hnsw(
    dataset: &DatasetBundle,
    truth: &[Vec<i64>],
    args: &CompareArgs,
) -> Result<AlgoSummary, Box<dyn Error>> {
    let mut params = IndexParams::hnsw(200, args.ef_search, 0.5);
    params.m = Some(16);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: dataset.dim,
        params,
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config)?;
    index.train(&dataset.base)?;
    index.add(&dataset.base, None)?;
    index.set_ef_search(args.ef_search);
    let build_secs = build_start.elapsed().as_secs_f64();

    let search_request = SearchRequest {
        top_k: args.top_k,
        nprobe: args.ef_search,
        ..Default::default()
    };
    let search_start = Instant::now();
    let result = index.search(&dataset.queries, &search_request)?;
    let search_secs = search_start.elapsed().as_secs_f64();

    Ok(AlgoSummary {
        name: "hnsw",
        build_secs,
        search_secs,
        qps: queries_per_second(dataset, search_secs),
        recall_at_k: recall_at_k(truth, &result.ids, args.top_k),
    })
}

fn run_rhtsdg(
    dataset: &DatasetBundle,
    truth: &[Vec<i64>],
    args: &CompareArgs,
) -> Result<AlgoSummary, Box<dyn Error>> {
    let mut config = IndexConfig::with_data_type(
        IndexType::Rhtsdg,
        MetricType::L2,
        dataset.dim,
        DataType::Float,
    );
    config.params.rhtsdg_knn_k = Some(
        args.ef_search
            .min(dataset.base.len() / dataset.dim - 1)
            .max(16),
    );
    config.params.rhtsdg_sample_count = Some(16);
    config.params.rhtsdg_iter_count = Some(8);
    config.params.rhtsdg_reverse_count = Some(8);

    let build_start = Instant::now();
    let mut index = RhtsdgIndex::new(&config)?;
    index.train(&dataset.base)?;
    index.add(&dataset.base, None)?;
    let build_secs = build_start.elapsed().as_secs_f64();

    let search_request = SearchRequest {
        top_k: args.top_k,
        nprobe: args.ef_search,
        ..Default::default()
    };
    let search_start = Instant::now();
    let result = index.search(&dataset.queries, &search_request)?;
    let search_secs = search_start.elapsed().as_secs_f64();

    Ok(AlgoSummary {
        name: "rhtsdg",
        build_secs,
        search_secs,
        qps: queries_per_second(dataset, search_secs),
        recall_at_k: recall_at_k(truth, &result.ids, args.top_k),
    })
}

fn queries_per_second(dataset: &DatasetBundle, search_secs: f64) -> f64 {
    let query_count = dataset.queries.len() / dataset.dim;
    if search_secs == 0.0 {
        query_count as f64
    } else {
        query_count as f64 / search_secs
    }
}

fn recall_at_k(truth: &[Vec<i64>], result_ids: &[i64], top_k: usize) -> f64 {
    let mut total = 0usize;
    let mut hits = 0usize;

    for (query_idx, truth_row) in truth.iter().enumerate() {
        let start = query_idx * top_k;
        let end = start + top_k;
        let row = &result_ids[start..end];
        total += top_k;
        for &id in truth_row.iter().take(top_k) {
            if row.contains(&id) {
                hits += 1;
            }
        }
    }

    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn brute_force_topk(base: &[f32], dim: usize, queries: &[f32], top_k: usize) -> Vec<Vec<i64>> {
    queries
        .chunks(dim)
        .map(|query| {
            let mut distances: Vec<(i64, f32)> = base
                .chunks(dim)
                .enumerate()
                .map(|(idx, vector)| (idx as i64, l2_distance(query, vector)))
                .collect();
            distances.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
            distances
                .into_iter()
                .take(top_k)
                .map(|(id, _)| id)
                .collect()
        })
        .collect()
}

fn l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
}

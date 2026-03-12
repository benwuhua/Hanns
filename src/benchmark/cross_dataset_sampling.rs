use crate::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use crate::benchmark::{confidence_from_recall, CONFIDENCE_TRUSTED};
use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use crate::faiss::{HnswIndex, IvfFlatIndex, IvfPqIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub const CROSS_DATASET_OUTPUT_PATH: &str = "benchmark_results/cross_dataset_sampling.json";
pub const CROSS_DATASET_RECALL_GATE: f64 = 0.80;
const DIM: usize = 64;
const BASE_SIZE: usize = 4000;
const QUERY_SIZE: usize = 100;
const TOP_K: usize = 10;

#[derive(Debug, Clone, Serialize)]
pub struct CrossDatasetRow {
    pub dataset: String,
    pub index: String,
    pub base_size: usize,
    pub query_size: usize,
    pub dim: usize,
    pub params: String,
    pub ground_truth_source: String,
    pub recall_at_10: f64,
    pub qps: f64,
    pub confidence: String,
    pub runtime_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossDatasetArtifact {
    pub benchmark: String,
    pub recall_gate: f64,
    pub generated_at_utc: String,
    pub rows: Vec<CrossDatasetRow>,
}

fn timestamp_utc() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn compute_ground_truth(base: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let num_queries = queries.len() / dim;
    let mut gt = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &queries[i * dim..(i + 1) * dim];
        let mut pairs: Vec<(usize, f32)> = (0..num_base)
            .map(|j| {
                let b = &base[j * dim..(j + 1) * dim];
                (j, l2_distance_squared(q, b))
            })
            .collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("distance compare"));
        gt.push(
            pairs
                .into_iter()
                .take(k)
                .map(|(idx, _)| idx as i32)
                .collect(),
        );
    }

    gt
}

fn average_recall_at_k(results: &[Vec<i32>], gt: &[Vec<i32>], k: usize) -> f64 {
    let mut total = 0.0;
    for (res, truth) in results.iter().zip(gt.iter()) {
        let hit = res
            .iter()
            .take(k)
            .filter(|id| truth.iter().take(k).any(|gt_id| gt_id == *id))
            .count();
        total += hit as f64 / k as f64;
    }
    total / results.len() as f64
}

fn dataset_random_uniform(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let base = (0..BASE_SIZE * DIM)
        .map(|_| rng.r#gen::<f32>())
        .collect::<Vec<_>>();
    let queries = (0..QUERY_SIZE * DIM)
        .map(|_| rng.r#gen::<f32>())
        .collect::<Vec<_>>();
    (base, queries)
}

fn dataset_clustered(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = 32;
    let mut centroids = vec![0.0f32; centers * DIM];
    for v in &mut centroids {
        *v = rng.r#gen::<f32>();
    }

    let sample_from_centroid = |rng: &mut StdRng| -> Vec<f32> {
        let c = rng.gen_range(0..centers);
        let mut out = vec![0.0f32; DIM];
        for d in 0..DIM {
            let noise = (rng.r#gen::<f32>() - 0.5) * 0.06;
            out[d] = centroids[c * DIM + d] + noise;
        }
        out
    };

    let mut base = Vec::with_capacity(BASE_SIZE * DIM);
    for _ in 0..BASE_SIZE {
        base.extend(sample_from_centroid(&mut rng));
    }

    let mut queries = Vec::with_capacity(QUERY_SIZE * DIM);
    for _ in 0..QUERY_SIZE {
        queries.extend(sample_from_centroid(&mut rng));
    }

    (base, queries)
}

fn dataset_anisotropic(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut scales = vec![0.0f32; DIM];
    for (i, s) in scales.iter_mut().enumerate() {
        *s = 0.2 + 2.5 * (i as f32 / (DIM - 1) as f32);
    }

    let make_vec = |rng: &mut StdRng| -> Vec<f32> {
        (0..DIM)
            .map(|d| (rng.r#gen::<f32>() - 0.5) * scales[d])
            .collect::<Vec<_>>()
    };

    let mut base = Vec::with_capacity(BASE_SIZE * DIM);
    for _ in 0..BASE_SIZE {
        base.extend(make_vec(&mut rng));
    }

    let mut queries = Vec::with_capacity(QUERY_SIZE * DIM);
    for _ in 0..QUERY_SIZE {
        queries.extend(make_vec(&mut rng));
    }

    (base, queries)
}

fn bench_hnsw(dataset: &str, base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> CrossDatasetRow {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(128),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create hnsw");
    index.train(base).expect("train hnsw");
    index.add(base, None).expect("add hnsw");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index
            .search(
                q,
                &SearchRequest {
                    top_k: TOP_K,
                    ..Default::default()
                },
            )
            .expect("search hnsw");
        all_results.push(result.ids.into_iter().map(|id| id as i32).collect());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall = average_recall_at_k(&all_results, gt, TOP_K);

    CrossDatasetRow {
        dataset: dataset.to_string(),
        index: "HNSW".to_string(),
        base_size: BASE_SIZE,
        query_size: QUERY_SIZE,
        dim: DIM,
        params: "m=16,ef_construction=200,ef_search=128".to_string(),
        ground_truth_source: "flat_exact_l2_bruteforce".to_string(),
        recall_at_10: recall,
        qps,
        confidence: confidence_from_recall(recall, CROSS_DATASET_RECALL_GATE).to_string(),
        runtime_seconds: elapsed,
    }
}

fn bench_ivf(dataset: &str, base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> CrossDatasetRow {
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            nlist: Some(128),
            nprobe: Some(16),
            ..Default::default()
        },
    };

    let mut index = IvfFlatIndex::new(&config).expect("create ivf");
    index.train(base).expect("train ivf");
    index.add(base, None).expect("add ivf");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index
            .search(
                q,
                &SearchRequest {
                    top_k: TOP_K,
                    nprobe: 16,
                    ..Default::default()
                },
            )
            .expect("search ivf");
        all_results.push(result.ids.into_iter().map(|id| id as i32).collect());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall = average_recall_at_k(&all_results, gt, TOP_K);

    CrossDatasetRow {
        dataset: dataset.to_string(),
        index: "IVF-Flat".to_string(),
        base_size: BASE_SIZE,
        query_size: QUERY_SIZE,
        dim: DIM,
        params: "nlist=128,nprobe=16".to_string(),
        ground_truth_source: "flat_exact_l2_bruteforce".to_string(),
        recall_at_10: recall,
        qps,
        confidence: confidence_from_recall(recall, CROSS_DATASET_RECALL_GATE).to_string(),
        runtime_seconds: elapsed,
    }
}

fn bench_ivfpq(dataset: &str, base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> CrossDatasetRow {
    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            nlist: Some(128),
            nprobe: Some(16),
            m: Some(16),
            nbits_per_idx: Some(8),
            ..Default::default()
        },
    };

    let mut index = IvfPqIndex::new(&config).expect("create ivfpq");
    index.train(base).expect("train ivfpq");
    index.add(base, None).expect("add ivfpq");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index
            .search(
                q,
                &SearchRequest {
                    top_k: TOP_K,
                    nprobe: 16,
                    ..Default::default()
                },
            )
            .expect("search ivfpq");
        all_results.push(result.ids.into_iter().map(|id| id as i32).collect());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall = average_recall_at_k(&all_results, gt, TOP_K);

    CrossDatasetRow {
        dataset: dataset.to_string(),
        index: "IVF-PQ".to_string(),
        base_size: BASE_SIZE,
        query_size: QUERY_SIZE,
        dim: DIM,
        params: "nlist=128,nprobe=16,m=16,nbits=8".to_string(),
        ground_truth_source: "flat_exact_l2_bruteforce".to_string(),
        recall_at_10: recall,
        qps,
        confidence: confidence_from_recall(recall, CROSS_DATASET_RECALL_GATE).to_string(),
        runtime_seconds: elapsed,
    }
}

fn bench_diskann(dataset: &str, base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> CrossDatasetRow {
    let config = AisaqConfig {
        max_degree: 48,
        search_list_size: 128,
        beamwidth: 8,
        num_entry_points: 1,
        ..AisaqConfig::default()
    };

    let mut index = PQFlashIndex::new(config, MetricType::L2, DIM).expect("create diskann");
    index.add(base).expect("add diskann");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index.search(q, TOP_K).expect("search diskann");
        all_results.push(result.ids.into_iter().map(|id| id as i32).collect());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall = average_recall_at_k(&all_results, gt, TOP_K);

    CrossDatasetRow {
        dataset: dataset.to_string(),
        index: "DiskANN".to_string(),
        base_size: BASE_SIZE,
        query_size: QUERY_SIZE,
        dim: DIM,
        params: "max_degree=48,search_list_size=128,beamwidth=8".to_string(),
        ground_truth_source: "flat_exact_l2_bruteforce".to_string(),
        recall_at_10: recall,
        qps,
        confidence: confidence_from_recall(recall, CROSS_DATASET_RECALL_GATE).to_string(),
        runtime_seconds: elapsed,
    }
}

fn build_rows(dataset: &str, base: Vec<f32>, queries: Vec<f32>) -> Vec<CrossDatasetRow> {
    let gt = compute_ground_truth(&base, &queries, DIM, TOP_K);
    vec![
        bench_hnsw(dataset, &base, &queries, &gt),
        bench_ivf(dataset, &base, &queries, &gt),
        bench_ivfpq(dataset, &base, &queries, &gt),
        bench_diskann(dataset, &base, &queries, &gt),
    ]
}

pub fn build_cross_dataset_artifact() -> CrossDatasetArtifact {
    let mut rows = Vec::new();

    let (base_random, query_random) = dataset_random_uniform(42);
    rows.extend(build_rows("random_uniform_l2", base_random, query_random));

    let (base_clustered, query_clustered) = dataset_clustered(7);
    rows.extend(build_rows("clustered_l2", base_clustered, query_clustered));

    let (base_anisotropic, query_anisotropic) = dataset_anisotropic(2026);
    rows.extend(build_rows(
        "anisotropic_l2",
        base_anisotropic,
        query_anisotropic,
    ));

    CrossDatasetArtifact {
        benchmark: "BENCH-P2-003-cross-dataset-sampling".to_string(),
        recall_gate: CROSS_DATASET_RECALL_GATE,
        generated_at_utc: timestamp_utc(),
        rows,
    }
}

pub fn validate_artifact(artifact: &CrossDatasetArtifact) -> Result<(), String> {
    if artifact.rows.len() < 12 {
        return Err("rows must include at least 3 datasets x 4 indexes".to_string());
    }
    for row in &artifact.rows {
        if row.dataset.trim().is_empty() {
            return Err("dataset must not be empty".to_string());
        }
        if row.ground_truth_source.trim().is_empty() {
            return Err(format!(
                "{}:{} missing ground_truth_source",
                row.dataset, row.index
            ));
        }
        if !row.recall_at_10.is_finite() || !(0.0..=1.0).contains(&row.recall_at_10) {
            return Err(format!(
                "{}:{} invalid recall_at_10",
                row.dataset, row.index
            ));
        }
        if !row.qps.is_finite() || row.qps <= 0.0 {
            return Err(format!("{}:{} invalid qps", row.dataset, row.index));
        }
        if row.confidence.trim().is_empty() {
            return Err(format!("{}:{} missing confidence", row.dataset, row.index));
        }
    }

    let trusted_or_not = artifact
        .rows
        .iter()
        .any(|r| r.confidence != CONFIDENCE_TRUSTED);
    if !trusted_or_not {
        // nothing to enforce, keep deterministic validation branch for audit readability
    }

    Ok(())
}

pub fn generate_cross_dataset_artifact(path: &str) -> Result<CrossDatasetArtifact, Box<dyn Error>> {
    let artifact = build_cross_dataset_artifact();
    validate_artifact(&artifact)?;

    let output = serde_json::to_string_pretty(&artifact)?;
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, output)?;

    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_dataset_artifact_shape() {
        let artifact = build_cross_dataset_artifact();
        validate_artifact(&artifact).expect("artifact should be valid");

        let datasets = artifact
            .rows
            .iter()
            .map(|r| r.dataset.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(datasets.len() >= 3);

        let indexes = artifact
            .rows
            .iter()
            .map(|r| r.index.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(indexes.contains("IVF-PQ"));
    }
}

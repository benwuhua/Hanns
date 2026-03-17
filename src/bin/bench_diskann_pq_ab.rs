use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::dataset::Dataset;
use knowhere_rs::faiss::DiskAnnIndex;
use knowhere_rs::index::Index;
use knowhere_rs::MetricType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Serialize)]
struct Row {
    pq_dims: usize,
    pq_expand_pct: usize,
    rerank_expand_pct: usize,
    saturate_after_prune: bool,
    intra_batch_candidates: usize,
    num_entry_points: usize,
    construction_l: usize,
    search_list_size: usize,
    base_size: usize,
    query_size: usize,
    dim: usize,
    top_k: usize,
    qps: f64,
    recall_at_10: f64,
    build_mode: String,
    build_seconds: f64,
    search_seconds: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    benchmark: String,
    metric: String,
    rows: Vec<Row>,
}

fn parse_usize_arg(name: &str, default: usize) -> usize {
    let key = format!("--{name}");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == key {
            if let Some(v) = args.next() {
                return v.parse::<usize>().unwrap_or(default);
            }
            break;
        }
    }
    default
}

fn parse_string_arg(name: &str, default: &str) -> String {
    let key = format!("--{name}");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == key {
            if let Some(v) = args.next() {
                return v;
            }
            break;
        }
    }
    default.to_string()
}

fn parse_bool_arg(name: &str, default: bool) -> bool {
    let raw = parse_string_arg(name, if default { "1" } else { "0" });
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => true,
        "0" | "false" | "no" | "n" | "off" => false,
        _ => default,
    }
}

fn parse_pq_dims_csv(raw: &str, default: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        if let Ok(v) = part.trim().parse::<usize>() {
            out.push(v);
        }
    }
    if out.is_empty() {
        return default.to_vec();
    }
    out
}

fn parse_pq_dims_arg(default: &[usize]) -> Vec<usize> {
    let raw = parse_string_arg("pq-dims", "0,2,4");
    parse_pq_dims_csv(&raw, default)
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

fn brute_force_topk(base: &[f32], query: &[f32], dim: usize, top_k: usize) -> Vec<i32> {
    let n = base.len() / dim;
    let mut scored = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dim;
        let dist = l2_sq(query, &base[start..start + dim]);
        scored.push((dist, i as i32));
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored.into_iter().map(|(_, id)| id).collect()
}

fn build_index_cache_path(
    cache_dir: Option<&Path>,
    pq_dims: usize,
    base_size: usize,
    dim: usize,
    max_degree: usize,
    construction_l: usize,
    saturate_after_prune: bool,
    intra_batch_candidates: usize,
    num_entry_points: usize,
) -> Option<PathBuf> {
    cache_dir.map(|dir| {
        dir.join(format!(
            "diskann_pq{}_b{}_d{}_r{}_cl{}_sat{}_intra{}_entry{}.dann",
            pq_dims,
            base_size,
            dim,
            max_degree,
            construction_l,
            if saturate_after_prune { 1 } else { 0 },
            intra_batch_candidates,
            num_entry_points
        ))
    })
}

fn main() {
    let base_size = parse_usize_arg("base-size", 100_000);
    let query_size = parse_usize_arg("query-size", 100);
    let dim = parse_usize_arg("dim", 128);
    let top_k = parse_usize_arg("top-k", 10);
    let max_degree = parse_usize_arg("max-degree", 48);
    let search_list_size = parse_usize_arg("search-list-size", 128);
    let construction_l = parse_usize_arg("construction-l", search_list_size);
    let beamwidth = parse_usize_arg("beamwidth", 8);
    let pq_expand_pct = parse_usize_arg("pq-expand-pct", 125);
    let rerank_expand_pct = parse_usize_arg("rerank-expand-pct", 100);
    let saturate_after_prune = parse_bool_arg("saturate-after-prune", true);
    let intra_batch_candidates = parse_usize_arg("intra-batch-candidates", 8);
    let num_entry_points = parse_usize_arg("num-entry-points", 1);
    let output = parse_string_arg("output", "benchmark_results/diskann_pq_ab.local.json");
    let pq_dims_list = parse_pq_dims_arg(&[0, 2, 4]);
    let reuse_index = parse_bool_arg("reuse-index", false);
    let index_cache_dir_raw = parse_string_arg("index-cache-dir", "");
    let index_cache_dir = if index_cache_dir_raw.trim().is_empty() {
        None
    } else {
        Some(PathBuf::from(index_cache_dir_raw))
    };
    if let Some(dir) = &index_cache_dir {
        fs::create_dir_all(dir).expect("create index cache dir");
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut base = Vec::with_capacity(base_size * dim);
    for _ in 0..base_size * dim {
        base.push(rng.gen_range(-1.0f32..1.0f32));
    }
    let mut queries = Vec::with_capacity(query_size * dim);
    for _ in 0..query_size * dim {
        queries.push(rng.gen_range(-1.0f32..1.0f32));
    }

    let mut gt = Vec::with_capacity(query_size);
    for q in 0..query_size {
        let qv = &queries[q * dim..(q + 1) * dim];
        gt.push(brute_force_topk(&base, qv, dim, top_k));
    }

    let mut rows = Vec::new();
    for pq_dims in pq_dims_list {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim,
            data_type: DataType::Float,
            params: IndexParams {
                max_degree: Some(max_degree),
                search_list_size: Some(search_list_size),
                construction_l: Some(construction_l),
                beamwidth: Some(beamwidth),
                disk_pq_dims: Some(pq_dims),
                disk_pq_candidate_expand_pct: Some(pq_expand_pct),
                disk_rerank_expand_pct: Some(rerank_expand_pct),
                disk_saturate_after_prune: Some(saturate_after_prune),
                disk_intra_batch_candidates: Some(intra_batch_candidates),
                disk_num_entry_points: Some(num_entry_points),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).expect("create DiskAnnIndex");

        let build_start = Instant::now();
        let cache_path =
            build_index_cache_path(
                index_cache_dir.as_deref(),
                pq_dims,
                base_size,
                dim,
                max_degree,
                construction_l,
                saturate_after_prune,
                intra_batch_candidates,
                num_entry_points,
            );
        let build_mode = if reuse_index && cache_path.as_ref().map(|p| p.exists()).unwrap_or(false)
        {
            index
                .load(cache_path.as_ref().expect("cache path"))
                .expect("load diskann index");
            "load".to_string()
        } else {
            Index::train(&mut index, &Dataset::from_vectors(base.clone(), dim))
                .expect("train diskann");
            if let Some(path) = &cache_path {
                index.save(path).expect("save diskann index cache");
            }
            "train".to_string()
        };
        let build_seconds = build_start.elapsed().as_secs_f64();

        let search_start = Instant::now();
        let mut result_rows = Vec::with_capacity(query_size);
        for q in 0..query_size {
            let qv = &queries[q * dim..(q + 1) * dim];
            let q_ds = Dataset::from_vectors(qv.to_vec(), dim);
            let res = Index::search(&index, &q_ds, top_k).expect("search");
            result_rows.push(res.ids);
        }
        let search_seconds = search_start.elapsed().as_secs_f64();
        let qps = query_size as f64 / search_seconds.max(1e-9);
        let recall = average_recall_at_k(&result_rows, &gt, top_k);

        rows.push(Row {
            pq_dims,
            pq_expand_pct,
            rerank_expand_pct,
            saturate_after_prune,
            intra_batch_candidates,
            num_entry_points,
            construction_l,
            search_list_size,
            base_size,
            query_size,
            dim,
            top_k,
            qps,
            recall_at_10: recall,
            build_mode,
            build_seconds,
            search_seconds,
        });
    }

    let report = Report {
        benchmark: "diskann_pq_ab".to_string(),
        metric: "L2".to_string(),
        rows,
    };
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    fs::write(&output, json).expect("write output");
    println!("wrote report to {}", output);
}

#[cfg(test)]
mod tests {
    use super::{build_index_cache_path, parse_bool_arg, parse_pq_dims_csv};
    use std::path::PathBuf;

    #[test]
    fn parse_pq_dims_csv_uses_default_when_invalid() {
        assert_eq!(parse_pq_dims_csv("x,y", &[0, 2, 4]), vec![0, 2, 4]);
    }

    #[test]
    fn parse_pq_dims_csv_parses_trimmed_numbers() {
        assert_eq!(parse_pq_dims_csv(" 0, 2 ,4 ", &[1]), vec![0, 2, 4]);
    }

    #[test]
    fn parse_bool_arg_parses_truthy_and_falsey() {
        assert!(parse_bool_arg("no-such-arg", true));
        assert!(!parse_bool_arg("no-such-arg", false));
    }

    #[test]
    fn build_index_cache_path_has_stable_name() {
        let p = build_index_cache_path(
            Some(PathBuf::from("tmp").as_path()),
            4,
            30000,
            128,
            48,
            128,
            true,
            8,
            1,
        )
        .expect("path");
        assert_eq!(
            p.to_string_lossy(),
            "tmp/diskann_pq4_b30000_d128_r48_cl128_sat1_intra8_entry1.dann"
        );
    }
}

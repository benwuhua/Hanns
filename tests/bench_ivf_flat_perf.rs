#![cfg(feature = "long-tests")]
//! BENCH-048: IVF-Flat Performance Regression Validation
//! Config:
//! - base vectors: 10K, 50K, 100K
//! - queries: 100
//! - nlist=100, nprobe=100
//! - 3 runs average

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::{IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const DIM: usize = 128;
const NQ: usize = 100;
const TOP_K: usize = 10;
const NLIST: usize = 100;
const NPROBE: usize = 100;
const RUNS: usize = 3;
const CPP_QPS_BASELINE: f64 = 5000.0;

#[derive(Debug, Clone)]
struct RunResult {
    qps: f64,
    recall_at_10: f64,
}

#[derive(Debug)]
struct ScaleSummary {
    nbase: usize,
    _runs: Vec<RunResult>,
    qps_mean: f64,
    qps_std: f64,
    r10_mean: f64,
    vs_cpp_ratio: f64,
}

fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let m = mean(values);
    let var = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64;
    var.sqrt()
}

fn run_once(nbase: usize, seed: u64) -> RunResult {
    let base = gen_vectors(seed, nbase, DIM);
    let query = gen_vectors(seed.wrapping_add(999_999), NQ, DIM);

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut flat = FlatIndex::new(&flat_cfg).unwrap();
    flat.add(&base, None).unwrap();

    let ivf_cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::ivf(NLIST, NPROBE),
    };

    let mut ivf = IvfFlatIndex::new(&ivf_cfg).unwrap();
    ivf.train(&base).unwrap();
    ivf.add(&base, None).unwrap();

    let mut gt_ids: Vec<Vec<i32>> = Vec::with_capacity(NQ);
    for i in 0..NQ {
        let q = &query[i * DIM..(i + 1) * DIM];
        let req = SearchRequest {
            top_k: TOP_K,
            ..Default::default()
        };
        let gt = flat.search(q, &req).unwrap();
        gt_ids.push(gt.ids.iter().map(|&x| x as i32).collect());
    }

    let search_req = SearchRequest {
        top_k: TOP_K,
        nprobe: NPROBE,
        ..Default::default()
    };

    let search_start = Instant::now();
    let mut pred_ids: Vec<Vec<i64>> = Vec::with_capacity(NQ);
    for i in 0..NQ {
        let q = &query[i * DIM..(i + 1) * DIM];
        let ret = ivf.search(q, &search_req).unwrap();
        pred_ids.push(ret.ids);
    }
    let elapsed_s = search_start.elapsed().as_secs_f64();

    let qps = NQ as f64 / elapsed_s;
    let recall_at_10 = average_recall_at_k(&pred_ids, &gt_ids, 10);

    RunResult { qps, recall_at_10 }
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn bench_048_ivf_flat_perf_regression() {
    let scales = [10_000usize, 50_000usize, 100_000usize];
    let mut summaries = Vec::new();

    println!("\n=== BENCH-048 IVF-Flat 性能回归验证 ===");
    println!(
        "Config: dim={}, nq={}, nlist={}, nprobe={}, runs={}\n",
        DIM, NQ, NLIST, NPROBE, RUNS
    );

    for &nbase in &scales {
        let mut runs = Vec::new();
        println!("--- Scale: {} ---", nbase);

        for run_id in 0..RUNS {
            let seed = 42 + (nbase as u64) * 100 + run_id as u64;
            let r = run_once(nbase, seed);
            println!(
                "run {}: QPS={:.0}, R@10={:.4}, vs_cpp={:.2}x",
                run_id + 1,
                r.qps,
                r.recall_at_10,
                r.qps / CPP_QPS_BASELINE
            );
            runs.push(r);
        }

        let qps_values: Vec<f64> = runs.iter().map(|r| r.qps).collect();
        let r10_values: Vec<f64> = runs.iter().map(|r| r.recall_at_10).collect();
        let qps_mean = mean(&qps_values);
        let qps_std = stddev(&qps_values);
        let r10_mean = mean(&r10_values);

        summaries.push(ScaleSummary {
            nbase,
            _runs: runs,
            qps_mean,
            qps_std,
            r10_mean,
            vs_cpp_ratio: qps_mean / CPP_QPS_BASELINE,
        });
    }

    println!("\n=== Summary ===");
    println!("| Scale | QPS(mean) | QPS(std) | R@10(mean) | vs C++ |");
    println!("|-------|-----------|----------|------------|--------|");
    for s in &summaries {
        println!(
            "| {} | {:.0} | {:.1} | {:.4} | {:.2}x |",
            s.nbase, s.qps_mean, s.qps_std, s.r10_mean, s.vs_cpp_ratio
        );
    }

    let target_100k = summaries
        .iter()
        .find(|s| s.nbase == 100_000)
        .map(|s| s.qps_mean >= 2000.0 && s.vs_cpp_ratio >= 0.40 && s.r10_mean >= 0.99)
        .unwrap_or(false);
    println!(
        "\nTarget check (100K: QPS >= 2000, vs C++ >= 0.40x, R@10 >= 99%): {}",
        if target_100k { "PASS" } else { "FAIL" }
    );

    for s in &summaries {
        if s.r10_mean < 0.99 {
            println!(
                "WARNING: R@10 below 99% at scale {}: {:.4}",
                s.nbase, s.r10_mean
            );
        }
    }
}

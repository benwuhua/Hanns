use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::IvfSq8Index;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_CLUSTERS: usize = 20;
const N_PER_CLUSTER: usize = 500;
const DIM: usize = 64;
const NOISE: f32 = 0.1;
const SEED: u64 = 42;
const TOP_K: usize = 10;
const NUM_QUERIES: usize = 200;

fn sample_standard_normal(rng: &mut StdRng) -> f32 {
    let u1 = rng.r#gen::<f32>().clamp(f32::MIN_POSITIVE, 1.0);
    let u2 = rng.r#gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

fn gen_clustered_vectors(
    n_clusters: usize,
    n_per_cluster: usize,
    dim: usize,
    noise: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = vec![0.0f32; n_clusters * dim];
    for c in centers.iter_mut() {
        *c = rng.r#gen::<f32>() * 10.0 - 5.0;
    }
    let mut vectors = vec![0.0f32; n_clusters * n_per_cluster * dim];
    for c in 0..n_clusters {
        let center = &centers[c * dim..(c + 1) * dim];
        for i in 0..n_per_cluster {
            let row = c * n_per_cluster + i;
            let out = &mut vectors[row * dim..(row + 1) * dim];
            for d in 0..dim {
                out[d] = center[d] + sample_standard_normal(&mut rng) * noise;
            }
        }
    }
    vectors
}

#[inline]
fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

fn brute_force_top_k(base: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<i64>> {
    let n = base.len() / dim;
    let mut all = Vec::with_capacity(queries.len() / dim);
    for q in queries.chunks(dim) {
        let mut scored: Vec<(i64, f32)> = (0..n)
            .map(|i| {
                let v = &base[i * dim..(i + 1) * dim];
                (i as i64, l2_sqr(q, v))
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        all.push(scored.into_iter().take(k).map(|(id, _)| id).collect());
    }
    all
}

fn recall_at_k(results: &[Vec<i64>], gt: &[Vec<i64>], k: usize) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (r, g) in results.iter().zip(gt.iter()) {
        for id in g.iter().take(k) {
            total += 1;
            if r.iter().take(k).any(|x| x == id) {
                hit += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = gen_clustered_vectors(N_CLUSTERS, N_PER_CLUSTER, DIM, NOISE, SEED);
    let queries = gen_clustered_vectors(N_CLUSTERS, 10, DIM, NOISE, SEED + 1);
    let queries = &queries[..NUM_QUERIES * DIM];

    let gt = brute_force_top_k(&base, queries, DIM, TOP_K);

    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params: IndexParams {
            nlist: Some(32),
            nprobe: Some(4),
            ..Default::default()
        },
    };

    let mut index = IvfSq8Index::new(&config)?;
    index.train(&base)?;
    index.add(&base, None)?;

    let nprobe_sweep = [4usize, 8, 16, 32];
    let mut full_scan_recall = 0.0f64;

    println!("=== IVF-SQ8 structured data (clustered, n=10K, dim=64) ===");
    for &nprobe in &nprobe_sweep {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            filter: None,
            params: None,
            radius: None,
        };

        let mut results = Vec::with_capacity(NUM_QUERIES);
        for q in queries.chunks(DIM) {
            let r = index.search(q, &req)?;
            results.push(r.ids);
        }
        let recall = recall_at_k(&results, &gt, TOP_K);
        println!("nprobe={:>3}  recall@10={:.3}", nprobe, recall);
        if nprobe == 32 {
            full_scan_recall = recall;
        }
    }

    std::fs::write(
        "/tmp/codex_status.txt",
        format!("DONE: full_scan_recall={:.3}", full_scan_recall),
    )?;

    Ok(())
}

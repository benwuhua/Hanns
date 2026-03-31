use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::IvfPqIndex;
use knowhere_rs::quantization::pq::{PQConfig, ProductQuantizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_CLUSTERS: usize = 20;
const N_PER_CLUSTER: usize = 500;
const _N: usize = N_CLUSTERS * N_PER_CLUSTER;
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
) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = vec![0.0f32; n_clusters * dim];
    for c in centers.iter_mut() {
        *c = rng.r#gen::<f32>() * 10.0 - 5.0;
    }
    let mut vectors = vec![0.0f32; n_clusters * n_per_cluster * dim];
    let mut labels = vec![0usize; n_clusters * n_per_cluster];
    for c in 0..n_clusters {
        let center = &centers[c * dim..(c + 1) * dim];
        for i in 0..n_per_cluster {
            let row = c * n_per_cluster + i;
            labels[row] = c;
            let out = &mut vectors[row * dim..(row + 1) * dim];
            for d in 0..dim {
                out[d] = center[d] + sample_standard_normal(&mut rng) * noise;
            }
        }
    }
    (vectors, centers, labels)
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

fn brute_force_single_with_dist(
    base: &[f32],
    query: &[f32],
    dim: usize,
    k: usize,
) -> Vec<(i64, f32)> {
    let n = base.len() / dim;
    let mut scored: Vec<(i64, f32)> = (0..n)
        .map(|i| {
            let v = &base[i * dim..(i + 1) * dim];
            (i as i64, l2_sqr(query, v))
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(k);
    scored
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
    let (base, centers, labels) =
        gen_clustered_vectors(N_CLUSTERS, N_PER_CLUSTER, DIM, NOISE, SEED);
    let (queries, _, _) = gen_clustered_vectors(N_CLUSTERS, 10, DIM, NOISE, SEED + 1);
    let queries = &queries[..NUM_QUERIES * DIM];

    let gt = brute_force_top_k(&base, queries, DIM, TOP_K);

    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params: IndexParams {
            nlist: Some(32),
            nprobe: Some(4),
            m: Some(8),
            nbits_per_idx: Some(8),
            ..Default::default()
        },
    };

    let mut index = IvfPqIndex::new(&config)?;
    index.train(&base)?;
    index.add(&base, None)?;

    let nprobe_sweep = [4usize, 8, 16, 32];

    println!("=== IVF-PQ structured data (clustered, n=10K, dim=64) ===");
    let mut full_scan_recall = 0.0f64;
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
    println!("full_scan_recall@10={:.3}", full_scan_recall);

    // Candidate-vs-ranking diagnosis on one query:
    // check whether GT top-10 appears in IVF-PQ top-100 when nprobe=32(full scan).
    let q0 = &queries[..DIM];
    let gt_top10_with_dist = brute_force_single_with_dist(&base, q0, DIM, 10);
    let gt_top10_ids: Vec<i64> = gt_top10_with_dist.iter().map(|(id, _)| *id).collect();

    let req_top100 = SearchRequest {
        top_k: 100,
        nprobe: 32,
        filter: None,
        params: None,
        radius: None,
    };
    let ivfpq_top100 = index.search(q0, &req_top100)?;
    let ivfpq_top10_ids: Vec<i64> = ivfpq_top100.ids.iter().copied().take(10).collect();
    let ivfpq_top100_ids: Vec<i64> = ivfpq_top100.ids.iter().copied().take(100).collect();

    let gt_in_top100 = gt_top10_ids
        .iter()
        .filter(|id| ivfpq_top100_ids.contains(id))
        .count();
    let gt_in_top10 = gt_top10_ids
        .iter()
        .filter(|id| ivfpq_top10_ids.contains(id))
        .count();

    println!("gt_in_top100: {}/10 found", gt_in_top100);
    println!("gt_in_top10: {}/10 found", gt_in_top10);
    println!("IVF-PQ top-5 (id, adc_dist):");
    for i in 0..5
        .min(ivfpq_top100.ids.len())
        .min(ivfpq_top100.distances.len())
    {
        println!(
            "  ({}, {:.4})",
            ivfpq_top100.ids[i], ivfpq_top100.distances[i]
        );
    }
    println!("GT top-5 (id, exact_dist):");
    for (id, dist) in gt_top10_with_dist.iter().take(5) {
        println!("  ({}, {:.4})", id, dist);
    }

    // ADC sanity check: neighbor vs random non-neighbor in the same cluster
    let neighbor_idx = 0usize;
    let query = &base[neighbor_idx * DIM..(neighbor_idx + 1) * DIM];
    let cluster = labels[neighbor_idx];
    let random_idx = cluster * N_PER_CLUSTER + (N_PER_CLUSTER - 1);
    let random_vec = &base[random_idx * DIM..(random_idx + 1) * DIM];

    let center = &centers[cluster * DIM..(cluster + 1) * DIM];
    let query_residual: Vec<f32> = query
        .iter()
        .zip(center.iter())
        .map(|(q, c)| q - c)
        .collect();

    let mut residuals = Vec::with_capacity(base.len());
    for i in 0..(base.len() / DIM) {
        let c = labels[i];
        let centroid = &centers[c * DIM..(c + 1) * DIM];
        let vec_i = &base[i * DIM..(i + 1) * DIM];
        for d in 0..DIM {
            residuals.push(vec_i[d] - centroid[d]);
        }
    }

    let mut pq = ProductQuantizer::new(PQConfig::new(DIM, 8, 8));
    pq.train(base.len() / DIM, &residuals)?;

    let neighbor_residual = &residuals[neighbor_idx * DIM..(neighbor_idx + 1) * DIM];
    let random_residual = &residuals[random_idx * DIM..(random_idx + 1) * DIM];
    let neighbor_code = pq.encode(neighbor_residual)?;
    let random_code = pq.encode(random_residual)?;

    let exact_neighbor = l2_sqr(query, query);
    let exact_random = l2_sqr(query, random_vec);
    let adc_neighbor = pq.compute_distance(&query_residual, &neighbor_code);
    let adc_random = pq.compute_distance(&query_residual, &random_code);
    let adc_ok = adc_neighbor <= adc_random;
    let adc_state = if adc_ok { "正确" } else { "颠倒" };

    println!(
        "ADC sanity: exact_neighbor={:.4} exact_random={:.4} adc_neighbor={:.4} adc_random={:.4} => ADC {}",
        exact_neighbor, exact_random, adc_neighbor, adc_random, adc_state
    );

    let cause = if gt_in_top100 < 10 {
        "候选集问题"
    } else {
        "ADC排名问题"
    };
    std::fs::write(
        "/tmp/codex_status.txt",
        format!(
            "DONE: gt_in_top100={}/10 gt_in_top10={}/10 — {}",
            gt_in_top100, gt_in_top10, cause
        ),
    )?;

    Ok(())
}

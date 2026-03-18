use knowhere_rs::api::MetricType;
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N: usize = 5_000;
const DIM: usize = 64;
const NUM_QUERIES: usize = 100;
const TOP_K: usize = 10;

#[inline]
fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<f32> = (0..N * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let queries: Vec<f32> = (0..NUM_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();

    let config = AisaqConfig {
        disk_pq_dims: 0,
        search_list_size: 64,
        ..AisaqConfig::default()
    };
    let mut index = PQFlashIndex::new(config, MetricType::L2, DIM)?;
    index.train(&vectors)?;
    index.add(&vectors)?;

    let mut recalls = Vec::with_capacity(NUM_QUERIES);
    let mut total_results = 0usize;

    for qi in 0..NUM_QUERIES {
        let q = &queries[qi * DIM..(qi + 1) * DIM];

        let mut scored: Vec<(usize, f32)> = (0..N)
            .map(|i| {
                let v = &vectors[i * DIM..(i + 1) * DIM];
                (i, l2_sqr(q, v))
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt_top10: Vec<(i64, f32)> = scored
            .iter()
            .take(TOP_K)
            .map(|(i, d)| (*i as i64, *d))
            .collect();

        let radius = gt_top10[TOP_K - 1].1 * 1.1;
        let rs = index.range_search_raw(q, radius)?;
        total_results += rs.len();

        let mut hit = 0usize;
        for (gt_id, _) in &gt_top10 {
            if rs.iter().any(|(id, _)| id == gt_id) {
                hit += 1;
            }
        }
        recalls.push(hit as f64 / TOP_K as f64);
    }

    let avg_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let min_recall = recalls
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let total_results_avg = total_results as f64 / NUM_QUERIES as f64;

    println!(
        "AISAQ range_search demo: avg_recall={:.3}, min_recall={:.3}, total_results_avg={:.2}",
        avg_recall, min_recall, total_results_avg
    );

    assert!(
        avg_recall >= 0.90,
        "avg_recall too low: {:.3} (expected >= 0.90)",
        avg_recall
    );

    Ok(())
}

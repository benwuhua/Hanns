/// Quick benchmark: measure QPS and recall of USQ-migrated IvfHvqIndex
use knowhere_rs::faiss::ivf_hvq::{IvfHvqConfig, IvfHvqIndex};
use knowhere_rs::api::{MetricType, SearchRequest};

fn generate_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut data = Vec::with_capacity(n * dim);
    let mut s = seed;
    for _ in 0..n * dim {
        // Simple LCG
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = ((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0;
        data.push(v);
    }
    data
}

fn brute_force_knn(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = data.len() / dim;
    let mut dists: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let d: f32 = query
                .iter()
                .zip(data[i * dim..(i + 1) * dim].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

#[test]
fn bench_ivf_hvq_usq_10k() {
    let dim = 128;
    let n = 10_000;
    let nlist = 64;
    let nbits = 4;
    let k = 10;
    let nq = 100;

    let data = generate_data(n, dim, 42);
    let queries = generate_data(nq, dim, 99);

    // Build index
    let config = IvfHvqConfig::new(dim, nlist, nbits)
        .with_nprobe(16)
        .with_metric(MetricType::L2);

    let mut index = IvfHvqIndex::new(config);
    let start = std::time::Instant::now();
    index.train(&data).unwrap();
    index.add(&data, None).unwrap();
    let build_time = start.elapsed();
    eprintln!("Build time: {:.2}s", build_time.as_secs_f64());

    // Search + measure recall (query by query)
    let mut total_recall = 0.0f64;
    let search_start = std::time::Instant::now();
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let req = SearchRequest {
            top_k: k,
            nprobe: 16,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();

        let gt = brute_force_knn(&data, dim, query, k);
        let gt_ids: Vec<i64> = gt.iter().map(|(id, _)| *id as i64).collect();
        let hits = result
            .ids
            .iter()
            .filter(|&id| gt_ids.contains(id))
            .count();
        total_recall += hits as f64 / k as f64;
    }
    let search_time = search_start.elapsed();
    let recall = total_recall / nq as f64;
    let qps = nq as f64 / search_time.as_secs_f64();

    eprintln!("Recall@{k}: {:.4}", recall);
    eprintln!(
        "QPS: {:.0}",
        qps
    );
    eprintln!(
        "Latency: {:.2}ms/query",
        search_time.as_secs_f64() / nq as f64 * 1000.0
    );

    // Basic sanity checks
    assert!(recall > 0.1, "recall too low: {recall}");
    // Note: debug build QPS is much lower than release; this threshold is intentionally low
    assert!(qps > 10.0, "QPS too low: {qps}");
}

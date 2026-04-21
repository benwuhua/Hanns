use hanns::api::{IndexConfig, IndexType, MetricType, DataType, IndexParams, SearchRequest};
use hanns::faiss::hnsw::HnswIndex;

fn main() {
    let dim = 128;
    let n = 50_000;
    let m = 16;
    let ef_construction = 128;
    let ef_search = 128;

    println!("=== Fast Deserialize Benchmark ===");
    println!("Dataset: {} vectors, dim={}, m={}, ef_construction={}", n, dim, m, ef_construction);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::Cosine,
        data_type: DataType::Float,
        dim,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(0.0),
            ..Default::default()
        },
    };
    let mut index = HnswIndex::new(&config).unwrap();

    let mut vectors = vec![0.0f32; n * dim];
    for (i, v) in vectors.iter_mut().enumerate() {
        *v = ((i as u32).wrapping_mul(1103515245).wrapping_add(12345)) as f32 / 2147483647.0;
    }
    for vec in vectors.chunks_exact_mut(dim) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in vec.iter_mut() { *x /= norm; } }
    }

    println!("\nBuilding index...");
    let t0 = std::time::Instant::now();
    index.train(&vectors).unwrap();
    let ids: Vec<i64> = (0..n as i64).collect();
    index.add(&vectors, Some(&ids)).unwrap();
    println!("Build: {:.2}s", t0.elapsed().as_secs_f64());

    let bytes = index.serialize_to_bytes().unwrap();
    println!("Serialized: {} bytes ({:.1} MB)", bytes.len(), bytes.len() as f64 / 1e6);

    let query = &vectors[0..dim];
    let req = SearchRequest { top_k: 10, nprobe: ef_search, ..Default::default() };
    let orig_result = index.search(query, &req).unwrap();
    let orig_ids = orig_result.ids.clone();

    println!("\n--- Old deserialize_from_bytes ---");
    let mut old_times = Vec::new();
    for i in 0..3 {
        let t0 = std::time::Instant::now();
        let idx = HnswIndex::deserialize_from_bytes(&bytes).unwrap();
        let dur = t0.elapsed();
        old_times.push(dur);
        let result = idx.search(query, &req).unwrap();
        let match_count = orig_ids.iter().zip(result.ids.iter()).filter(|(a, b)| a == b).count();
        println!("  run {}: {:.3}s, top-10 match={}/10", i, dur.as_secs_f64(), match_count);
    }

    println!("\n--- New fast_deserialize_from_bytes ---");
    let mut new_times = Vec::new();
    for i in 0..3 {
        let t0 = std::time::Instant::now();
        let idx = HnswIndex::fast_deserialize_from_bytes(&bytes).unwrap();
        let dur = t0.elapsed();
        new_times.push(dur);
        let result = idx.search(query, &req).unwrap();
        let match_count = orig_ids.iter().zip(result.ids.iter()).filter(|(a, b)| a == b).count();
        println!("  run {}: {:.3}s, top-10 match={}/10", i, dur.as_secs_f64(), match_count);
    }

    let old_avg: f64 = old_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / old_times.len() as f64;
    let new_avg: f64 = new_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / new_times.len() as f64;
    println!("\n=== Result ===");
    println!("Old: {:.3}s avg", old_avg);
    println!("New: {:.3}s avg", new_avg);
    if new_avg > 0.0 {
        println!("Speedup: {:.1}x", old_avg / new_avg);
    }
}

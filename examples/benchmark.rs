//! Benchmark for KnowHere RS

use std::time::Instant;

use knowhere_rs::faiss::{MemIndex, HnswIndex, IvfPqIndex, DiskAnnIndex};
use knowhere_rs::api::{IndexConfig, IndexType, MetricType, IndexParams, SearchRequest};

const NUM_VECTORS: usize = 1_000;
const DIM: usize = 128;
const TOP_K: usize = 10;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut vectors = vec![0.0f32; n * dim];
    for i in 0..vectors.len() {
        vectors[i] = (i as f32 * 0.01).sin().abs();
    }
    vectors
}

fn benchmark_flat_index() {
    println!("\n=== Flat Index Benchmark ===");
    
    let config = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut index = MemIndex::new(&config).unwrap();
    
    let vectors = generate_vectors(NUM_VECTORS, DIM);
    
    // Benchmark add
    let start = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_time = start.elapsed();
    println!("Add {} vectors: {:?}", NUM_VECTORS, add_time);
    println!("  Throughput: {:.2} vectors/sec", 
        NUM_VECTORS as f64 / add_time.as_secs_f64());
    
    // Benchmark search
    let query = generate_vectors(100, DIM);
    
    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 1,
        filter: None,
        params: None,
    };
    
    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!("  QPS: {:.2} queries/sec", 100.0 / search_time.as_secs_f64());
}

fn benchmark_hnsw_index() {
    println!("\n=== HNSW Index Benchmark ===");
    
    let params = IndexParams::hnsw(200, 50, 0.5);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: DIM,
        params,
    };
    
    let mut index = HnswIndex::new(&config).unwrap();
    
    let vectors = generate_vectors(NUM_VECTORS, DIM);
    
    // Train
    let start = Instant::now();
    index.train(&vectors).unwrap();
    let train_time = start.elapsed();
    println!("Train: {:?}", train_time);
    
    // Benchmark add
    let start = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_time = start.elapsed();
    println!("Add {} vectors: {:?}", NUM_VECTORS, add_time);
    println!("  Throughput: {:.2} vectors/sec", 
        NUM_VECTORS as f64 / add_time.as_secs_f64());
    
    // Benchmark search
    let query = generate_vectors(100, DIM);
    
    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 50,
        filter: None,
        params: None,
    };
    
    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!("  QPS: {:.2} queries/sec", 100.0 / search_time.as_secs_f64());
}

fn benchmark_ivfpq_index() {
    println!("\n=== IVF-PQ Index Benchmark ===");
    
    let params = IndexParams::ivf(100, 10);
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim: DIM,
        params,
    };
    
    let mut index = IvfPqIndex::new(&config).unwrap();
    
    let vectors = generate_vectors(NUM_VECTORS, DIM);
    
    // Train
    let start = Instant::now();
    index.train(&vectors).unwrap();
    let train_time = start.elapsed();
    println!("Train (k-means): {:?}", train_time);
    
    // Benchmark add
    let start = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_time = start.elapsed();
    println!("Add {} vectors: {:?}", NUM_VECTORS, add_time);
    println!("  Throughput: {:.2} vectors/sec", 
        NUM_VECTORS as f64 / add_time.as_secs_f64());
    
    // Benchmark search
    let query = generate_vectors(100, DIM);
    
    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 10,
        filter: None,
        params: None,
    };
    
    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!("  QPS: {:.2} queries/sec", 100.0 / search_time.as_secs_f64());
}

fn benchmark_diskann_index() {
    println!("\n=== DiskANN Index Benchmark ===");
    
    let config = IndexConfig {
        index_type: IndexType::DiskAnn,
        metric_type: MetricType::L2,
        dim: DIM,
        params: IndexParams::default(),
    };
    
    let mut index = DiskAnnIndex::new(&config).unwrap();
    
    let vectors = generate_vectors(NUM_VECTORS, DIM);
    
    // Train (build graph)
    let start = Instant::now();
    index.train(&vectors).unwrap();
    let train_time = start.elapsed();
    println!("Train (graph build): {:?}", train_time);
    
    // Benchmark search
    let query = generate_vectors(100, DIM);
    
    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 50,
        filter: None,
        params: None,
    };
    
    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!("  QPS: {:.2} queries/sec", 100.0 / search_time.as_secs_f64());
}

fn main() {
    println!("KnowHere RS Benchmark");
    println!("=====================");
    println!("Vectors: {}", NUM_VECTORS);
    println!("Dimension: {}", DIM);
    println!("Top-K: {}", TOP_K);
    
    benchmark_flat_index();
    benchmark_hnsw_index();
    benchmark_ivfpq_index();
    benchmark_diskann_index();
    
    println!("\n✅ Benchmark complete!");
}

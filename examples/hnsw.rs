//! Example: HNSW vector search with KnowHere RS

use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::api::{IndexConfig, IndexType, MetricType, IndexParams, SearchRequest};

fn main() {
    // Create HNSW index
    let params = IndexParams::hnsw(200, 50, 0.5);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: 4,
        params,
    };
    
    let mut index = HnswIndex::new(&config).unwrap();
    
    // Train
    let train_vectors = vec![
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ];
    index.train(&train_vectors).unwrap();
    println!("✓ Trained HNSW index");
    
    // Add vectors
    let vectors = vec![
        0.0, 0.0, 0.0, 1.0,  // id=0
        0.0, 0.0, 1.0, 0.0,  // id=1
        0.0, 1.0, 0.0, 0.0,  // id=2
        1.0, 0.0, 0.0, 0.0,  // id=3
    ];
    index.add(&vectors, None).unwrap();
    println!("✓ Added 4 vectors to HNSW");
    
    // Search
    let query = vec![0.1, 0.1, 0.1, 0.1];
    let req = SearchRequest {
        top_k: 2,
        nprobe: 10,
        filter: None,
        params: None,
        radius: None,
    };
    
    let result = index.search(&query, &req).unwrap();
    
    println!("\n🔍 HNSW Search results (top 2):");
    for i in 0..result.ids.len() {
        println!("  id={}, distance={:.4}", result.ids[i], result.distances[i]);
    }
    
    // Save and reload
    index.save(std::path::Path::new("/tmp/hnsw.idx")).unwrap();
    println!("\n💾 Saved HNSW to /tmp/hnsw.idx");
    
    let mut index2 = HnswIndex::new(&config).unwrap();
    index2.load(std::path::Path::new("/tmp/hnsw.idx")).unwrap();
    println!("✓ Loaded HNSW from /tmp/hnsw.idx");
    
    let result2 = index2.search(&query, &req).unwrap();
    println!("\n🔍 Search after reload:");
    for i in 0..result2.ids.len() {
        println!("  id={}, distance={:.4}", result2.ids[i], result2.distances[i]);
    }
    
    println!("\n✅ HNSW E2E test passed!");
}

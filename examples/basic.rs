//! Example: Basic vector search with KnowHere RS

use knowhere_rs::faiss::MemIndex;
use knowhere_rs::api::{IndexConfig, IndexType, MetricType, SearchRequest};

fn main() {
    // Create index
    let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
    let mut index = MemIndex::new(&config).unwrap();
    
    // Add vectors (4 vectors, dim 4)
    let vectors = vec![
        0.0, 0.0, 0.0, 1.0,  // id=0
        0.0, 0.0, 1.0, 0.0,  // id=1
        0.0, 1.0, 0.0, 0.0,  // id=2
        1.0, 0.0, 0.0, 0.0,  // id=3
    ];
    index.add(&vectors, None).unwrap();
    
    println!("✓ Added 4 vectors to index");
    
    // Search
    let query = vec![0.1, 0.1, 0.1, 0.1];
    let req = SearchRequest {
        top_k: 2,
        nprobe: 1,
        filter: None,
        params: None,
        radius: None,
    };
    
    let result = index.search(&query, &req).unwrap();
    
    println!("\n🔍 Search results (top 2):");
    for i in 0..result.ids.len() {
        println!("  id={}, distance={:.4}", result.ids[i], result.distances[i]);
    }
    
    // Save and reload
    index.save(std::path::Path::new("/tmp/test.idx")).unwrap();
    println!("\n💾 Saved to /tmp/test.idx");
    
    let mut index2 = MemIndex::new(&config).unwrap();
    index2.load(std::path::Path::new("/tmp/test.idx")).unwrap();
    println!("✓ Loaded from /tmp/test.idx");
    
    let result2 = index2.search(&query, &req).unwrap();
    println!("\n🔍 Search after reload:");
    for i in 0..result2.ids.len() {
        println!("  id={}, distance={:.4}", result2.ids[i], result2.distances[i]);
    }
    
    println!("\n✅ E2E test passed!");
}

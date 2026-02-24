//! CLI for KnowHere RS

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use clap::{Parser, Subcommand};
use once_cell::sync::Lazy;
use knowhere_rs::faiss::MemIndex;
use knowhere_rs::api::{IndexConfig, IndexType, MetricType, SearchRequest};

#[derive(Parser)]
#[command(name = "knowhere")]
#[command(about = "KnowHere RS - Vector Search CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new index
    Create {
        /// Index name
        name: String,
        /// Vector dimension
        dim: usize,
        /// Index type (flat, hnsw)
        #[arg(default_value = "flat")]
        index_type: String,
        /// Metric type (l2, ip, cosine)
        #[arg(default_value = "l2")]
        metric: String,
    },
    /// Add vectors to index
    Add {
        /// Index name
        name: String,
        /// Path to vectors file (binary f32)
        vectors_file: PathBuf,
    },
    /// Search vectors
    Search {
        /// Index name
        name: String,
        /// Query vector (comma-separated f32)
        query: String,
        /// Number of results
        #[arg(short, default_value = "10")]
        top_k: usize,
    },
    /// Save index to disk
    Save {
        /// Index name
        name: String,
        /// Output path
        path: PathBuf,
    },
    /// Load index from disk
    Load {
        /// Index name
        name: String,
        /// Input path
        path: PathBuf,
        /// Vector dimension
        dim: usize,
    },
}

// Global in-memory state
static INDICES: Lazy<Mutex<HashMap<String, MemIndex>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Create { name, dim, index_type, metric } => {
            let idx_type = IndexType::from_str(&index_type).unwrap_or(IndexType::Flat);
            let met = MetricType::from_str(&metric).unwrap_or(MetricType::L2);
            
            let config = IndexConfig::new(idx_type, met, dim);
            let index = MemIndex::new(&config).expect("Failed to create index");
            
            let mut indices = INDICES.lock().unwrap();
            indices.insert(name.clone(), index);
            println!("Created index '{}' (dim={}, type={}, metric={})", name, dim, index_type, metric);
        }
        
        Commands::Add { name, vectors_file } => {
            // Read binary vectors
            let data = std::fs::read(&vectors_file).expect("Failed to read vectors file");
            let vectors: Vec<f32> = data
                .chunks(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            let mut indices = INDICES.lock().unwrap();
            if let Some(index) = indices.get_mut(&name) {
                let n = index.add(&vectors, None).expect("Failed to add vectors");
                println!("Added {} vectors to index '{}'", n, name);
            } else {
                eprintln!("Index '{}' not found", name);
            }
        }
        
        Commands::Search { name, query, top_k } => {
            let query_vec: Vec<f32> = query
                .split(',')
                .map(|s| s.trim().parse::<f32>().expect("Invalid query"))
                .collect();
            
            let indices = INDICES.lock().unwrap();
            if let Some(index) = indices.get(&name) {
                let req = SearchRequest {
                    top_k,
                    nprobe: 1,
                    filter: None,
                    params: None,
            radius: None,
                };
                
                let result = index.search(&query_vec, &req).expect("Search failed");
                
                println!("Search results ({}ms):", result.elapsed_ms);
                for i in 0..result.ids.len() {
                    println!("  id={}, distance={:.4}", result.ids[i], result.distances[i]);
                }
            } else {
                eprintln!("Index '{}' not found", name);
            }
        }
        
        Commands::Save { name, path } => {
            let indices = INDICES.lock().unwrap();
            if let Some(index) = indices.get(&name) {
                index.save(&path).expect("Failed to save index");
                println!("Saved index '{}' to {:?}", name, path);
            } else {
                eprintln!("Index '{}' not found", name);
            }
        }
        
        Commands::Load { name, path, dim } => {
            let mut index = MemIndex::new(&IndexConfig::new(IndexType::Flat, MetricType::L2, dim))
                .expect("Failed to create index");
            index.load(&path).expect("Failed to load index");
            
            let mut indices = INDICES.lock().unwrap();
            indices.insert(name.clone(), index);
            println!("Loaded index '{}' from {:?} (dim={})", name, path, dim);
        }
    }
}

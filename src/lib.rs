//! KnowHere RS - Vector Search Engine
//! 
//! A Rust-native vector search engine, designed as a replacement for KnowHere (C++).

pub mod api;
pub mod bitset;
pub mod dataset;
pub mod metrics;
pub mod simd;
pub mod index;
pub mod faiss;
pub mod storage;
pub mod codec;
pub mod executor;
pub mod quantization;
pub mod disk_io;
pub mod serialize;
pub mod benchmark;
pub mod memory;
pub mod integration;
pub mod bloom;
pub mod layout;
pub mod stats;
pub mod skiplist;
pub mod lru_cache;
pub mod arena;
pub mod atomic_utils;
pub mod version;
pub mod error;
pub mod types;
pub mod utils;
pub mod prealloc;
pub mod ring;
pub mod once_cell;
pub mod ffi;

pub use api::{SearchRequest, SearchResult, KnowhereError, Result, IndexConfig, IndexType, MetricType};
pub use executor::Executor;
pub use bitset::BitsetView;
pub use dataset::{Dataset, DataType};
pub use metrics::{Distance, get_distance_calculator, L2Distance, InnerProductDistance, CosineDistance, HammingDistance};
pub use index::{Index, IndexError, SearchResult as IndexSearchResult};

// Export all index types
pub use faiss::{FaissIndex, MemIndex, HnswIndex, IvfPqIndex, DiskAnnIndex};

use tracing::info;

pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();
    
    info!("KnowHere RS initialized");
}

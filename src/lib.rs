//! KnowHere RS - Vector Search Engine
//!
//! A Rust-native vector search engine, designed as a replacement for KnowHere (C++).

pub mod api;
pub mod arena;
pub mod atomic_utils;
pub mod benchmark;
pub mod bitset;
pub mod bloom;
pub mod clustering;
pub mod codec;
pub mod comp; // components (bloomfilter, etc.)
pub mod dataset;
pub mod disk_io;
pub mod error;
pub mod executor;
pub mod faiss;
pub mod federation;
pub mod ffi;
pub mod half; // fp16/bf16 support
pub mod index;
pub mod integration;
pub mod interrupt;
pub mod layout;
pub mod lru_cache;
pub mod memory;
pub mod metrics;
pub mod once_cell;
pub mod prealloc;
pub mod quantization;
pub mod ring;
pub mod search;
pub mod serialize;
pub mod simd;
pub mod skiplist;
pub mod stats;
pub mod storage;
pub mod types;
pub mod utils;
pub mod version;

#[cfg(feature = "jni-bindings")]
pub mod jni;

#[cfg(feature = "python")]
pub mod python;

pub use api::{
    IndexConfig, IndexType, KnowhereError, MetricType, Result, SearchRequest, SearchResult,
    SqMode,
};
pub use bitset::BitsetView;
pub use comp::BloomFilter;
pub use dataset::{DataType, Dataset};
pub use executor::Executor;
pub use half::{bf16_l2, bf16_to_f32, f32_to_bf16, f32_to_fp16, fp16_l2, fp16_to_f32, Bf16, Fp16};
pub use index::{Index, IndexError, SearchResult as IndexSearchResult};
pub use interrupt::Interrupt;
pub use metrics::{
    get_distance_calculator, CosineDistance, Distance, HammingDistance, InnerProductDistance,
    L2Distance,
};
pub use quantization::{pick_refine_index, RefineIndex, RefineType};

// Export all index types
pub use faiss::{
    AsyncReadEngine, BeamSearchIO, BeamSearchStats, DiskAnnIndex, FaissIndex, FileGroup,
    FlashLayout, HnswIndex, IvfPqIndex, IvfSq8Index, MemIndex, PQFlashIndex, PageCache,
    PageCacheStats, ScaNNConfig, ScaNNIndex,
};
pub use faiss::{
    IvfExRaBitqConfig, IvfExRaBitqIndex, IvfRaBitqConfig, IvfRaBitqIndex, IvfSqCcIndex,
    IvfTurboQuantConfig, IvfTurboQuantIndex,
};

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

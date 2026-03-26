//! Faiss binding layer

pub mod annoy;
pub mod bin_flat;
pub mod bin_ivf_flat;
pub mod binary;
pub mod binary_hnsw;
pub mod diskann;
pub mod diskann_aisaq;
pub mod diskann_sq;
pub mod hnsw;
pub mod hnsw_pq;
pub mod hnsw_prq;
pub mod hnsw_quantized;
pub mod index;
pub mod ivf;
pub mod ivf_flat;
pub mod ivf_flat_cc;
pub mod ivf_opq;
pub mod ivf_rabitq;
pub mod ivf_sq8;
pub mod ivf_turboquant;
pub mod ivf_sq_cc;
pub mod ivfpq;
pub mod lazy_index;
pub mod mem_index;
pub mod pq;
pub mod pq_simd;
pub mod raw;
pub mod scann;
pub mod sparse;
pub mod sparse_inverted;
pub mod sparse_inverted_cc;
pub mod sparse_wand;
pub mod sparse_wand_cc;

pub use bin_flat::BinFlatIndex;
pub use bin_ivf_flat::BinIvfFlatIndex;
pub use binary::BinaryIndex;
pub use binary_hnsw::BinaryHnswIndex;
pub use diskann::DiskAnnIndex;
pub use diskann_aisaq::{
    AsyncReadEngine, BeamSearchIO, BeamSearchStats, FileGroup, FlashLayout, PQFlashIndex,
    PageCache, PageCacheStats,
};
pub use diskann_sq::{DiskAnnSqConfig, DiskAnnSqIndex};
pub use hnsw::HnswIndex;
pub use hnsw_pq::{HnswPqConfig, HnswPqIndex};
pub use hnsw_prq::{HnswPrqConfig, HnswPrqIndex};
pub use hnsw_quantized::{HnswQuantizeConfig, HnswSqIndex};
pub use index::FaissIndex;
pub use ivf::IvfIndex;
pub use ivf_flat::IvfFlatIndex;
pub use ivf_flat_cc::IvfFlatCcIndex;
pub use ivf_opq::{IvfOpqConfig, IvfOpqIndex, IvfOpqIndexWrapper};
pub use ivf_rabitq::{IvfRaBitqConfig, IvfRaBitqIndex};
pub use ivf_sq8::IvfSq8Index;
pub use ivf_turboquant::{IvfTurboQuantConfig, IvfTurboQuantIndex};
pub use ivf_sq_cc::IvfSqCcIndex;
pub use ivfpq::IvfPqIndex;
pub use mem_index::MemIndex;
pub use pq::PqEncoder;
pub use scann::{ScaNNConfig, ScaNNIndex};
pub use sparse::{SparseIndex, SparseVector};
pub use sparse_inverted::{
    ApproxSearchParams, InvertedIndexAlgo, SparseInvertedIndex, SparseInvertedSearcher,
    SparseMetricType,
};
pub use sparse_inverted_cc::{SparseInvertedIndexCC, SparseMetricType as SparseMetricTypeCC};
pub use sparse_wand::SparseWandIndex;
pub use sparse_wand_cc::SparseWandIndexCC;

#[cfg(feature = "ffi")]
pub mod rabitq_ffi;

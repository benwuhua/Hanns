//! Faiss binding layer

pub mod annoy;
pub mod bin_flat;
pub mod bin_ivf_flat;
pub mod binary;
pub mod binary_hnsw;
pub mod diskann_aisaq;
pub mod diskann_sq;
pub mod hnsw;
pub mod hnsw_hvq;
pub mod hnsw_pq;
pub mod hnsw_prq;
pub mod hnsw_quantized;
pub mod index;
pub mod ivf;
pub mod ivf_flat;
pub mod ivf_flat_cc;
pub mod ivf_opq;
pub mod ivf_usq;

pub mod ivf_sq8;
pub mod ivf_sq_cc;
pub mod ivf_turboquant;
pub mod ivfpq;
pub mod lazy_index;
pub mod mem_index;
pub mod pq_simd;
pub mod raw;
pub mod rhtsdg;
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
pub use diskann_aisaq::{
    AisaqConfig, AsyncReadEngine, BeamSearchIO, BeamSearchStats, FileGroup, FlashLayout,
    PQFlashIndex, PageCache, PageCacheStats,
};
pub use diskann_sq::{DiskAnnSqConfig, DiskAnnSqIndex};
pub use hnsw::HnswIndex;
pub use hnsw_hvq::{HnswHvqConfig, HnswHvqIndex};
pub use hnsw_pq::{HnswPqConfig, HnswPqIndex};
pub use hnsw_prq::{HnswPrqConfig, HnswPrqIndex};
pub use hnsw_quantized::{HnswQuantizeConfig, HnswSqIndex};
pub use index::FaissIndex;
#[allow(deprecated)]
// Re-exported for compatibility with callers that still name the legacy IVF scaffold.
pub use ivf::IvfIndex;
pub use ivf_flat::IvfFlatIndex;
pub use ivf_flat_cc::IvfFlatCcIndex;
pub use ivf_opq::{IvfOpqConfig, IvfOpqIndex, IvfOpqIndexWrapper};
pub use ivf_usq::{IvfUsqConfig, IvfUsqIndex};

pub use ivf_sq8::IvfSq8Index;
pub use ivf_sq_cc::IvfSqCcIndex;
pub use ivf_turboquant::{IvfTurboQuantConfig, IvfTurboQuantIndex};
pub use ivfpq::IvfPqIndex;
pub use mem_index::MemIndex;
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
pub mod exrabitq_ffi;
